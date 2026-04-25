//! Broad canonicalization invariant search over corpus and generated SMARTS.

use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    str::FromStr,
};

use serde_json::Value;
use smarts_rs::QueryMol;

#[test]
fn corpus_smarts_satisfy_canonicalization_invariants() {
    let mut cases = BTreeSet::new();
    collect_corpus_smarts(&mut cases);

    assert!(
        !cases.is_empty(),
        "canonicalization search should collect corpus SMARTS"
    );

    for smarts in cases {
        let query = QueryMol::from_str(&smarts)
            .unwrap_or_else(|error| panic!("corpus SMARTS should parse: {smarts:?}: {error}"));
        assert_canonicalization_is_stable(&smarts, &query);
    }
}

#[test]
fn generated_equivalent_smarts_converge() {
    let mut groups = Vec::new();
    collect_generated_equivalence_groups(&mut groups);

    assert!(
        !groups.is_empty(),
        "canonicalization search should generate equivalence groups"
    );

    for group in groups {
        let expected = canonical_string(&group[0]);
        for candidate in group.iter().skip(1) {
            assert_eq!(
                expected,
                canonical_string(candidate),
                "generated equivalence group diverged: {group:?}"
            );
        }
    }
}

fn assert_canonicalization_is_stable(source: &str, query: &QueryMol) {
    let canonical = query.canonicalize();
    assert_eq!(
        query.atom_count(),
        canonical.atom_count(),
        "canonicalization changed atom count for {source:?}"
    );
    assert_eq!(
        query.bond_count(),
        canonical.bond_count(),
        "canonicalization changed bond count for {source:?}"
    );
    assert_eq!(
        query.component_count(),
        canonical.component_count(),
        "canonicalization changed component count for {source:?}"
    );
    assert!(
        canonical.is_canonical(),
        "canonical result is not stable for {source:?}: {canonical}"
    );
    assert_eq!(
        canonical,
        canonical.canonicalize(),
        "canonicalize is not idempotent for {source:?}"
    );

    let canonical_smarts = query.to_canonical_smarts();
    assert_eq!(
        canonical_smarts,
        canonical.to_string(),
        "canonical string and canonical query display diverged for {source:?}"
    );

    let reparsed = QueryMol::from_str(&canonical_smarts).unwrap_or_else(|error| {
        panic!("canonical SMARTS should parse again for {source:?}: {canonical_smarts:?}: {error}")
    });
    let recanonicalized = reparsed.canonicalize();
    assert!(
        recanonicalized.is_canonical(),
        "reparsed canonical SMARTS is not stable for {source:?}: {canonical_smarts:?}"
    );
    assert_eq!(
        canonical, recanonicalized,
        "canonical SMARTS roundtrip changed canonical query for {source:?}"
    );
    assert_eq!(
        canonical_smarts,
        recanonicalized.to_string(),
        "canonical SMARTS roundtrip changed canonical text for {source:?}"
    );

    let displayed = query.to_string();
    let displayed_query = QueryMol::from_str(&displayed).unwrap_or_else(|error| {
        panic!("displayed SMARTS should parse again for {source:?}: {displayed:?}: {error}")
    });
    assert_eq!(
        canonical,
        displayed_query.canonicalize(),
        "display roundtrip changed canonical query for {source:?}: {displayed:?}"
    );
}

fn canonical_string(smarts: &str) -> String {
    let query = QueryMol::from_str(smarts)
        .unwrap_or_else(|error| panic!("generated SMARTS should parse: {smarts:?}: {error}"));
    assert_canonicalization_is_stable(smarts, &query);
    query.to_canonical_smarts()
}

fn collect_corpus_smarts(cases: &mut BTreeSet<String>) {
    let root = repo_root().join("corpus");
    collect_corpus_smarts_from_dir(&root, cases);
}

fn collect_corpus_smarts_from_dir(dir: &Path, cases: &mut BTreeSet<String>) {
    let mut entries = fs::read_dir(dir)
        .unwrap_or_else(|error| {
            panic!("failed to read corpus directory {}: {error}", dir.display())
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_else(|error| {
            panic!("failed to read corpus directory {}: {error}", dir.display())
        });
    entries.sort_by_key(fs::DirEntry::path);

    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            collect_corpus_smarts_from_dir(&path, cases);
            continue;
        }

        match path.extension().and_then(|extension| extension.to_str()) {
            Some("json") => collect_json_smarts(&path, cases),
            Some("smarts") => collect_line_smarts(&path, cases),
            _ => {}
        }
    }
}

fn collect_json_smarts(path: &Path, cases: &mut BTreeSet<String>) {
    if path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.contains("invalid"))
    {
        return;
    }

    let raw = fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read corpus JSON {}: {error}", path.display()));
    let value: Value = serde_json::from_str(&raw)
        .unwrap_or_else(|error| panic!("failed to parse corpus JSON {}: {error}", path.display()));
    collect_json_smarts_value(&value, cases);
}

fn collect_json_smarts_value(value: &Value, cases: &mut BTreeSet<String>) {
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                if key == "smarts" {
                    if let Some(smarts) = value.as_str() {
                        cases.insert(smarts.to_owned());
                    }
                    continue;
                }
                if key == "members" {
                    if let Some(members) = value.as_array() {
                        for member in members.iter().filter_map(Value::as_str) {
                            cases.insert(member.to_owned());
                        }
                    }
                    continue;
                }
                collect_json_smarts_value(value, cases);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_json_smarts_value(item, cases);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {}
    }
}

fn collect_line_smarts(path: &Path, cases: &mut BTreeSet<String>) {
    let raw = fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read SMARTS corpus {}: {error}", path.display()));
    for line in raw.lines() {
        let smarts = line.trim();
        if !smarts.is_empty() && !smarts.starts_with('#') {
            cases.insert(smarts.to_owned());
        }
    }
}

fn collect_generated_equivalence_groups(groups: &mut Vec<Vec<String>>) {
    let atoms = [
        "C", "N", "O", "S", "c", "n", "*", "[#6]", "[#7]", "[#8]", "[C;H1]", "[N,O]", "[C,N;H1]",
        "[!#1]", "[R]", "[r5]", "[+]", "[-]",
    ];
    let bonds = ["-", "=", "#", "~", ":", "@", "-,=", "-;@"];

    for left in atoms {
        for right in atoms {
            for bond in bonds {
                groups.push(vec![
                    format!("{left}{bond}{right}"),
                    format!("{right}{bond}{left}"),
                ]);
            }
        }
    }

    for center in atoms {
        for first in atoms {
            for second in atoms {
                groups.push(vec![
                    format!("{center}({first}){second}"),
                    format!("{center}({second}){first}"),
                ]);
            }
        }
    }

    for first in atoms {
        for second in atoms {
            groups.push(vec![
                format!("{first}.{second}"),
                format!("{second}.{first}"),
            ]);
            groups.push(vec![
                format!("({first}.{second}).C"),
                format!("C.({second}.{first})"),
            ]);
        }
    }

    for left in ["C", "N", "O", "c", "n", "[#6]", "[#7]", "[#8]"] {
        for right in ["C", "N", "O", "c", "n", "[#6]", "[#7]", "[#8]"] {
            groups.push(vec![
                format!("[$({left}-{right})]"),
                format!("[$({right}-{left})]"),
            ]);
        }
    }
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}
