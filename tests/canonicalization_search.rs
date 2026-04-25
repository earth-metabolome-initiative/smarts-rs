//! Broad canonicalization invariant search over corpus and generated SMARTS.

use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    str::FromStr,
};

use serde_json::Value;
use smarts_rs::{
    AtomExpr, AtomPrimitive, BondExpr, BondExprTree, BondPrimitive, BracketExprTree, QueryAtom,
    QueryBond, QueryMol,
};
use smiles_parser::bond::Bond;

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

#[test]
#[ignore = "deterministic stress search for local canonicalization investigation"]
fn generated_smarts_satisfy_canonicalization_invariants() {
    let mut cases = BTreeSet::new();
    collect_generated_invariant_cases(&mut cases);

    let mut checked = 0usize;
    for smarts in cases {
        let Ok(query) = QueryMol::from_str(&smarts) else {
            continue;
        };
        assert_canonicalization_is_stable(&smarts, &query);
        checked += 1;
    }

    assert!(
        checked > 5_000,
        "stress search should check many generated SMARTS; checked {checked}"
    );
    eprintln!("checked {checked} generated canonicalization candidates");
}

#[test]
#[ignore = "deterministic stress search for local canonicalization investigation"]
fn corpus_relabelings_keep_canonicalization_stable() {
    let mut cases = BTreeSet::new();
    collect_corpus_smarts(&mut cases);

    let mut checked = 0usize;
    for smarts in cases {
        let Ok(query) = QueryMol::from_str(&smarts) else {
            continue;
        };
        if query.atom_count() <= 1 || query.atom_count() > 20 || query_contains_chirality(&query) {
            continue;
        }

        let canonical = query.canonicalize();
        for order in relabel_orders(query.atom_count()) {
            let relabeled = relabel_query(&query, &order);
            assert_eq!(
                canonical,
                relabeled.canonicalize(),
                "atom relabeling changed canonical form for {smarts:?}: {order:?}"
            );
            checked += 1;
        }
    }

    assert!(
        checked > 1_000,
        "relabel search should check many corpus relabelings; checked {checked}"
    );
    eprintln!("checked {checked} corpus relabeling canonicalization candidates");
}

#[test]
#[ignore = "deterministic stress search for local canonicalization investigation"]
fn generated_relabelings_keep_canonicalization_stable() {
    let mut cases = BTreeSet::new();
    collect_generated_invariant_cases(&mut cases);

    let mut checked = 0usize;
    for smarts in cases {
        let Ok(query) = QueryMol::from_str(&smarts) else {
            continue;
        };
        if query.atom_count() <= 1 || query.atom_count() > 12 || query_contains_chirality(&query) {
            continue;
        }

        let canonical = query.canonicalize();
        for order in relabel_orders(query.atom_count()) {
            let relabeled = relabel_query(&query, &order);
            assert_eq!(
                canonical,
                relabeled.canonicalize(),
                "atom relabeling changed canonical form for {smarts:?}: {order:?}"
            );
            checked += 1;
        }
    }

    assert!(
        checked > 10_000,
        "generated relabel search should check many relabelings; checked {checked}"
    );
    eprintln!("checked {checked} generated relabeling canonicalization candidates");
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

fn collect_generated_invariant_cases(cases: &mut BTreeSet<String>) {
    let atoms = generated_atoms();
    let pair_atoms = generated_pair_atoms();
    let chain_atoms = generated_chain_atoms();
    let bonds = ["", "-", "=", "#", "~", ":", "@", "-,=", "-;@", "!,=", "!@"];

    for &atom in atoms {
        cases.insert(atom.to_owned());
    }

    for &left in pair_atoms {
        for &right in pair_atoms {
            for bond in bonds {
                if bond.is_empty() && ambiguous_elided_pair(left, right) {
                    continue;
                }
                cases.insert(format!("{left}{bond}{right}"));
            }
        }
    }

    for &first in chain_atoms {
        for &second in chain_atoms {
            for &third in chain_atoms {
                cases.insert(format!("{first}-{second}-{third}"));
                cases.insert(format!("{first}({second}){third}"));
                cases.insert(format!("({first}.{second}).{third}"));
                cases.insert(format!("{first}.{second}.{third}"));
            }
        }
    }

    for &first in chain_atoms {
        for &second in chain_atoms {
            for &third in chain_atoms {
                cases.insert(format!("{first}1{second}{third}1"));
                cases.insert(format!("{first}-1{second}-{third}-1"));
                cases.insert(format!("{first}1({second}){third}1"));
            }
        }
    }

    for &left in chain_atoms {
        for &right in chain_atoms {
            cases.insert(format!("[$({left}-{right})]"));
            cases.insert(format!("[!$({left}-{right})]"));
            cases.insert(format!("[$({left}.{right})]"));
            cases.insert(format!("[$([{left}]-{right})]"));
        }
    }

    for base in [
        "C", "N", "O", "S", "#6", "#7", "#8", "#16", "H", "h", "D2", "X3", "v4", "R", "r5", "x2",
        "+", "-", "+0", "!#1", "!H0", "A", "a",
    ] {
        for qualifier in [
            "H", "H0", "H1", "D1", "D2", "X1", "X2", "R", "R0", "r5", "x1", "+", "-", "!#6", "!H0",
            "A", "a",
        ] {
            cases.insert(format!("[{base}&{qualifier}]"));
            cases.insert(format!("[{base};{qualifier}]"));
            cases.insert(format!("[{base},{qualifier}]"));
            cases.insert(format!("[!{base},{qualifier}]"));
        }
    }
}

const fn generated_atoms() -> &'static [&'static str] {
    &[
        "*",
        "C",
        "N",
        "O",
        "S",
        "P",
        "F",
        "Cl",
        "Br",
        "I",
        "B",
        "c",
        "n",
        "o",
        "s",
        "p",
        "[#1]",
        "[#6]",
        "[#7]",
        "[#8]",
        "[#15]",
        "[#16]",
        "[H]",
        "[H+]",
        "[H-]",
        "[2H]",
        "[2H+]",
        "[3H-]",
        "[H0]",
        "[H1]",
        "[h]",
        "[h0]",
        "[He]",
        "[Hf]",
        "[Hg]",
        "[Ho]",
        "[Ra]",
        "[Rg]",
        "[C;H1]",
        "[C&H1]",
        "[C,N;H1]",
        "[N,O]",
        "[!#1]",
        "[!H]",
        "[!H0]",
        "[D{2-4}]",
        "[X{1-2}]",
        "[v{-0}]",
        "[R{1-}]",
        "[r{5-6}]",
        "[x{1-2}]",
        "[z{1-}]",
        "[Z{-2}]",
        "[+]",
        "[-]",
        "[+2]",
        "[-2]",
        "[+0]",
        "[!+0]",
        "[C+]",
        "[N-]",
        "[O-]",
        "[Na+]",
        "[Cl-]",
        "[C:1]",
        "[#6:2]",
        "[$(CO)]",
        "[$([OH])]",
    ]
}

const fn generated_pair_atoms() -> &'static [&'static str] {
    &[
        "*", "C", "N", "O", "S", "c", "n", "[#1]", "[#6]", "[#7]", "[H]", "[H+]", "[2H]", "[He]",
        "[C;H1]", "[N,O]", "[!#1]", "[D{2-4}]", "[+]", "[C:1]",
    ]
}

const fn generated_chain_atoms() -> &'static [&'static str] {
    &[
        "*", "C", "N", "O", "S", "c", "n", "[#1]", "[#6]", "[H]", "[H+]", "[C;H1]",
    ]
}

fn ambiguous_elided_pair(left: &str, right: &str) -> bool {
    left.bytes()
        .last()
        .is_some_and(|byte| byte.is_ascii_alphabetic())
        && right
            .bytes()
            .next()
            .is_some_and(|byte| byte.is_ascii_alphabetic())
}

fn relabel_orders(atom_count: usize) -> Vec<Vec<usize>> {
    let mut orders = Vec::new();
    orders.push((0..atom_count).rev().collect());

    let mut rotated = (0..atom_count).collect::<Vec<_>>();
    rotated.rotate_left(1);
    orders.push(rotated);

    let mut interleaved = Vec::with_capacity(atom_count);
    for atom_id in (0..atom_count).step_by(2) {
        interleaved.push(atom_id);
    }
    for atom_id in (1..atom_count).step_by(2) {
        interleaved.push(atom_id);
    }
    orders.push(interleaved);

    orders
}

fn relabel_query(query: &QueryMol, order: &[usize]) -> QueryMol {
    let mut new_index_of_old_atom = vec![usize::MAX; query.atom_count()];
    for (new_index, old_atom) in order.iter().copied().enumerate() {
        new_index_of_old_atom[old_atom] = new_index;
    }

    let mut new_component_of_old = vec![usize::MAX; query.component_count()];
    let mut next_component = 0usize;
    for &old_atom in order {
        let old_component = query.atoms()[old_atom].component;
        if new_component_of_old[old_component] == usize::MAX {
            new_component_of_old[old_component] = next_component;
            next_component += 1;
        }
    }

    let atoms = order
        .iter()
        .copied()
        .map(|old_atom| {
            let atom = &query.atoms()[old_atom];
            QueryAtom {
                id: new_index_of_old_atom[old_atom],
                component: new_component_of_old[atom.component],
                expr: atom.expr.clone(),
            }
        })
        .collect::<Vec<_>>();

    let mut bonds = query
        .bonds()
        .iter()
        .map(|bond| {
            let src = new_index_of_old_atom[bond.src];
            let dst = new_index_of_old_atom[bond.dst];
            let expr = if src <= dst {
                bond.expr.clone()
            } else {
                flipped_directional_bond_expr(&bond.expr)
            };
            let (src, dst) = if src <= dst { (src, dst) } else { (dst, src) };
            (src, dst, expr)
        })
        .collect::<Vec<_>>();
    bonds.sort_unstable_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then(left.1.cmp(&right.1))
            .then(left.2.cmp(&right.2))
    });
    let bonds = bonds
        .into_iter()
        .enumerate()
        .map(|(bond_id, (src, dst, expr))| QueryBond {
            id: bond_id,
            src,
            dst,
            expr,
        })
        .collect::<Vec<_>>();

    let mut component_groups = vec![None; query.component_count()];
    for (old_component, &new_component) in new_component_of_old.iter().enumerate() {
        component_groups[new_component] = query.component_group(old_component);
    }

    QueryMol::from_parts(atoms, bonds, query.component_count(), component_groups)
}

fn flipped_directional_bond_expr(expr: &BondExpr) -> BondExpr {
    match expr {
        BondExpr::Elided => BondExpr::Elided,
        BondExpr::Query(tree) => BondExpr::Query(flipped_directional_bond_tree(tree)),
    }
}

fn flipped_directional_bond_tree(tree: &BondExprTree) -> BondExprTree {
    match tree {
        BondExprTree::Primitive(primitive) => {
            BondExprTree::Primitive(flip_bond_primitive(*primitive))
        }
        BondExprTree::Not(inner) => {
            BondExprTree::Not(Box::new(flipped_directional_bond_tree(inner)))
        }
        BondExprTree::HighAnd(items) => {
            BondExprTree::HighAnd(items.iter().map(flipped_directional_bond_tree).collect())
        }
        BondExprTree::Or(items) => {
            BondExprTree::Or(items.iter().map(flipped_directional_bond_tree).collect())
        }
        BondExprTree::LowAnd(items) => {
            BondExprTree::LowAnd(items.iter().map(flipped_directional_bond_tree).collect())
        }
    }
}

const fn flip_bond_primitive(primitive: BondPrimitive) -> BondPrimitive {
    match primitive {
        BondPrimitive::Bond(Bond::Up) => BondPrimitive::Bond(Bond::Down),
        BondPrimitive::Bond(Bond::Down) => BondPrimitive::Bond(Bond::Up),
        other => other,
    }
}

fn query_contains_chirality(query: &QueryMol) -> bool {
    query.atoms().iter().any(|atom| match &atom.expr {
        AtomExpr::Wildcard | AtomExpr::Bare { .. } => false,
        AtomExpr::Bracket(bracket) => tree_contains_chirality(&bracket.tree),
    })
}

fn tree_contains_chirality(tree: &BracketExprTree) -> bool {
    match tree {
        BracketExprTree::Primitive(AtomPrimitive::Chirality(_)) => true,
        BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(query)) => {
            query_contains_chirality(query)
        }
        BracketExprTree::Primitive(_) => false,
        BracketExprTree::Not(inner) => tree_contains_chirality(inner),
        BracketExprTree::HighAnd(items)
        | BracketExprTree::Or(items)
        | BracketExprTree::LowAnd(items) => items.iter().any(tree_contains_chirality),
    }
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}
