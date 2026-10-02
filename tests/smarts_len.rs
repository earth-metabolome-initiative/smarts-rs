//! Confirms `QueryMol::smarts_len` equals `to_string().len()` across the shared
//! rendering corpora, including recursive and multi-component queries, in both
//! parsed and canonical forms.

use serde::Deserialize;
use smarts_rs::QueryMol;
use std::str::FromStr;

#[derive(Debug, Deserialize)]
struct Case {
    id: String,
    smarts: String,
}

fn load_cases(file_name: &str) -> Vec<Case> {
    let path = format!("{}/corpus/{}", env!("CARGO_MANIFEST_DIR"), file_name);
    let raw =
        std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("failed to read {path}: {err}"));
    serde_json::from_str(&raw).unwrap_or_else(|err| panic!("failed to parse {path}: {err}"))
}

#[test]
fn smarts_len_matches_to_string_len() {
    let mut cases = load_cases("parser/parse-valid-v0.json");
    cases.extend(load_cases("benchmark/parse-extra-cases.json"));
    assert!(!cases.is_empty(), "rendering corpus should not be empty");

    for case in cases {
        let query = QueryMol::from_str(&case.smarts)
            .unwrap_or_else(|err| panic!("failed to parse `{}` ({}): {err}", case.smarts, case.id));

        assert_eq!(
            query.smarts_len(),
            query.to_string().len(),
            "case {} `{}`",
            case.id,
            case.smarts
        );

        let canonical = query.canonicalize();
        assert_eq!(
            canonical.smarts_len(),
            canonical.to_string().len(),
            "case {} `{}` (canonical)",
            case.id,
            case.smarts
        );
    }
}
