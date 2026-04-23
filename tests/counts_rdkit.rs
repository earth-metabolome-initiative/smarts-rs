//! RDKit-grounded connectivity and valence SMARTS parity tests.

use core::str::FromStr;

use serde::Deserialize;
use smarts_rs::QueryMol;

#[derive(Debug, Deserialize)]
struct RdkitCountsCase {
    name: String,
    smarts: String,
    smiles: String,
    expected_match: bool,
}

fn load_cases() -> Vec<RdkitCountsCase> {
    serde_json::from_str(include_str!("../corpus/matching/counts-v0.rdkit.json"))
        .expect("valid RDKit counts fixture")
}

#[test]
fn counts_matches_agree_with_frozen_rdkit_fixture() {
    let cases = load_cases();
    for case in cases {
        let query = QueryMol::from_str(&case.smarts)
            .unwrap_or_else(|error| panic!("failed to parse SMARTS {:?}: {error}", case.smarts));
        let actual = query
            .matches(&case.smiles)
            .unwrap_or_else(|error| panic!("{}: matcher error: {error}", case.name));
        assert_eq!(
            actual, case.expected_match,
            "{}: SMARTS {:?} against SMILES {:?}",
            case.name, case.smarts, case.smiles
        );
    }
}
