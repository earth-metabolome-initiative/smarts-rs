//! `RDKit`-grounded red tests for the next stereo gap slice.

use core::str::FromStr;

use serde::Deserialize;
use smarts_parser::QueryMol;
use smarts_validator::matches;

#[derive(Debug, Deserialize)]
struct RdkitStereoGapCase {
    name: String,
    smarts: String,
    smiles: String,
    expected_match: bool,
}

fn load_cases() -> Vec<RdkitStereoGapCase> {
    serde_json::from_str(include_str!(
        "../../../corpus/validator/stereo-gap-v3.rdkit.json"
    ))
    .expect("valid RDKit stereo gap v3 fixture")
}

#[test]
fn stereo_gap_v3_cases_should_agree_with_frozen_rdkit_fixture() {
    let cases = load_cases();
    for case in cases {
        let query = QueryMol::from_str(&case.smarts)
            .unwrap_or_else(|error| panic!("failed to parse SMARTS {:?}: {error}", case.smarts));
        let actual = matches(&query, &case.smiles)
            .unwrap_or_else(|error| panic!("{}: validator error: {error}", case.name));
        assert_eq!(
            actual, case.expected_match,
            "{}: SMARTS {:?} against SMILES {:?}",
            case.name, case.smarts, case.smiles
        );
    }
}
