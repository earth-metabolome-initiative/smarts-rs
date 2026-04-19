//! Additional RDKit-frozen Daylight-style SMARTS examples.

use core::str::FromStr;

use smarts_parser::QueryMol;
use smarts_validator::{match_count, matches, substructure_matches};

#[derive(Debug, Clone, Copy)]
struct ExampleCase {
    name: &'static str,
    smarts: &'static str,
    smiles: &'static str,
    expected_count: usize,
}

const CASES: &[ExampleCase] = &[
    ExampleCase {
        name: "kekule_smarts_does_not_match_benzene_target",
        smarts: "C1=CC=CC=C1",
        smiles: "c1ccccc1",
        expected_count: 0,
    },
    ExampleCase {
        name: "kekule_smarts_matches_nonaromatic_cationic_ring",
        smarts: "C1=CC=CC=C1",
        smiles: "C1=CC=CC=[CH+]1",
        expected_count: 1,
    },
    ExampleCase {
        name: "grouped_disconnected_query_matches_separate_components",
        smarts: "([Cl]).([c])",
        smiles: "Cl.c1ccccc1",
        expected_count: 6,
    },
    ExampleCase {
        name: "grouped_disconnected_query_rejects_same_component_match",
        smarts: "([Cl]).([c])",
        smiles: "Clc1ccccc1",
        expected_count: 0,
    },
    ExampleCase {
        name: "simple_and_normalization_matches_cationic_carbon",
        smarts: "[C&+]",
        smiles: "[CH3+]",
        expected_count: 1,
    },
    ExampleCase {
        name: "simple_charge_query_matches_cationic_carbon",
        smarts: "[C+]",
        smiles: "[CH3+]",
        expected_count: 1,
    },
    ExampleCase {
        name: "simple_and_normalization_rejects_neutral_carbon",
        smarts: "[C&+]",
        smiles: "C",
        expected_count: 0,
    },
    ExampleCase {
        name: "simple_charge_query_rejects_neutral_carbon",
        smarts: "[C+]",
        smiles: "C",
        expected_count: 0,
    },
];

#[test]
fn additional_daylight_examples_should_agree_with_rdkit() {
    for case in CASES {
        let query = QueryMol::from_str(case.smarts)
            .unwrap_or_else(|error| panic!("{}: failed to parse SMARTS: {error}", case.name));
        let count = match_count(&query, case.smiles)
            .unwrap_or_else(|error| panic!("{}: validator error: {error}", case.name));
        let matches_out = substructure_matches(&query, case.smiles)
            .unwrap_or_else(|error| panic!("{}: validator error: {error}", case.name));

        assert_eq!(
            count, case.expected_count,
            "{}: SMARTS {:?} against SMILES {:?}",
            case.name, case.smarts, case.smiles
        );
        assert_eq!(
            matches_out.len(),
            count,
            "{}: materialized match count mismatch",
            case.name
        );
        assert_eq!(
            matches(&query, case.smiles).unwrap(),
            count > 0,
            "{}: boolean/count mismatch",
            case.name
        );
    }
}
