//! RDKit-frozen counted-match parity tests.

use core::str::FromStr;

use smarts_parser::QueryMol;
use smarts_validator::{match_count, matches};

#[derive(Debug, Clone, Copy)]
struct CountCase {
    name: &'static str,
    smarts: &'static str,
    smiles: &'static str,
    expected_count: usize,
}

const CASES: &[CountCase] = &[
    CountCase {
        name: "symmetric_edge_dedupes_to_one_embedding",
        smarts: "CC",
        smiles: "CC",
        expected_count: 1,
    },
    CountCase {
        name: "overlapping_edges_are_counted",
        smarts: "CC",
        smiles: "CCC",
        expected_count: 2,
    },
    CountCase {
        name: "branched_pattern_counts_three_unique_embeddings",
        smarts: "C(C)C",
        smiles: "CC(C)C",
        expected_count: 3,
    },
    CountCase {
        name: "disconnected_two_atom_query_dedupes_symmetric_mapping",
        smarts: "C.C",
        smiles: "CC",
        expected_count: 1,
    },
    CountCase {
        name: "disconnected_query_counts_all_unique_pairs",
        smarts: "C.C",
        smiles: "C.C.C",
        expected_count: 3,
    },
    CountCase {
        name: "symmetric_path_pattern_dedupes_reversed_mapping",
        smarts: "*C*",
        smiles: "CCC",
        expected_count: 1,
    },
    CountCase {
        name: "single_atom_hetero_query_counts_each_embedding",
        smarts: "[#8]",
        smiles: "O=CO",
        expected_count: 2,
    },
    CountCase {
        name: "asymmetric_query_keeps_query_order_mapping",
        smarts: "C(O)N",
        smiles: "NCO",
        expected_count: 1,
    },
];

#[test]
fn counted_matches_agree_with_frozen_rdkit_examples() {
    for case in CASES {
        let query = QueryMol::from_str(case.smarts)
            .unwrap_or_else(|error| panic!("{}: failed to parse SMARTS: {error}", case.name));
        let count = match_count(&query, case.smiles)
            .unwrap_or_else(|error| panic!("{}: validator error: {error}", case.name));
        assert_eq!(
            count, case.expected_count,
            "{}: SMARTS {:?} against SMILES {:?}",
            case.name, case.smarts, case.smiles
        );
        assert_eq!(
            matches(&query, case.smiles).unwrap(),
            count > 0,
            "{}: boolean/count mismatch",
            case.name
        );
    }
}
