//! RDKit-frozen counted-match parity tests.

use core::str::FromStr;

use smarts_parser::QueryMol;
use smarts_validator::{match_count, matches, substructure_matches};

#[derive(Debug, Clone, Copy)]
struct CountCase {
    name: &'static str,
    smarts: &'static str,
    smiles: &'static str,
    expected_count: usize,
    expected_matches: &'static [&'static [usize]],
}

const CASES: &[CountCase] = &[
    CountCase {
        name: "symmetric_edge_dedupes_to_one_embedding",
        smarts: "CC",
        smiles: "CC",
        expected_count: 1,
        expected_matches: &[&[0, 1]],
    },
    CountCase {
        name: "overlapping_edges_are_counted",
        smarts: "CC",
        smiles: "CCC",
        expected_count: 2,
        expected_matches: &[&[0, 1], &[1, 2]],
    },
    CountCase {
        name: "branched_pattern_counts_three_unique_embeddings",
        smarts: "C(C)C",
        smiles: "CC(C)C",
        expected_count: 3,
        expected_matches: &[&[1, 0, 2], &[1, 0, 3], &[1, 2, 3]],
    },
    CountCase {
        name: "disconnected_two_atom_query_dedupes_symmetric_mapping",
        smarts: "C.C",
        smiles: "CC",
        expected_count: 1,
        expected_matches: &[&[0, 1]],
    },
    CountCase {
        name: "disconnected_query_counts_all_unique_pairs",
        smarts: "C.C",
        smiles: "C.C.C",
        expected_count: 3,
        expected_matches: &[&[0, 1], &[0, 2], &[1, 2]],
    },
    CountCase {
        name: "disconnected_query_can_match_connected_target",
        smarts: "C.C",
        smiles: "CCC",
        expected_count: 3,
        expected_matches: &[&[0, 1], &[0, 2], &[1, 2]],
    },
    CountCase {
        name: "three_component_query_dedupes_all_component_permutations",
        smarts: "C.C.C",
        smiles: "CCC",
        expected_count: 1,
        expected_matches: &[&[0, 1, 2]],
    },
    CountCase {
        name: "symmetric_path_pattern_dedupes_reversed_mapping",
        smarts: "*C*",
        smiles: "CCC",
        expected_count: 1,
        expected_matches: &[&[0, 1, 2]],
    },
    CountCase {
        name: "any_bond_path_dedupes_valence_distinct_ring_traversals",
        smarts: "C~C~C~C",
        smiles: "C1CCC=1",
        expected_count: 1,
        expected_matches: &[&[0, 1, 2, 3]],
    },
    CountCase {
        name: "single_atom_hetero_query_counts_each_embedding",
        smarts: "[#8]",
        smiles: "O=CO",
        expected_count: 2,
        expected_matches: &[&[0], &[2]],
    },
    CountCase {
        name: "asymmetric_query_keeps_query_order_mapping",
        smarts: "C(O)N",
        smiles: "NCO",
        expected_count: 1,
        expected_matches: &[&[1, 2, 0]],
    },
    CountCase {
        name: "query_order_is_preserved_for_longer_asymmetric_matches",
        smarts: "NCO",
        smiles: "CCOCN",
        expected_count: 1,
        expected_matches: &[&[4, 3, 2]],
    },
    CountCase {
        name: "query_order_is_preserved_for_multiple_carboxyl_matches",
        smarts: "C(=O)O",
        smiles: "OC(=O)CC(=O)O",
        expected_count: 2,
        expected_matches: &[&[1, 2, 0], &[4, 5, 6]],
    },
    CountCase {
        name: "recursive_ring_query_counts_outer_anchor_atoms",
        smarts: "[$(c1ccccc1)]",
        smiles: "c1ccccc1",
        expected_count: 6,
        expected_matches: &[&[0], &[1], &[2], &[3], &[4], &[5]],
    },
    CountCase {
        name: "recursive_small_ring_query_counts_all_ring_anchors",
        smarts: "[$(*1~[CH2]~[CH2]1)]",
        smiles: "C1CC1",
        expected_count: 3,
        expected_matches: &[&[0], &[1], &[2]],
    },
    CountCase {
        name: "recursive_linear_anchor_query_counts_terminal_anchors_only",
        smarts: "[$(*~[CH2]~[CH2]~*)]",
        smiles: "CCCC",
        expected_count: 2,
        expected_matches: &[&[0], &[3]],
    },
    CountCase {
        name: "symmetric_ring_query_dedupes_full_automorphism_class",
        smarts: "c1ccccc1",
        smiles: "c1ccccc1",
        expected_count: 1,
        expected_matches: &[&[0, 1, 2, 3, 4, 5]],
    },
];

#[test]
fn counted_matches_agree_with_frozen_rdkit_examples() {
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
        let expected_matches: Vec<Box<[usize]>> = case
            .expected_matches
            .iter()
            .map(|mapping| Box::<[usize]>::from(*mapping))
            .collect();
        assert_eq!(
            matches_out.as_ref(),
            expected_matches.as_slice(),
            "{}: materialized matches differ for SMARTS {:?} against SMILES {:?}",
            case.name,
            case.smarts,
            case.smiles
        );
        assert_eq!(
            matches(&query, case.smiles).unwrap(),
            count > 0,
            "{}: boolean/count mismatch",
            case.name
        );
    }
}
