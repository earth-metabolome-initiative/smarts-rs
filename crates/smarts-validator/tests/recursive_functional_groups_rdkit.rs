//! RDKit-frozen recursive functional-group SMARTS parity tests.

use core::str::FromStr;

use smarts_parser::QueryMol;
use smarts_validator::{match_count, matches, substructure_matches};

#[derive(Debug, Clone, Copy)]
struct RecursiveCase {
    name: &'static str,
    smarts: &'static str,
    smiles: &'static str,
    expected_count: usize,
    expected_matches: &'static [&'static [usize]],
}

const CASES: &[RecursiveCase] = &[
    RecursiveCase {
        name: "alpha_amino_recursive_query_matches_cysteine_like_target",
        smarts: "[$([NX3!H0][CX4][CX3](=[OX1])[OD1])]",
        smiles: "NC(CS)C(O)=O",
        expected_count: 1,
        expected_matches: &[&[0]],
    },
    RecursiveCase {
        name: "alpha_amino_recursive_query_matches_glycine_like_target",
        smarts: "[$([NX3!H0][CX4][CX3](=[OX1])[OD1])]",
        smiles: "O=C(O)CN",
        expected_count: 1,
        expected_matches: &[&[4]],
    },
    RecursiveCase {
        name: "alpha_amino_recursive_query_rejects_simple_alcohol",
        smarts: "[$([NX3!H0][CX4][CX3](=[OX1])[OD1])]",
        smiles: "CCO",
        expected_count: 0,
        expected_matches: &[],
    },
    RecursiveCase {
        name: "alpha_amino_carboxyl_query_matches_cysteine_like_target",
        smarts: "[$([CX3](=[OX1])[CX4][NX3!H0])]-[OD1]",
        smiles: "NC(CS)C(O)=O",
        expected_count: 1,
        expected_matches: &[&[4, 5]],
    },
    RecursiveCase {
        name: "alpha_amino_carboxyl_query_matches_glycine_like_target",
        smarts: "[$([CX3](=[OX1])[CX4][NX3!H0])]-[OD1]",
        smiles: "O=C(O)CN",
        expected_count: 1,
        expected_matches: &[&[1, 2]],
    },
    RecursiveCase {
        name: "alpha_amino_carboxyl_query_rejects_simple_acid",
        smarts: "[$([CX3](=[OX1])[CX4][NX3!H0])]-[OD1]",
        smiles: "CC(=O)O",
        expected_count: 0,
        expected_matches: &[],
    },
    RecursiveCase {
        name: "carboxylic_acid_query_matches_neutral_acid",
        smarts: "C(=O)[O;H,-]",
        smiles: "CC(=O)O",
        expected_count: 1,
        expected_matches: &[&[1, 2, 3]],
    },
    RecursiveCase {
        name: "carboxylic_acid_query_matches_carboxylate",
        smarts: "C(=O)[O;H,-]",
        smiles: "CC(=O)[O-]",
        expected_count: 1,
        expected_matches: &[&[1, 2, 3]],
    },
    RecursiveCase {
        name: "carboxylic_acid_query_rejects_ester",
        smarts: "C(=O)[O;H,-]",
        smiles: "CC(=O)OC",
        expected_count: 0,
        expected_matches: &[],
    },
    RecursiveCase {
        name: "carboxylic_acid_query_counts_both_acid_and_carboxylate_sites",
        smarts: "C(=O)[O;H,-]",
        smiles: "O=C([O-])C(=O)O",
        expected_count: 2,
        expected_matches: &[&[1, 0, 2], &[3, 4, 5]],
    },
];

#[test]
fn recursive_functional_group_queries_should_agree_with_rdkit() {
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
