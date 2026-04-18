//! RDKit-frozen SMARTS boolean precedence parity tests.

use core::str::FromStr;

use smarts_parser::QueryMol;
use smarts_validator::{match_count, matches, substructure_matches};

#[derive(Debug, Clone, Copy)]
struct PrecedenceCase {
    name: &'static str,
    smarts: &'static str,
    smiles: &'static str,
    expected_count: usize,
    expected_matches: &'static [&'static [usize]],
}

const CASES: &[PrecedenceCase] = &[
    PrecedenceCase {
        name: "high_precedence_and_or_matches_nitrogen",
        smarts: "[N,O&H1]",
        smiles: "N",
        expected_count: 1,
        expected_matches: &[&[0]],
    },
    PrecedenceCase {
        name: "high_precedence_and_or_rejects_neutral_water_oxygen",
        smarts: "[N,O&H1]",
        smiles: "O",
        expected_count: 0,
        expected_matches: &[],
    },
    PrecedenceCase {
        name: "high_precedence_and_or_matches_hydroxyl_oxygen",
        smarts: "[N,O&H1]",
        smiles: "CO",
        expected_count: 1,
        expected_matches: &[&[1]],
    },
    PrecedenceCase {
        name: "low_precedence_and_or_requires_hydrogen_on_both_branches_and_rejects_bare_nitrogen",
        smarts: "[N,O;H1]",
        smiles: "N",
        expected_count: 0,
        expected_matches: &[],
    },
    PrecedenceCase {
        name: "low_precedence_and_or_requires_hydrogen_on_both_branches_and_matches_hydroxyl_oxygen",
        smarts: "[N,O;H1]",
        smiles: "CO",
        expected_count: 1,
        expected_matches: &[&[1]],
    },
    PrecedenceCase {
        name: "low_precedence_and_or_matches_secondary_amine_with_one_hydrogen",
        smarts: "[N,O;H1]",
        smiles: "CNC",
        expected_count: 1,
        expected_matches: &[&[1]],
    },
    PrecedenceCase {
        name: "low_precedence_and_or_rejects_tertiary_amine_without_hydrogen",
        smarts: "[N,O;H1]",
        smiles: "CN(C)C",
        expected_count: 0,
        expected_matches: &[],
    },
    PrecedenceCase {
        name: "semicolon_and_equivalent_to_explicit_repeated_and_for_hydrogen_filter",
        smarts: "[N,O;!H0]",
        smiles: "C[NH2]",
        expected_count: 1,
        expected_matches: &[&[1]],
    },
    PrecedenceCase {
        name: "explicit_and_variant_matches_same_primary_amine",
        smarts: "[N&!H0,O&!H0]",
        smiles: "C[NH2]",
        expected_count: 1,
        expected_matches: &[&[1]],
    },
    PrecedenceCase {
        name: "bond_or_matches_single_c_n_bond",
        smarts: "[#6]-,:[#7]",
        smiles: "CN",
        expected_count: 1,
        expected_matches: &[&[0, 1]],
    },
    PrecedenceCase {
        name: "bond_or_rejects_double_c_n_bond",
        smarts: "[#6]-,:[#7]",
        smiles: "C=N",
        expected_count: 0,
        expected_matches: &[],
    },
    PrecedenceCase {
        name: "bond_or_matches_aromatic_c_n_bonds_in_pyridine",
        smarts: "[#6]-,:[#7]",
        smiles: "c1ccncc1",
        expected_count: 2,
        expected_matches: &[&[2, 3], &[4, 3]],
    },
];

#[test]
fn boolean_precedence_should_agree_with_rdkit() {
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
