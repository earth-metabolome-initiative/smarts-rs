//! RDKit-frozen explicit-hydrogen topology parity tests.

use core::str::FromStr;

use smarts_rs::QueryMol;

#[derive(Debug, Clone, Copy)]
struct ExplicitHydrogenCase {
    name: &'static str,
    smarts: &'static str,
    smiles: &'static str,
    expected_count: usize,
    expected_matches: Option<&'static [&'static [usize]]>,
}

const CASES: &[ExplicitHydrogenCase] = &[
    ExplicitHydrogenCase {
        name: "recursive_explicit_hydrogen_neighbor_query_does_not_match_default_hydroxyl_target",
        smarts: "[$([#1]O)]",
        smiles: "[H]O",
        expected_count: 0,
        expected_matches: Some(&[]),
    },
    ExplicitHydrogenCase {
        name: "recursive_explicit_hydrogen_neighbor_query_does_not_match_alcohol_target",
        smarts: "[$([#1]O)]",
        smiles: "[H]OC",
        expected_count: 0,
        expected_matches: Some(&[]),
    },
    ExplicitHydrogenCase {
        name: "recursive_explicit_hydrogen_neighbor_query_does_not_match_plain_oxygen",
        smarts: "[$([#1]O)]",
        smiles: "O",
        expected_count: 0,
        expected_matches: Some(&[]),
    },
    ExplicitHydrogenCase {
        name: "literal_oxygen_hydrogen_bond_query_does_not_match_default_hydroxyl_target",
        smarts: "[O]-[H]",
        smiles: "[H]O",
        expected_count: 0,
        expected_matches: Some(&[]),
    },
    ExplicitHydrogenCase {
        name: "literal_oxygen_hydrogen_bond_query_does_not_match_default_alcohol_target",
        smarts: "[O]-[H]",
        smiles: "[H]OC",
        expected_count: 0,
        expected_matches: Some(&[]),
    },
    ExplicitHydrogenCase {
        name: "literal_hydrogen_oxygen_bond_query_does_not_match_default_water_target",
        smarts: "[#1]-O",
        smiles: "[H][O][H]",
        expected_count: 0,
        expected_matches: Some(&[]),
    },
    ExplicitHydrogenCase {
        name: "literal_water_query_does_not_match_default_explicit_water_target",
        smarts: "[H]O[H]",
        smiles: "[H][O][H]",
        expected_count: 0,
        expected_matches: Some(&[]),
    },
    ExplicitHydrogenCase {
        name: "oxygen_h1_query_matches_explicit_hydrogen_acid_target",
        smarts: "[O;H1]",
        smiles: "[H]OC(=O)C",
        expected_count: 1,
        expected_matches: None,
    },
    ExplicitHydrogenCase {
        name: "oxygen_h1_query_matches_normal_neutral_acid_target",
        smarts: "[O;H1]",
        smiles: "CC(=O)O",
        expected_count: 1,
        expected_matches: Some(&[&[3]]),
    },
    ExplicitHydrogenCase {
        name: "carboxylic_acid_query_matches_explicit_hydrogen_acid_target",
        smarts: "C(=O)[O;H,-]",
        smiles: "[H]OC(=O)C",
        expected_count: 1,
        expected_matches: None,
    },
];

#[test]
fn explicit_hydrogen_topology_should_agree_with_rdkit_default_target_parsing() {
    for case in CASES {
        let query = QueryMol::from_str(case.smarts)
            .unwrap_or_else(|error| panic!("{}: failed to parse SMARTS: {error}", case.name));
        let count = query
            .match_count(case.smiles)
            .unwrap_or_else(|error| panic!("{}: matcher error: {error}", case.name));
        let matches_out = query
            .substructure_matches(case.smiles)
            .unwrap_or_else(|error| panic!("{}: matcher error: {error}", case.name));

        assert_eq!(
            count, case.expected_count,
            "{}: SMARTS {:?} against SMILES {:?}",
            case.name, case.smarts, case.smiles
        );

        if let Some(expected_matches) = case.expected_matches {
            let expected_matches: Vec<Box<[usize]>> = expected_matches
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
        }
        assert_eq!(
            query.matches(case.smiles).unwrap(),
            count > 0,
            "{}: boolean/count mismatch",
            case.name
        );
    }
}
