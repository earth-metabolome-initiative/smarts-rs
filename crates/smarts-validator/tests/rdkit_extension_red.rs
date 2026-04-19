//! `RDKit` extension parity tests.

use core::str::FromStr;

use smarts_parser::QueryMol;
use smarts_validator::match_count;

#[derive(Debug, Clone, Copy)]
struct ExtensionCase {
    name: &'static str,
    smarts: &'static str,
    smiles: &'static str,
    expected_count: usize,
}

const CASES: &[ExtensionCase] = &[
    ExtensionCase {
        name: "hetero_neighbor_z2_matches_three_atoms",
        smarts: "[z2]",
        smiles: "O=C(O)c1nc(O)ccn1",
        expected_count: 3,
    },
    ExtensionCase {
        name: "hetero_neighbor_z_uppercase_two_matches_one_atom",
        smarts: "[Z2]",
        smiles: "O=C(O)c1nc(O)ccn1",
        expected_count: 1,
    },
    ExtensionCase {
        name: "hetero_neighbor_z_uppercase_one_matches_one_atom",
        smarts: "[Z1]",
        smiles: "O=C(O)c1nc(O)ccn1",
        expected_count: 1,
    },
    ExtensionCase {
        name: "hetero_neighbor_z_lowercase_one_matches_one_atom",
        smarts: "[z1]",
        smiles: "O=C(O)c1nc(O)ccn1",
        expected_count: 1,
    },
    ExtensionCase {
        name: "hybridization_sp_matches_nitrile_atoms",
        smarts: "[^1]",
        smiles: "CC#N",
        expected_count: 2,
    },
    ExtensionCase {
        name: "hybridization_sp_matches_cumulene_center",
        smarts: "[^1]",
        smiles: "C=C=C",
        expected_count: 1,
    },
    ExtensionCase {
        name: "hybridization_sp2_matches_alkene_atoms",
        smarts: "[^2]",
        smiles: "CC=CF",
        expected_count: 2,
    },
    ExtensionCase {
        name: "hybridization_sp2_matches_aromatic_atoms",
        smarts: "[^2]",
        smiles: "c1ccccc1",
        expected_count: 6,
    },
    ExtensionCase {
        name: "hybridization_sp2_matches_conjugated_hetero_atoms",
        smarts: "[^2]",
        smiles: "CC(=O)NC",
        expected_count: 3,
    },
    ExtensionCase {
        name: "hybridization_sp2_matches_vinyl_amine_atoms",
        smarts: "[^2]",
        smiles: "NC=C",
        expected_count: 3,
    },
    ExtensionCase {
        name: "hybridization_sp2_matches_formic_acid_like_atoms",
        smarts: "[^2]",
        smiles: "OC=O",
        expected_count: 3,
    },
    ExtensionCase {
        name: "hybridization_sp3_matches_alkane_atoms",
        smarts: "[^3]",
        smiles: "CC",
        expected_count: 2,
    },
    ExtensionCase {
        name: "hybridization_sp3_matches_remaining_atoms_in_alkene",
        smarts: "[^3]",
        smiles: "CC=CF",
        expected_count: 2,
    },
];

#[test]
fn rdkit_extension_cases_should_agree_with_reference_behavior() {
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
    }
}
