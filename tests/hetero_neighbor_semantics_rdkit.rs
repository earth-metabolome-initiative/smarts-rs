//! `RDKit`-frozen hetero-neighbor SMARTS parity tests.

use core::str::FromStr;

use smarts_rs::QueryMol;

#[derive(Debug, Clone, Copy)]
struct HeteroNeighborCase {
    name: &'static str,
    smarts: &'static str,
    smiles: &'static str,
    expected_count: usize,
    expected_matches: &'static [&'static [usize]],
}

const CASES: &[HeteroNeighborCase] = &[
    HeteroNeighborCase {
        name: "aza_phenol_z2_matches_three_atoms",
        smarts: "[z2]",
        smiles: "O=C(O)c1nc(O)ccn1",
        expected_count: 3,
        expected_matches: &[&[1], &[3], &[5]],
    },
    HeteroNeighborCase {
        name: "aza_phenol_z1_matches_one_atom",
        smarts: "[z1]",
        smiles: "O=C(O)c1nc(O)ccn1",
        expected_count: 1,
        expected_matches: &[&[8]],
    },
    HeteroNeighborCase {
        name: "aza_phenol_z0_matches_remaining_atoms",
        smarts: "[z0]",
        smiles: "O=C(O)c1nc(O)ccn1",
        expected_count: 6,
        expected_matches: &[&[0], &[2], &[4], &[6], &[7], &[9]],
    },
    HeteroNeighborCase {
        name: "aza_phenol_z_uppercase_two_matches_one_atom",
        smarts: "[Z2]",
        smiles: "O=C(O)c1nc(O)ccn1",
        expected_count: 1,
        expected_matches: &[&[1]],
    },
    HeteroNeighborCase {
        name: "aza_phenol_z_uppercase_one_matches_one_atom",
        smarts: "[Z1]",
        smiles: "O=C(O)c1nc(O)ccn1",
        expected_count: 1,
        expected_matches: &[&[5]],
    },
    HeteroNeighborCase {
        name: "aza_phenol_z_uppercase_zero_matches_remaining_atoms",
        smarts: "[Z0]",
        smiles: "O=C(O)c1nc(O)ccn1",
        expected_count: 8,
        expected_matches: &[&[0], &[2], &[3], &[4], &[6], &[7], &[8], &[9]],
    },
    HeteroNeighborCase {
        name: "pyridine_like_ring_z1_matches_two_carbons",
        smarts: "[z1]",
        smiles: "c1ncccc1",
        expected_count: 2,
        expected_matches: &[&[0], &[2]],
    },
    HeteroNeighborCase {
        name: "diazine_like_ring_z2_matches_one_carbon",
        smarts: "[z2]",
        smiles: "c1ncncc1",
        expected_count: 1,
        expected_matches: &[&[2]],
    },
    HeteroNeighborCase {
        name: "amide_z2_matches_carboxamide_carbon",
        smarts: "[z2]",
        smiles: "CC(=O)NC",
        expected_count: 1,
        expected_matches: &[&[1]],
    },
    HeteroNeighborCase {
        name: "amide_z_uppercase_one_matches_alkylated_nitrogen",
        smarts: "[Z1]",
        smiles: "CC(=O)NC",
        expected_count: 1,
        expected_matches: &[&[4]],
    },
    HeteroNeighborCase {
        name: "sulfoxide_z1_matches_all_atoms",
        smarts: "[z1]",
        smiles: "CS(=O)C",
        expected_count: 4,
        expected_matches: &[&[0], &[1], &[2], &[3]],
    },
    HeteroNeighborCase {
        name: "sulfoxide_z_uppercase_one_matches_all_atoms",
        smarts: "[Z1]",
        smiles: "CS(=O)C",
        expected_count: 4,
        expected_matches: &[&[0], &[1], &[2], &[3]],
    },
    HeteroNeighborCase {
        name: "sulfone_z2_matches_sulfur_only",
        smarts: "[z2]",
        smiles: "CS(=O)(=O)C",
        expected_count: 1,
        expected_matches: &[&[1]],
    },
    HeteroNeighborCase {
        name: "sulfone_z_uppercase_two_matches_sulfur_only",
        smarts: "[Z2]",
        smiles: "CS(=O)(=O)C",
        expected_count: 1,
        expected_matches: &[&[1]],
    },
    HeteroNeighborCase {
        name: "phosphate_z1_matches_all_oxygens",
        smarts: "[z1]",
        smiles: "O=P(O)(O)O",
        expected_count: 4,
        expected_matches: &[&[0], &[2], &[3], &[4]],
    },
    HeteroNeighborCase {
        name: "phosphate_z_uppercase_one_matches_all_oxygens",
        smarts: "[Z1]",
        smiles: "O=P(O)(O)O",
        expected_count: 4,
        expected_matches: &[&[0], &[2], &[3], &[4]],
    },
    HeteroNeighborCase {
        name: "sulfate_z1_matches_all_oxygens",
        smarts: "[z1]",
        smiles: "O=S(=O)(O)O",
        expected_count: 4,
        expected_matches: &[&[0], &[2], &[3], &[4]],
    },
    HeteroNeighborCase {
        name: "sulfate_z_uppercase_one_matches_all_oxygens",
        smarts: "[Z1]",
        smiles: "O=S(=O)(O)O",
        expected_count: 4,
        expected_matches: &[&[0], &[2], &[3], &[4]],
    },
    HeteroNeighborCase {
        name: "nitrobenzene_z2_matches_substituted_ipso_carbon",
        smarts: "[z2]",
        smiles: "c1cc([N+](=O)[O-])ccc1",
        expected_count: 1,
        expected_matches: &[&[3]],
    },
    HeteroNeighborCase {
        name: "nitrobenzene_z1_matches_adjacent_and_remote_carbons",
        smarts: "[z1]",
        smiles: "c1cc([N+](=O)[O-])ccc1",
        expected_count: 3,
        expected_matches: &[&[2], &[4], &[5]],
    },
    HeteroNeighborCase {
        name: "nitrobenzene_z_uppercase_two_matches_substituted_ipso_carbon",
        smarts: "[Z2]",
        smiles: "c1cc([N+](=O)[O-])ccc1",
        expected_count: 1,
        expected_matches: &[&[3]],
    },
    HeteroNeighborCase {
        name: "nitrobenzene_z_uppercase_one_matches_adjacent_and_remote_carbons",
        smarts: "[Z1]",
        smiles: "c1cc([N+](=O)[O-])ccc1",
        expected_count: 3,
        expected_matches: &[&[2], &[4], &[5]],
    },
];

#[test]
fn hetero_neighbor_queries_should_agree_with_rdkit() {
    for case in CASES {
        let query = QueryMol::from_str(case.smarts)
            .unwrap_or_else(|error| panic!("{}: failed to parse SMARTS: {error}", case.name));
        let count = query
            .match_count(case.smiles)
            .unwrap_or_else(|error| panic!("{}: matcher error: {error}", case.name));
        let matches = query
            .substructure_matches(case.smiles)
            .unwrap_or_else(|error| panic!("{}: matcher error: {error}", case.name));
        let expected_matches: Vec<Box<[usize]>> = case
            .expected_matches
            .iter()
            .map(|mapping| Box::<[usize]>::from(*mapping))
            .collect();

        assert_eq!(
            count, case.expected_count,
            "{}: count differs for SMARTS {:?} against SMILES {:?}",
            case.name, case.smarts, case.smiles
        );
        assert_eq!(
            matches.as_ref(),
            expected_matches.as_slice(),
            "{}: materialized matches differ for SMARTS {:?} against SMILES {:?}",
            case.name,
            case.smarts,
            case.smiles
        );
    }
}
