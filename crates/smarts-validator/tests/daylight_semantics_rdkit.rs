//! RDKit-frozen Daylight-style SMARTS semantic examples.

use core::str::FromStr;

use smarts_parser::QueryMol;
use smarts_validator::{match_count, matches, substructure_matches};

#[derive(Debug, Clone, Copy)]
struct DaylightCase {
    name: &'static str,
    smarts: &'static str,
    smiles: &'static str,
    expected_count: usize,
    expected_matches: &'static [&'static [usize]],
}

const CASES: &[DaylightCase] = &[
    DaylightCase {
        name: "not_aromatic_carbon_rejects_benzene",
        smarts: "[!c]",
        smiles: "c1ccccc1",
        expected_count: 0,
        expected_matches: &[],
    },
    DaylightCase {
        name: "not_aromatic_carbon_matches_each_aliphatic_carbon",
        smarts: "[!c]",
        smiles: "CC",
        expected_count: 2,
        expected_matches: &[&[0], &[1]],
    },
    DaylightCase {
        name: "atom_or_matches_nitrogen",
        smarts: "[N,O]",
        smiles: "N",
        expected_count: 1,
        expected_matches: &[&[0]],
    },
    DaylightCase {
        name: "atom_or_matches_oxygen",
        smarts: "[N,O]",
        smiles: "O",
        expected_count: 1,
        expected_matches: &[&[0]],
    },
    DaylightCase {
        name: "atom_or_rejects_carbon",
        smarts: "[N,O]",
        smiles: "C",
        expected_count: 0,
        expected_matches: &[],
    },
    DaylightCase {
        name: "high_precedence_or_keeps_all_aromatic_carbons_in_toluene",
        smarts: "[c,n&H1]",
        smiles: "Cc1ccccc1",
        expected_count: 6,
        expected_matches: &[&[1], &[2], &[3], &[4], &[5], &[6]],
    },
    DaylightCase {
        name: "low_precedence_and_excludes_the_ipso_toluene_carbon",
        smarts: "[c,n;H1]",
        smiles: "Cc1ccccc1",
        expected_count: 5,
        expected_matches: &[&[2], &[3], &[4], &[5], &[6]],
    },
    DaylightCase {
        name: "not_aliphatic_carbon_and_ring_matches_each_benzene_atom",
        smarts: "[!C;R]",
        smiles: "c1ccccc1",
        expected_count: 6,
        expected_matches: &[&[0], &[1], &[2], &[3], &[4], &[5]],
    },
    DaylightCase {
        name: "not_aliphatic_carbon_and_ring_rejects_cyclohexane",
        smarts: "[!C;R]",
        smiles: "C1CCCCC1",
        expected_count: 0,
        expected_matches: &[],
    },
    DaylightCase {
        name: "aromatic_n_h1_matches_pyrrolic_nitrogen",
        smarts: "[n;H1]",
        smiles: "c1cc[nH]c1",
        expected_count: 1,
        expected_matches: &[&[3]],
    },
    DaylightCase {
        name: "aromatic_n_h1_rejects_pyridine_nitrogen",
        smarts: "[n;H1]",
        smiles: "n1ccccc1",
        expected_count: 0,
        expected_matches: &[],
    },
    DaylightCase {
        name: "bond_or_matches_double_bond",
        smarts: "[#6]=,:[#6]",
        smiles: "C=C",
        expected_count: 1,
        expected_matches: &[&[0, 1]],
    },
    DaylightCase {
        name: "bond_or_matches_aromatic_bonds",
        smarts: "[#6]=,:[#6]",
        smiles: "c1ccccc1",
        expected_count: 6,
        expected_matches: &[&[0, 1], &[0, 5], &[1, 2], &[2, 3], &[3, 4], &[4, 5]],
    },
    DaylightCase {
        name: "bond_or_rejects_single_bond",
        smarts: "[#6]=,:[#6]",
        smiles: "CC",
        expected_count: 0,
        expected_matches: &[],
    },
    DaylightCase {
        name: "amine_not_amide_matches_primary_amine",
        smarts: "[NX3;H2,H1;!$(NC=O)]",
        smiles: "CCN",
        expected_count: 1,
        expected_matches: &[&[2]],
    },
    DaylightCase {
        name: "amine_not_amide_matches_secondary_amine",
        smarts: "[NX3;H2,H1;!$(NC=O)]",
        smiles: "CCNC",
        expected_count: 1,
        expected_matches: &[&[2]],
    },
    DaylightCase {
        name: "amine_not_amide_rejects_amide",
        smarts: "[NX3;H2,H1;!$(NC=O)]",
        smiles: "CC(=O)NC",
        expected_count: 0,
        expected_matches: &[],
    },
    DaylightCase {
        name: "amine_not_amide_rejects_bare_ammonia_query_target",
        smarts: "[NX3;H2,H1;!$(NC=O)]",
        smiles: "N",
        expected_count: 0,
        expected_matches: &[],
    },
    DaylightCase {
        name: "ring_bond_query_matches_each_ring_edge",
        smarts: "[#6]@[#6]",
        smiles: "C1CCCCC1",
        expected_count: 6,
        expected_matches: &[&[0, 1], &[0, 5], &[1, 2], &[2, 3], &[3, 4], &[4, 5]],
    },
    DaylightCase {
        name: "ring_bond_query_rejects_chain",
        smarts: "[#6]@[#6]",
        smiles: "CCC",
        expected_count: 0,
        expected_matches: &[],
    },
    DaylightCase {
        name: "non_ring_bond_query_matches_each_chain_edge",
        smarts: "*!@*",
        smiles: "CCC",
        expected_count: 2,
        expected_matches: &[&[0, 1], &[1, 2]],
    },
    DaylightCase {
        name: "non_ring_bond_query_rejects_cycle",
        smarts: "*!@*",
        smiles: "C1CCCCC1",
        expected_count: 0,
        expected_matches: &[],
    },
    DaylightCase {
        name: "macrocycle_query_matches_each_eight_membered_ring_atom",
        smarts: "[r;!r3;!r4;!r5;!r6;!r7]",
        smiles: "C1CCCCCCC1",
        expected_count: 8,
        expected_matches: &[&[0], &[1], &[2], &[3], &[4], &[5], &[6], &[7]],
    },
    DaylightCase {
        name: "macrocycle_query_rejects_six_membered_ring",
        smarts: "[r;!r3;!r4;!r5;!r6;!r7]",
        smiles: "C1CCCCC1",
        expected_count: 0,
        expected_matches: &[],
    },
    DaylightCase {
        name: "unfused_benzene_query_matches_benzene",
        smarts: "[cR1]1[cR1][cR1][cR1][cR1][cR1]1",
        smiles: "c1ccccc1",
        expected_count: 1,
        expected_matches: &[&[0, 1, 2, 3, 4, 5]],
    },
    DaylightCase {
        name: "unfused_benzene_query_rejects_naphthalene",
        smarts: "[cR1]1[cR1][cR1][cR1][cR1][cR1]1",
        smiles: "c1cccc2ccccc12",
        expected_count: 0,
        expected_matches: &[],
    },
    DaylightCase {
        name: "ring_membership_two_matches_only_fusion_atoms",
        smarts: "[R2]",
        smiles: "c1cccc2ccccc12",
        expected_count: 2,
        expected_matches: &[&[4], &[9]],
    },
    DaylightCase {
        name: "ring_membership_two_rejects_simple_benzene",
        smarts: "[R2]",
        smiles: "c1ccccc1",
        expected_count: 0,
        expected_matches: &[],
    },
    DaylightCase {
        name: "recursive_conjunction_matches_only_terminal_carbon_with_oxygen_and_two_carbon_path",
        smarts: "[$(*O);$(*CC)]",
        smiles: "CCCO",
        expected_count: 1,
        expected_matches: &[&[2]],
    },
    DaylightCase {
        name: "recursive_conjunction_rejects_short_alcohol",
        smarts: "[$(*O);$(*CC)]",
        smiles: "CCO",
        expected_count: 0,
        expected_matches: &[],
    },
    DaylightCase {
        name: "resonance_style_or_matches_neutral_carbonyl_carbon",
        smarts: "[$([CX3]=[OX1]),$([CX3+]-[OX1-])]",
        smiles: "CC(=O)O",
        expected_count: 1,
        expected_matches: &[&[1]],
    },
    DaylightCase {
        name: "resonance_style_or_matches_charge_separated_carbonyl_carbon",
        smarts: "[$([CX3]=[OX1]),$([CX3+]-[OX1-])]",
        smiles: "C[C+]([O-])O",
        expected_count: 1,
        expected_matches: &[&[1]],
    },
];

#[test]
fn daylight_style_semantics_should_agree_with_rdkit() {
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
