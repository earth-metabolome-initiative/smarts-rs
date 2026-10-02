//! RDKit-frozen aromatic atom and bond semantics.

use core::str::FromStr;

use smarts_rs::QueryMol;

#[derive(Debug, Clone, Copy)]
struct AromaticCase {
    name: &'static str,
    smarts: &'static str,
    smiles: &'static str,
    expected_count: usize,
    expected_matches: &'static [&'static [usize]],
}

const CASES: &[AromaticCase] = &[
    AromaticCase {
        name: "neutral_pyridine_carbons_match_aromatic_nitrogen_bond_query",
        smarts: "c:n",
        smiles: "c1ccncc1",
        expected_count: 2,
        expected_matches: &[&[2, 3], &[4, 3]],
    },
    AromaticCase {
        name: "biaryl_single_bond_matches_c_dash_c_once",
        smarts: "c-c",
        smiles: "c1ccccc1-c1ccccc1",
        expected_count: 1,
        expected_matches: &[&[5, 6]],
    },
    AromaticCase {
        name: "biaryl_ring_bonds_match_c_colon_c_but_not_central_single_bond",
        smarts: "c:c",
        smiles: "c1ccccc1-c1ccccc1",
        expected_count: 12,
        expected_matches: &[
            &[0, 1],
            &[0, 5],
            &[1, 2],
            &[2, 3],
            &[3, 4],
            &[4, 5],
            &[6, 7],
            &[6, 11],
            &[7, 8],
            &[8, 9],
            &[9, 10],
            &[10, 11],
        ],
    },
    AromaticCase {
        name: "fused_kekule_system_exposes_nonaromatic_fused_bonds_between_aromatic_atoms",
        smarts: "a-a",
        smiles: "C1=CC2=C(C=C1)C1=CC=CC=C21",
        expected_count: 2,
        expected_matches: &[&[2, 11], &[3, 6]],
    },
    AromaticCase {
        name: "fused_kekule_system_still_has_aromatic_ring_bonds",
        smarts: "a:a",
        smiles: "C1=CC2=C(C=C1)C1=CC=CC=C21",
        expected_count: 12,
        expected_matches: &[
            &[0, 1],
            &[0, 5],
            &[1, 2],
            &[2, 3],
            &[3, 4],
            &[4, 5],
            &[6, 7],
            &[6, 11],
            &[7, 8],
            &[8, 9],
            &[9, 10],
            &[10, 11],
        ],
    },
    AromaticCase {
        name: "fully_aromatic_fused_system_has_no_a_dash_a_bonds",
        smarts: "a-a",
        smiles: "c1ccc2ccccc2c1",
        expected_count: 0,
        expected_matches: &[],
    },
    AromaticCase {
        name: "fully_aromatic_fused_system_has_only_a_colon_a_bonds",
        smarts: "a:a",
        smiles: "c1ccc2ccccc2c1",
        expected_count: 11,
        expected_matches: &[
            &[0, 1],
            &[0, 9],
            &[1, 2],
            &[2, 3],
            &[3, 4],
            &[3, 8],
            &[4, 5],
            &[5, 6],
            &[6, 7],
            &[7, 8],
            &[8, 9],
        ],
    },
    AromaticCase {
        name: "exocyclic_carbonyl_fused_system_has_no_a_dash_a_bonds",
        smarts: "a-a",
        smiles: "O=C1C=CC(=O)C2=C1OC=CO2",
        expected_count: 0,
        expected_matches: &[],
    },
    AromaticCase {
        name: "exocyclic_carbonyl_fused_system_keeps_only_true_aromatic_bonds",
        smarts: "a:a",
        smiles: "O=C1C=CC(=O)C2=C1OC=CO2",
        expected_count: 10,
        expected_matches: &[
            &[1, 2],
            &[1, 7],
            &[2, 3],
            &[3, 4],
            &[4, 6],
            &[6, 11],
            &[7, 8],
            &[8, 9],
            &[9, 10],
            &[10, 11],
        ],
    },
];

#[test]
fn aromatic_bond_semantics_should_agree_with_rdkit() {
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
            query.matches(case.smiles).unwrap(),
            count > 0,
            "{}: boolean/count mismatch",
            case.name
        );
    }
}
