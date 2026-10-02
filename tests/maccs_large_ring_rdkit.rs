//! RDKit-frozen large-ring recursive MACCS SMARTS parity tests.

use core::str::FromStr;

use smarts_rs::QueryMol;

const MACCS_101_SMARTS: &str = "[$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1)]";

#[derive(Debug, Clone, Copy)]
struct LargeRingCase {
    name: &'static str,
    smiles: &'static str,
    expected_count: usize,
    expected_matches: &'static [&'static [usize]],
}

const CASES: &[LargeRingCase] = &[
    LargeRingCase {
        name: "eight_membered_ring_matches_every_anchor_atom",
        smiles: "C1CCCCCCC1",
        expected_count: 8,
        expected_matches: &[&[0], &[1], &[2], &[3], &[4], &[5], &[6], &[7]],
    },
    LargeRingCase {
        name: "seven_membered_ring_does_not_match",
        smiles: "C1CCCCCC1",
        expected_count: 0,
        expected_matches: &[],
    },
    LargeRingCase {
        name: "nine_membered_ring_matches_every_anchor_atom",
        smiles: "C1CCCCCCCC1",
        expected_count: 9,
        expected_matches: &[&[0], &[1], &[2], &[3], &[4], &[5], &[6], &[7], &[8]],
    },
    LargeRingCase {
        name: "bicyclic_bridge_system_still_matches_all_ring_anchor_atoms",
        smiles: "C1CCC2CCCC2C1",
        expected_count: 9,
        expected_matches: &[&[0], &[1], &[2], &[3], &[4], &[5], &[6], &[7], &[8]],
    },
];

#[test]
fn large_ring_maccs_pattern_should_agree_with_rdkit() {
    let query = QueryMol::from_str(MACCS_101_SMARTS).unwrap();

    for case in CASES {
        let count = query
            .match_count(case.smiles)
            .unwrap_or_else(|error| panic!("{}: matcher error: {error}", case.name));
        let matches_out = query
            .substructure_matches(case.smiles)
            .unwrap_or_else(|error| panic!("{}: matcher error: {error}", case.name));

        assert_eq!(
            count, case.expected_count,
            "{}: SMARTS {:?} against SMILES {:?}",
            case.name, MACCS_101_SMARTS, case.smiles
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
            MACCS_101_SMARTS,
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
