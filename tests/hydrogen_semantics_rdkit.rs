//! RDKit-frozen hydrogen SMARTS semantics.

use core::str::FromStr;

use smarts_rs::QueryMol;

#[derive(Debug, Clone, Copy)]
struct HydrogenCase {
    name: &'static str,
    smarts: &'static str,
    smiles: &'static str,
    expected_count: usize,
    expected_matches: &'static [&'static [usize]],
}

const CASES: &[HydrogenCase] = &[
    HydrogenCase {
        name: "atomic_hydrogen_matches_each_explicit_hydrogen_atom",
        smarts: "[H]",
        smiles: "[H][H]",
        expected_count: 2,
        expected_matches: &[&[0], &[1]],
    },
    HydrogenCase {
        name: "historical_hydrogen_or_halogen_form_matches_halogen",
        smarts: "[H,Cl]",
        smiles: "Cl",
        expected_count: 1,
        expected_matches: &[&[0]],
    },
    HydrogenCase {
        name: "historical_hydrogen_or_halogen_form_matches_hydrogen",
        smarts: "[H,Cl]",
        smiles: "[H][H]",
        expected_count: 2,
        expected_matches: &[&[0], &[1]],
    },
    HydrogenCase {
        name: "double_hydrogen_bracket_form_is_hydrogen_count_not_literal_hh",
        smarts: "[HH]",
        smiles: "[H][H]",
        expected_count: 2,
        expected_matches: &[&[0], &[1]],
    },
    HydrogenCase {
        name: "isotopic_charged_hydrogen_matches_exactly",
        smarts: "[2H+]",
        smiles: "[2H+]",
        expected_count: 1,
        expected_matches: &[&[0]],
    },
];

#[test]
fn hydrogen_semantics_should_agree_with_rdkit() {
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
