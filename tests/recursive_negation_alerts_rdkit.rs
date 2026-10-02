//! `RDKit`-frozen recursive SMARTS with nested negation and conjunction.

use core::str::FromStr;

use smarts_rs::QueryMol;

#[derive(Debug, Clone, Copy)]
struct RecursiveAlertCase {
    name: &'static str,
    smarts: &'static str,
    smiles: &'static str,
    expected_count: usize,
    expected_matches: &'static [&'static [usize]],
}

const CASES: &[RecursiveAlertCase] = &[
    RecursiveAlertCase {
        name: "primary_or_secondary_amine_alert_matches_simple_amine",
        smarts: "[$([NX3;H2,H1;!$(NC=O)])]",
        smiles: "NCC",
        expected_count: 1,
        expected_matches: &[&[0]],
    },
    RecursiveAlertCase {
        name: "primary_or_secondary_amine_alert_matches_aniline_like_amine",
        smarts: "[$([NX3;H2,H1;!$(NC=O)])]",
        smiles: "Nc1ccccc1",
        expected_count: 1,
        expected_matches: &[&[0]],
    },
    RecursiveAlertCase {
        name: "primary_or_secondary_amine_alert_rejects_formamide_like_target",
        smarts: "[$([NX3;H2,H1;!$(NC=O)])]",
        smiles: "NC=O",
        expected_count: 0,
        expected_matches: &[],
    },
    RecursiveAlertCase {
        name: "hydroxyl_alert_excludes_phosphate_and_carboxylate_oxygen",
        smarts: "[$([O;H1;!$(O[C,S,P]=O)])]",
        smiles: "CCO",
        expected_count: 1,
        expected_matches: &[&[2]],
    },
    RecursiveAlertCase {
        name: "hydroxyl_alert_rejects_phosphate_oxygen",
        smarts: "[$([O;H1;!$(O[C,S,P]=O)])]",
        smiles: "COP(=O)(O)O",
        expected_count: 0,
        expected_matches: &[],
    },
    RecursiveAlertCase {
        name: "hydroxyl_alert_rejects_carboxylic_acid_oxygen",
        smarts: "[$([O;H1;!$(O[C,S,P]=O)])]",
        smiles: "OC(=O)C",
        expected_count: 0,
        expected_matches: &[],
    },
    RecursiveAlertCase {
        name: "amine_alert_excludes_heteroatom_substituted_and_amide_nitrogen_but_keeps_nitrile_nitrogen",
        smarts: "[$([N;!$(N-[!#6;!#1]);!$(N-C=[O,N,S])])]",
        smiles: "NC#N",
        expected_count: 2,
        expected_matches: &[&[0], &[2]],
    },
    RecursiveAlertCase {
        name: "amine_alert_rejects_amide_nitrogen",
        smarts: "[$([N;!$(N-[!#6;!#1]);!$(N-C=[O,N,S])])]",
        smiles: "NC=O",
        expected_count: 0,
        expected_matches: &[],
    },
    RecursiveAlertCase {
        name: "carboxyl_alert_matches_neutral_acid_center",
        smarts: "[$([CX3](=O)[OX1H0-,OX2H1])]",
        smiles: "CC(=O)O",
        expected_count: 1,
        expected_matches: &[&[1]],
    },
    RecursiveAlertCase {
        name: "carboxyl_alert_matches_carboxylate_center",
        smarts: "[$([CX3](=O)[OX1H0-,OX2H1])]",
        smiles: "CC(=O)[O-]",
        expected_count: 1,
        expected_matches: &[&[1]],
    },
    RecursiveAlertCase {
        name: "carboxyl_alert_rejects_ester_center",
        smarts: "[$([CX3](=O)[OX1H0-,OX2H1])]",
        smiles: "CC(=O)OC",
        expected_count: 0,
        expected_matches: &[],
    },
    RecursiveAlertCase {
        name: "amide_exclusion_alert_keeps_thioamide_like_nitrogen",
        smarts: "[$([#7;!$([#7]-[#6](=O))])]",
        smiles: "NC(=S)C",
        expected_count: 1,
        expected_matches: &[&[0]],
    },
    RecursiveAlertCase {
        name: "amide_exclusion_alert_rejects_amide_nitrogen",
        smarts: "[$([#7;!$([#7]-[#6](=O))])]",
        smiles: "CC(=O)N",
        expected_count: 0,
        expected_matches: &[],
    },
];

#[test]
fn recursive_negation_alerts_should_agree_with_rdkit() {
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
