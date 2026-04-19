//! RDKit-frozen functional-group hierarchy SMARTS parity tests.

use core::str::FromStr;

use smarts_parser::QueryMol;
use smarts_validator::{match_count, matches, substructure_matches};

#[derive(Debug, Clone, Copy)]
struct HierarchyCase {
    name: &'static str,
    smarts: &'static str,
    smiles: &'static str,
    expected_count: usize,
}

const CASES: &[HierarchyCase] = &[
    HierarchyCase {
        name: "alpha_amino_acid_matches_glycine",
        smarts: "[$(C-[C;!$(C=[!#6])]-[N;!H0;!$(N-[!#6;!#1]);!$(N-C=[O,N,S])])](=O)([O;H,-])",
        smiles: "NCC(=O)O",
        expected_count: 1,
    },
    HierarchyCase {
        name: "alpha_amino_acid_matches_carboxylate_variant",
        smarts: "[$(C-[C;!$(C=[!#6])]-[N;!H0;!$(N-[!#6;!#1]);!$(N-C=[O,N,S])])](=O)([O;H,-])",
        smiles: "NCC(=O)[O-]",
        expected_count: 1,
    },
    HierarchyCase {
        name: "alpha_amino_acid_rejects_amide",
        smarts: "[$(C-[C;!$(C=[!#6])]-[N;!H0;!$(N-[!#6;!#1]);!$(N-C=[O,N,S])])](=O)([O;H,-])",
        smiles: "NCC(=O)N",
        expected_count: 0,
    },
    HierarchyCase {
        name: "alpha_amino_acid_rejects_hydroxy_acid",
        smarts: "[$(C-[C;!$(C=[!#6])]-[N;!H0;!$(N-[!#6;!#1]);!$(N-C=[O,N,S])])](=O)([O;H,-])",
        smiles: "OCC(=O)O",
        expected_count: 0,
    },
    HierarchyCase {
        name: "primary_aromatic_amine_matches_aniline",
        smarts: "[N;H2;D1;$(N-!@c);!$(N-C=[O,N,S])]",
        smiles: "Nc1ccccc1",
        expected_count: 1,
    },
    HierarchyCase {
        name: "primary_aromatic_amine_rejects_acetanilide",
        smarts: "[N;H2;D1;$(N-!@c);!$(N-C=[O,N,S])]",
        smiles: "CC(=O)Nc1ccccc1",
        expected_count: 0,
    },
    HierarchyCase {
        name: "primary_aromatic_amine_rejects_aliphatic_amine",
        smarts: "[N;H2;D1;$(N-!@c);!$(N-C=[O,N,S])]",
        smiles: "CCN",
        expected_count: 0,
    },
    HierarchyCase {
        name: "aromatic_sulfonyl_chloride_matches_aryl_target",
        smarts: "[$(S-!@c)](=O)(=O)(Cl)",
        smiles: "O=S(=O)(Cl)c1ccccc1",
        expected_count: 1,
    },
    HierarchyCase {
        name: "aromatic_sulfonyl_chloride_rejects_alkyl_target",
        smarts: "[$(S-!@c)](=O)(=O)(Cl)",
        smiles: "CCS(=O)(=O)Cl",
        expected_count: 0,
    },
    HierarchyCase {
        name: "sulfonamide_charge_logic_good_pattern_matches",
        smarts: "[NH2]-S(=,-[OX1;+0,-1])(=,-[OX1;+0,-1])-[#6]",
        smiles: "CCCCS(=O)(=O)[NH2]",
        expected_count: 1,
    },
    HierarchyCase {
        name: "sulfonamide_charge_logic_bad_pattern_rejects",
        smarts: "[NH2]-S(=,-[OX1;+0;-1])(=,-[OX1;+0;-1])-[#6]",
        smiles: "CCCCS(=O)(=O)[NH2]",
        expected_count: 0,
    },
];

#[test]
fn functional_group_hierarchy_patterns_should_agree_with_rdkit() {
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
        assert_eq!(
            matches_out.len(),
            count,
            "{}: materialized match count mismatch",
            case.name
        );
        assert_eq!(
            matches(&query, case.smiles).unwrap(),
            count > 0,
            "{}: boolean/count mismatch",
            case.name
        );
    }
}
