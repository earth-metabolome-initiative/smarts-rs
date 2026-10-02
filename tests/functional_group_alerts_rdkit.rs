//! RDKit-frozen functional-group and alert SMARTS parity tests.

use core::str::FromStr;

use smarts_rs::QueryMol;

#[derive(Debug, Clone, Copy)]
struct AlertCase {
    name: &'static str,
    smarts: &'static str,
    smiles: &'static str,
    expected_count: usize,
    expected_matches: &'static [&'static [usize]],
}

const CASES: &[AlertCase] = &[
    AlertCase {
        name: "aryl_halide_attachment_matches_bromobenzene",
        smarts: "[F,Cl,Br,I;$(*-!@c)]",
        smiles: "c1ccccc1Br",
        expected_count: 1,
        expected_matches: &[&[6]],
    },
    AlertCase {
        name: "aryl_halide_attachment_rejects_alkyl_bromide",
        smarts: "[F,Cl,Br,I;$(*-!@c)]",
        smiles: "CCBr",
        expected_count: 0,
        expected_matches: &[],
    },
    AlertCase {
        name: "aryl_halide_attachment_counts_each_halogen_on_dihalobenzene",
        smarts: "[F,Cl,Br,I;$(*-!@c)]",
        smiles: "Brc1ccccc1Cl",
        expected_count: 2,
        expected_matches: &[&[0], &[7]],
    },
    AlertCase {
        name: "acid_halide_pattern_matches_acid_chloride",
        smarts: "[S,C](=[O,S])[F,Br,Cl,I]",
        smiles: "CC(=O)Cl",
        expected_count: 1,
        expected_matches: &[&[1, 2, 3]],
    },
    AlertCase {
        name: "acid_halide_pattern_matches_thioacyl_halide",
        smarts: "[S,C](=[O,S])[F,Br,Cl,I]",
        smiles: "CC(=S)Cl",
        expected_count: 1,
        expected_matches: &[&[1, 2, 3]],
    },
    AlertCase {
        name: "acid_halide_pattern_rejects_carboxylic_acid",
        smarts: "[S,C](=[O,S])[F,Br,Cl,I]",
        smiles: "CC(=O)O",
        expected_count: 0,
        expected_matches: &[],
    },
    AlertCase {
        name: "isocyanate_pattern_matches_isocyanate",
        smarts: "N=C=[S,O]",
        smiles: "CCN=C=O",
        expected_count: 1,
        expected_matches: &[&[2, 3, 4]],
    },
    AlertCase {
        name: "isocyanate_pattern_matches_isothiocyanate",
        smarts: "N=C=[S,O]",
        smiles: "CCN=C=S",
        expected_count: 1,
        expected_matches: &[&[2, 3, 4]],
    },
    AlertCase {
        name: "isocyanate_pattern_rejects_amide_like_target",
        smarts: "N=C=[S,O]",
        smiles: "CCNC=O",
        expected_count: 0,
        expected_matches: &[],
    },
    AlertCase {
        name: "michael_acceptor_matches_enone",
        smarts: "C=!@CC=[O,S]",
        smiles: "C=CC(=O)C",
        expected_count: 1,
        expected_matches: &[&[0, 1, 2, 3]],
    },
    AlertCase {
        name: "michael_acceptor_matches_thioenone",
        smarts: "C=!@CC=[O,S]",
        smiles: "C=CC(=S)C",
        expected_count: 1,
        expected_matches: &[&[0, 1, 2, 3]],
    },
    AlertCase {
        name: "michael_acceptor_rejects_plain_alkene",
        smarts: "C=!@CC=[O,S]",
        smiles: "CC=CC",
        expected_count: 0,
        expected_matches: &[],
    },
    AlertCase {
        name: "reactive_alkyl_halide_matches_primary_alkyl_halide",
        smarts: "[Br,Cl,I][CX4;CH,CH2]",
        smiles: "CCCl",
        expected_count: 1,
        expected_matches: &[&[2, 1]],
    },
    AlertCase {
        name: "reactive_alkyl_halide_matches_secondary_alkyl_halide",
        smarts: "[Br,Cl,I][CX4;CH,CH2]",
        smiles: "CC(C)Cl",
        expected_count: 1,
        expected_matches: &[&[3, 1]],
    },
    AlertCase {
        name: "reactive_alkyl_halide_rejects_aryl_halide",
        smarts: "[Br,Cl,I][CX4;CH,CH2]",
        smiles: "Clc1ccccc1",
        expected_count: 0,
        expected_matches: &[],
    },
    AlertCase {
        name: "charged_quaternary_nitrogen_alert_matches_methyl_pyridinium",
        smarts: "[s,S,c,C,n,N,o,O]~[nX3+,NX3+](~[s,S,c,C,n,N])~[s,S,c,C,n,N]",
        smiles: "C[n+]1ccccc1",
        expected_count: 1,
        expected_matches: &[&[0, 1, 2, 6]],
    },
    AlertCase {
        name: "acyl_azinium_alert_matches_n_acyl_pyridinium_both_ring_directions",
        smarts: "[#7+](:[!#1]:[!#1]:[!#1])-[!#1]=[#8]",
        smiles: "O=C[n+]1ccccc1",
        expected_count: 2,
        expected_matches: &[&[2, 3, 4, 5, 1, 0], &[2, 7, 6, 5, 1, 0]],
    },
    AlertCase {
        name: "thioaryl_alert_matches_both_aromatic_directions_from_sulfur",
        smarts: "[#16]!:*:*",
        smiles: "Sc1ccccc1",
        expected_count: 2,
        expected_matches: &[&[0, 1, 2], &[0, 1, 6]],
    },
    AlertCase {
        name: "quinone_mixed_bond_pattern_matches_simple_quinone",
        smarts: "[!#6&!#1]=[#6]1[#6]=,:[#6][#6](=[!#6&!#1])[#6]=,:[#6]1",
        smiles: "O=C1C=CC(=O)C=C1",
        expected_count: 1,
        expected_matches: &[&[0, 1, 2, 3, 4, 5, 6, 7]],
    },
    AlertCase {
        name: "quinone_mixed_bond_pattern_matches_fused_quinone",
        smarts: "[!#6&!#1]=[#6]1[#6]=,:[#6][#6](=[!#6&!#1])[#6]=,:[#6]1",
        smiles: "O=C1C=CC(=O)c2ccccc12",
        expected_count: 1,
        expected_matches: &[&[0, 1, 2, 3, 4, 5, 6, 11]],
    },
    AlertCase {
        name: "quinone_mixed_bond_pattern_rejects_phenol",
        smarts: "[!#6&!#1]=[#6]1[#6]=,:[#6][#6](=[!#6&!#1])[#6]=,:[#6]1",
        smiles: "Oc1ccccc1",
        expected_count: 0,
        expected_matches: &[],
    },
    AlertCase {
        name: "recursive_amino_acid_like_pattern_matches_glycine_like_target",
        smarts: "[$(C-[C;!$(C=[!#6])]-[N;!H0;!$(N-[!#6;!#1]);!$(N-C=[O,N,S])])](=O)([O;H,-])",
        smiles: "NCC(=O)O",
        expected_count: 1,
        expected_matches: &[&[2, 3, 4]],
    },
    AlertCase {
        name: "recursive_amino_acid_like_pattern_matches_carboxylate_variant",
        smarts: "[$(C-[C;!$(C=[!#6])]-[N;!H0;!$(N-[!#6;!#1]);!$(N-C=[O,N,S])])](=O)([O;H,-])",
        smiles: "NCC(=O)[O-]",
        expected_count: 1,
        expected_matches: &[&[2, 3, 4]],
    },
    AlertCase {
        name: "recursive_amino_acid_like_pattern_matches_alpha_substituted_amino_acid",
        smarts: "[$(C-[C;!$(C=[!#6])]-[N;!H0;!$(N-[!#6;!#1]);!$(N-C=[O,N,S])])](=O)([O;H,-])",
        smiles: "NC(C)C(=O)O",
        expected_count: 1,
        expected_matches: &[&[3, 4, 5]],
    },
    AlertCase {
        name: "recursive_amino_acid_like_pattern_rejects_hydroxy_acid",
        smarts: "[$(C-[C;!$(C=[!#6])]-[N;!H0;!$(N-[!#6;!#1]);!$(N-C=[O,N,S])])](=O)([O;H,-])",
        smiles: "OCC(=O)O",
        expected_count: 0,
        expected_matches: &[],
    },
];

#[test]
fn functional_group_and_alert_patterns_should_agree_with_rdkit() {
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
