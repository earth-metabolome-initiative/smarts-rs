//! RDKit-frozen PAINS and medicinal-alert SMARTS parity tests.

use core::str::FromStr;

use smarts_rs::QueryMol;

#[derive(Debug, Clone, Copy)]
struct AlertCase {
    name: &'static str,
    smarts: &'static str,
    smiles: &'static str,
    expected_count: usize,
}

const CASES: &[AlertCase] = &[
    AlertCase {
        name: "quinone_alert_matches_simple_quinone",
        smarts: "[!#6&!#1]=[#6]1[#6]=,:[#6][#6](=[!#6&!#1])[#6]=,:[#6]1",
        smiles: "O=C1C=CC(=O)C=C1",
        expected_count: 1,
    },
    AlertCase {
        name: "quinone_alert_matches_fused_quinone",
        smarts: "[!#6&!#1]=[#6]1[#6]=,:[#6][#6](=[!#6&!#1])[#6]=,:[#6]1",
        smiles: "O=C1C=CC(=O)c2ccccc12",
        expected_count: 1,
    },
    AlertCase {
        name: "quinone_alert_rejects_phenol",
        smarts: "[!#6&!#1]=[#6]1[#6]=,:[#6][#6](=[!#6&!#1])[#6]=,:[#6]1",
        smiles: "Oc1ccccc1",
        expected_count: 0,
    },
    AlertCase {
        name: "azo_alert_matches_azo_linkage",
        smarts: "[#7;!R]=[#7]",
        smiles: "c1ccccc1N=Nc1ccccc1",
        expected_count: 1,
    },
    AlertCase {
        name: "azo_alert_rejects_hydrazo_linkage",
        smarts: "[#7;!R]=[#7]",
        smiles: "c1ccccc1NNc1ccccc1",
        expected_count: 0,
    },
    AlertCase {
        name: "nitrosamine_alert_matches_dimethylnitrosamine",
        smarts: "N-[N;X2](=O)",
        smiles: "CN(N=O)C",
        expected_count: 1,
    },
    AlertCase {
        name: "nitrosamine_alert_rejects_formamide",
        smarts: "N-[N;X2](=O)",
        smiles: "CN(C)C=O",
        expected_count: 0,
    },
    AlertCase {
        name: "aliphatic_n_oxide_alert_matches_trialkyl_n_oxide",
        smarts: "[N+!$(N=O)][O-X1]",
        smiles: "C[N+](C)([O-])C",
        expected_count: 1,
    },
    AlertCase {
        name: "aliphatic_n_oxide_alert_rejects_nitro",
        smarts: "[N+!$(N=O)][O-X1]",
        smiles: "C[N+](=O)[O-]",
        expected_count: 0,
    },
    AlertCase {
        name: "thiol_alert_matches_aliphatic_thiol",
        smarts: "[!a][SX2;H1]",
        smiles: "CCS",
        expected_count: 1,
    },
    AlertCase {
        name: "thiol_alert_rejects_aryl_thiol",
        smarts: "[!a][SX2;H1]",
        smiles: "Sc1ccccc1",
        expected_count: 0,
    },
    AlertCase {
        name: "oxime_alert_matches_oxime",
        smarts: "[#6]C(=!@N[$(OC),$([OH])])[#6]",
        smiles: "CC(=NO)c1ccccc1",
        expected_count: 1,
    },
    AlertCase {
        name: "oxime_alert_rejects_ketone",
        smarts: "[#6]C(=!@N[$(OC),$([OH])])[#6]",
        smiles: "CC(=O)c1ccccc1",
        expected_count: 0,
    },
    AlertCase {
        name: "three_membered_heterocycle_matches_epoxide",
        smarts: "*1[O,S]*1",
        smiles: "C1CO1",
        expected_count: 1,
    },
    AlertCase {
        name: "three_membered_heterocycle_rejects_five_membered_ring",
        smarts: "*1[O,S]*1",
        smiles: "C1CCCO1",
        expected_count: 0,
    },
];

#[test]
fn pains_and_alert_patterns_should_agree_with_rdkit() {
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
        assert_eq!(
            matches_out.len(),
            count,
            "{}: materialized match count mismatch",
            case.name
        );
        assert_eq!(
            query.matches(case.smiles).unwrap(),
            count > 0,
            "{}: boolean/count mismatch",
            case.name
        );
    }
}
