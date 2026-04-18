//! `RDKit` parity checks for representative sugar SMARTS queries.

use core::str::FromStr;

use smarts_parser::QueryMol;
use smarts_validator::matches;

const SUGAR_SMARTS: [(&str, &str); 6] = [
    (
        "sugar1",
        "[OX2;$([r5]1@C@C@C(O)@C1),$([r6]1@C@C@C(O)@C(O)@C1)]",
    ),
    (
        "sugar2",
        "[OX2;$([r5]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C1),$([r6]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C@C1)]",
    ),
    (
        "sugar3",
        "[OX2;$([r5]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C(O)@C1),$([r6]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C(O)@C(O)@C1)]",
    ),
    (
        "sugar4",
        "[OX2;$([r5]1@C(!@[OX2H1])@C@C@C1),$([r6]1@C(!@[OX2H1])@C@C@C@C1)]",
    ),
    (
        "sugar5",
        "[OX2;$([r5]1@[C@@](!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C1),$([r6]1@[C@@](!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C@C1)]",
    ),
    (
        "sugar6",
        "[OX2;$([r5]1@[C@](!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C1),$([r6]1@[C@](!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C@C1)]",
    ),
];

const TARGETS: [(&str, &str); 6] = [
    ("ribofuranose", "OC[C@H]1O[C@@H](O)[C@H](O)[C@H]1O"),
    (
        "glucopyranose",
        "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O",
    ),
    ("tetrahydrofuran", "C1CCOC1"),
    ("tetrahydropyran", "C1CCOCC1"),
    ("cyclohexanol", "OC1CCCCC1"),
    ("dioxolane", "C1OCOC1"),
];

#[test]
fn sugar_smarts_1_to_4_match_rdkit_reference_panel() {
    let expected = [
        ("sugar1", "ribofuranose", true),
        ("sugar1", "glucopyranose", true),
        ("sugar1", "tetrahydrofuran", false),
        ("sugar1", "tetrahydropyran", false),
        ("sugar1", "cyclohexanol", false),
        ("sugar1", "dioxolane", false),
        ("sugar2", "ribofuranose", true),
        ("sugar2", "glucopyranose", true),
        ("sugar2", "tetrahydrofuran", false),
        ("sugar2", "tetrahydropyran", false),
        ("sugar2", "cyclohexanol", false),
        ("sugar2", "dioxolane", false),
        ("sugar3", "ribofuranose", true),
        ("sugar3", "glucopyranose", true),
        ("sugar3", "tetrahydrofuran", false),
        ("sugar3", "tetrahydropyran", false),
        ("sugar3", "cyclohexanol", false),
        ("sugar3", "dioxolane", false),
        ("sugar4", "ribofuranose", true),
        ("sugar4", "glucopyranose", true),
        ("sugar4", "tetrahydrofuran", false),
        ("sugar4", "tetrahydropyran", false),
        ("sugar4", "cyclohexanol", false),
        ("sugar4", "dioxolane", false),
    ];

    for (query_name, query_smarts) in SUGAR_SMARTS.into_iter().take(4) {
        let query = QueryMol::from_str(query_smarts).unwrap_or_else(|error| {
            panic!("{query_name} should parse because RDKit accepts it: {error}")
        });
        for (target_name, target_smiles) in TARGETS {
            let expected_match = expected
                .iter()
                .find_map(|(expected_query, expected_target, expected_match)| {
                    (*expected_query == query_name && *expected_target == target_name)
                        .then_some(*expected_match)
                })
                .unwrap_or_else(|| panic!("missing expectation for {query_name} × {target_name}"));

            let matched = matches(&query, target_smiles).unwrap_or_else(|error| {
                panic!(
                    "{query_name} should stay inside the current validator slice against {target_name}: {error}"
                )
            });
            assert_eq!(
                matched, expected_match,
                "{query_name} should match {target_name} = {expected_match}",
            );
        }
    }
}

#[test]
fn sugar_smarts_5_and_6_match_rdkit_reference_panel() {
    let expected = [
        ("sugar5", "ribofuranose", true),
        ("sugar5", "glucopyranose", true),
        ("sugar5", "tetrahydrofuran", false),
        ("sugar5", "tetrahydropyran", false),
        ("sugar5", "cyclohexanol", false),
        ("sugar5", "dioxolane", false),
        ("sugar6", "ribofuranose", false),
        ("sugar6", "glucopyranose", false),
        ("sugar6", "tetrahydrofuran", false),
        ("sugar6", "tetrahydropyran", false),
        ("sugar6", "cyclohexanol", false),
        ("sugar6", "dioxolane", false),
    ];

    for (query_name, query_smarts) in SUGAR_SMARTS.into_iter().skip(4) {
        let query = QueryMol::from_str(query_smarts).unwrap_or_else(|error| {
            panic!("{query_name} should parse because RDKit accepts it: {error}")
        });
        for (target_name, target_smiles) in TARGETS {
            let expected_match = expected
                .iter()
                .find_map(|(expected_query, expected_target, expected_match)| {
                    (*expected_query == query_name && *expected_target == target_name)
                        .then_some(*expected_match)
                })
                .unwrap_or_else(|| panic!("missing expectation for {query_name} × {target_name}"));

            let matched = matches(&query, target_smiles).unwrap_or_else(|error| {
                panic!(
                    "{query_name} should stay inside the current validator slice against {target_name}: {error}"
                )
            });
            assert_eq!(
                matched, expected_match,
                "{query_name} should match {target_name} = {expected_match}",
            );
        }
    }
}
