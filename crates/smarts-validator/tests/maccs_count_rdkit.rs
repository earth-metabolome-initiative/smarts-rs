//! RDKit-frozen counted-match parity tests for thresholded MACCS keys.

use core::str::FromStr;

use smarts_parser::QueryMol;
use smarts_validator::{match_count_compiled, CompiledQuery, PreparedTarget};

#[derive(Debug, Clone, Copy)]
struct MaccsCountCase {
    key: u16,
    smarts: &'static str,
    threshold: usize,
    smiles: &'static str,
    expected_count: usize,
}

const CASES: &[MaccsCountCase] = &[
    MaccsCountCase {
        key: 118,
        smarts: "[$(*~[CH2]~[CH2]~*),$(*1~[CH2]~[CH2]1)]",
        threshold: 1,
        smiles: "CCOC",
        expected_count: 0,
    },
    MaccsCountCase {
        key: 118,
        smarts: "[$(*~[CH2]~[CH2]~*),$(*1~[CH2]~[CH2]1)]",
        threshold: 1,
        smiles: "CCCO",
        expected_count: 2,
    },
    MaccsCountCase {
        key: 120,
        smarts: "[!#6;R]",
        threshold: 1,
        smiles: "C1CCOC1",
        expected_count: 1,
    },
    MaccsCountCase {
        key: 120,
        smarts: "[!#6;R]",
        threshold: 1,
        smiles: "O1CCOCC1",
        expected_count: 2,
    },
    MaccsCountCase {
        key: 127,
        smarts: "*@*!@[#8]",
        threshold: 1,
        smiles: "C1CCOC1",
        expected_count: 0,
    },
    MaccsCountCase {
        key: 127,
        smarts: "*@*!@[#8]",
        threshold: 1,
        smiles: "OC1CCCCC1",
        expected_count: 2,
    },
    MaccsCountCase {
        key: 130,
        smarts: "[!#6;!#1]~[!#6;!#1]",
        threshold: 1,
        smiles: "NN",
        expected_count: 1,
    },
    MaccsCountCase {
        key: 130,
        smarts: "[!#6;!#1]~[!#6;!#1]",
        threshold: 1,
        smiles: "NNO",
        expected_count: 2,
    },
    MaccsCountCase {
        key: 131,
        smarts: "[!#6;!#1;!H0]",
        threshold: 1,
        smiles: "CN",
        expected_count: 1,
    },
    MaccsCountCase {
        key: 131,
        smarts: "[!#6;!#1;!H0]",
        threshold: 1,
        smiles: "NCCN",
        expected_count: 2,
    },
    MaccsCountCase {
        key: 136,
        smarts: "[#8]=*",
        threshold: 1,
        smiles: "NC(N)=O",
        expected_count: 1,
    },
    MaccsCountCase {
        key: 136,
        smarts: "[#8]=*",
        threshold: 1,
        smiles: "O=C(O)C(=O)O",
        expected_count: 2,
    },
    MaccsCountCase {
        key: 138,
        smarts: "[!#6;!#1]~[CH2]~*",
        threshold: 1,
        smiles: "OCC",
        expected_count: 1,
    },
    MaccsCountCase {
        key: 138,
        smarts: "[!#6;!#1]~[CH2]~*",
        threshold: 1,
        smiles: "OCCO",
        expected_count: 2,
    },
    MaccsCountCase {
        key: 140,
        smarts: "[#8]",
        threshold: 3,
        smiles: "OCC(O)CO",
        expected_count: 3,
    },
    MaccsCountCase {
        key: 140,
        smarts: "[#8]",
        threshold: 3,
        smiles: "O=C(O)C(O)CO",
        expected_count: 4,
    },
    MaccsCountCase {
        key: 141,
        smarts: "[CH3]",
        threshold: 2,
        smiles: "CCCC",
        expected_count: 2,
    },
    MaccsCountCase {
        key: 141,
        smarts: "[CH3]",
        threshold: 2,
        smiles: "CC(C)C",
        expected_count: 3,
    },
    MaccsCountCase {
        key: 142,
        smarts: "[#7]",
        threshold: 1,
        smiles: "CN",
        expected_count: 1,
    },
    MaccsCountCase {
        key: 142,
        smarts: "[#7]",
        threshold: 1,
        smiles: "NCCN",
        expected_count: 2,
    },
    MaccsCountCase {
        key: 145,
        smarts: "*1~*~*~*~*~*~1",
        threshold: 1,
        smiles: "C1CCCCC1",
        expected_count: 1,
    },
    MaccsCountCase {
        key: 145,
        smarts: "*1~*~*~*~*~*~1",
        threshold: 1,
        smiles: "C1CCCCC1C2CCCCC2",
        expected_count: 2,
    },
    MaccsCountCase {
        key: 146,
        smarts: "[#8]",
        threshold: 2,
        smiles: "OCO",
        expected_count: 2,
    },
    MaccsCountCase {
        key: 146,
        smarts: "[#8]",
        threshold: 2,
        smiles: "OCC(O)CO",
        expected_count: 3,
    },
    MaccsCountCase {
        key: 149,
        smarts: "[C;H3,H4]",
        threshold: 1,
        smiles: "CCO",
        expected_count: 1,
    },
    MaccsCountCase {
        key: 149,
        smarts: "[C;H3,H4]",
        threshold: 1,
        smiles: "CCCC",
        expected_count: 2,
    },
    MaccsCountCase {
        key: 159,
        smarts: "[#8]",
        threshold: 1,
        smiles: "CO",
        expected_count: 1,
    },
    MaccsCountCase {
        key: 159,
        smarts: "[#8]",
        threshold: 1,
        smiles: "OCO",
        expected_count: 2,
    },
];

#[test]
fn thresholded_maccs_key_counts_agree_with_frozen_rdkit_examples() {
    let mut compiled_queries = std::collections::BTreeMap::new();

    for case in CASES {
        let compiled = compiled_queries.entry(case.key).or_insert_with(|| {
            let query = QueryMol::from_str(case.smarts).unwrap_or_else(|error| {
                panic!(
                    "key {}: failed to parse SMARTS {:?}: {error}",
                    case.key, case.smarts
                )
            });
            CompiledQuery::new(query).unwrap_or_else(|error| {
                panic!("key {}: failed to compile SMARTS: {error}", case.key)
            })
        });
        let prepared = PreparedTarget::new(case.smiles.parse().unwrap());
        let actual_count = match_count_compiled(compiled, &prepared);

        assert_eq!(
            actual_count, case.expected_count,
            "MACCS key {} SMARTS {:?} against SMILES {:?}",
            case.key, case.smarts, case.smiles
        );
        assert_eq!(
            actual_count > case.threshold,
            case.expected_count > case.threshold,
            "MACCS threshold disagreement for key {} against {:?}",
            case.key,
            case.smiles
        );
    }
}
