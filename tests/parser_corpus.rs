//! Corpus-driven parser tests for valid and invalid SMARTS fixtures.

use serde::Deserialize;
use smarts_rs::parse_smarts;

#[derive(Debug, Deserialize)]
struct ValidCase {
    id: String,
    smarts: String,
    source: String,
    expected_atoms: usize,
    expected_bonds: usize,
    expected_components: usize,
}

#[derive(Debug, Deserialize)]
struct InvalidCase {
    id: String,
    smarts: String,
    source: String,
    expected_code: String,
}

#[test]
fn parse_valid_corpus_v0() {
    let cases: Vec<ValidCase> = load_json("parse-valid-v0.json");

    for case in cases {
        let query = parse_smarts(&case.smarts).unwrap_or_else(|err| {
            panic!(
                "expected valid SMARTS `{}` from {} but got {} ({})",
                case.smarts,
                case.source,
                err,
                err.code(),
            )
        });

        assert_eq!(query.atom_count(), case.expected_atoms, "case {}", case.id);
        assert_eq!(query.bond_count(), case.expected_bonds, "case {}", case.id);
        assert_eq!(
            query.component_count(),
            case.expected_components,
            "case {}",
            case.id
        );
    }
}

#[test]
fn parse_invalid_corpus_v0() {
    let cases: Vec<InvalidCase> = load_json("parse-invalid-v0.json");

    for case in cases {
        let err = parse_smarts(&case.smarts).unwrap_err();
        assert_eq!(err.code(), case.expected_code, "case {}", case.id);
        assert!(
            !err.to_string().is_empty(),
            "case {} from {} should have a diagnostic message",
            case.id,
            case.source
        );
    }
}

fn load_json<T>(file_name: &str) -> T
where
    T: for<'de> Deserialize<'de>,
{
    let path = format!("{}/corpus/parser/{}", env!("CARGO_MANIFEST_DIR"), file_name);
    let raw = std::fs::read_to_string(path).expect("failed to read corpus fixture");
    serde_json::from_str(&raw).expect("failed to parse corpus fixture JSON")
}
