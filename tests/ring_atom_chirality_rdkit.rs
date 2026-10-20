//! Tetrahedral ring-atom chirality against the frozen `RDKit` fixture.

use core::str::FromStr;

use serde::Deserialize;
use smarts_rs::{CompiledQuery, MatchScratch, PreparedTarget, QueryMol};
use smiles_rs::Smiles;

#[derive(Debug, Deserialize)]
struct RingAtomChiralityCase {
    name: String,
    smarts: String,
    smiles: String,
    expected_match: bool,
    expected_count: usize,
}

fn load_cases() -> Vec<RingAtomChiralityCase> {
    serde_json::from_str(include_str!(
        "../corpus/matching/ring-atom-chirality.rdkit.json"
    ))
    .expect("valid ring atom chirality fixture")
}

#[test]
fn ring_atom_chirality_cases_should_agree_with_frozen_rdkit_fixture() {
    let cases = load_cases();
    for case in cases {
        let query = QueryMol::from_str(&case.smarts)
            .unwrap_or_else(|error| panic!("failed to parse SMARTS {:?}: {error}", case.smarts));
        let canonical = query.canonicalize();
        let displayed = QueryMol::from_str(&query.to_string()).unwrap();
        let canonical_displayed = QueryMol::from_str(&canonical.to_string()).unwrap();
        for (form, query) in [
            ("parsed", &query),
            ("displayed", &displayed),
            ("canonical", &canonical),
            ("canonical displayed", &canonical_displayed),
        ] {
            let actual = query
                .matches(&case.smiles)
                .unwrap_or_else(|error| panic!("{}: {form} matcher error: {error}", case.name));
            let actual_count = query
                .match_count(&case.smiles)
                .unwrap_or_else(|error| panic!("{}: {form} match_count error: {error}", case.name));
            assert_eq!(
                actual, case.expected_match,
                "{}: {form} SMARTS {:?} against SMILES {:?}",
                case.name, case.smarts, case.smiles
            );
            assert_eq!(
                actual_count, case.expected_count,
                "{}: {form} SMARTS {:?} against SMILES {:?}: match count",
                case.name, case.smarts, case.smiles
            );
        }
    }
}

#[test]
fn ring_atom_chirality_should_survive_target_canonicalization() {
    let cases = load_cases();
    for case in cases {
        let target = Smiles::from_str(&case.smiles).unwrap_or_else(|error| {
            panic!(
                "{}: failed to parse SMILES {:?}: {error}",
                case.name, case.smiles
            )
        });
        let canonical = target.canonicalize().to_string();
        let query = QueryMol::from_str(&case.smarts)
            .unwrap_or_else(|error| panic!("failed to parse SMARTS {:?}: {error}", case.smarts));
        let actual = query.matches(&canonical).unwrap_or_else(|error| {
            panic!(
                "{}: canonical matcher error ({canonical:?}): {error}",
                case.name
            )
        });
        let actual_count = query.match_count(&canonical).unwrap_or_else(|error| {
            panic!(
                "{}: canonical match_count error ({canonical:?}): {error}",
                case.name
            )
        });
        assert_eq!(
            actual, case.expected_match,
            "{}: SMARTS {:?} against canonical {canonical:?}",
            case.name, case.smarts
        );
        assert_eq!(
            actual_count, case.expected_count,
            "{}: SMARTS {:?} against canonical {canonical:?}: match count",
            case.name, case.smarts
        );
    }
}

#[test]
fn ring_atom_chirality_should_survive_target_display_reparse() {
    let cases = load_cases();
    for case in cases {
        let target = Smiles::from_str(&case.smiles).unwrap_or_else(|error| {
            panic!(
                "{}: failed to parse SMILES {:?}: {error}",
                case.name, case.smiles
            )
        });
        let reparsed = target.to_string();
        let query = QueryMol::from_str(&case.smarts)
            .unwrap_or_else(|error| panic!("failed to parse SMARTS {:?}: {error}", case.smarts));
        let actual = query.matches(&reparsed).unwrap_or_else(|error| {
            panic!(
                "{}: reparse matcher error ({reparsed:?}): {error}",
                case.name
            )
        });
        assert_eq!(
            actual, case.expected_match,
            "{}: SMARTS {:?} against reparse {reparsed:?}",
            case.name, case.smarts
        );
    }
}

#[test]
fn ring_atom_chirality_should_hold_through_compiled_query_and_shared_scratch() {
    let cases = load_cases();
    let mut scratch = MatchScratch::new();
    for case in cases {
        let target = PreparedTarget::new(Smiles::from_str(&case.smiles).unwrap_or_else(|error| {
            panic!(
                "{}: failed to parse SMILES {:?}: {error}",
                case.name, case.smiles
            )
        }));
        let query = QueryMol::from_str(&case.smarts)
            .unwrap_or_else(|error| panic!("failed to parse SMARTS {:?}: {error}", case.smarts));
        let compiled = CompiledQuery::new(query).unwrap_or_else(|error| {
            panic!(
                "{}: failed to compile SMARTS {:?}: {error}",
                case.name, case.smarts
            )
        });
        let actual = compiled.matches_with_scratch(&target, &mut scratch);
        let actual_count = compiled.match_count(&target);
        assert_eq!(
            actual, case.expected_match,
            "{}: SMARTS {:?} against SMILES {:?}",
            case.name, case.smarts, case.smiles
        );
        assert_eq!(
            actual_count, case.expected_count,
            "{}: SMARTS {:?} against SMILES {:?}: match count",
            case.name, case.smarts, case.smiles
        );
    }
}

const RECURSIVE_CHIRAL_COMPONENT_CASES: &[(&str, &str, &str, bool, usize)] = &[
    (
        "at query matches @@ target",
        "([C@H&$(CO)](F)(O)(Cl)).N",
        "F[C@@H](O)Cl.N",
        true,
        1,
    ),
    (
        "at query misses @ target",
        "([C@H&$(CO)](F)(O)(Cl)).N",
        "F[C@H](O)Cl.N",
        false,
        0,
    ),
    (
        "at query misses plain target",
        "([C@H&$(CO)](F)(O)(Cl)).N",
        "FC(O)Cl.N",
        false,
        0,
    ),
    (
        "atat query matches @ target",
        "([C@@H&$(CO)](F)(O)(Cl)).N",
        "F[C@H](O)Cl.N",
        true,
        1,
    ),
    (
        "atat query misses @@ target",
        "([C@@H&$(CO)](F)(O)(Cl)).N",
        "F[C@@H](O)Cl.N",
        false,
        0,
    ),
    (
        "at query counts two free carbons",
        "([C@H&$(CO)](F)(O)(Cl)).N.C",
        "F[C@@H](O)Cl.N.CC",
        true,
        2,
    ),
    (
        "atat query counts two free carbons",
        "([C@@H&$(CO)](F)(O)(Cl)).N.C",
        "F[C@H](O)Cl.N.CC",
        true,
        2,
    ),
    (
        "at query misses missing carbon",
        "([C@H&$(CO)](F)(O)(Cl)).N.C",
        "F[C@@H](O)Cl.N",
        false,
        0,
    ),
];

#[test]
fn recursive_chiral_component_should_keep_enantiomer_selectivity() {
    let mut scratch = MatchScratch::new();
    for (name, smarts, smiles, expected_match, expected_count) in RECURSIVE_CHIRAL_COMPONENT_CASES {
        let query = QueryMol::from_str(smarts)
            .unwrap_or_else(|error| panic!("{name}: failed to parse SMARTS {smarts:?}: {error}"));
        let actual = query
            .matches(smiles)
            .unwrap_or_else(|error| panic!("{name}: matcher error: {error}"));
        let actual_count = query
            .match_count(smiles)
            .unwrap_or_else(|error| panic!("{name}: match_count error: {error}"));
        assert_eq!(
            actual, *expected_match,
            "{name}: SMARTS {smarts:?} against SMILES {smiles:?}"
        );
        assert_eq!(
            actual_count, *expected_count,
            "{name}: SMARTS {smarts:?} against SMILES {smiles:?}: match count"
        );

        let target =
            PreparedTarget::new(Smiles::from_str(smiles).unwrap_or_else(|error| {
                panic!("{name}: failed to parse SMILES {smiles:?}: {error}")
            }));
        let compiled = CompiledQuery::new(query)
            .unwrap_or_else(|error| panic!("{name}: failed to compile SMARTS {smarts:?}: {error}"));
        let compiled_actual = compiled.matches_with_scratch(&target, &mut scratch);
        let compiled_count = compiled.match_count(&target);
        assert_eq!(
            compiled_actual, *expected_match,
            "{name}: compiled SMARTS {smarts:?} against SMILES {smiles:?}"
        );
        assert_eq!(
            compiled_count, *expected_count,
            "{name}: compiled SMARTS {smarts:?} against SMILES {smiles:?}: match count"
        );
    }
}
