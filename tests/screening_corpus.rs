//! Corpus-grounded screening soundness tests.

use core::str::FromStr;

use serde::Deserialize;
use smarts_rs::{
    CompiledQuery, MatchScratch, PreparedTarget, QueryMol, QueryScreen, TargetCorpusIndex,
    TargetScreen,
};
use smiles_rs::Smiles;

#[derive(Debug, Deserialize)]
struct ExpectedCase {
    smarts: String,
    smiles: String,
    expected_match: bool,
}

#[test]
fn screen_never_filters_true_matches_from_frozen_fixtures() {
    for fixture in [
        include_str!("../corpus/matching/single-atom-v0.rdkit.json"),
        include_str!("../corpus/matching/connected-v0.rdkit.json"),
        include_str!("../corpus/matching/ring-v0.rdkit.json"),
        include_str!("../corpus/matching/counts-v0.rdkit.json"),
        include_str!("../corpus/matching/disconnected-v0.rdkit.json"),
        include_str!("../corpus/matching/recursive-v0.rdkit.json"),
        include_str!("../corpus/matching/stereo-v0.rdkit.json"),
    ] {
        let cases: Vec<ExpectedCase> = serde_json::from_str(fixture).expect("valid frozen fixture");
        for case in cases {
            if !case.expected_match {
                continue;
            }
            let query = QueryMol::from_str(&case.smarts).expect("valid SMARTS");
            let target = PreparedTarget::new(case.smiles.parse::<Smiles>().expect("valid SMILES"));
            let query_screen = QueryScreen::new(&query);
            let target_screen = TargetScreen::new(&target);
            let index = TargetCorpusIndex::new(core::slice::from_ref(&target));
            assert!(
                query_screen.may_match(&target_screen),
                "screen rejected known true match: SMARTS {:?} vs SMILES {:?}",
                case.smarts,
                case.smiles
            );
            assert_eq!(
                index.candidate_ids(&query_screen),
                vec![0],
                "index rejected known true match: SMARTS {:?} vs SMILES {:?}",
                case.smarts,
                case.smiles
            );
        }
    }
}

#[test]
fn index_never_filters_scalar_matches_from_frozen_fixtures() {
    for fixture in [
        include_str!("../corpus/matching/single-atom-v0.rdkit.json"),
        include_str!("../corpus/matching/connected-v0.rdkit.json"),
        include_str!("../corpus/matching/ring-v0.rdkit.json"),
        include_str!("../corpus/matching/counts-v0.rdkit.json"),
        include_str!("../corpus/matching/disconnected-v0.rdkit.json"),
        include_str!("../corpus/matching/recursive-v0.rdkit.json"),
        include_str!("../corpus/matching/stereo-v0.rdkit.json"),
        include_str!("../corpus/matching/stereo-gap-v1.rdkit.json"),
        include_str!("../corpus/matching/stereo-gap-v2.rdkit.json"),
        include_str!("../corpus/matching/stereo-gap-v3.rdkit.json"),
        include_str!("../corpus/matching/stereo-gap-v4.rdkit.json"),
        include_str!("../corpus/matching/stereo-gap-v5.rdkit.json"),
        include_str!("../corpus/matching/stereo-gap-v6.rdkit.json"),
        include_str!("../corpus/matching/stereo-gap-v7.rdkit.json"),
        include_str!("../corpus/matching/benchmark-alerts-v0.rdkit.json"),
        include_str!("../corpus/matching/benchmark-aromaticity-v0.rdkit.json"),
        include_str!("../corpus/matching/benchmark-counted-v0.rdkit.json"),
        include_str!("../corpus/matching/benchmark-extensions-v0.rdkit.json"),
    ] {
        let cases: Vec<ExpectedCase> = serde_json::from_str(fixture).expect("valid frozen fixture");
        for case in cases {
            let query = QueryMol::from_str(&case.smarts).expect("valid SMARTS");
            let compiled = CompiledQuery::new(query.clone()).expect("supported SMARTS");
            let target = PreparedTarget::new(case.smiles.parse::<Smiles>().expect("valid SMILES"));
            let mut match_scratch = MatchScratch::new();
            if !compiled.matches_with_scratch(&target, &mut match_scratch) {
                continue;
            }

            let query_screen = QueryScreen::new(&query);
            let index = TargetCorpusIndex::new(core::slice::from_ref(&target));
            assert_eq!(
                index.candidate_ids(&query_screen),
                vec![0],
                "index rejected scalar match: SMARTS {:?} vs SMILES {:?}",
                case.smarts,
                case.smiles
            );
        }
    }
}
