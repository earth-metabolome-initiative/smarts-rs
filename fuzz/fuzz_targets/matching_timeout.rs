#![no_main]

mod matching_timeout_common;

use std::time::Duration;

use libfuzzer_sys::fuzz_target;
use matching_timeout_common::{
    for_each_fuzz_input, query_is_too_large, run_phase_with_slow_guard, selected_target_ids,
    split_query_and_target, target_fixture, MAX_INPUT_LEN,
};
use smarts_rs::{
    parse_smarts, CompiledQuery, MatchLimitResult, MatchScratch, PreparedTarget, QueryMol,
    QueryScreen, TargetCorpusScratch,
};
use smiles_parser::Smiles;

const MATCH_TIME_LIMIT: Duration = Duration::from_secs(30);

struct MatchRun<'a> {
    input: &'a str,
    variant: &'a str,
    mode: &'a str,
    query: &'a QueryMol,
    compiled: &'a CompiledQuery,
    target_smiles: &'a str,
    target: &'a PreparedTarget,
}

fn evaluate_query_variant(input: &str, variant: &str, query: &QueryMol, data: &[u8]) {
    let Ok(compiled) = run_phase_with_slow_guard("compile", input, query, || {
        CompiledQuery::new(query.clone())
    }) else {
        return;
    };

    let fixture = target_fixture();
    let screen = QueryScreen::new(compiled.query());
    let mut corpus_scratch = TargetCorpusScratch::new();
    let mut candidate_ids = Vec::new();
    run_phase_with_slow_guard("index_candidates", input, query, || {
        fixture.index.candidate_ids_with_scratch_into(
            &screen,
            &mut corpus_scratch,
            &mut candidate_ids,
        );
    });

    let mut scratch = MatchScratch::new();
    for &target_id in &candidate_ids {
        let target = &fixture.targets[target_id];
        assert_time_limited_match_finishes(
            MatchRun {
                input,
                variant,
                mode: "indexed",
                query,
                compiled: &compiled,
                target_smiles: target.smiles,
                target: &target.prepared,
            },
            &mut scratch,
        );
    }

    for target_id in selected_target_ids(data, fixture.targets.len()) {
        let target = &fixture.targets[target_id];
        assert_time_limited_match_finishes(
            MatchRun {
                input,
                variant,
                mode: "sampled",
                query,
                compiled: &compiled,
                target_smiles: target.smiles,
                target: &target.prepared,
            },
            &mut scratch,
        );
    }
}

fn assert_time_limited_match_finishes(run: MatchRun<'_>, scratch: &mut MatchScratch) {
    let result =
        run.compiled
            .matches_with_scratch_and_time_limit(run.target, scratch, MATCH_TIME_LIMIT);
    assert!(
        !matches!(result, MatchLimitResult::Exceeded),
        "SMARTS matching time limit exceeded in {mode} mode after {:?}; variant={variant}; target=`{}`; input=`{}`; canonical=`{}`; atoms={}; bonds={}; components={}",
        MATCH_TIME_LIMIT,
        run.target_smiles,
        run.input,
        run.query.to_canonical_smarts(),
        run.query.atom_count(),
        run.query.bond_count(),
        run.query.component_count(),
        mode = run.mode,
        variant = run.variant,
    );
}

fn evaluate_smarts_input(input: &str, data: &[u8]) {
    if input.is_empty() || input.len() > MAX_INPUT_LEN {
        return;
    }

    let (query_text, target_override) = split_query_and_target(input);
    let Ok(query) = parse_smarts(query_text) else {
        return;
    };
    if query_is_too_large(query_text, &query) {
        return;
    }

    let canonical =
        run_phase_with_slow_guard("canonicalize", input, &query, || query.canonicalize());
    if query_is_too_large(query_text, &canonical) {
        return;
    }

    if let Some(target_text) = target_override {
        if let Ok(target) = target_text.parse::<Smiles>() {
            let target = PreparedTarget::new(target);
            if let Ok(compiled) = CompiledQuery::new(canonical.clone()) {
                let mut scratch = MatchScratch::new();
                assert_time_limited_match_finishes(
                    MatchRun {
                        input,
                        variant: "explicit-target",
                        mode: "explicit",
                        query: &canonical,
                        compiled: &compiled,
                        target_smiles: target_text,
                        target: &target,
                    },
                    &mut scratch,
                );
            }
        }
    }

    evaluate_query_variant(input, "raw", &query, data);
    if canonical != query {
        evaluate_query_variant(input, "canonical", &canonical, data);
    }
}

fuzz_target!(|data: &[u8]| {
    for_each_fuzz_input(data, evaluate_smarts_input);
});
