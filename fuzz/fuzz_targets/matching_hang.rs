#![no_main]

mod matching_timeout_common;

use libfuzzer_sys::fuzz_target;
use matching_timeout_common::{
    for_each_fuzz_input, query_is_too_large, run_phase_with_slow_guard, selected_target_ids,
    split_query_and_target, target_fixture,
};
use smarts_rs::{
    parse_smarts, CompiledQuery, MatchScratch, PreparedTarget, QueryMol, QueryScreen,
    TargetCorpusScratch,
};
use smiles_parser::Smiles;

struct MatchRun<'a> {
    compiled: &'a CompiledQuery,
    target: &'a PreparedTarget,
}

fn evaluate_query_variant(input: &str, _variant: &str, query: &QueryMol, data: &[u8]) {
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
        let _target_smiles = target.smiles;
        assert_plain_match_returns(
            MatchRun {
                compiled: &compiled,
                target: &target.prepared,
            },
            &mut scratch,
        );
    }

    for target_id in selected_target_ids(data, fixture.targets.len()) {
        let target = &fixture.targets[target_id];
        let _target_smiles = target.smiles;
        assert_plain_match_returns(
            MatchRun {
                compiled: &compiled,
                target: &target.prepared,
            },
            &mut scratch,
        );
    }
}

fn assert_plain_match_returns(run: MatchRun<'_>, scratch: &mut MatchScratch) {
    let _ = run.compiled.matches_with_scratch(run.target, scratch);
}

fn evaluate_smarts_input(input: &str, data: &[u8]) {
    if input.is_empty() || input.len() > matching_timeout_common::MAX_INPUT_LEN {
        return;
    }

    let (query_text, target_override) = split_query_and_target(input);
    let Ok(query) = parse_smarts(query_text) else {
        return;
    };
    if query_is_too_large(query_text, &query) {
        return;
    }

    let canonical = run_phase_with_slow_guard("canonicalize", input, &query, || query.canonicalize());
    if query_is_too_large(query_text, &canonical) {
        return;
    }

    if let Some(target_text) = target_override {
        if let Ok(target) = target_text.parse::<Smiles>() {
            let target = PreparedTarget::new(target);
            if let Ok(compiled) = CompiledQuery::new(canonical.clone()) {
                let mut scratch = MatchScratch::new();
                assert_plain_match_returns(
                    MatchRun {
                        compiled: &compiled,
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
