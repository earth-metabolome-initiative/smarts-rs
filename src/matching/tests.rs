use alloc::{format, string::String, vec, vec::Vec};
use core::str::FromStr;

use smiles_parser::Smiles;

use super::{
    component_constraints_match, component_embedding_assignment_exists, select_next_query_atom,
    select_next_query_atom_for_target, should_use_disconnected_component_search, AtomStereoCache,
    CompiledQuery, ComponentEmbedding, ComponentEmbeddingSet, ComponentMatcher, ComponentPlanEntry,
    FreshSearchBuffers, MatchScratch, RecursiveMatchCache, NO_MATCH_LIMIT,
};
use super::{AtomFastPredicate, MatchLimitResult, MatchOutcomeLimitResult};
use crate::error::SmartsMatchError;
use crate::prepared::PreparedTarget;
use crate::QueryMol;

fn query_matches_smiles(smarts: &str, smiles: &str) -> bool {
    QueryMol::from_str(smarts).unwrap().matches(smiles).unwrap()
}

fn recursive_component_chain(depth: usize) -> String {
    let mut nested = String::from("CsO");
    for _ in 0..depth {
        nested = format!("Cs[$({nested})]O.OO");
    }
    format!("[$({nested})]O.O")
}

fn assert_coverage_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-12,
        "coverage {actual} != expected {expected}"
    );
}

#[test]
fn rejects_empty_target() {
    let query = QueryMol::from_str("C").unwrap();
    let error = query.matches("").unwrap_err();
    assert!(matches!(error, SmartsMatchError::EmptyTarget));
}

#[test]
fn rejects_invalid_target_smiles() {
    let query = QueryMol::from_str("C").unwrap();
    let error = query.matches("C)").unwrap_err();
    assert!(matches!(
        error,
        SmartsMatchError::InvalidTargetSmiles { .. }
    ));
}

#[test]
fn negated_directional_bond_forms_match_nothing_in_current_rdkit_slice() {
    assert!(!QueryMol::from_str("F/!\\C=C/F")
        .unwrap()
        .matches("F/C=C/F")
        .unwrap());
    assert!(!QueryMol::from_str("F/!\\C=C/F")
        .unwrap()
        .matches("F\\C=C\\F")
        .unwrap());
    assert!(!QueryMol::from_str("F\\!/C=C/F")
        .unwrap()
        .matches("F/C=C\\F")
        .unwrap());
    assert!(!QueryMol::from_str("F\\!/C=C/F")
        .unwrap()
        .matches("FC=CF")
        .unwrap());
}

#[test]
fn single_atom_supported_query_matches() {
    let query = QueryMol::from_str("[O;H1]").unwrap();
    assert!(query.matches("CCO").unwrap());
}

#[test]
fn compiled_query_matches_existing_prepared_path() {
    let compiled = CompiledQuery::new(QueryMol::from_str("F/C=C/F").unwrap()).unwrap();
    let prepared = PreparedTarget::new(Smiles::from_str("F/C=C/F").unwrap());
    assert!(compiled.matches(&prepared));
    assert!(compiled.query().matches("F/C=C/F").unwrap());
}

#[test]
fn compiled_query_matches_with_reusable_scratch() {
    let compiled = CompiledQuery::new(QueryMol::from_str("COCOC").unwrap()).unwrap();
    let good = PreparedTarget::new(Smiles::from_str("CCOCOCC").unwrap());
    let bad = PreparedTarget::new(Smiles::from_str("CCCCCCC").unwrap());
    let mut scratch = MatchScratch::new();

    assert!(compiled.matches_with_scratch(&good, &mut scratch));
    assert!(!compiled.matches_with_scratch(&bad, &mut scratch));
    assert_eq!(
        compiled.matches(&good),
        compiled.matches_with_scratch(&good, &mut scratch)
    );
}

#[test]
fn compiled_query_interrupt_reports_exhaustion_without_poisoning_scratch() {
    let compiled = CompiledQuery::new(QueryMol::from_str("[$([#6]-[#6])].[#7]").unwrap()).unwrap();
    let target = PreparedTarget::new(Smiles::from_str("CCC.N").unwrap());
    let mut scratch = MatchScratch::new();

    assert_eq!(
        compiled.matches_with_scratch_and_interrupt(&target, &mut scratch, || true),
        MatchLimitResult::Exceeded
    );
    assert_eq!(
        compiled.matches_with_scratch_and_interrupt(&target, &mut scratch, || false),
        MatchLimitResult::Complete(compiled.matches(&target))
    );
    assert!(compiled.matches(&target));
}

#[test]
#[cfg(all(
    feature = "std",
    not(all(target_family = "wasm", target_os = "unknown"))
))]
fn compiled_query_time_limit_reports_exhaustion_without_poisoning_scratch() {
    use std::time::Duration;

    let compiled = CompiledQuery::new(QueryMol::from_str("[$([#6]-[#6])].[#7]").unwrap()).unwrap();
    let target = PreparedTarget::new(Smiles::from_str("CCC.N").unwrap());
    let mut scratch = MatchScratch::new();

    assert_eq!(
        compiled.matches_with_scratch_and_time_limit(&target, &mut scratch, Duration::ZERO),
        MatchLimitResult::Exceeded
    );
    assert_eq!(
        compiled.matches_with_scratch_and_time_limit(&target, &mut scratch, Duration::from_mins(1)),
        MatchLimitResult::Complete(compiled.matches(&target))
    );
    assert!(compiled.matches(&target));
}

#[test]
fn compiled_query_outcome_interrupt_reports_exhaustion_without_poisoning_scratch() {
    let compiled = CompiledQuery::new(QueryMol::from_str("CC").unwrap()).unwrap();
    let target = PreparedTarget::new(Smiles::from_str("CCC").unwrap());
    let mut scratch = MatchScratch::new();
    let expected = compiled.match_outcome(&target);

    assert_eq!(
        compiled.match_outcome_with_scratch_and_interrupt(&target, &mut scratch, || true),
        MatchOutcomeLimitResult::Exceeded
    );
    assert_eq!(
        compiled.match_outcome_with_scratch_and_interrupt(&target, &mut scratch, || false),
        MatchOutcomeLimitResult::Complete(expected)
    );
    assert_eq!(
        compiled.match_outcome_with_interrupt(&target, || false),
        MatchOutcomeLimitResult::Complete(expected)
    );
    assert!(expected.matched);
    assert_coverage_close(expected.coverage, 1.0);
}

#[test]
#[cfg(all(
    feature = "std",
    not(all(target_family = "wasm", target_os = "unknown"))
))]
fn compiled_query_outcome_time_limit_reports_exhaustion_without_poisoning_scratch() {
    use std::time::Duration;

    let compiled = CompiledQuery::new(QueryMol::from_str("CC").unwrap()).unwrap();
    let target = PreparedTarget::new(Smiles::from_str("CCC").unwrap());
    let mut scratch = MatchScratch::new();
    let expected = compiled.match_outcome(&target);

    assert_eq!(
        compiled.match_outcome_with_scratch_and_time_limit(&target, &mut scratch, Duration::ZERO),
        MatchOutcomeLimitResult::Exceeded
    );
    assert_eq!(
        compiled.match_outcome_with_scratch_and_time_limit(
            &target,
            &mut scratch,
            Duration::from_mins(1)
        ),
        MatchOutcomeLimitResult::Complete(expected)
    );
    assert_eq!(
        compiled.match_outcome_with_time_limit(&target, Duration::from_mins(1)),
        MatchOutcomeLimitResult::Complete(expected)
    );
    assert!(expected.matched);
    assert_coverage_close(expected.coverage, 1.0);
}

#[test]
fn reusable_scratch_matches_fast_path_query_shapes() {
    let cases = [
        ("O", "CCO", true),
        ("O", "CCC", false),
        ("C=O", "CC(=O)O", true),
        ("C=O", "CCO", false),
        ("COC", "CCOC", true),
        ("COC", "CCO", false),
        ("COCO", "CCOCO", true),
        ("COCO", "CCOCC", false),
        ("[#6]-[#6]-[#8]-[#6]-[#8]", "CCOCO", true),
        ("[#6]-[#6]-[#8]-[#6]-[#8]", "CCOCC", false),
    ];
    let mut scratch = MatchScratch::new();

    for (smarts, smiles, expected) in cases {
        let compiled = CompiledQuery::new(QueryMol::from_str(smarts).unwrap()).unwrap();
        let target = PreparedTarget::new(Smiles::from_str(smiles).unwrap());

        assert_eq!(compiled.matches(&target), expected);
        assert_eq!(
            compiled.matches_with_scratch(&target, &mut scratch),
            expected,
            "scratch mismatch for {smarts} against {smiles}"
        );
    }
}

#[test]
fn counted_matches_follow_rdkit_uniquify_for_symmetric_queries() {
    let query = QueryMol::from_str("CC").unwrap();
    assert_eq!(query.match_count("CC").unwrap(), 1);
    assert_eq!(
        query.substructure_matches("CC").unwrap().as_ref(),
        &[alloc::boxed::Box::<[usize]>::from([0, 1])]
    );
}

#[test]
fn counted_matches_include_overlapping_embeddings() {
    let query = QueryMol::from_str("CC").unwrap();
    assert_eq!(query.match_count("CCC").unwrap(), 2);
    assert_eq!(
        query.substructure_matches("CCC").unwrap().as_ref(),
        &[
            alloc::boxed::Box::<[usize]>::from([0, 1]),
            alloc::boxed::Box::<[usize]>::from([1, 2]),
        ]
    );
}

#[test]
fn match_outcome_reports_combined_atom_and_bond_coverage() {
    let ring = QueryMol::from_str("c1ccccc1").unwrap();
    let outcome = ring.match_outcome("Cc1ccccc1").unwrap();
    assert!(outcome.matched);
    assert_coverage_close(outcome.coverage, 12.0 / 14.0);

    let atom_only = QueryMol::from_str("[#6]").unwrap();
    let outcome = atom_only.match_outcome("c1ccccc1").unwrap();
    assert!(outcome.matched);
    assert_coverage_close(outcome.coverage, 6.0 / 12.0);

    let no_match = QueryMol::from_str("[#7]")
        .unwrap()
        .match_outcome("CC")
        .unwrap();
    assert!(!no_match.matched);
    assert_coverage_close(no_match.coverage, 0.0);
}

#[test]
fn match_outcome_unions_coverage_across_full_embeddings() {
    let query = CompiledQuery::new(QueryMol::from_str("CC").unwrap()).unwrap();
    let target = PreparedTarget::new(Smiles::from_str("CCC").unwrap());
    let mut scratch = MatchScratch::new();

    let outcome = query.match_outcome_with_scratch(&target, &mut scratch);
    assert!(outcome.matched);
    assert_coverage_close(outcome.coverage, 1.0);
}

#[test]
fn match_outcome_counts_bonds_from_all_valid_embeddings() {
    let query = CompiledQuery::new(QueryMol::from_str("CCC").unwrap()).unwrap();
    let target = PreparedTarget::new(Smiles::from_str("C1CC1").unwrap());

    let outcome = query.match_outcome(&target);
    assert!(outcome.matched);
    assert_coverage_close(outcome.coverage, 1.0);
}

#[test]
fn counted_matches_choose_a_canonical_query_order_representative() {
    let query = QueryMol::from_str("C~C~C~C").unwrap();
    assert_eq!(query.match_count("C1CCC=1").unwrap(), 1);
    assert_eq!(
        query.substructure_matches("C1CCC=1").unwrap().as_ref(),
        &[alloc::boxed::Box::<[usize]>::from([0, 1, 2, 3])]
    );
}

#[test]
fn three_atom_tree_fast_path_preserves_counting_and_materialization() {
    let symmetric = CompiledQuery::new(QueryMol::from_str("C(C)C").unwrap()).unwrap();
    let branched_target = PreparedTarget::new(Smiles::from_str("CC(C)C").unwrap());
    assert_eq!(symmetric.match_count(&branched_target), 3);
    assert_eq!(
        symmetric.substructure_matches(&branched_target).as_ref(),
        &[
            alloc::boxed::Box::<[usize]>::from([1, 0, 2]),
            alloc::boxed::Box::<[usize]>::from([1, 0, 3]),
            alloc::boxed::Box::<[usize]>::from([1, 2, 3]),
        ]
    );

    let asymmetric = CompiledQuery::new(QueryMol::from_str("C(=O)O").unwrap()).unwrap();
    let anhydride_like = PreparedTarget::new(Smiles::from_str("OC(=O)OC(=O)O").unwrap());
    assert_eq!(asymmetric.match_count(&anhydride_like), 4);
    assert_eq!(
        asymmetric.substructure_matches(&anhydride_like).as_ref(),
        &[
            alloc::boxed::Box::<[usize]>::from([1, 2, 0]),
            alloc::boxed::Box::<[usize]>::from([1, 2, 3]),
            alloc::boxed::Box::<[usize]>::from([4, 5, 3]),
            alloc::boxed::Box::<[usize]>::from([4, 5, 6]),
        ]
    );
}

#[test]
fn counted_matches_handle_simple_single_atom_cases() {
    assert_eq!(
        QueryMol::from_str("C").unwrap().match_count("CCC").unwrap(),
        3
    );
    assert_eq!(
        QueryMol::from_str("[#8]")
            .unwrap()
            .match_count("O=CO")
            .unwrap(),
        2
    );
    assert_eq!(
        QueryMol::from_str("[#8]")
            .unwrap()
            .match_count("CC.O")
            .unwrap(),
        1
    );
}

#[test]
fn counted_matches_agree_with_boolean_matching() {
    let query = QueryMol::from_str("C.C").unwrap();
    let prepared = PreparedTarget::new(Smiles::from_str("CC").unwrap());
    let compiled = CompiledQuery::new(query).unwrap();
    assert_eq!(
        compiled.matches(&prepared),
        compiled.match_count(&prepared) > 0
    );
}

#[test]
fn bondless_boolean_fast_path_matches_counting_reference() {
    let cases = [
        ("C.C", "CC"),
        ("C.C", "C"),
        ("(C.C)", "CC"),
        ("(C.C)", "C.C"),
        ("(C).(C)", "CC"),
        ("(C).(C)", "C.C"),
        ("[$([#6,#7])].[R]", "c1ccccc1"),
        (
            "[!!12C,16O,D,D11,h,v14].([!r6&$([#6,#7]):57773].[R:30837])",
            "c1ccccc1",
        ),
    ];
    let mut scratch = MatchScratch::new();

    for (smarts, smiles) in cases {
        let query = QueryMol::from_str(smarts).unwrap();
        let target = PreparedTarget::new(Smiles::from_str(smiles).unwrap());
        let compiled = CompiledQuery::new(query).unwrap();
        let expected = compiled.match_count(&target) > 0;

        assert_eq!(
            compiled.matches(&target),
            expected,
            "compiled boolean mismatch for {smarts} against {smiles}"
        );
        assert_eq!(
            compiled.matches_with_scratch(&target, &mut scratch),
            expected,
            "scratch boolean mismatch for {smarts} against {smiles}"
        );
    }

    let long_anchored_target = format!("{}.Cl.[Na]", "C".repeat(64));
    let query = QueryMol::from_str("[Cl].[Na]").unwrap();
    let target = PreparedTarget::new(Smiles::from_str(&long_anchored_target).unwrap());
    let compiled = CompiledQuery::new(query).unwrap();
    let expected = compiled.match_count(&target) > 0;
    assert_eq!(compiled.matches(&target), expected);
    assert_eq!(
        compiled.matches_with_scratch(&target, &mut scratch),
        expected
    );
}

#[test]
fn target_aware_root_selection_prefers_more_selective_unanchored_component() {
    let compiled = CompiledQuery::new(QueryMol::from_str("C.[Cl-].[Na+]").unwrap()).unwrap();
    let target = PreparedTarget::new(Smiles::from_str("CC.[Cl-].[Na+]").unwrap());
    let mut atom_stereo_cache = AtomStereoCache::new();
    let mut recursive_cache =
        RecursiveMatchCache::new(compiled.recursive_cache_slots, target.atom_count());
    let mut scratch = FreshSearchBuffers::new(&compiled, &target, &mut recursive_cache);
    let context =
        scratch
            .view()
            .into_context(&compiled, &mut atom_stereo_cache, &mut recursive_cache);

    assert_eq!(
        select_next_query_atom(
            context.query_neighbors,
            context.query_atom_scores,
            context.query_to_target,
        ),
        0
    );
    assert_eq!(select_next_query_atom_for_target(&context), 1);
}

#[test]
fn target_aware_root_selection_scans_unanchored_recursive_misses() {
    let compiled = CompiledQuery::new(QueryMol::from_str("C.[!$([!#1])]").unwrap()).unwrap();
    let target = PreparedTarget::new(Smiles::from_str("CCCC").unwrap());
    let mut atom_stereo_cache = AtomStereoCache::new();
    let mut recursive_cache =
        RecursiveMatchCache::new(compiled.recursive_cache_slots, target.atom_count());
    let mut scratch = FreshSearchBuffers::new(&compiled, &target, &mut recursive_cache);
    let context =
        scratch
            .view()
            .into_context(&compiled, &mut atom_stereo_cache, &mut recursive_cache);

    assert_eq!(context.query_atom_anchor_widths[1], 0);
    assert_eq!(select_next_query_atom_for_target(&context), 1);
    assert!(!compiled.matches(&target));
}

#[test]
fn anchored_complete_matchers_only_skip_rechecks_when_every_predicate_is_covered() {
    let query = QueryMol::from_str("[A;X1-2].O").unwrap();
    assert!(!query.matches("CC(C)C.O").unwrap());
    assert_eq!(query.match_count("CC(C)C.O").unwrap(), 0);
}

#[test]
fn benchmark_screened_pair_does_not_regress_to_false_positive_match() {
    let query =
        QueryMol::from_str("[#6;D1;H3]-[#6;D2]-[#6;D3]=[#6;D3]-[#6;D3](=[#8;D1])-[#8;D1;H1]")
            .unwrap();
    let target = "C[C@@H]1C2[C@H](C(=O)N2C(=C1OC)C(=O)O)[C@@H](C)O";

    assert!(!query.matches(target).unwrap());
    assert_eq!(query.match_count(target).unwrap(), 0);
}

#[test]
fn disconnected_boolean_precheck_matches_counting_reference() {
    let cases = [
        (
            "*@[B,X{16-}].[!R,b].([!#16]~[!#8].[!107*&r6]-[#6&H]=&@[D3])",
            "C1CCCCC1O.CC",
        ),
        (
            "*-,:,=C.(*(!=[!-]-&@1)!-[#6]-&@1~[!#6&H]-[!Cl].[!#6&H]-[!Cl]-,/&@[#6&R])",
            "ClCC1=CCCCC1.CC",
        ),
        ("C.O.N", "CCO"),
        ("(C.O).N", "CCO.N"),
    ];
    let mut scratch = MatchScratch::new();

    for (smarts, smiles) in cases {
        let query = QueryMol::from_str(smarts).unwrap();
        let target = PreparedTarget::new(Smiles::from_str(smiles).unwrap());
        let compiled = CompiledQuery::new(query).unwrap();
        let expected = compiled.match_count(&target) > 0;

        assert_eq!(
            compiled.matches(&target),
            expected,
            "compiled boolean mismatch for {smarts} against {smiles}"
        );
        assert_eq!(
            compiled.matches_with_scratch(&target, &mut scratch),
            expected,
            "scratch boolean mismatch for {smarts} against {smiles}"
        );
    }
}

#[test]
fn disconnected_prechecks_skip_recursive_components() {
    let compiled = CompiledQuery::new(QueryMol::from_str("[$(C.O)].N").unwrap()).unwrap();
    let precheck_count = compiled
        .component_plan
        .iter()
        .filter(|component| component.supports_precheck())
        .count();

    assert_eq!(precheck_count, 1);
}

#[test]
fn disconnected_component_search_handles_repeated_component_fuzz_artifact() {
    let smarts = concat!(
        "[!!12C,16O,D,D11,h,v14].",
        "([!r6&$([#6,#7]):57773].[R:30837]).",
        "[+0,z11,+0&+0,+0]-;@[+0&+0,+0,z11,+0](!:[+0,z11,+0&+0,+0]).",
        "[+0&+0,+0,z11,+0]!:[+0,z11,+0&+0,+0](-;@[+0&+0,+0,z11,+0]).",
        "[z11,+0&+0,+0,z11]!:[+0,+0,z11,+0](!:[+0,z11,+0&+0,+0]).",
        "[+0,+0,z11,+0]!:[+0,z11,+0&+0,+0](-;@[+0&+0,+0,z11,+0])",
    );
    let target = PreparedTarget::new(
        Smiles::from_str("CCN1C(=O)C(=CNC2=CC=C(C=C2)CCN3CCN(CC3)C)SC1=C(C#N)C(=O)OCC").unwrap(),
    );
    let compiled = CompiledQuery::new(QueryMol::from_str(smarts).unwrap()).unwrap();
    let mut scratch = MatchScratch::new();

    let mut polls = 0usize;
    assert_eq!(
        compiled.matches_with_scratch_and_interrupt(&target, &mut scratch, || {
            polls += 1;
            polls > 1_000
        }),
        MatchLimitResult::Complete(false)
    );
}

#[test]
fn disconnected_component_search_handles_recursive_component_timeout_artifact() {
    let smarts = concat!(
        "*~[!!#6&!!#6&!!#6&!!#6]",
        "(~[!!#6&!!#6&!!#6&!!#6]~[!!#6&!!#6&!!#6&!!#6]).",
        "[!!#6&!!#6&!!#6&!!#6].",
        "[!!#6&!#6&$([#6,#7])&r]~[!!R&!!#6&!!#6].",
        "[!!#6&!!#6&!!#6&!!#6].",
        "[!!#6&!!#6&!!#6&!!#6]",
    );
    let target = PreparedTarget::new(
        Smiles::from_str(
            "C1CCN(C1)CCOC2=CC=C(C=C2)CC3=C(SC4=CC=CC=C43)C5=CC=C(C=C5)CCNCC6=CN=CC=C6",
        )
        .unwrap(),
    );
    let compiled = CompiledQuery::new(QueryMol::from_str(smarts).unwrap()).unwrap();
    let mut scratch = MatchScratch::new();
    let mut polls = 0usize;

    assert!(compiled
        .component_plan
        .iter()
        .all(ComponentPlanEntry::supports_component_search));
    assert_eq!(
        compiled.matches_with_scratch_and_interrupt(&target, &mut scratch, || {
            polls += 1;
            polls > 1_000
        }),
        MatchLimitResult::Complete(false)
    );
}

#[test]
fn double_negated_bracket_atoms_keep_fast_anchors() {
    let compiled =
        CompiledQuery::new(QueryMol::from_str("[!!#6&!!#6&!!#6&!!#6]").unwrap()).unwrap();
    let matcher = &compiled.atom_matchers[0];

    assert!(matcher.complete);
    assert!(matcher
        .predicates
        .iter()
        .any(|predicate| { matches!(predicate, AtomFastPredicate::AtomicNumber(6)) }));
    assert!(compiled.matches(&PreparedTarget::new(Smiles::from_str("C").unwrap())));
    assert!(!compiled.matches(&PreparedTarget::new(Smiles::from_str("O").unwrap())));
}

#[test]
fn bracket_expression_matching_polls_interrupt_limit() {
    let mut smarts = String::from("[");
    for index in 0..5000 {
        if index > 0 {
            smarts.push('&');
        }
        smarts.push_str("!#7");
    }
    smarts.push(']');

    let compiled = CompiledQuery::new(QueryMol::from_str(&smarts).unwrap()).unwrap();
    let target = PreparedTarget::new(Smiles::from_str("C").unwrap());
    let mut scratch = MatchScratch::new();
    let mut polls = 0usize;

    assert_eq!(
        compiled.matches_with_scratch_and_interrupt(&target, &mut scratch, || {
            polls += 1;
            polls > 1
        }),
        MatchLimitResult::Exceeded
    );
    assert!(polls > 1);
}

#[test]
fn small_disconnected_tail_uses_component_assignment_fuzz_artifacts() {
    let cases = [
        concat!(
            "*[$([#6])&!!#6](~[!!#6&!!#6&!!#6&!!#6]~[!!#6&!!#6&!!#6&!!#6]).",
            "[!!#6&!!#6&!!#6&!!#6].",
            "[!!#6&!$([#6])&!!#6&!!#6].",
            "[!!#6&!!#6&!!#6&!!#6].",
            "[!!#6&!!#6&!!#6&!!#6]",
        ),
        concat!(
            "*~[D3&!D3&!a,$([#6,#7]),c:0]([!*:0]~[!R,!!R,!!R&!!#6]).",
            "[!!#6&!!#6&!!#6&!!#6].",
            "[!!#6&!!#6&!!#6&!!#6].",
            "[!!#6&!!#6&!!#6&!!#6].",
            "[!!#6&!!#6&!!#6&!!#6]",
        ),
        concat!(
            "[!!#6&!!#6&!!#6&!!#6]~[!!#6&!!#6&!!#6&!!#6].",
            "[!!#6&!!#6&*&!!#6].",
            "[!!#6&!!#6&!!#6,!*:0][!*:0]([!*:0][!$([#7])]).",
            "[$([#7])].",
            "[!!#6&!!#6&!!#6&!!#6]",
        ),
    ];
    let target = PreparedTarget::new(
        Smiles::from_str(
            "C1CCN(C1)CCOC2=CC=C(C=C2)CC3=C(SC4=CC=CC=C43)C5=CC=C(C=C5)CCNCC6=CN=CC=C6",
        )
        .unwrap(),
    );

    for smarts in cases {
        let compiled = CompiledQuery::new(QueryMol::from_str(smarts).unwrap()).unwrap();
        let mut scratch = MatchScratch::new();
        let mut polls = 0usize;

        assert!(
            should_use_disconnected_component_search(&compiled),
            "{smarts}"
        );
        assert!(matches!(
            compiled.matches_with_scratch_and_interrupt(&target, &mut scratch, || {
                polls += 1;
                polls > 1_000
            }),
            MatchLimitResult::Complete(_)
        ));
    }
}

#[test]
fn large_ungrouped_wildcard_components_stay_on_general_search_path() {
    let compiled =
        CompiledQuery::new(QueryMol::from_str("*-***.*.*-***.*.***.***-*.****-*******").unwrap())
            .unwrap();

    assert!(!should_use_disconnected_component_search(&compiled));
}

#[test]
fn ungrouped_disconnected_wildcard_query_does_not_enumerate_all_component_embeddings() {
    let query = QueryMol::from_str("*-***.*.*-***.*.***.***-*.****-*******").unwrap();
    let target = PreparedTarget::new(
        Smiles::from_str(
            "C1CCN(C1)CCOC2=CC=C(C=C2)CC3=C(SC4=CC=CC=C43)C5=CC=C(C=C5)CCNCC6=CN=CC=C6",
        )
        .unwrap(),
    );
    let compiled = CompiledQuery::new(query).unwrap();
    let mut scratch = MatchScratch::new();
    let mut polls = 0usize;

    assert!(matches!(
        compiled.matches_with_scratch_and_interrupt(&target, &mut scratch, || {
            polls += 1;
            polls > 1_000
        }),
        MatchLimitResult::Complete(_)
    ));
}

#[test]
fn disconnected_component_search_respects_zero_level_groups() {
    let compiled = CompiledQuery::new(QueryMol::from_str("(C-C.N)").unwrap()).unwrap();
    let same_component = PreparedTarget::new(Smiles::from_str("CCN").unwrap());
    let separate_components = PreparedTarget::new(Smiles::from_str("CC.N").unwrap());

    assert!(compiled.matches(&same_component));
    assert!(!compiled.matches(&separate_components));
}

#[test]
fn disconnected_component_assignment_handles_deep_component_lists_without_recursing() {
    let component_count = 50_000usize;
    let entries = (0..component_count)
        .map(|component_id| ComponentPlanEntry {
            component_id,
            group: None,
            atom_count: 1,
            precheckable: true,
            matcher: ComponentMatcher::SingleAtom(0),
        })
        .collect::<Vec<_>>();
    let component_sets = entries
        .iter()
        .enumerate()
        .map(|(target_atom, entry)| ComponentEmbeddingSet {
            entry,
            embeddings: vec![ComponentEmbedding {
                target_atoms: alloc::boxed::Box::<[usize]>::from([target_atom]),
                target_component: Some(0),
            }],
        })
        .collect::<Vec<_>>();
    let mut used_target_atoms = vec![false; component_count];
    let mut group_targets = Vec::new();

    assert!(component_embedding_assignment_exists(
        &component_sets,
        &mut used_target_atoms,
        &mut group_targets,
        &NO_MATCH_LIMIT,
    ));
}

#[test]
fn compiled_query_handles_deep_recursive_component_chain_regression() {
    let smarts = recursive_component_chain(12);
    let compiled = CompiledQuery::new(QueryMol::from_str(&smarts).unwrap()).unwrap();
    let precheck_count = compiled
        .component_plan
        .iter()
        .filter(|component| component.supports_precheck())
        .count();

    assert_eq!(compiled.query().component_count(), 2);
    assert_eq!(precheck_count, 1);
}

#[test]
fn low_level_boolean_tree_is_respected() {
    let query = QueryMol::from_str("[!O]").unwrap();
    assert!(query.matches("C").unwrap());
    assert!(!query.matches("O").unwrap());
}

#[test]
fn bracket_or_is_respected() {
    let query = QueryMol::from_str("[N,O]").unwrap();
    assert!(query.matches("CCO").unwrap());
}

#[test]
fn connected_query_does_not_reuse_target_atoms() {
    let query = QueryMol::from_str("CCC").unwrap();
    assert!(!query.matches("CC").unwrap());
}

#[test]
fn connected_query_matches_simple_cycle() {
    let query = QueryMol::from_str("C1CC1").unwrap();
    assert!(query.matches("C1CC1").unwrap());
    assert!(!query.matches("CCC").unwrap());
}

#[test]
fn disconnected_query_matches_without_grouping() {
    let query = QueryMol::from_str("C.C").unwrap();
    assert!(query.matches("CC").unwrap());
    assert!(query.matches("C.C").unwrap());
    assert!(!query.matches("C").unwrap());
}

#[test]
fn bond_boolean_logic_is_respected() {
    let query = QueryMol::from_str("C!#N").unwrap();
    assert!(query.matches("CN").unwrap());
    assert!(!query.matches("CC#N").unwrap());
}

#[test]
fn connectivity_and_valence_primitives_are_respected() {
    assert!(QueryMol::from_str("[X4]").unwrap().matches("C").unwrap());
    assert!(!QueryMol::from_str("[X3]").unwrap().matches("C").unwrap());
    assert!(QueryMol::from_str("[N&X]").unwrap().matches("C#N").unwrap());

    assert!(QueryMol::from_str("[v4]")
        .unwrap()
        .matches("c1ccccc1")
        .unwrap());
    assert!(!QueryMol::from_str("[v3]")
        .unwrap()
        .matches("c1ccccc1")
        .unwrap());
    assert!(QueryMol::from_str("[Cl&v]")
        .unwrap()
        .matches("CCCl")
        .unwrap());
}

#[test]
fn hybridization_primitives_are_respected() {
    assert_eq!(
        QueryMol::from_str("[^1]")
            .unwrap()
            .match_count("CC#N")
            .unwrap(),
        2
    );
    assert_eq!(
        QueryMol::from_str("[^2]")
            .unwrap()
            .match_count("CC=CF")
            .unwrap(),
        2
    );
    assert_eq!(
        QueryMol::from_str("[^2]")
            .unwrap()
            .match_count("c1ccccc1")
            .unwrap(),
        6
    );
    assert_eq!(
        QueryMol::from_str("[^2]")
            .unwrap()
            .match_count("CC(=O)NC")
            .unwrap(),
        3
    );
    assert_eq!(
        QueryMol::from_str("[^3]")
            .unwrap()
            .match_count("CC")
            .unwrap(),
        2
    );
    assert_eq!(
        QueryMol::from_str("[^3]")
            .unwrap()
            .match_count("CC=CF")
            .unwrap(),
        2
    );
}

#[test]
fn ring_atom_primitives_are_respected() {
    assert!(QueryMol::from_str("[R]").unwrap().matches("C1CC1").unwrap());
    assert!(!QueryMol::from_str("[R]").unwrap().matches("CCC").unwrap());
    assert!(QueryMol::from_str("[R0]").unwrap().matches("CCC").unwrap());
    assert!(QueryMol::from_str("[r3]")
        .unwrap()
        .matches("C1CC1")
        .unwrap());
    assert!(QueryMol::from_str("[x2]")
        .unwrap()
        .matches("C1CC1")
        .unwrap());
}

#[test]
fn ring_bond_primitives_are_respected() {
    assert!(QueryMol::from_str("C@C").unwrap().matches("C1CC1").unwrap());
    assert!(!QueryMol::from_str("C@C").unwrap().matches("CC").unwrap());
    assert!(QueryMol::from_str("C!@C").unwrap().matches("CC").unwrap());
    assert!(!QueryMol::from_str("C!@C")
        .unwrap()
        .matches("C1CC1")
        .unwrap());
}

#[test]
fn ordinary_attached_hydrogens_are_not_matchable_as_hydrogen_atoms() {
    assert!(!QueryMol::from_str("[H]").unwrap().matches("[H]Cl").unwrap());
    assert!(!QueryMol::from_str("[#1]")
        .unwrap()
        .matches("[H]Cl")
        .unwrap());
    assert!(!QueryMol::from_str("[H]Cl")
        .unwrap()
        .matches("[H]Cl")
        .unwrap());
    assert!(QueryMol::from_str("[H][H]")
        .unwrap()
        .matches("[H][H]")
        .unwrap());
    assert!(QueryMol::from_str("[2H]")
        .unwrap()
        .matches("[2H]Cl")
        .unwrap());
}

#[test]
fn zero_level_grouping_requires_same_target_component() {
    let query = QueryMol::from_str("(C.C)").unwrap();
    let target = PreparedTarget::new(Smiles::from_str("CC").unwrap());
    let compiled = CompiledQuery::new(query.clone()).unwrap();
    let mapping = [Some(0), None];
    assert_eq!(query.component_groups(), &[Some(0), Some(0)]);
    assert_eq!(target.connected_component(0), Some(0));
    assert_eq!(target.connected_component(1), Some(0));
    assert!(component_constraints_match(1, 1, &query, &target, &mapping));
    assert!(compiled.matches(&target));
    assert!(query.matches("CC").unwrap());
    assert!(QueryMol::from_str("(C.C)")
        .unwrap()
        .matches("CC.C")
        .unwrap());
    assert!(!QueryMol::from_str("(C.C)").unwrap().matches("C.C").unwrap());
}

#[test]
fn separate_groups_require_different_target_components() {
    assert!(!QueryMol::from_str("(C).(C)")
        .unwrap()
        .matches("CC")
        .unwrap());
    assert!(QueryMol::from_str("(C).(C)")
        .unwrap()
        .matches("C.C")
        .unwrap());
    assert!(QueryMol::from_str("(C).(C)")
        .unwrap()
        .matches("CC.C")
        .unwrap());
}

#[test]
fn grouped_and_ungrouped_components_interact_correctly() {
    assert!(QueryMol::from_str("(C).C").unwrap().matches("CC").unwrap());
    assert!(QueryMol::from_str("(C).C").unwrap().matches("C.C").unwrap());
    assert!(QueryMol::from_str("(C).(C).C")
        .unwrap()
        .matches("CC.C")
        .unwrap());
    assert!(!QueryMol::from_str("(C).(C.C)")
        .unwrap()
        .matches("CCC")
        .unwrap());
    assert!(QueryMol::from_str("(C).(C.C)")
        .unwrap()
        .matches("CC.C")
        .unwrap());
}

#[test]
fn recursive_queries_are_respected() {
    assert!(QueryMol::from_str("[$(C)]").unwrap().matches("C").unwrap());
    assert!(!QueryMol::from_str("[$(O)]").unwrap().matches("C").unwrap());
    assert!(QueryMol::from_str("[$([#6;R])]")
        .unwrap()
        .matches("C1CCCCC1")
        .unwrap());
    assert!(!QueryMol::from_str("[$([#6;R])]")
        .unwrap()
        .matches("CC")
        .unwrap());
    assert!(QueryMol::from_str("[$(CO)]")
        .unwrap()
        .matches("CO")
        .unwrap());
    assert!(!QueryMol::from_str("[$(CO)]")
        .unwrap()
        .matches("CC")
        .unwrap());
    assert!(QueryMol::from_str("[C&$(*O)]")
        .unwrap()
        .matches("CO")
        .unwrap());
    assert!(!QueryMol::from_str("[C&$(*O)]")
        .unwrap()
        .matches("CC")
        .unwrap());
}

#[test]
fn explicit_tetrahedral_chirality_is_respected() {
    assert!(query_matches_smiles("F[C@](Cl)Br", "F[C@H](Cl)Br"));
    assert!(!query_matches_smiles("F[C@](Cl)Br", "F[C@@H](Cl)Br"));
    assert!(query_matches_smiles("F[C@@](Cl)Br", "F[C@@H](Cl)Br"));
    assert!(!query_matches_smiles("F[C@@](Cl)Br", "F[C@H](Cl)Br"));
    assert!(query_matches_smiles("F[C@H](Cl)Br", "F[C@H](Cl)Br"));
    assert!(!query_matches_smiles("F[C@H](Cl)Br", "F[C@@H](Cl)Br"));
    assert!(query_matches_smiles("F[C@@H](Cl)Br", "F[C@@H](Cl)Br"));
    assert!(!query_matches_smiles("F[C@@H](Cl)Br", "F[C@H](Cl)Br"));
    assert!(query_matches_smiles("F[C@TH1](Cl)Br", "F[C@H](Cl)Br"));
    assert!(!query_matches_smiles("F[C@TH1](Cl)Br", "F[C@@H](Cl)Br"));
    assert!(query_matches_smiles("F[C@TH2](Cl)Br", "F[C@@H](Cl)Br"));
    assert!(!query_matches_smiles("F[C@TH2](Cl)Br", "FC(Cl)Br"));
    assert!(query_matches_smiles("Br[C@TH1](Cl)F", "F[C@@H](Cl)Br"));
    assert!(!query_matches_smiles("Br[C@TH1](Cl)F", "F[C@H](Cl)Br"));
    assert!(query_matches_smiles("[C@TH2](Br)(Cl)F", "F[C@H](Cl)Br"));
    assert!(!query_matches_smiles("[C@TH2](Br)(Cl)F", "F[C@@H](Cl)Br"));
    assert!(query_matches_smiles("Br[C@](Cl)F", "F[C@@H](Cl)Br"));
    assert!(query_matches_smiles("Br[C@@](Cl)F", "F[C@H](Cl)Br"));
    assert!(query_matches_smiles("Br[C@H](Cl)F", "F[C@@H](Cl)Br"));
    assert!(!query_matches_smiles("Br[C@H](Cl)F", "F[C@H](Cl)Br"));
    assert!(query_matches_smiles("[C@H](F)(Cl)Br", "F[C@@H](Cl)Br"));
    assert!(!query_matches_smiles("[C@H](F)(Cl)Br", "F[C@H](Cl)Br"));
    assert!(query_matches_smiles("[C@@H](F)(Cl)Br", "F[C@H](Cl)Br"));
    assert!(!query_matches_smiles("[C@@H](F)(Cl)Br", "F[C@@H](Cl)Br"));
}

#[test]
fn underconstrained_tetrahedral_queries_do_not_overconstrain() {
    assert!(QueryMol::from_str("[C@](F)Cl")
        .unwrap()
        .matches("F[C@H](Cl)Br")
        .unwrap());
    assert!(QueryMol::from_str("[C@](F)Cl")
        .unwrap()
        .matches("F[C@@H](Cl)Br")
        .unwrap());
    assert!(QueryMol::from_str("[C@@](F)Cl")
        .unwrap()
        .matches("F[C@H](Cl)Br")
        .unwrap());
    assert!(QueryMol::from_str("[C@@](F)Cl")
        .unwrap()
        .matches("F[C@@H](Cl)Br")
        .unwrap());
    assert!(QueryMol::from_str("[C@H](F)Cl")
        .unwrap()
        .matches("F[C@H](Cl)Br")
        .unwrap());
    assert!(QueryMol::from_str("[C@H](F)Cl")
        .unwrap()
        .matches("F[C@@H](Cl)Br")
        .unwrap());
    assert!(QueryMol::from_str("[C@@H](F)Cl")
        .unwrap()
        .matches("F[C@H](Cl)Br")
        .unwrap());
    assert!(QueryMol::from_str("[C@@H](F)Cl")
        .unwrap()
        .matches("F[C@@H](Cl)Br")
        .unwrap());
}

#[test]
fn supported_nontetrahedral_query_classes_do_not_constrain_matches() {
    for smarts in [
        "[C@AL1](F)(Cl)Br",
        "[C@AL2](F)(Cl)Br",
        "[C@SP1](F)(Cl)Br",
        "[C@SP2](F)(Cl)Br",
        "[C@SP3](F)(Cl)Br",
        "[C@TB1](F)(Cl)Br",
        "[C@TB20](F)(Cl)Br",
        "[C@OH1](F)(Cl)Br",
        "[C@OH30](F)(Cl)Br",
    ] {
        let query = QueryMol::from_str(smarts).unwrap();
        assert!(query.matches("F[C@H](Cl)Br").unwrap(), "{smarts}");
        assert!(query.matches("F[C@@H](Cl)Br").unwrap(), "{smarts}");
        assert!(query.matches("FC(Cl)Br").unwrap(), "{smarts}");
    }
}

#[test]
fn multidirectional_endpoint_queries_are_respected() {
    assert!(query_matches_smiles("F/C=C(/Cl)\\Br", "F/C=C(/Cl)Br"));
    assert!(!query_matches_smiles("F/C=C(/Cl)\\Br", "F/C=C(\\Cl)Br"));
    assert!(query_matches_smiles("F/,\\C=C/F", "F/C=C/F"));
    assert!(query_matches_smiles("F/,\\C=C/F", "F\\C=C\\F"));
    assert!(!query_matches_smiles("F/,\\C=C/F", "F/C=C\\F"));
    assert!(query_matches_smiles("F/;\\C=C/F", "F/C=C/F"));
    assert!(query_matches_smiles("F\\,/C=C/F", "F/C=C\\F"));
    assert!(query_matches_smiles("F/&\\C=C/F", "F/C=C/F"));
    assert!(query_matches_smiles("F\\&/C=C/F", "F/C=C\\F"));
}

#[test]
fn semantic_double_bond_stereo_is_respected() {
    assert!(QueryMol::from_str("C/C").unwrap().matches("CC").unwrap());
    assert!(QueryMol::from_str("F/C=C")
        .unwrap()
        .matches("F/C=C/F")
        .unwrap());
    assert!(QueryMol::from_str("F/C=C")
        .unwrap()
        .matches("F/C=C\\F")
        .unwrap());
    assert!(QueryMol::from_str("F/C=C")
        .unwrap()
        .matches("FC=CF")
        .unwrap());
    assert!(QueryMol::from_str("C=C/F")
        .unwrap()
        .matches("F/C=C/F")
        .unwrap());
    assert!(QueryMol::from_str("C=C/F")
        .unwrap()
        .matches("FC=CF")
        .unwrap());
    assert!(QueryMol::from_str("F/C=C/F")
        .unwrap()
        .matches("F/C=C/F")
        .unwrap());
    assert!(QueryMol::from_str("F/C=C/F")
        .unwrap()
        .matches("F\\C=C\\F")
        .unwrap());
    assert!(!QueryMol::from_str("F/C=C/F")
        .unwrap()
        .matches("F/C=C\\F")
        .unwrap());
    assert!(!QueryMol::from_str("F/C=C/F")
        .unwrap()
        .matches("FC=CF")
        .unwrap());
    assert!(QueryMol::from_str("F/C=C\\F")
        .unwrap()
        .matches("F\\C=C/F")
        .unwrap());
    assert!(QueryMol::from_str("C/C=C/C")
        .unwrap()
        .matches("C/C=C/C")
        .unwrap());
    assert!(!QueryMol::from_str("C/C=C/C")
        .unwrap()
        .matches("C/C=C\\C")
        .unwrap());
    assert!(!QueryMol::from_str("C/C=C/C")
        .unwrap()
        .matches("CC=CC")
        .unwrap());
}
