#![no_main]

use libfuzzer_sys::fuzz_target;
use query_fuzz_common::{
    assert_query_render_is_stable, assert_recursive_queries_lowered, query_exceeds_budget,
    QueryBudgetLimits,
};
use smarts_rs::{fuzz_parse_bracket_text, parse_smarts, AtomExpr, AtomPrimitive, BracketExprTree};

mod query_fuzz_common;

const QUERY_LIMITS: QueryBudgetLimits = QueryBudgetLimits {
    atom_count: 48,
    bond_count: 72,
    component_count: 12,
    total_atoms: 128,
    total_bonds: 192,
    total_components: 32,
    recursive_queries: 16,
    recursive_work: usize::MAX,
    max_depth: 8,
};

fuzz_target!(|data: &[u8]| {
    if data.len() > 192 {
        return;
    }

    let candidate = String::from_utf8_lossy(data);
    let bracket_text = format!("$({})", candidate);
    let Ok(raw_bracket) = fuzz_parse_bracket_text(&bracket_text) else {
        return;
    };
    let Ok(expected_nested) = parse_smarts(candidate.as_ref()) else {
        return;
    };

    let wrapped = format!("[{}]", raw_bracket);
    let query = parse_smarts(&wrapped).expect("validated recursive bracket must parse as SMARTS");
    if query_exceeds_budget(&query, QUERY_LIMITS)
        || query_exceeds_budget(&expected_nested, QUERY_LIMITS)
    {
        return;
    }

    assert_eq!(query.atom_count(), 1);
    assert_eq!(query.bond_count(), 0);
    assert_recursive_queries_lowered(&query);

    let AtomExpr::Bracket(bracket) = &query.atoms()[0].expr else {
        panic!("wrapped recursive SMARTS must stay a bracket atom");
    };
    let BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(nested)) = &bracket.tree else {
        panic!("wrapped recursive SMARTS must lower into a nested query");
    };
    assert_eq!(nested.atom_count(), expected_nested.atom_count());
    assert_eq!(nested.bond_count(), expected_nested.bond_count());
    assert_eq!(nested.component_count(), expected_nested.component_count());
    assert_eq!(nested.component_groups(), expected_nested.component_groups());
    assert_query_render_is_stable(nested);

    assert_query_render_is_stable(&query);
});
