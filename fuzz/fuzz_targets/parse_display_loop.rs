#![no_main]

use libfuzzer_sys::fuzz_target;
use query_fuzz_common::{
    assert_query_render_is_stable, assert_recursive_queries_lowered, query_exceeds_budget,
    QueryBudgetLimits,
};
use smarts_rs::parse_smarts;

mod query_fuzz_common;

const QUERY_LIMITS: QueryBudgetLimits = QueryBudgetLimits {
    atom_count: 96,
    bond_count: 144,
    component_count: 24,
    total_atoms: 192,
    total_bonds: 288,
    total_components: 48,
    recursive_queries: 24,
    recursive_work: usize::MAX,
    max_depth: 8,
};

fuzz_target!(|data: &[u8]| {
    if data.len() > 256 {
        return;
    }

    let candidate = String::from_utf8_lossy(data);
    let input = candidate.as_ref();

    let Ok(query) = parse_smarts(input) else {
        return;
    };
    if query_exceeds_budget(&query, QUERY_LIMITS) {
        return;
    }

    assert_recursive_queries_lowered(&query);
    assert_query_render_is_stable(&query);
});
