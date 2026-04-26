#![no_main]

use libfuzzer_sys::fuzz_target;
use query_fuzz_common::{
    assert_query_render_is_stable, assert_recursive_queries_lowered, query_exceeds_budget,
    QueryBudgetLimits,
};
use smarts_rs::parse_smarts;

mod query_fuzz_common;

const QUERY_LIMITS: QueryBudgetLimits = QueryBudgetLimits {
    atom_count: 64,
    bond_count: 96,
    component_count: 16,
    total_atoms: 128,
    total_bonds: 192,
    total_components: 32,
    recursive_queries: 16,
    recursive_work: usize::MAX,
    max_depth: 8,
};

fuzz_target!(|data: &[u8]| {
    if data.len() > 512 {
        return;
    }

    let candidate = String::from_utf8_lossy(data);
    let wrapped = format!("[{}]", candidate);

    let Ok(query) = parse_smarts(&wrapped) else {
        return;
    };
    if query_exceeds_budget(&query, QUERY_LIMITS) {
        return;
    }

    assert_recursive_queries_lowered(&query);

    let rendered = query.to_string();
    let reparsed = parse_smarts(&rendered).expect("displayed SMARTS must parse again");

    assert_recursive_queries_lowered(&reparsed);
    assert_query_render_is_stable(&query);
});
