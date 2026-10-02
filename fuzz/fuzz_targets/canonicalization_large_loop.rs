#![no_main]

use std::time::{Duration, Instant};

use libfuzzer_sys::fuzz_target;
use query_fuzz_common::{query_exceeds_budget, QueryBudgetLimits};
use smarts_rs::{parse_smarts, QueryMol};

mod query_fuzz_common;

const PHASE_SLOW_LIMIT: Duration = Duration::from_millis(250);
const QUERY_LIMITS: QueryBudgetLimits = QueryBudgetLimits {
    atom_count: 48,
    bond_count: 64,
    component_count: 8,
    total_atoms: 80,
    total_bonds: 96,
    total_components: 12,
    recursive_queries: 6,
    recursive_work: 24,
    max_depth: 3,
};

fn run_phase_with_slow_guard<R>(
    phase: &str,
    input: &str,
    query: &QueryMol,
    f: impl FnOnce() -> R,
) -> R {
    let started = Instant::now();
    let result = f();
    let elapsed = started.elapsed();
    assert!(
        elapsed <= PHASE_SLOW_LIMIT,
        "slow canonicalization phase `{phase}` after {:?} on input `{input}` (atoms={}, bonds={}, components={})",
        elapsed,
        query.atom_count(),
        query.bond_count(),
        query.component_count()
    );
    result
}

fn assert_reduced_canonicalization_is_stable(input: &str, query: &QueryMol) {
    let canonical = run_phase_with_slow_guard("canonicalize", input, query, || {
        query.canonicalize()
    });
    assert_eq!(query.atom_count(), canonical.atom_count());
    assert_eq!(query.bond_count(), canonical.bond_count());
    assert_eq!(query.component_count(), canonical.component_count());
    assert!(canonical.is_canonical());
    assert_eq!(
        canonical,
        run_phase_with_slow_guard("canonicalize_again", input, &canonical, || {
            canonical.canonicalize()
        })
    );

    let canonical_smarts =
        run_phase_with_slow_guard("to_canonical_smarts", input, query, || {
            query.to_canonical_smarts()
        });
    assert_eq!(canonical_smarts, canonical.to_string());

    let reparsed =
        parse_smarts(&canonical_smarts).expect("canonical SMARTS must parse again");
    let recanonicalized = run_phase_with_slow_guard(
        "reparsed_canonicalize",
        input,
        &reparsed,
        || reparsed.canonicalize(),
    );
    assert_eq!(canonical, recanonicalized);
    assert_eq!(canonical_smarts, recanonicalized.to_string());
}

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

    assert_reduced_canonicalization_is_stable(input, &query);
});
