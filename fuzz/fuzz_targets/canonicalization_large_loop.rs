#![no_main]

use std::time::{Duration, Instant};

use libfuzzer_sys::fuzz_target;
use smarts_parser::{parse_smarts, AtomExpr, AtomPrimitive, BracketExprTree, QueryMol};

#[derive(Default)]
struct QueryBudget {
    total_atoms: usize,
    total_bonds: usize,
    total_components: usize,
    recursive_queries: usize,
    recursive_work: usize,
    max_depth: usize,
}

const PHASE_SLOW_LIMIT: Duration = Duration::from_millis(250);

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

fn query_is_too_large(query: &QueryMol) -> bool {
    let mut budget = QueryBudget::default();
    accumulate_query_budget(query, 1, &mut budget);
    query.atom_count() > 48
        || query.bond_count() > 64
        || query.component_count() > 8
        || budget.total_atoms > 80
        || budget.total_bonds > 96
        || budget.total_components > 12
        || budget.recursive_queries > 6
        || budget.recursive_work > 24
        || budget.max_depth > 3
}

fn accumulate_query_budget(query: &QueryMol, depth: usize, budget: &mut QueryBudget) {
    budget.total_atoms += query.atom_count();
    budget.total_bonds += query.bond_count();
    budget.total_components += query.component_count();
    budget.recursive_work += depth
        * (query.atom_count()
            + query.bond_count()
            + 4 * query.component_count().max(1));
    budget.max_depth = budget.max_depth.max(depth);

    for atom in query.atoms() {
        let AtomExpr::Bracket(bracket) = &atom.expr else {
            continue;
        };
        accumulate_tree_budget(&bracket.tree, depth, budget);
    }
}

fn accumulate_tree_budget(tree: &BracketExprTree, depth: usize, budget: &mut QueryBudget) {
    budget.recursive_work += depth;
    match tree {
        BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(nested)) => {
            budget.recursive_queries += 1;
            accumulate_query_budget(nested, depth + 1, budget);
        }
        BracketExprTree::Primitive(_) => {}
        BracketExprTree::Not(inner) => accumulate_tree_budget(inner, depth, budget),
        BracketExprTree::HighAnd(items)
        | BracketExprTree::Or(items)
        | BracketExprTree::LowAnd(items) => {
            for item in items {
                accumulate_tree_budget(item, depth, budget);
            }
        }
    }
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
    if query_is_too_large(&query) {
        return;
    }

    assert_reduced_canonicalization_is_stable(input, &query);
});
