#![no_main]

use libfuzzer_sys::fuzz_target;
use smarts_rs::{parse_smarts, AtomExpr, AtomPrimitive, BracketExprTree, QueryMol};

#[derive(Default)]
struct QueryBudget {
    total_atoms: usize,
    total_bonds: usize,
    total_components: usize,
    recursive_queries: usize,
    max_depth: usize,
}

fn assert_recursive_queries_lowered(query: &QueryMol) {
    for atom in query.atoms() {
        let AtomExpr::Bracket(bracket) = &atom.expr else {
            continue;
        };
        assert_recursive_tree_lowered(&bracket.tree);
    }
}

fn assert_recursive_tree_lowered(tree: &BracketExprTree) {
    match tree {
        BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(nested)) => {
            assert_query_render_is_stable(nested);
            assert_recursive_queries_lowered(nested);
        }
        BracketExprTree::Primitive(_) => {}
        BracketExprTree::Not(inner) => assert_recursive_tree_lowered(inner),
        BracketExprTree::HighAnd(items)
        | BracketExprTree::Or(items)
        | BracketExprTree::LowAnd(items) => {
            for item in items {
                assert_recursive_tree_lowered(item);
            }
        }
    }
}

fn assert_query_render_is_stable(query: &QueryMol) {
    let rendered = query.to_string();
    let reparsed = parse_smarts(&rendered).expect("displayed SMARTS must parse again");
    let rerendered = reparsed.to_string();
    let reparsed_again = parse_smarts(&rerendered).expect("normalized SMARTS must reparse again");

    assert_eq!(rerendered, reparsed_again.to_string());
    assert_eq!(reparsed.atom_count(), reparsed_again.atom_count());
    assert_eq!(reparsed.bond_count(), reparsed_again.bond_count());
    assert_eq!(reparsed.component_count(), reparsed_again.component_count());
}

fn query_is_too_large(query: &QueryMol) -> bool {
    let mut budget = QueryBudget::default();
    accumulate_query_budget(query, 1, &mut budget);
    query.atom_count() > 64
        || query.bond_count() > 96
        || query.component_count() > 16
        || budget.total_atoms > 128
        || budget.total_bonds > 192
        || budget.total_components > 32
        || budget.recursive_queries > 16
        || budget.max_depth > 8
}

fn accumulate_query_budget(query: &QueryMol, depth: usize, budget: &mut QueryBudget) {
    budget.total_atoms += query.atom_count();
    budget.total_bonds += query.bond_count();
    budget.total_components += query.component_count();
    budget.max_depth = budget.max_depth.max(depth);

    for atom in query.atoms() {
        let AtomExpr::Bracket(bracket) = &atom.expr else {
            continue;
        };
        accumulate_tree_budget(&bracket.tree, depth, budget);
    }
}

fn accumulate_tree_budget(tree: &BracketExprTree, depth: usize, budget: &mut QueryBudget) {
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
    if data.len() > 512 {
        return;
    }

    let candidate = String::from_utf8_lossy(data);
    let wrapped = format!("[{}]", candidate);

    let Ok(query) = parse_smarts(&wrapped) else {
        return;
    };
    if query_is_too_large(&query) {
        return;
    }

    assert_recursive_queries_lowered(&query);

    let rendered = query.to_string();
    let reparsed = parse_smarts(&rendered).expect("displayed SMARTS must parse again");

    assert_recursive_queries_lowered(&reparsed);
    assert_query_render_is_stable(&query);
});
