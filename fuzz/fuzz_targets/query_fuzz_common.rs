#![allow(dead_code)]

use smarts_rs::{parse_smarts, AtomExpr, AtomPrimitive, BracketExprTree, QueryMol};

#[derive(Debug, Clone, Copy)]
pub struct QueryBudgetLimits {
    pub atom_count: usize,
    pub bond_count: usize,
    pub component_count: usize,
    pub total_atoms: usize,
    pub total_bonds: usize,
    pub total_components: usize,
    pub recursive_queries: usize,
    pub recursive_work: usize,
    pub max_depth: usize,
}

#[derive(Debug, Default, Clone, Copy)]
struct QueryBudget {
    total_atoms: usize,
    total_bonds: usize,
    total_components: usize,
    recursive_queries: usize,
    recursive_work: usize,
    max_depth: usize,
}

pub fn assert_recursive_queries_lowered(query: &QueryMol) {
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

pub fn assert_query_render_is_stable(query: &QueryMol) {
    let rendered = query.to_string();
    let reparsed = parse_smarts(&rendered).expect("displayed SMARTS must parse again");
    let rerendered = reparsed.to_string();
    let reparsed_again = parse_smarts(&rerendered).expect("normalized SMARTS must reparse again");

    assert_eq!(rerendered, reparsed_again.to_string());
    assert_eq!(reparsed.atom_count(), reparsed_again.atom_count());
    assert_eq!(reparsed.bond_count(), reparsed_again.bond_count());
    assert_eq!(reparsed.component_count(), reparsed_again.component_count());
    assert_eq!(reparsed.component_groups(), reparsed_again.component_groups());
}

pub fn query_exceeds_budget(query: &QueryMol, limits: QueryBudgetLimits) -> bool {
    let mut budget = QueryBudget::default();
    accumulate_query_budget(query, 1, &mut budget);
    query.atom_count() > limits.atom_count
        || query.bond_count() > limits.bond_count
        || query.component_count() > limits.component_count
        || budget.total_atoms > limits.total_atoms
        || budget.total_bonds > limits.total_bonds
        || budget.total_components > limits.total_components
        || budget.recursive_queries > limits.recursive_queries
        || budget.recursive_work > limits.recursive_work
        || budget.max_depth > limits.max_depth
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

pub fn query_contains_chirality(query: &QueryMol) -> bool {
    query.atoms().iter().any(|atom| match &atom.expr {
        AtomExpr::Wildcard | AtomExpr::Bare { .. } => false,
        AtomExpr::Bracket(bracket) => tree_contains_chirality(&bracket.tree),
    })
}

fn tree_contains_chirality(tree: &BracketExprTree) -> bool {
    match tree {
        BracketExprTree::Primitive(AtomPrimitive::Chirality(_)) => true,
        BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(nested)) => {
            query_contains_chirality(nested)
        }
        BracketExprTree::Primitive(_) => false,
        BracketExprTree::Not(inner) => tree_contains_chirality(inner),
        BracketExprTree::HighAnd(items)
        | BracketExprTree::Or(items)
        | BracketExprTree::LowAnd(items) => items.iter().any(tree_contains_chirality),
    }
}
