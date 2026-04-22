#![no_main]

use libfuzzer_sys::fuzz_target;
use smarts_parser::{
    parse_smarts, AtomExpr, AtomPrimitive, BracketExprTree, QueryCanonicalLabeling, QueryMol,
};

#[derive(Default)]
struct QueryBudget {
    total_atoms: usize,
    total_bonds: usize,
    total_components: usize,
    recursive_queries: usize,
    recursive_work: usize,
    max_depth: usize,
}

fn assert_labeling_is_permutation(labeling: &QueryCanonicalLabeling, atom_count: usize) {
    assert_eq!(labeling.order().len(), atom_count);
    assert_eq!(labeling.new_index_of_old_atom().len(), atom_count);

    let mut seen_old = vec![false; atom_count];
    let mut seen_new = vec![false; atom_count];

    for &old_atom in labeling.order() {
        assert!(old_atom < atom_count);
        assert!(!seen_old[old_atom]);
        seen_old[old_atom] = true;
    }

    for &new_atom in labeling.new_index_of_old_atom() {
        assert!(new_atom < atom_count);
        assert!(!seen_new[new_atom]);
        seen_new[new_atom] = true;
    }
}

fn assert_canonicalization_is_stable(query: &QueryMol) {
    let labeling = query.canonical_labeling();
    assert_labeling_is_permutation(&labeling, query.atom_count());

    let canonical = query.canonicalize();
    assert_eq!(canonical.atom_count(), query.atom_count());
    assert_eq!(canonical.bond_count(), query.bond_count());
    assert_eq!(canonical.component_count(), query.component_count());
    assert!(canonical.is_canonical());
    assert_eq!(canonical, canonical.canonicalize());
    assert_labeling_is_permutation(&canonical.canonical_labeling(), canonical.atom_count());

    let canonical_smarts = query.to_canonical_smarts();
    assert_eq!(canonical_smarts, canonical.to_string());

    let reparsed = parse_smarts(&canonical_smarts).expect("canonical SMARTS must parse again");
    let recanonicalized = reparsed.canonicalize();

    assert!(recanonicalized.is_canonical());
    assert_labeling_is_permutation(&reparsed.canonical_labeling(), reparsed.atom_count());
    assert_labeling_is_permutation(
        &recanonicalized.canonical_labeling(),
        recanonicalized.atom_count(),
    );
    assert_eq!(canonical, recanonicalized);
    assert_eq!(canonical_smarts, recanonicalized.to_string());
}

fn assert_recursive_canonicalization(tree: &BracketExprTree) {
    match tree {
        BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(nested)) => {
            assert_canonicalization_is_stable(nested);
            assert_recursive_query_children_are_stable(nested);
        }
        BracketExprTree::Primitive(_) => {}
        BracketExprTree::Not(inner) => assert_recursive_canonicalization(inner),
        BracketExprTree::HighAnd(items)
        | BracketExprTree::Or(items)
        | BracketExprTree::LowAnd(items) => {
            for item in items {
                assert_recursive_canonicalization(item);
            }
        }
    }
}

fn assert_recursive_query_children_are_stable(query: &QueryMol) {
    for atom in query.atoms() {
        let AtomExpr::Bracket(bracket) = &atom.expr else {
            continue;
        };
        assert_recursive_canonicalization(&bracket.tree);
    }
}

fn query_is_too_large(query: &QueryMol) -> bool {
    let mut budget = QueryBudget::default();
    accumulate_query_budget(query, 1, &mut budget);
    query.atom_count() > 96
        || query.bond_count() > 144
        || query.component_count() > 24
        || budget.total_atoms > 192
        || budget.total_bonds > 288
        || budget.total_components > 48
        || budget.recursive_queries > 24
        || budget.recursive_work > 40
        || budget.max_depth > 8
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

    assert_canonicalization_is_stable(&query);
    assert_recursive_query_children_are_stable(&query);
});
