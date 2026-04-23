#![no_main]

use std::time::{Duration, Instant};

use libfuzzer_sys::fuzz_target;
use smarts_rs::{
    parse_smarts, AtomExpr, AtomPrimitive, BondExpr, BondExprTree, BondPrimitive, BracketExprTree,
    QueryAtom, QueryBond, QueryCanonicalLabeling, QueryMol,
};
use smiles_parser::bond::Bond;

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
    let deep_labeling_checks = query_allows_deep_labeling_checks(query);

    let canonical = query.canonicalize();
    assert_eq!(query.canonicalize(), canonical);
    if deep_labeling_checks {
        assert_eq!(relabel_query(query, &labeling).canonicalize(), canonical);
    }
    assert_eq!(canonical.atom_count(), query.atom_count());
    assert_eq!(canonical.bond_count(), query.bond_count());
    assert_eq!(canonical.component_count(), query.component_count());
    assert!(canonical.is_canonical());
    assert_eq!(canonical, canonical.canonicalize());
    assert_labeling_is_permutation(&canonical.canonical_labeling(), canonical.atom_count());
    if deep_labeling_checks {
        assert_eq!(
            relabel_query(&canonical, &canonical.canonical_labeling()).canonicalize(),
            canonical
        );
    }

    let canonical_smarts = query.to_canonical_smarts();
    assert_eq!(canonical_smarts, canonical.to_string());

    let reparsed = parse_smarts(&canonical_smarts).expect("canonical SMARTS must parse again");
    assert_eq!(canonical_smarts, reparsed.to_canonical_smarts());
    let recanonicalized = reparsed.canonicalize();

    assert!(recanonicalized.is_canonical());
    assert_labeling_is_permutation(&reparsed.canonical_labeling(), reparsed.atom_count());
    assert_labeling_is_permutation(
        &recanonicalized.canonical_labeling(),
        recanonicalized.atom_count(),
    );
    if deep_labeling_checks {
        assert_eq!(
            relabel_query(&reparsed, &reparsed.canonical_labeling()).canonicalize(),
            recanonicalized
        );
    }
    assert_eq!(canonical, recanonicalized);
    assert_eq!(canonical_smarts, recanonicalized.to_string());
}

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

fn query_contains_chirality(query: &QueryMol) -> bool {
    query.atoms().iter().any(atom_contains_chirality)
}

fn query_allows_deep_labeling_checks(query: &QueryMol) -> bool {
    if query_contains_chirality(query) {
        return false;
    }

    let mut budget = QueryBudget::default();
    accumulate_query_budget(query, 1, &mut budget);
    query.atom_count() <= 32
        && query.bond_count() <= 40
        && query.component_count() <= 6
        && budget.total_atoms <= 48
        && budget.total_bonds <= 56
        && budget.total_components <= 8
        && budget.recursive_queries == 0
        && budget.recursive_work <= 20
        && budget.max_depth <= 1
}

fn atom_contains_chirality(atom: &QueryAtom) -> bool {
    match &atom.expr {
        AtomExpr::Wildcard => false,
        AtomExpr::Bare { .. } => false,
        AtomExpr::Bracket(bracket) => tree_contains_chirality(&bracket.tree),
    }
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

fn relabel_query(query: &QueryMol, labeling: &QueryCanonicalLabeling) -> QueryMol {
    let order = labeling.order();
    let new_index_of_old_atom = labeling.new_index_of_old_atom();
    let mut new_component_of_old = vec![usize::MAX; query.component_count()];
    let mut next_component = 0;

    for &old_atom in order {
        let old_component = query.atoms()[old_atom].component;
        if new_component_of_old[old_component] == usize::MAX {
            new_component_of_old[old_component] = next_component;
            next_component += 1;
        }
    }

    let atoms = order
        .iter()
        .copied()
        .map(|old_atom| {
            let atom = &query.atoms()[old_atom];
            QueryAtom {
                id: new_index_of_old_atom[old_atom],
                component: new_component_of_old[atom.component],
                expr: atom.expr.clone(),
            }
        })
        .collect::<Vec<_>>();

    let mut bonds = query
        .bonds()
        .iter()
        .map(|bond| {
            let src = new_index_of_old_atom[bond.src];
            let dst = new_index_of_old_atom[bond.dst];
            let expr = if src <= dst {
                bond.expr.clone()
            } else {
                flipped_directional_bond_expr(&bond.expr)
            };
            let (src, dst) = if src <= dst { (src, dst) } else { (dst, src) };
            (src, dst, expr)
        })
        .collect::<Vec<_>>();
    bonds.sort_unstable_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then(left.1.cmp(&right.1))
            .then(left.2.cmp(&right.2))
    });
    let bonds = bonds
        .into_iter()
        .enumerate()
        .map(|(bond_id, (src, dst, expr))| QueryBond {
            id: bond_id,
            src,
            dst,
            expr,
        })
        .collect::<Vec<_>>();

    let mut component_groups = vec![None; query.component_count()];
    for (old_component, &new_component) in new_component_of_old.iter().enumerate() {
        if new_component != usize::MAX {
            component_groups[new_component] = query.component_group(old_component);
        }
    }

    QueryMol::from_parts(atoms, bonds, query.component_count(), component_groups)
}

fn flipped_directional_bond_expr(expr: &BondExpr) -> BondExpr {
    match expr {
        BondExpr::Elided => BondExpr::Elided,
        BondExpr::Query(tree) => BondExpr::Query(flipped_directional_bond_tree(tree)),
    }
}

fn flipped_directional_bond_tree(tree: &BondExprTree) -> BondExprTree {
    match tree {
        BondExprTree::Primitive(primitive) => BondExprTree::Primitive(flip_bond_primitive(*primitive)),
        BondExprTree::Not(inner) => {
            BondExprTree::Not(Box::new(flipped_directional_bond_tree(inner)))
        }
        BondExprTree::HighAnd(items) => BondExprTree::HighAnd(
            items.iter().map(flipped_directional_bond_tree).collect(),
        ),
        BondExprTree::Or(items) => {
            BondExprTree::Or(items.iter().map(flipped_directional_bond_tree).collect())
        }
        BondExprTree::LowAnd(items) => BondExprTree::LowAnd(
            items.iter().map(flipped_directional_bond_tree).collect(),
        ),
    }
}

const fn flip_bond_primitive(primitive: BondPrimitive) -> BondPrimitive {
    match primitive {
        BondPrimitive::Bond(Bond::Up) => BondPrimitive::Bond(Bond::Down),
        BondPrimitive::Bond(Bond::Down) => BondPrimitive::Bond(Bond::Up),
        other => other,
    }
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
    query.atom_count() > 24
        || query.bond_count() > 32
        || query.component_count() > 4
        || budget.total_atoms > 32
        || budget.total_bonds > 40
        || budget.total_components > 6
        || budget.recursive_queries > 0
        || budget.recursive_work > 12
        || budget.max_depth > 1
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

    run_phase_with_slow_guard("canonicalization_stability", input, &query, || {
        assert_canonicalization_is_stable(&query);
    });
    run_phase_with_slow_guard("recursive_children_stability", input, &query, || {
        assert_recursive_query_children_are_stable(&query);
    });
});
