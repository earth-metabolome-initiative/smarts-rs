#![no_main]

use libfuzzer_sys::fuzz_target;
use smarts_parser::{
    fuzz_parse_bracket_text, parse_smarts, AtomExpr, AtomPrimitive, BracketExprTree, QueryMol,
};
use std::collections::BTreeMap;

fn assert_same_query_structure(left: &QueryMol, right: &QueryMol) {
    assert_eq!(left.atom_count(), right.atom_count());
    assert_eq!(left.bond_count(), right.bond_count());
    assert_eq!(left.component_count(), right.component_count());
    assert_eq!(left.component_groups(), right.component_groups());

    let left_atoms = left
        .atoms()
        .iter()
        .map(|atom| (atom.component, atom.expr.to_string()))
        .collect::<Vec<_>>();
    let right_atoms = right
        .atoms()
        .iter()
        .map(|atom| (atom.component, atom.expr.to_string()))
        .collect::<Vec<_>>();
    assert_eq!(left_atoms, right_atoms);

    let left_bonds = left
        .bonds()
        .iter()
        .map(|bond| (bond.src.min(bond.dst), bond.src.max(bond.dst), bond.expr.clone()))
        .collect::<Vec<_>>();
    let right_bonds = right
        .bonds()
        .iter()
        .map(|bond| (bond.src.min(bond.dst), bond.src.max(bond.dst), bond.expr.clone()))
        .collect::<Vec<_>>();
    assert_eq!(multiset_counts(&left_bonds), multiset_counts(&right_bonds));
}

fn multiset_counts<T>(items: &[T]) -> BTreeMap<T, usize>
where
    T: Ord + Clone,
{
    let mut counts = BTreeMap::new();
    for item in items {
        *counts.entry(item.clone()).or_insert(0) += 1;
    }
    counts
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
            let reparsed = parse_smarts(&nested.to_string())
                .expect("nested recursive SMARTS must reparse");
            assert_same_query_structure(&reparsed, nested);
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

fuzz_target!(|data: &[u8]| {
    if data.len() > 2048 {
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

    assert_eq!(query.atom_count(), 1);
    assert_eq!(query.bond_count(), 0);
    assert_recursive_queries_lowered(&query);

    let AtomExpr::Bracket(bracket) = &query.atoms()[0].expr else {
        panic!("wrapped recursive SMARTS must stay a bracket atom");
    };
    let BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(nested)) = &bracket.tree else {
        panic!("wrapped recursive SMARTS must lower into a nested query");
    };
    assert_same_query_structure(nested, &expected_nested);

    let rendered = query.to_string();
    let reparsed = parse_smarts(&rendered).expect("displayed SMARTS must parse again");
    assert_recursive_queries_lowered(&reparsed);
    assert_same_query_structure(&query, &reparsed);
});
