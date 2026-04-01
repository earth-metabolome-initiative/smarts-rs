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
    let input = candidate.as_ref();

    let Ok(expr) = fuzz_parse_bracket_text(input) else {
        return;
    };

    let rendered_expr = expr.to_string();
    let reparsed = fuzz_parse_bracket_text(&rendered_expr).expect("bracket text must reparse");
    assert_eq!(rendered_expr, reparsed.to_string());

    let wrapped = format!("[{}]", rendered_expr);
    let Ok(query) = parse_smarts(&wrapped) else {
        return;
    };

    assert_eq!(query.atom_count(), 1);
    assert_eq!(query.bond_count(), 0);
    assert_recursive_queries_lowered(&query);

    let AtomExpr::Bracket(_) = &query.atoms()[0].expr else {
        panic!("wrapped bracket text must produce a bracket atom");
    };

    let reparsed_query = parse_smarts(&query.to_string()).expect("query display must reparse");
    assert_same_query_structure(&query, &reparsed_query);
});
