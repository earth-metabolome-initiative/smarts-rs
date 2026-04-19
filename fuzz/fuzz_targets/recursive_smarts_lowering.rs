#![no_main]

use libfuzzer_sys::fuzz_target;
use smarts_parser::{
    fuzz_parse_bracket_text, parse_smarts, AtomExpr, AtomPrimitive, BracketExprTree, QueryMol,
};

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
    assert_eq!(reparsed.component_groups(), reparsed_again.component_groups());
}

fuzz_target!(|data: &[u8]| {
    if data.len() > 512 {
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
    assert_eq!(nested.atom_count(), expected_nested.atom_count());
    assert_eq!(nested.bond_count(), expected_nested.bond_count());
    assert_eq!(nested.component_count(), expected_nested.component_count());
    assert_eq!(nested.component_groups(), expected_nested.component_groups());
    assert_query_render_is_stable(nested);

    assert_query_render_is_stable(&query);
});
