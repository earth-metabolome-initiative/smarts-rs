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
}

fn query_is_too_large(query: &QueryMol) -> bool {
    query.atom_count() > 64 || query.bond_count() > 96 || query.component_count() > 16
}

fuzz_target!(|data: &[u8]| {
    if data.len() > 128 {
        return;
    }

    let candidate = String::from_utf8_lossy(data);
    let input = candidate.as_ref();

    let Ok(expr) = fuzz_parse_bracket_text(input) else {
        return;
    };

    let rendered_expr = expr.to_string();
    let reparsed = fuzz_parse_bracket_text(&rendered_expr).expect("bracket text must reparse");
    let rerendered_expr = reparsed.to_string();
    let reparsed_again =
        fuzz_parse_bracket_text(&rerendered_expr).expect("normalized bracket text must reparse");
    assert_eq!(rerendered_expr, reparsed_again.to_string());

    let wrapped = format!("[{}]", rerendered_expr);
    let Ok(query) = parse_smarts(&wrapped) else {
        return;
    };
    if query_is_too_large(&query) {
        return;
    }

    assert_eq!(query.atom_count(), 1);
    assert_eq!(query.bond_count(), 0);
    assert_recursive_queries_lowered(&query);

    let AtomExpr::Bracket(_) = &query.atoms()[0].expr else {
        panic!("wrapped bracket text must produce a bracket atom");
    };

    assert_query_render_is_stable(&query);
});
