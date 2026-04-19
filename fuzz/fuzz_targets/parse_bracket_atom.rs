#![no_main]

use libfuzzer_sys::fuzz_target;
use smarts_parser::{parse_smarts, AtomExpr, AtomPrimitive, BracketExprTree, QueryMol};

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

fuzz_target!(|data: &[u8]| {
    if data.len() > 512 {
        return;
    }

    let candidate = String::from_utf8_lossy(data);
    let wrapped = format!("[{}]", candidate);

    let Ok(query) = parse_smarts(&wrapped) else {
        return;
    };

    assert_recursive_queries_lowered(&query);

    let rendered = query.to_string();
    let reparsed = parse_smarts(&rendered).expect("displayed SMARTS must parse again");

    assert_recursive_queries_lowered(&reparsed);
    assert_query_render_is_stable(&query);
});
