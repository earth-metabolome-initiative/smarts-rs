#![no_main]

use libfuzzer_sys::fuzz_target;
use query_fuzz_common::{assert_query_render_is_stable, assert_recursive_queries_lowered};
use smarts_rs::{fuzz_parse_bracket_text, parse_smarts, AtomExpr, QueryMol};

mod query_fuzz_common;

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
