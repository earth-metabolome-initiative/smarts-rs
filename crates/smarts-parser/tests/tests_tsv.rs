//! Compatibility test for the large SMARTS TSV corpus shipped at the repo root.

use core::fmt::Write as _;

use smarts_parser::parse_smarts;

#[test]
fn parser_accepts_all_tests_tsv_smarts() {
    let corpus_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests.tsv");
    let corpus = std::fs::read_to_string(corpus_path).expect("read tests.tsv");

    let mut unexpected_failures = Vec::new();
    for (line_no, line) in corpus.lines().enumerate().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let mut fields = line.splitn(4, '\t');
        let Some(id) = fields.next() else {
            continue;
        };
        let _cd_id = fields.next();
        let name = fields.next().unwrap_or("<missing-name>");
        let smarts = fields.next().unwrap_or("");
        if let Err(err) = parse_smarts(smarts) {
            unexpected_failures.push((
                line_no + 1,
                id.to_owned(),
                name.to_owned(),
                err.to_string(),
            ));
        }
    }

    if !unexpected_failures.is_empty() {
        let mut message = String::new();
        writeln!(
            &mut message,
            "{} SMARTS from tests.tsv still fail to parse:",
            unexpected_failures.len()
        )
        .unwrap();
        for (line_no, id, name, err) in unexpected_failures.iter().take(50) {
            writeln!(&mut message, "line {line_no} id {id} {name}: {err}").unwrap();
        }
        if unexpected_failures.len() > 50 {
            writeln!(
                &mut message,
                "... and {} more",
                unexpected_failures.len() - 50
            )
            .unwrap();
        }
        panic!("{message}");
    }
}
