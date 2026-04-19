//! `RDKit` extension parse-regression tests.

use smarts_parser::parse_smarts;

#[test]
fn rdkit_hetero_neighbor_queries_should_parse() {
    for smarts in ["[z2]", "[Z2]", "[Z1]"] {
        assert!(
            parse_smarts(smarts).is_ok(),
            "`RDKit` accepts hetero-neighbor SMARTS {smarts}, but our parser still rejects it"
        );
    }
}

#[test]
fn rdkit_hybridization_queries_should_parse() {
    for smarts in ["[^1]", "[^2]", "[^3]"] {
        assert!(
            parse_smarts(smarts).is_ok(),
            "`RDKit` accepts hybridization SMARTS {smarts}, but our parser still rejects it"
        );
    }
}
