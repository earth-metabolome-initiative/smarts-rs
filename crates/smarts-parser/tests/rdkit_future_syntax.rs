//! Focused RDKit-compatibility regressions for historically missing parser features.

use smarts_parser::parse_smarts;

#[test]
fn rdkit_range_queries_should_parse() {
    for smarts in ["[h{1-}]", "[D{2-3}]", "[R{1-}]", "[x{2-}]"] {
        assert!(
            parse_smarts(smarts).is_ok(),
            "RDKit accepts range-query SMARTS {smarts}, but our parser still rejects it"
        );
    }
}

#[test]
fn rdkit_atom_maps_should_parse() {
    for smarts in ["[C:1]", "[C,N,O:1]", "[#6:1](=O)N"] {
        assert!(
            parse_smarts(smarts).is_ok(),
            "RDKit accepts atom-map SMARTS {smarts}, but our parser still rejects it"
        );
    }
}

#[test]
fn rdkit_numeric_wildcard_forms_should_parse() {
    for smarts in [
        "[89*]",
        "[0*]",
        "[89*,57*]",
        "[0*,-,-2]",
        "[#24&X1][89*,57*]",
        "[#80&+2][0*,-,-2]",
    ] {
        assert!(
            parse_smarts(smarts).is_ok(),
            "RDKit accepts numeric-wildcard SMARTS {smarts}, but our parser still rejects it"
        );
    }
}
