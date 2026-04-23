//! Regression coverage for MACCS key 49 style charge normalization cases.

use core::str::FromStr;

use smarts_rs::{CompiledQuery, PreparedTarget, QueryMol};
use smiles_parser::Smiles;

const KEY_49_COUNTEREXAMPLE: &str = "CCOCl(=O)(=O)=O";

#[test]
fn bromic_acid_fragment_exposes_rdkit_like_non_neutral_atoms() {
    let compiled = CompiledQuery::new(QueryMol::from_str("[!+0]").unwrap()).unwrap();
    let prepared = PreparedTarget::new(Smiles::from_str("OBr(=O)=O").unwrap());

    assert!(compiled.matches(&prepared));
    assert_eq!(compiled.match_count(&prepared), 3);
}

#[test]
fn alkyl_oxychloride_fragment_exposes_rdkit_like_non_neutral_atoms() {
    let compiled = CompiledQuery::new(QueryMol::from_str("[!+0]").unwrap()).unwrap();
    let prepared = PreparedTarget::new(Smiles::from_str(KEY_49_COUNTEREXAMPLE).unwrap());

    assert!(compiled.matches(&prepared));
    assert_eq!(compiled.match_count(&prepared), 4);
}
