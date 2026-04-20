//! Regression coverage for MACCS key 49 style charge normalization cases.

use core::str::FromStr;

use smarts_parser::QueryMol;
use smarts_validator::{match_count_compiled, matches_compiled, CompiledQuery, PreparedTarget};
use smiles_parser::Smiles;

const KEY_49_COUNTEREXAMPLE: &str = "CCOCl(=O)(=O)=O";

#[test]
fn bromic_acid_fragment_exposes_rdkit_like_non_neutral_atoms() {
    let compiled = CompiledQuery::new(QueryMol::from_str("[!+0]").unwrap()).unwrap();
    let prepared = PreparedTarget::new(Smiles::from_str("OBr(=O)=O").unwrap());

    assert!(matches_compiled(&compiled, &prepared));
    assert_eq!(match_count_compiled(&compiled, &prepared), 3);
}

#[test]
fn alkyl_oxychloride_fragment_exposes_rdkit_like_non_neutral_atoms() {
    let compiled = CompiledQuery::new(QueryMol::from_str("[!+0]").unwrap()).unwrap();
    let prepared = PreparedTarget::new(Smiles::from_str(KEY_49_COUNTEREXAMPLE).unwrap());

    assert!(matches_compiled(&compiled, &prepared));
    assert_eq!(match_count_compiled(&compiled, &prepared), 4);
}
