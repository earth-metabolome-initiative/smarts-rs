//! Regression coverage for the MACCS key 144 aromatic-path counterexample.

use core::str::FromStr;

use smarts_rs::{target::BondLabel, CompiledQuery, PreparedTarget, QueryMol};
use smiles_parser::Smiles;

const MACCS_KEY_144_COUNTEREXAMPLE: &str = "CC.CC.CC.C/C=C\\1/C=CC=C/C1=C/C=C";

#[test]
fn maccs_key_144_counterexample_matches_aromatic_path_query() {
    let compiled = CompiledQuery::new(QueryMol::from_str("*!:*:*!:*").unwrap()).unwrap();
    let prepared = PreparedTarget::new(Smiles::from_str(MACCS_KEY_144_COUNTEREXAMPLE).unwrap());

    assert_eq!(prepared.bond(7, 8), Some(BondLabel::Double));
    assert_eq!(prepared.bond(8, 13), Some(BondLabel::Aromatic));
    assert_eq!(prepared.bond(13, 14), Some(BondLabel::Double));
    assert!(compiled.matches(&prepared));
}
