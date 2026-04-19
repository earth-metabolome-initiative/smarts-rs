//! Regression coverage for the MACCS key 17 large-molecule triple-bond case.

use core::str::FromStr;
use std::boxed::Box;

use smarts_parser::QueryMol;
use smarts_validator::{
    match_count_compiled, matches_compiled, substructure_matches_compiled, CompiledQuery,
    PreparedTarget,
};
use smiles_parser::Smiles;

const MACCS_KEY_17_COUNTEREXAMPLE: &str = concat!(
    "CC[C@@H]1[C@@H](N1)C(=O)N(C)CC(=O)N(C)C(=C(C)C)C(=O)N[C@H]2CC3=CC(=CC(=C3)O)",
    "C4=CC5=C(C=C4)N(C(=C5CC(COC(=O)[C@@H]6CCCN(C2=O)N6)(C)C)C7=C(N=CC#C7)[C@H](C)OC)CC"
);

#[test]
fn maccs_key_17_counterexample_matches_carbon_carbon_triple_bond_query() {
    let compiled = CompiledQuery::new(QueryMol::from_str("[#6]#[#6]").unwrap()).unwrap();
    let prepared = PreparedTarget::new(Smiles::from_str(MACCS_KEY_17_COUNTEREXAMPLE).unwrap());

    assert!(matches_compiled(&compiled, &prepared));
    assert_eq!(match_count_compiled(&compiled, &prepared), 1);
    assert_eq!(
        substructure_matches_compiled(&compiled, &prepared).as_ref(),
        &[Box::<[usize]>::from([59, 60])]
    );
}
