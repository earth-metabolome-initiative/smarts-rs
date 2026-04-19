//! Regression coverage for the MACCS key 136 oxyhalogen count counterexample.

use core::str::FromStr;

use smarts_parser::QueryMol;
use smarts_validator::{match_count_compiled, matches_compiled, CompiledQuery, PreparedTarget};
use smiles_parser::Smiles;

const MACCS_KEY_136_COUNTEREXAMPLE: &str = concat!(
    "CCN1/C(=C/C=C/C2=[N+](C3=CC=CC=C3C=C2)CC)/C=CC4=CC=CC=C41.",
    "CCN\\1C2=CC=CC=C2S/C1=C/C=C/3\\CCC(=C3N(C4=CC=CC=C4)C5=CC=CC=C5)/C=C/C6=[N+](C7=CC=CC=C7S6)CC.",
    "CCN\\1C2=CC=CC=C2S/C1=C/C=C/3\\CCC(=C3SC4=NN=NN4C5=CC=CC=C5)/C=C/C6=[N+](C7=CC=CC=C7S6)CC.",
    "CC1=CC=C(C=C1)C(=C2C=CC(=C)C=C2)/C=C\\C=C(C3=CC=C(C=C3)C)C4=CC=C(C=C4)C.",
    "CC1(C(=[N+](C2=C1C3=CC=CC=C3C=C2)C)C=CC4=C(C(=CC=C5C(C6=C(N5C)C=CC7=CC=CC=C76)(C)C)CCC4)Cl)C.",
    "C=CCN1C(=[N+](C2=NC3=CC=CC=C3N=C21)CC=C)/C=C/C4=C(/C(=C/C=C5N(C6=NC7=CC=CC=C7N=C6N5CC=C)CC=C)/CC4)",
    "SC8=NN=NN8C9=CC=CC=C9.[O-]Cl(=O)(=O)=O.[I-].[I-]"
);

#[test]
fn maccs_key_136_counterexample_does_not_count_oxyhalogen_oxo_bonds_like_rdkit() {
    let compiled = CompiledQuery::new(QueryMol::from_str("[#8]=*").unwrap()).unwrap();
    let prepared = PreparedTarget::new(Smiles::from_str(MACCS_KEY_136_COUNTEREXAMPLE).unwrap());

    assert!(!matches_compiled(&compiled, &prepared));
    assert_eq!(match_count_compiled(&compiled, &prepared), 0);
}
