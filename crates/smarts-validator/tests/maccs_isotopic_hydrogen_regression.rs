//! Regression coverage for the MACCS key 44 / 28 isotopic-hydrogen mismatch.

use core::str::FromStr;

use smarts_parser::QueryMol;
use smarts_validator::{
    match_count_compiled, matches_compiled, substructure_matches_compiled, CompiledQuery,
    PreparedTarget,
};
use smiles_parser::Smiles;

const SMILES: &str = "[H][C@@]1(C[C@@H]([C@H](O1)CO[Si](C2=CC=CC=C2)(C3=CC=CC=C3)C(C)(C)C)CI)[3H]";

#[test]
fn non_isotopic_attached_hydrogens_do_not_trigger_key_44_or_28() {
    let prepared = PreparedTarget::new(Smiles::from_str(SMILES).unwrap());
    let key_44 = CompiledQuery::new(
        QueryMol::from_str("[!#1;!#6;!#7;!#8;!#9;!#14;!#15;!#16;!#17;!#35;!#53]").unwrap(),
    )
    .unwrap();
    let key_28 =
        CompiledQuery::new(QueryMol::from_str("[!#6;!#1]~[CH2]~[!#6;!#1]").unwrap()).unwrap();

    assert!(!matches_compiled(&key_44, &prepared));
    assert_eq!(match_count_compiled(&key_44, &prepared), 0);
    assert_eq!(&*substructure_matches_compiled(&key_44, &prepared), &[]);

    assert!(!matches_compiled(&key_28, &prepared));
    assert_eq!(match_count_compiled(&key_28, &prepared), 0);
    assert_eq!(&*substructure_matches_compiled(&key_28, &prepared), &[]);
}
