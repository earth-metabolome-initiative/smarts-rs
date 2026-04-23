//! Regression coverage for the MACCS key 44 / 28 isotopic-hydrogen mismatch.

use core::str::FromStr;

use smarts_rs::{CompiledQuery, PreparedTarget, QueryMol};
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

    assert!(!key_44.matches(&prepared));
    assert_eq!(key_44.match_count(&prepared), 0);
    assert_eq!(&*key_44.substructure_matches(&prepared), &[]);

    assert!(!key_28.matches(&prepared));
    assert_eq!(key_28.match_count(&prepared), 0);
    assert_eq!(&*key_28.substructure_matches(&prepared), &[]);
}
