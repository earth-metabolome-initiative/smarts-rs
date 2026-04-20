//! Red regressions for the current live MACCS mismatch signatures from the fresh
//! full `PubChem` run.

use core::str::FromStr;

use smarts_parser::QueryMol;
use smarts_validator::{match_count_compiled, CompiledQuery, PreparedTarget};
use smiles_parser::Smiles;

const EXTRA_136_MISSING_49_ROW_24345: &str = "OI(=O)=O";
const EXTRA_136_ROW_24619: &str = "[O-]I(=O)=O.[O-]I(=O)=O.[Ca+2]";
const MISSING_49_ROW_25289: &str = "OI(=O)(O)(O)(O)O";
const EXTRA_49_ROW_19872324: &str = "C=P(=O)O";
const EXTRA_49_MISSING_136_ROW_70419781: &str = "CCOC1C(=NC(=O)O1)P(=N)=O";
const MISSING_136_ROW_59972358: &str = "C[P+](=O)CCCP(=C)=O";
const EXTRA_136_MISSING_49_ROW_53471212: &str = "C1NN1P(=NC(=O)C2=CC(=CC=C2)F)=O";
const MISSING_49_ROW_70453091: &str =
    "C[C@@H](C(=O)N1CCC[C@H]1C(=O)N2CCC[C@H]2C(=O)O)N=P(=O)CC3=CC=CC=C3";
const EXTRA_49_ROW_68957045: &str = "C1=CC=C(C=C1)P(=N)=O";
const EXTRA_49_MISSING_136_ROW_130266756: &str = "CCO/N=C(/C1=NSC(=N1)OP(=N)=O)\\C(=O)N";

fn compile(smarts: &str) -> CompiledQuery {
    CompiledQuery::new(QueryMol::from_str(smarts).unwrap()).unwrap()
}

fn assert_key_49_and_136_counts(smiles: &str, key_49_count: usize, key_136_count: usize) {
    let prepared = PreparedTarget::new(Smiles::from_str(smiles).unwrap());
    let key_49 = compile("[!+0]");
    let key_136 = compile("[#8]=*");

    assert_eq!(match_count_compiled(&key_49, &prepared), key_49_count);
    assert_eq!(match_count_compiled(&key_136, &prepared), key_136_count);
}

#[test]
fn row_24345_should_have_three_non_neutral_atoms_and_no_oxygen_double_bonds() {
    assert_key_49_and_136_counts(EXTRA_136_MISSING_49_ROW_24345, 3, 0);
}

#[test]
fn row_24619_should_have_nine_non_neutral_atoms_and_no_oxygen_double_bonds() {
    assert_key_49_and_136_counts(EXTRA_136_ROW_24619, 9, 0);
}

#[test]
fn row_25289_should_have_two_non_neutral_atoms_and_no_oxygen_double_bonds() {
    assert_key_49_and_136_counts(MISSING_49_ROW_25289, 2, 0);
}

#[test]
fn row_19872324_should_not_report_non_neutral_atoms_and_should_keep_one_oxygen_double_bond() {
    assert_key_49_and_136_counts(EXTRA_49_ROW_19872324, 0, 1);
}

#[test]
fn row_70419781_should_not_report_non_neutral_atoms_and_should_keep_two_oxygen_double_bonds() {
    assert_key_49_and_136_counts(EXTRA_49_MISSING_136_ROW_70419781, 0, 2);
}

#[test]
fn row_59972358_should_report_one_non_neutral_atom_and_two_oxygen_double_bonds() {
    assert_key_49_and_136_counts(MISSING_136_ROW_59972358, 1, 2);
}

#[test]
fn row_53471212_should_have_two_non_neutral_atoms_and_one_oxygen_double_bond() {
    assert_key_49_and_136_counts(EXTRA_136_MISSING_49_ROW_53471212, 2, 1);
}

#[test]
fn row_70453091_should_have_two_non_neutral_atoms_and_three_oxygen_double_bonds() {
    assert_key_49_and_136_counts(MISSING_49_ROW_70453091, 2, 3);
}

#[test]
fn row_68957045_should_not_report_non_neutral_atoms_and_should_keep_one_oxygen_double_bond() {
    assert_key_49_and_136_counts(EXTRA_49_ROW_68957045, 0, 1);
}

#[test]
fn row_130266756_should_not_report_non_neutral_atoms_and_should_keep_two_oxygen_double_bonds() {
    assert_key_49_and_136_counts(EXTRA_49_MISSING_136_ROW_130266756, 0, 2);
}
