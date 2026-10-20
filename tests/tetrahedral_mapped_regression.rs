//! Tetrahedral parity against mapped substituents.

use core::str::FromStr;

use smarts_rs::QueryMol;

#[test]
fn same_element_substituents_keep_their_mapped_orientation() {
    let query = QueryMol::from_str("C[C@H](O)CC").unwrap();
    assert!(query.matches("CC[C@@H](O)C").unwrap());
    assert!(!query.matches("CC[C@H](O)C").unwrap());
}

#[test]
fn ring_closure_neighbors_keep_their_written_orientation() {
    let query = QueryMol::from_str("C[C@@H]1CCCCO1").unwrap();
    assert!(query.matches("O1CCCC[C@H]1C").unwrap());
    assert!(!query.matches("O1CCCC[C@@H]1C").unwrap());
}

#[test]
fn underconstrained_chirality_requires_a_stereo_target() {
    for smarts in [
        "[C@]",
        "[C@@]F",
        "[C@](F)Cl",
        "[C@].O",
        "[$([C@])]",
        "[C;$([C@])]",
    ] {
        let query = QueryMol::from_str(smarts).unwrap();
        for (smiles, expected) in [
            ("F[C@H](Cl)Br.O", true),
            ("F[C@@H](Cl)Br.O", true),
            ("FC(Cl)Br.O", false),
        ] {
            assert_eq!(
                query.matches(smiles).unwrap(),
                expected,
                "{smarts} against {smiles}"
            );
            assert_eq!(
                query.match_count(smiles).unwrap(),
                usize::from(expected),
                "{smarts} against {smiles}"
            );
        }
    }
}
