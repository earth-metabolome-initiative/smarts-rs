//! RDKit-frozen charged aromaticity parity tests.

use core::str::FromStr;

use smarts_rs::{CompiledQuery, PreparedTarget, QueryMol};
use smiles_parser::Smiles;

#[test]
fn charged_thiophene_matches_aromatic_atom_query_like_rdkit() {
    let target = PreparedTarget::new(Smiles::from_str("Cc1cc(C)[s+]s1").unwrap());
    let aromatic_query = CompiledQuery::new(QueryMol::from_str("a").unwrap()).unwrap();

    assert!(aromatic_query.matches(&target));
    assert_eq!(aromatic_query.match_count(&target), 5);
    assert_eq!(
        aromatic_query.substructure_matches(&target).as_ref(),
        &[
            Box::<[usize]>::from([1]),
            Box::<[usize]>::from([2]),
            Box::<[usize]>::from([3]),
            Box::<[usize]>::from([5]),
            Box::<[usize]>::from([6]),
        ]
    );
}

#[test]
fn pyridinium_matches_aromatic_atom_and_cn_queries_like_rdkit() {
    let target = PreparedTarget::new(Smiles::from_str("C[n+]1ccccc1C=NO").unwrap());
    let aromatic_query = CompiledQuery::new(QueryMol::from_str("a").unwrap()).unwrap();
    let cn_query = CompiledQuery::new(QueryMol::from_str("c:n").unwrap()).unwrap();

    assert!(aromatic_query.matches(&target));
    assert_eq!(aromatic_query.match_count(&target), 6);
    assert_eq!(
        aromatic_query.substructure_matches(&target).as_ref(),
        &[
            Box::<[usize]>::from([1]),
            Box::<[usize]>::from([2]),
            Box::<[usize]>::from([3]),
            Box::<[usize]>::from([4]),
            Box::<[usize]>::from([5]),
            Box::<[usize]>::from([6]),
        ]
    );

    assert!(cn_query.matches(&target));
    assert_eq!(cn_query.match_count(&target), 2);
    assert_eq!(
        cn_query.substructure_matches(&target).as_ref(),
        &[Box::<[usize]>::from([2, 1]), Box::<[usize]>::from([6, 1])]
    );
}
