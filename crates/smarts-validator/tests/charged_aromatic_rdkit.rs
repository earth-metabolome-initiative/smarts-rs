//! RDKit-frozen charged aromaticity parity tests.

use core::str::FromStr;

use smarts_parser::QueryMol;
use smarts_validator::{
    match_count_compiled, matches_compiled, substructure_matches_compiled, CompiledQuery,
    PreparedTarget,
};
use smiles_parser::Smiles;

#[test]
fn charged_thiophene_matches_aromatic_atom_query_like_rdkit() {
    let target = PreparedTarget::new(Smiles::from_str("Cc1cc(C)[s+]s1").unwrap());
    let aromatic_query = CompiledQuery::new(QueryMol::from_str("a").unwrap()).unwrap();

    assert!(matches_compiled(&aromatic_query, &target));
    assert_eq!(match_count_compiled(&aromatic_query, &target), 5);
    assert_eq!(
        substructure_matches_compiled(&aromatic_query, &target).as_ref(),
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

    assert!(matches_compiled(&aromatic_query, &target));
    assert_eq!(match_count_compiled(&aromatic_query, &target), 6);
    assert_eq!(
        substructure_matches_compiled(&aromatic_query, &target).as_ref(),
        &[
            Box::<[usize]>::from([1]),
            Box::<[usize]>::from([2]),
            Box::<[usize]>::from([3]),
            Box::<[usize]>::from([4]),
            Box::<[usize]>::from([5]),
            Box::<[usize]>::from([6]),
        ]
    );

    assert!(matches_compiled(&cn_query, &target));
    assert_eq!(match_count_compiled(&cn_query, &target), 2);
    assert_eq!(
        substructure_matches_compiled(&cn_query, &target).as_ref(),
        &[Box::<[usize]>::from([2, 1]), Box::<[usize]>::from([6, 1])]
    );
}
