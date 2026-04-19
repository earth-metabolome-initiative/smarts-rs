#![no_main]

use libfuzzer_sys::fuzz_target;
use smarts_parser::{parse_smarts, AtomExpr};
use smarts_validator::{
    match_count, match_count_compiled, match_count_prepared, matches, matches_compiled,
    matches_prepared, substructure_matches, substructure_matches_compiled,
    substructure_matches_prepared, CompiledQuery, PreparedTarget, SmartsMatchError,
};
use smiles_parser::Smiles;

fn split_input(data: &[u8]) -> (&[u8], &[u8]) {
    let split_at = data
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(data.len());
    if split_at == data.len() {
        return (data, &[]);
    }
    (&data[..split_at], &data[split_at + 1..])
}

fn same_error_kind(left: &SmartsMatchError, right: &SmartsMatchError) -> bool {
    match (left, right) {
        (SmartsMatchError::EmptyTarget, SmartsMatchError::EmptyTarget)
        | (
            SmartsMatchError::InvalidTargetSmiles { .. },
            SmartsMatchError::InvalidTargetSmiles { .. },
        ) => true,
        (
            SmartsMatchError::UnsupportedAtomPrimitive { primitive: left },
            SmartsMatchError::UnsupportedAtomPrimitive { primitive: right },
        )
        | (
            SmartsMatchError::UnsupportedBondPrimitive { primitive: left },
            SmartsMatchError::UnsupportedBondPrimitive { primitive: right },
        ) => left == right,
        _ => false,
    }
}

fn assert_materialized_matches_are_well_formed(
    matches_out: &[Box<[usize]>],
    query_atom_count: usize,
    target_atom_count: usize,
) {
    for mapping in matches_out {
        assert_eq!(mapping.len(), query_atom_count);

        let mut sorted = mapping.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), mapping.len());
        assert!(mapping.iter().all(|atom_id| *atom_id < target_atom_count));
    }
}

fn query_is_too_large(query: &smarts_parser::QueryMol) -> bool {
    query.atom_count() > 24 || query.bond_count() > 36 || query.component_count() > 8
}

fn query_is_too_generic(query: &smarts_parser::QueryMol) -> bool {
    let wildcard_atoms = query
        .atoms()
        .iter()
        .filter(|atom| matches!(atom.expr, AtomExpr::Wildcard))
        .count();
    query.component_count() > 3 || wildcard_atoms > 2
}

fuzz_target!(|data: &[u8]| {
    if data.len() > 256 {
        return;
    }

    let (query_bytes, target_bytes) = split_input(data);
    if query_bytes.len() > 96 || target_bytes.len() > 96 {
        return;
    }
    let query_text = String::from_utf8_lossy(query_bytes);
    let target_text = String::from_utf8_lossy(target_bytes);

    let Ok(query) = parse_smarts(query_text.as_ref()) else {
        return;
    };
    if query_is_too_large(&query) || query_is_too_generic(&query) {
        return;
    }

    let Ok(target) = target_text.as_ref().parse::<Smiles>() else {
        let bool_result = matches(&query, target_text.as_ref());
        let count_result = match_count(&query, target_text.as_ref());
        let materialized_result = substructure_matches(&query, target_text.as_ref());

        match (&bool_result, &count_result, &materialized_result) {
            (Err(left), Err(right), Err(third)) => {
                assert!(same_error_kind(left, right));
                assert!(same_error_kind(left, third));
            }
            _ => panic!("validator entrypoints must agree on invalid target errors"),
        }
        return;
    };
    if target.nodes().len() > 32 {
        return;
    }

    let bool_result = matches(&query, target_text.as_ref());
    let count_result = match_count(&query, target_text.as_ref());
    let materialized_result = substructure_matches(&query, target_text.as_ref());

    match (&bool_result, &count_result, &materialized_result) {
        (Ok(boolean), Ok(count), Ok(matches_out)) => {
            assert_eq!(*boolean, *count > 0);
            assert_eq!(*count, matches_out.len());
        }
        (Err(left), Err(right), Err(third)) => {
            assert!(same_error_kind(left, right));
            assert!(same_error_kind(left, third));
        }
        _ => panic!("validator entrypoints must agree on success vs error"),
    }

    let prepared = PreparedTarget::new(target);

    match CompiledQuery::new(query.clone()) {
        Ok(compiled) => {
            let prepared_bool =
                matches_prepared(&query, &prepared).expect("prepared API must match compiled API");
            let prepared_count = match_count_prepared(&query, &prepared)
                .expect("prepared count API must match compiled API");
            let prepared_matches = substructure_matches_prepared(&query, &prepared)
                .expect("prepared match collection API must match compiled API");

            assert_eq!(prepared_bool, matches_compiled(&compiled, &prepared));
            assert_eq!(prepared_count, match_count_compiled(&compiled, &prepared));
            assert_eq!(
                prepared_matches.as_ref(),
                substructure_matches_compiled(&compiled, &prepared).as_ref()
            );

            assert_eq!(prepared_bool, prepared_count > 0);
            assert_eq!(prepared_count, prepared_matches.len());
            assert_materialized_matches_are_well_formed(
                &prepared_matches,
                compiled.query().atom_count(),
                prepared.atom_count(),
            );

            if let (Ok(boolean), Ok(count), Ok(matches_out)) =
                (&bool_result, &count_result, &materialized_result)
            {
                assert_eq!(*boolean, prepared_bool);
                assert_eq!(*count, prepared_count);
                assert_eq!(matches_out.as_ref(), prepared_matches.as_ref());
            }
        }
        Err(compilation_error) => {
            let prepared_bool = matches_prepared(&query, &prepared)
                .expect_err("prepared API must reject same query");
            let prepared_count = match_count_prepared(&query, &prepared)
                .expect_err("prepared count API must reject same query");
            let prepared_matches = substructure_matches_prepared(&query, &prepared)
                .expect_err("prepared match collection API must reject same query");

            assert!(same_error_kind(&compilation_error, &prepared_bool));
            assert!(same_error_kind(&compilation_error, &prepared_count));
            assert!(same_error_kind(&compilation_error, &prepared_matches));
        }
    }
});
