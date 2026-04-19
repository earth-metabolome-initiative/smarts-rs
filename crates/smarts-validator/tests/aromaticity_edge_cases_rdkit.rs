//! RDKit-frozen aromaticity edge-case parity tests.

use core::str::FromStr;

use smarts_parser::QueryMol;
use smarts_validator::{
    match_count_compiled, substructure_matches_compiled, CompiledQuery, PreparedTarget,
};
use smiles_parser::Smiles;

#[derive(Debug, Clone, Copy)]
struct AromaticityCase {
    name: &'static str,
    smiles: &'static str,
    aromatic_atoms: &'static [usize],
    aromatic_matches: &'static [&'static [usize]],
    carbon_nitrogen_matches: &'static [&'static [usize]],
}

const CASES: &[AromaticityCase] = &[
    AromaticityCase {
        name: "neutral_radical_pyrrolyl_like_ring_is_not_aromatic",
        smiles: "C1=C[N]C=C1",
        aromatic_atoms: &[],
        aromatic_matches: &[],
        carbon_nitrogen_matches: &[],
    },
    AromaticityCase {
        name: "cycloheptatrienyl_cation_variant_is_not_aromatic",
        smiles: "C1=CC=CC=C[C+]1",
        aromatic_atoms: &[],
        aromatic_matches: &[],
        carbon_nitrogen_matches: &[],
    },
    AromaticityCase {
        name: "neutral_carbon_radical_ring_is_aromatic_under_rdkit",
        smiles: "C1=[C]NC=C1",
        aromatic_atoms: &[0, 1, 2, 3, 4],
        aromatic_matches: &[&[0], &[1], &[2], &[3], &[4]],
        carbon_nitrogen_matches: &[&[1, 2], &[3, 2]],
    },
    AromaticityCase {
        name: "charged_thiophene_variant_retains_aromatic_atoms",
        smiles: "Cc1cc(C)[s+]s1",
        aromatic_atoms: &[1, 2, 3, 5, 6],
        aromatic_matches: &[&[1], &[2], &[3], &[5], &[6]],
        carbon_nitrogen_matches: &[],
    },
    AromaticityCase {
        name: "pyridinium_variant_retains_aromatic_atoms",
        smiles: "C[n+]1ccccc1C=NO",
        aromatic_atoms: &[1, 2, 3, 4, 5, 6],
        aromatic_matches: &[&[1], &[2], &[3], &[4], &[5], &[6]],
        carbon_nitrogen_matches: &[&[2, 1], &[6, 1]],
    },
];

fn compile_query(smarts: &str) -> CompiledQuery {
    let query = QueryMol::from_str(smarts)
        .unwrap_or_else(|error| panic!("failed to parse SMARTS {smarts:?}: {error}"));
    CompiledQuery::new(query)
        .unwrap_or_else(|error| panic!("failed to compile SMARTS {smarts:?}: {error}"))
}

fn collect_aromatic_atoms(target: &PreparedTarget) -> Vec<usize> {
    (0..target.atom_count())
        .filter(|&atom_id| target.is_aromatic(atom_id))
        .collect()
}

fn boxed_mappings(mappings: &'static [&'static [usize]]) -> Vec<Box<[usize]>> {
    mappings
        .iter()
        .map(|mapping| Box::<[usize]>::from(*mapping))
        .collect()
}

#[test]
fn aromaticity_edge_cases_should_agree_with_rdkit() {
    let aromatic_query = compile_query("a");
    let carbon_nitrogen_query = compile_query("c:n");

    for case in CASES {
        let smiles = case
            .smiles
            .parse::<Smiles>()
            .unwrap_or_else(|error| panic!("{}: invalid SMILES: {error}", case.name));
        let target = PreparedTarget::new(smiles);

        assert_eq!(
            collect_aromatic_atoms(&target),
            case.aromatic_atoms,
            "{}: aromatic atom set differs",
            case.name
        );

        let aromatic_matches = substructure_matches_compiled(&aromatic_query, &target);
        assert_eq!(
            aromatic_matches.as_ref(),
            boxed_mappings(case.aromatic_matches).as_slice(),
            "{}: aromatic atom matches differ",
            case.name
        );
        assert_eq!(
            match_count_compiled(&aromatic_query, &target),
            case.aromatic_matches.len(),
            "{}: aromatic atom match count differs",
            case.name
        );

        let carbon_nitrogen_matches =
            substructure_matches_compiled(&carbon_nitrogen_query, &target);
        assert_eq!(
            carbon_nitrogen_matches.as_ref(),
            boxed_mappings(case.carbon_nitrogen_matches).as_slice(),
            "{}: c:n matches differ",
            case.name
        );
        assert_eq!(
            match_count_compiled(&carbon_nitrogen_query, &target),
            case.carbon_nitrogen_matches.len(),
            "{}: c:n match count differs",
            case.name
        );
    }
}
