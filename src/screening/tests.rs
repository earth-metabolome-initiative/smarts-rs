use alloc::string::String;
use core::str::FromStr;

use crate::QueryMol;
use serde::Deserialize;
use smiles_parser::Smiles;

use super::{
    count_slice_get, AtomFeature, BondCountScreen, EdgeBondFeature, EdgeFeature,
    QueryFeatureFilter, QueryScreen, ShardedTargetCorpusIndex, ShardedTargetCorpusIndexError,
    TargetCandidateSet, TargetCorpusIndex, TargetCorpusIndexShard, TargetCorpusScratch,
    TargetScreen,
};
use crate::prepared::PreparedTarget;
use crate::{CompiledQuery, MatchScratch};

#[derive(Debug, Deserialize)]
struct ExpectedCase {
    smarts: String,
    smiles: String,
    expected_match: bool,
}

#[test]
fn query_screen_extracts_conservative_lower_bounds() {
    let query = QueryMol::from_str("([c].[c]).[R]").unwrap();
    let screen = QueryScreen::new(&query);

    assert_eq!(screen.min_atom_count, 3);
    assert_eq!(screen.min_target_component_count, 1);
    assert_eq!(
        screen.required_element_counts.get(&elements_rs::Element::C),
        Some(&2)
    );
    assert_eq!(screen.min_aromatic_atom_count, 2);
    assert_eq!(screen.min_ring_atom_count, 1);
    assert_eq!(screen.required_bond_counts, BondCountScreen::default());

    let recursive_hydroxyl = QueryScreen::new(&QueryMol::from_str("[#8&$([O&H1&X2])]").unwrap());
    assert_eq!(
        recursive_hydroxyl.required_total_hydrogen_counts.get(&1),
        Some(&1)
    );

    let recursive_ring = QueryScreen::new(&QueryMol::from_str("[!#1&$(C1COCCN1)]").unwrap());
    assert_eq!(recursive_ring.min_atom_count, 6);
    assert_eq!(recursive_ring.min_ring_atom_count, 6);
    assert_eq!(
        recursive_ring
            .required_element_counts
            .get(&elements_rs::Element::C),
        Some(&4)
    );
    assert_eq!(
        recursive_ring
            .required_element_counts
            .get(&elements_rs::Element::N),
        Some(&1)
    );
    assert_eq!(
        recursive_ring
            .required_element_counts
            .get(&elements_rs::Element::O),
        Some(&1)
    );

    let recursive_alternatives = QueryScreen::new(
        &QueryMol::from_str("[!#1;$([#6]1-[#8]-[#7]-1),$([#6]1-[#8]-[#7]-[#6]-1)]").unwrap(),
    );
    assert_eq!(recursive_alternatives.min_atom_count, 3);
    assert_eq!(recursive_alternatives.min_ring_atom_count, 3);
    assert_eq!(
        recursive_alternatives
            .required_element_counts
            .get(&elements_rs::Element::C),
        Some(&1)
    );
    assert_eq!(
        recursive_alternatives
            .required_element_counts
            .get(&elements_rs::Element::N),
        Some(&1)
    );
    assert_eq!(
        recursive_alternatives
            .required_element_counts
            .get(&elements_rs::Element::O),
        Some(&1)
    );

    let heteroaromatic_ring = QueryScreen::new(&QueryMol::from_str("c1ncccc1").unwrap());
    assert_eq!(
        heteroaromatic_ring
            .required_ring_element_counts
            .get(&elements_rs::Element::C),
        Some(&5)
    );
    assert_eq!(
        heteroaromatic_ring
            .required_ring_element_counts
            .get(&elements_rs::Element::N),
        Some(&1)
    );
    assert_eq!(
        heteroaromatic_ring
            .required_aromatic_element_counts
            .get(&elements_rs::Element::N),
        Some(&1)
    );

    let exact_ring_predicates = QueryScreen::new(&QueryMol::from_str("[#6;R2;r6;x3]").unwrap());
    assert_eq!(
        exact_ring_predicates
            .required_ring_membership_counts
            .get(&2),
        Some(&1)
    );
    assert_eq!(
        exact_ring_predicates.required_ring_size_counts.get(&6),
        Some(&1)
    );
    assert_eq!(
        exact_ring_predicates
            .required_ring_connectivity_counts
            .get(&3),
        Some(&1)
    );
}

#[test]
fn query_screen_extracts_conservative_bond_bounds() {
    let double = QueryScreen::new(&QueryMol::from_str("C=C").unwrap());
    let triple = QueryScreen::new(&QueryMol::from_str("C#N").unwrap());
    let aromatic = QueryScreen::new(&QueryMol::from_str("c:c").unwrap());
    let ring = QueryScreen::new(&QueryMol::from_str("C@C").unwrap());
    let topological_ring = QueryScreen::new(&QueryMol::from_str("C1CCCCC1").unwrap());

    assert_eq!(double.required_bond_counts.double, 1);
    assert_eq!(triple.required_bond_counts.triple, 1);
    assert_eq!(aromatic.required_bond_counts.aromatic, 1);
    assert_eq!(ring.required_bond_counts.ring, 1);
    assert_eq!(topological_ring.required_bond_counts.ring, 6);
    assert_eq!(topological_ring.min_ring_atom_count, 6);
}

#[test]
fn target_screen_summarizes_prepared_target() {
    let prepared = PreparedTarget::new(Smiles::from_str("c1ccccc1.O").unwrap());
    let screen = TargetScreen::new(&prepared);

    assert_eq!(screen.atom_count, 7);
    assert_eq!(screen.connected_component_count, 2);
    assert_eq!(
        screen.element_counts.get(&elements_rs::Element::C),
        Some(&6)
    );
    assert_eq!(
        screen.element_counts.get(&elements_rs::Element::O),
        Some(&1)
    );
    assert_eq!(screen.aromatic_atom_count, 6);
    assert_eq!(screen.ring_atom_count, 6);
    assert_eq!(
        count_slice_get(&screen.ring_element_counts, &elements_rs::Element::C),
        Some(&6)
    );
    assert_eq!(
        count_slice_get(&screen.aromatic_element_counts, &elements_rs::Element::C),
        Some(&6)
    );
    assert_eq!(
        count_slice_get(&screen.ring_membership_counts, &1),
        Some(&6)
    );
    assert_eq!(count_slice_get(&screen.ring_size_counts, &6), Some(&6));
    assert_eq!(
        count_slice_get(&screen.ring_connectivity_counts, &2),
        Some(&6)
    );
    assert_eq!(screen.bond_counts.aromatic, 6);
    assert_eq!(screen.bond_counts.ring, 6);
}

#[test]
fn screen_rejects_obvious_non_matches() {
    let query = QueryScreen::new(&QueryMol::from_str("(C).(C)").unwrap());
    let same_component = TargetScreen::new(&PreparedTarget::new(Smiles::from_str("CC").unwrap()));
    let two_components = TargetScreen::new(&PreparedTarget::new(Smiles::from_str("C.C").unwrap()));

    assert!(!query.may_match(&same_component));
    assert!(query.may_match(&two_components));
}

#[test]
fn screen_rejects_missing_atoms_elements_aromaticity_ring_membership_and_bond_types() {
    let atom_count_query = QueryScreen::new(&QueryMol::from_str("CCC").unwrap());
    let aromatic_query = QueryScreen::new(&QueryMol::from_str("c").unwrap());
    let ring_query = QueryScreen::new(&QueryMol::from_str("[R]").unwrap());
    let element_query = QueryScreen::new(&QueryMol::from_str("[Cl]").unwrap());
    let double_bond_query = QueryScreen::new(&QueryMol::from_str("C=C").unwrap());
    let ring_bond_query = QueryScreen::new(&QueryMol::from_str("C@C").unwrap());
    let topological_ring_query = QueryScreen::new(&QueryMol::from_str("C1CCCCC1").unwrap());
    let recursive_ring_query = QueryScreen::new(&QueryMol::from_str("[!#1&$(C1COCCN1)]").unwrap());
    let ring_nitrogen_query = QueryScreen::new(&QueryMol::from_str("[#7&R]").unwrap());
    let aromatic_nitrogen_query = QueryScreen::new(&QueryMol::from_str("[n]").unwrap());
    let ring_membership_two_query = QueryScreen::new(&QueryMol::from_str("[#6;R2]").unwrap());
    let ring_size_six_query = QueryScreen::new(&QueryMol::from_str("[#6;r6]").unwrap());
    let ring_connectivity_three_query = QueryScreen::new(&QueryMol::from_str("[#6;x3]").unwrap());

    let small_target = TargetScreen::new(&PreparedTarget::new(Smiles::from_str("CC").unwrap()));
    let aliphatic_target = TargetScreen::new(&PreparedTarget::new(Smiles::from_str("CC").unwrap()));
    let acyclic_target = TargetScreen::new(&PreparedTarget::new(Smiles::from_str("CCC").unwrap()));
    let oxygen_target = TargetScreen::new(&PreparedTarget::new(Smiles::from_str("O").unwrap()));
    let single_bond_target =
        TargetScreen::new(&PreparedTarget::new(Smiles::from_str("CC").unwrap()));
    let cyclohexane_target =
        TargetScreen::new(&PreparedTarget::new(Smiles::from_str("C1CCCCC1").unwrap()));
    let morpholine_target =
        TargetScreen::new(&PreparedTarget::new(Smiles::from_str("C1COCCN1").unwrap()));
    let benzene_target =
        TargetScreen::new(&PreparedTarget::new(Smiles::from_str("c1ccccc1").unwrap()));
    let pyridine_target =
        TargetScreen::new(&PreparedTarget::new(Smiles::from_str("n1ccccc1").unwrap()));
    let naphthalene_target = TargetScreen::new(&PreparedTarget::new(
        Smiles::from_str("c1ccc2ccccc2c1").unwrap(),
    ));
    let cyclopropane_target =
        TargetScreen::new(&PreparedTarget::new(Smiles::from_str("C1CC1").unwrap()));

    assert!(!atom_count_query.may_match(&small_target));
    assert!(!aromatic_query.may_match(&aliphatic_target));
    assert!(!ring_query.may_match(&acyclic_target));
    assert!(!element_query.may_match(&oxygen_target));
    assert!(!double_bond_query.may_match(&single_bond_target));
    assert!(!ring_bond_query.may_match(&single_bond_target));
    assert!(!topological_ring_query.may_match(&acyclic_target));
    assert!(topological_ring_query.may_match(&cyclohexane_target));
    assert!(!recursive_ring_query.may_match(&acyclic_target));
    assert!(recursive_ring_query.may_match(&morpholine_target));
    assert!(!ring_nitrogen_query.may_match(&benzene_target));
    assert!(ring_nitrogen_query.may_match(&pyridine_target));
    assert!(!aromatic_nitrogen_query.may_match(&morpholine_target));
    assert!(aromatic_nitrogen_query.may_match(&pyridine_target));
    assert!(!ring_membership_two_query.may_match(&benzene_target));
    assert!(ring_membership_two_query.may_match(&naphthalene_target));
    assert!(!ring_size_six_query.may_match(&cyclopropane_target));
    assert!(ring_size_six_query.may_match(&cyclohexane_target));
    assert!(!ring_connectivity_three_query.may_match(&benzene_target));
    assert!(ring_connectivity_three_query.may_match(&naphthalene_target));
}

#[test]
fn screen_rejects_missing_exact_degree_and_total_hydrogen_counts() {
    let degree_query = QueryScreen::new(&QueryMol::from_str("[#6;D4][#6;D1]").unwrap());
    let hydrogen_query = QueryScreen::new(&QueryMol::from_str("[#6;H3]").unwrap());

    let ethane = TargetScreen::new(&PreparedTarget::new(Smiles::from_str("CC").unwrap()));
    let neopentane =
        TargetScreen::new(&PreparedTarget::new(Smiles::from_str("CC(C)(C)C").unwrap()));
    let double_bonded = TargetScreen::new(&PreparedTarget::new(Smiles::from_str("C=C").unwrap()));

    assert!(!degree_query.may_match(&ethane));
    assert!(degree_query.may_match(&neopentane));
    assert!(!hydrogen_query.may_match(&double_bonded));
    assert!(hydrogen_query.may_match(&ethane));
}

#[test]
fn corpus_index_filters_exact_degree_and_total_hydrogen_counts() {
    let prepared_targets = ["CC", "C=C", "CC(C)(C)C"]
        .into_iter()
        .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
        .collect::<alloc::vec::Vec<_>>();
    let index = TargetCorpusIndex::new(&prepared_targets);

    let terminal_methyl = QueryScreen::new(&QueryMol::from_str("[#6;D1;H3]").unwrap());
    let quaternary_carbon = QueryScreen::new(&QueryMol::from_str("[#6;D4]").unwrap());

    assert_eq!(index.candidate_ids(&terminal_methyl), alloc::vec![0, 2]);
    assert_eq!(index.candidate_ids(&quaternary_carbon), alloc::vec![2]);
}

#[test]
fn corpus_index_filters_positive_recursive_alternatives() {
    let prepared_targets = ["CO", "CS", "CC"]
        .into_iter()
        .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
        .collect::<alloc::vec::Vec<_>>();
    let index = TargetCorpusIndex::new(&prepared_targets);
    let query = QueryScreen::new(&QueryMol::from_str("[!#1;$([#6]-[#8]),$([#6]-[#16])]").unwrap());

    assert_eq!(index.candidate_ids(&query), alloc::vec![0, 1]);
}

#[test]
fn corpus_index_filters_disjunctive_bond_pair_counts() {
    let prepared_targets = ["CCC", "C#CC", "C=CC", "C=C=C"]
        .into_iter()
        .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
        .collect::<alloc::vec::Vec<_>>();
    let index = TargetCorpusIndex::new(&prepared_targets);
    let query = QueryScreen::new(&QueryMol::from_str("[#6]-,=[#6]-,=[#6]").unwrap());

    assert_eq!(index.candidate_ids(&query), alloc::vec![0, 2, 3]);
}

#[test]
fn corpus_index_filters_ring_and_aromatic_atom_property_counts() {
    let prepared_targets = [
        "CCN",
        "c1ccccc1",
        "n1ccccc1",
        "C1CCCCC1",
        "c1ccc2ccccc2c1",
        "C1CC1",
    ]
    .into_iter()
    .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
    .collect::<alloc::vec::Vec<_>>();
    let index = TargetCorpusIndex::new(&prepared_targets);

    let ring_nitrogen = QueryScreen::new(&QueryMol::from_str("[#7&R]").unwrap());
    let aromatic_nitrogen = QueryScreen::new(&QueryMol::from_str("[n]").unwrap());
    let fused_ring_carbon = QueryScreen::new(&QueryMol::from_str("[#6;R2]").unwrap());
    let six_membered_ring_carbon = QueryScreen::new(&QueryMol::from_str("[#6;r6]").unwrap());
    let fused_ring_connectivity = QueryScreen::new(&QueryMol::from_str("[#6;x3]").unwrap());
    let three_membered_ring_carbon = QueryScreen::new(&QueryMol::from_str("[#6;r3]").unwrap());
    let acyclic_carbon = QueryScreen::new(&QueryMol::from_str("[#6;R0]").unwrap());

    assert_eq!(index.candidate_ids(&ring_nitrogen), alloc::vec![2]);
    assert_eq!(index.candidate_ids(&aromatic_nitrogen), alloc::vec![2]);
    assert_eq!(index.candidate_ids(&fused_ring_carbon), alloc::vec![4]);
    assert_eq!(
        index.candidate_ids(&six_membered_ring_carbon),
        alloc::vec![1, 2, 3, 4]
    );
    assert_eq!(
        index.candidate_ids(&fused_ring_connectivity),
        alloc::vec![4]
    );
    assert_eq!(
        index.candidate_ids(&three_membered_ring_carbon),
        alloc::vec![5]
    );
    assert_eq!(index.candidate_ids(&acyclic_carbon), alloc::vec![0]);
}

#[test]
fn corpus_index_counts_candidates_without_materializing_ids() {
    let prepared_targets = ["CC", "C=C", "CC(C)(C)C"]
        .into_iter()
        .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
        .collect::<alloc::vec::Vec<_>>();
    let index = TargetCorpusIndex::new(&prepared_targets);
    let query = QueryScreen::new(&QueryMol::from_str("[#6;D1;H3]").unwrap());
    let mut scratch = TargetCorpusScratch::new();

    assert_eq!(
        index.candidate_count(&query),
        index.candidate_ids(&query).len()
    );
    assert_eq!(
        index.candidate_count_with_scratch(&query, &mut scratch),
        index.candidate_ids(&query).len()
    );
}

#[test]
fn sharded_corpus_index_matches_monolithic_candidate_results() {
    let prepared_targets = [
        "CC",
        "C.C",
        "CCO",
        "CC(C)(C)C",
        "C=C",
        "C#N",
        "c1ccccc1",
        "C1CCCCC1",
        "O=C(O)c1ccccc1",
        "CC(=O)N",
        "ClCCl",
        "O",
    ]
    .into_iter()
    .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
    .collect::<alloc::vec::Vec<_>>();
    let monolithic = TargetCorpusIndex::new(&prepared_targets);
    let sharded =
        ShardedTargetCorpusIndex::from_prepared_target_chunks(prepared_targets.chunks(3)).unwrap();
    let queries = [
        "C",
        "CCC",
        "(C).(O)",
        "c1ccccc1",
        "[#6;D1;H3]-[#6;D4]",
        "[#6;D1;H2]=[#6;D1;H2]",
        "[#8;D1;H1]",
        "[R]",
        "C@C",
        "C#N",
        "[$([#6]=[#8])]",
    ]
    .into_iter()
    .map(|smarts| QueryScreen::new(&QueryMol::from_str(smarts).unwrap()))
    .collect::<alloc::vec::Vec<_>>();

    assert_eq!(sharded.len(), monolithic.len());
    assert_eq!(sharded.shards().len(), 4);
    assert_eq!(
        sharded.stats().target_count,
        monolithic.stats().target_count
    );

    let mut monolithic_scratch = TargetCorpusScratch::new();
    let mut sharded_scratch = TargetCorpusScratch::new();
    for query in &queries {
        assert_eq!(
            sharded.candidate_ids(query),
            monolithic.candidate_ids(query)
        );
        assert_eq!(
            sharded.candidate_count(query),
            monolithic.candidate_count(query)
        );

        let mut streamed = alloc::vec::Vec::new();
        sharded.for_each_candidate_id_with_scratch(query, &mut sharded_scratch, |target_id| {
            streamed.push(target_id);
        });
        let mut expected = alloc::vec::Vec::new();
        monolithic.candidate_ids_with_scratch_into(query, &mut monolithic_scratch, &mut expected);
        assert_eq!(streamed, expected);
    }

    let monolithic_sets = monolithic.candidate_sets(&queries);
    let sharded_sets = sharded.candidate_sets(&queries);
    assert_eq!(
        sharded_sets
            .iter()
            .map(TargetCandidateSet::target_ids)
            .collect::<alloc::vec::Vec<_>>(),
        monolithic_sets
            .iter()
            .map(TargetCandidateSet::target_ids)
            .collect::<alloc::vec::Vec<_>>()
    );
}

#[test]
fn sharded_corpus_index_rejects_overlapping_shards() {
    let prepared_targets = ["CC", "CO"]
        .into_iter()
        .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
        .collect::<alloc::vec::Vec<_>>();
    let first = TargetCorpusIndexShard::new(0, TargetCorpusIndex::new(&prepared_targets[..1]));
    let second = TargetCorpusIndexShard::new(0, TargetCorpusIndex::new(&prepared_targets[1..]));

    assert_eq!(
        ShardedTargetCorpusIndex::from_shards(alloc::vec![first, second]).unwrap_err(),
        ShardedTargetCorpusIndexError::OverlappingShard {
            shard_index: 1,
            previous_end_target_id: 1,
            shard_base_target_id: 0,
        }
    );
}

#[test]
fn corpus_index_new_drops_retained_screens_after_building_indexes() {
    let prepared_targets = ["CC", "CO"]
        .into_iter()
        .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
        .collect::<alloc::vec::Vec<_>>();
    let compact = TargetCorpusIndex::new(&prepared_targets);
    assert_eq!(compact.len(), 2);
    assert!(compact.screen(0).is_none());

    let screens = prepared_targets
        .iter()
        .map(TargetScreen::new)
        .collect::<alloc::vec::Vec<_>>();
    let retained = TargetCorpusIndex::from_screens(screens);
    assert_eq!(retained.len(), 2);
    assert!(retained.screen(0).is_some());
}

#[test]
fn corpus_index_candidates_are_a_subset_of_pairwise_screening() {
    let prepared_targets = ["CC", "C.C", "c1ccccc1", "ClCCl", "C1CCCCC1", "C=C", "C#N"]
        .into_iter()
        .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
        .collect::<alloc::vec::Vec<_>>();
    let target_screens = prepared_targets
        .iter()
        .map(TargetScreen::new)
        .collect::<alloc::vec::Vec<_>>();
    let index = TargetCorpusIndex::new(&prepared_targets);
    let mut scratch = TargetCorpusScratch::new();

    for smarts in ["CCC", "(C).(C)", "c", "[Cl]", "[R]", "C=C", "C#N", "C@C"] {
        let query = QueryScreen::new(&QueryMol::from_str(smarts).unwrap());
        let expected = target_screens
            .iter()
            .enumerate()
            .filter_map(|(target_id, target)| query.may_match(target).then_some(target_id))
            .collect::<alloc::vec::Vec<_>>();
        let mut actual = alloc::vec::Vec::new();
        index.candidate_ids_with_scratch_into(&query, &mut scratch, &mut actual);
        assert!(
            actual.iter().all(|target_id| expected.contains(target_id)),
            "index admitted target outside the coarse screen for {smarts}: actual={actual:?} expected={expected:?}"
        );
    }
}

#[test]
fn indexed_execution_matches_naive_exact_matrix() {
    let smarts_cases = [
        "C",
        "CC",
        "CCC",
        "(C).(O)",
        "c1ccccc1",
        "[#6;D1;H3]-[#6;D4]",
        "[#6;D1;H2]=[#6;D1;H2]",
        "[#8;D1;H1]",
        "[R]",
        "C@C",
        "C#N",
        "[$([#6]=[#8])]",
    ];
    let target_cases = [
        "CC",
        "C.C",
        "CCO",
        "CC(C)(C)C",
        "C=C",
        "C#N",
        "c1ccccc1",
        "C1CCCCC1",
        "O=C(O)c1ccccc1",
        "CC(=O)N",
        "ClCCl",
        "O",
    ];

    let queries = smarts_cases
        .into_iter()
        .map(|smarts| {
            let query = QueryMol::from_str(smarts).unwrap();
            CompiledQuery::new(query).unwrap()
        })
        .collect::<alloc::vec::Vec<_>>();
    let targets = target_cases
        .into_iter()
        .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
        .collect::<alloc::vec::Vec<_>>();
    let index = TargetCorpusIndex::new(&targets);

    let mut match_scratch = MatchScratch::new();
    let mut naive = alloc::vec::Vec::new();
    for (query_id, query) in queries.iter().enumerate() {
        for (target_id, target) in targets.iter().enumerate() {
            if query.matches_with_scratch(target, &mut match_scratch) {
                naive.push((query_id, target_id));
            }
        }
    }

    let mut index_scratch = TargetCorpusScratch::new();
    let mut match_scratch = MatchScratch::new();
    let mut indexed = alloc::vec::Vec::new();
    for (query_id, query) in queries.iter().enumerate() {
        let mut matches = alloc::vec::Vec::new();
        index.matching_target_ids_with_scratch_into(
            query,
            &targets,
            &mut index_scratch,
            &mut match_scratch,
            &mut matches,
        );
        indexed.extend(matches.into_iter().map(|target_id| (query_id, target_id)));
    }

    assert_eq!(indexed, naive);
}

#[test]
fn sharded_indexed_execution_matches_naive_exact_matrix() {
    let target_cases = [
        "CC",
        "C.C",
        "CCO",
        "CC(C)(C)C",
        "C=C",
        "C#N",
        "c1ccccc1",
        "C1CCCCC1",
        "O=C(O)c1ccccc1",
        "CC(=O)N",
        "ClCCl",
        "O",
    ];
    let targets = target_cases
        .into_iter()
        .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
        .collect::<alloc::vec::Vec<_>>();
    let index = ShardedTargetCorpusIndex::from_prepared_target_chunks(targets.chunks(4)).unwrap();

    for smarts in ["C", "CCC", "(C).(O)", "c1ccccc1", "C#N", "[$([#6]=[#8])]"] {
        let query = CompiledQuery::new(QueryMol::from_str(smarts).unwrap()).unwrap();
        let mut match_scratch = MatchScratch::new();
        let expected = targets
            .iter()
            .enumerate()
            .filter_map(|(target_id, target)| {
                query
                    .matches_with_scratch(target, &mut match_scratch)
                    .then_some(target_id)
            })
            .collect::<alloc::vec::Vec<_>>();

        assert_eq!(index.matching_target_ids(&query, &targets), expected);
    }
}

#[test]
fn indexed_execution_keeps_reported_amphetamine_scalar_matches() {
    let smarts = "[#6](~[#7])(~[#7])=[#8]";
    let target_cases = [
        "CCCN1C2=CC=CC=C2N(C1=O)CCC(=O)NC(C)CC3=CC=CC=C3F",
        "CC1=CC=CC=C1C[C@@H](C(=O)NC2=NC(=CS2)C(=O)OC)N3C(=C(NC3=O)C4=CC5=C(C=C4)OCO5)O",
        "C1[C@@H](O[C@@H]([C@@]1(C(=O)[C@H](CC2=CC=C(C=C2)F)N)O)C(C(=O)[C@H](CC3=CC=C(C=C3)F)N)O)N4C=C(C(=O)NC4=O)/C=C/Br",
        "CC(C)(CC1=CC=CC=C1)N(C)C(=O)CN(CCOC(=O)N2CCC(=CC2)N3C4=CC=CC=C4NC3=O)CC(=O)N(C)C(C)(C)CC5=CC=CC=C5",
        "CC(C)(C)OC(=O)N(CC1=CC=CC=C1)[C@H](CC2=CC=CC=C2)COC(=O)N3CCC(CC3)N4C5=CC=CC=C5NC4=O",
        "COC1=NC(=NC(=C1)C(=O)NC(CC2=CC=CC=C2)C=NN3C=C(NC3=O)O)OC",
        "C1=CC(=CC=C1CCN)CNC(=O)C2=CNC(=O)N2",
        "CN1C(=O)C2=C(N=NC(=N2)C3=CC=C(C=C3)CCN(C)C)N(C1=O)C4=CC=CC=C4",
        "CN1C(=C(C(=O)NC1=O)NCCC2=CC=C(C=C2)OC)N",
    ];

    let query = QueryMol::from_str(smarts).unwrap();
    let compiled = CompiledQuery::new(query.clone()).unwrap();
    let query_screen = QueryScreen::new(&query);
    let targets = target_cases
        .into_iter()
        .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
        .collect::<alloc::vec::Vec<_>>();
    let index = TargetCorpusIndex::new(&targets);

    let mut match_scratch = MatchScratch::new();
    let scalar_matches = targets
        .iter()
        .enumerate()
        .filter_map(|(target_id, target)| {
            compiled
                .matches_with_scratch(target, &mut match_scratch)
                .then_some(target_id)
        })
        .collect::<alloc::vec::Vec<_>>();
    assert_eq!(
        scalar_matches,
        (0..target_cases.len()).collect::<alloc::vec::Vec<_>>()
    );

    let candidate_ids = index.candidate_ids(&query_screen);
    for &target_id in &scalar_matches {
        assert!(
            candidate_ids.contains(&target_id),
            "index missed scalar match target_id={target_id}: {}",
            target_cases[target_id]
        );
    }

    let mut match_scratch = MatchScratch::new();
    let indexed_matches = candidate_ids
        .iter()
        .copied()
        .filter(|&target_id| compiled.matches_with_scratch(&targets[target_id], &mut match_scratch))
        .collect::<alloc::vec::Vec<_>>();
    assert_eq!(indexed_matches, scalar_matches);
}

#[test]
fn edge_feature_count_filter_respects_required_multiplicity() {
    let prepared_targets = ["CCO", "CCC", "CC(C)C"]
        .into_iter()
        .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
        .collect::<alloc::vec::Vec<_>>();
    let index = TargetCorpusIndex::new(&prepared_targets);
    let query = QueryScreen::new(&QueryMol::from_str("CCC").unwrap());

    assert_eq!(
        query.required_edge_feature_counts.as_ref(),
        &[(
            EdgeFeature::new(
                AtomFeature {
                    element: Some(elements_rs::Element::C),
                    aromatic: Some(false),
                    requires_ring: false,
                    ..AtomFeature::default()
                },
                EdgeBondFeature {
                    kind: None,
                    requires_ring: false,
                },
                AtomFeature {
                    element: Some(elements_rs::Element::C),
                    aromatic: Some(false),
                    requires_ring: false,
                    ..AtomFeature::default()
                },
            ),
            2
        )]
    );
    assert_eq!(index.candidate_ids(&query), alloc::vec![1, 2]);
}

#[test]
fn edge_feature_filter_uses_atomic_number_as_local_element_identity() {
    let prepared_targets = ["CC", "CO", "OO"]
        .into_iter()
        .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
        .collect::<alloc::vec::Vec<_>>();
    let index = TargetCorpusIndex::new(&prepared_targets);
    let query = QueryScreen::new(&QueryMol::from_str("[#6]-[#8]").unwrap());

    assert_eq!(query.required_edge_features.len(), 1);
    assert_eq!(index.candidate_ids(&query), alloc::vec![1]);
}

#[test]
fn edge_feature_filter_uses_exact_degree_and_total_hydrogen_identity() {
    let prepared_targets = ["CC", "C=C", "CC(C)(C)C"]
        .into_iter()
        .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
        .collect::<alloc::vec::Vec<_>>();
    let index = TargetCorpusIndex::new(&prepared_targets);

    let terminal_methyl_edge = QueryScreen::new(&QueryMol::from_str("[#6;D1;H3]-[#6;D4]").unwrap());
    let alkene_edge = QueryScreen::new(&QueryMol::from_str("[#6;D1;H2]=[#6;D1;H2]").unwrap());

    assert_eq!(terminal_methyl_edge.required_edge_features.len(), 1);
    assert_eq!(index.candidate_ids(&terminal_methyl_edge), alloc::vec![2]);
    assert_eq!(alkene_edge.required_edge_features.len(), 1);
    assert_eq!(index.candidate_ids(&alkene_edge), alloc::vec![1]);
}

#[test]
fn path3_feature_filter_rejects_missing_three_atom_context() {
    let prepared_targets = ["CCO", "COC", "CCC", "CCOC"]
        .into_iter()
        .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
        .collect::<alloc::vec::Vec<_>>();
    let index = TargetCorpusIndex::new(&prepared_targets);
    let query = QueryScreen::new(&QueryMol::from_str("COC").unwrap());

    assert_eq!(query.required_path3_features.len(), 1);
    assert_eq!(index.candidate_ids(&query), alloc::vec![1, 3]);
}

#[test]
fn path3_feature_filter_uses_exact_degree_and_ring_identity() {
    let prepared_targets = ["CCC", "CC(C)C", "C1CCCCC1"]
        .into_iter()
        .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
        .collect::<alloc::vec::Vec<_>>();
    let index = TargetCorpusIndex::new(&prepared_targets);

    let branched = QueryScreen::new(&QueryMol::from_str("[#6;D1]-[#6;D3]-[#6;D1]").unwrap());
    let ring = QueryScreen::new(&QueryMol::from_str("[#6;R;D2]-[#6;R;D2]-[#6;R;D2]").unwrap());

    assert_eq!(branched.required_path3_features.len(), 1);
    assert_eq!(index.candidate_ids(&branched), alloc::vec![1]);
    assert_eq!(ring.required_path3_features.len(), 1);
    assert_eq!(index.candidate_ids(&ring), alloc::vec![2]);
}

#[test]
fn path3_feature_count_filter_respects_required_multiplicity() {
    let prepared_targets = ["COC", "COCOC", "COCOCOC"]
        .into_iter()
        .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
        .collect::<alloc::vec::Vec<_>>();
    let index = TargetCorpusIndex::new(&prepared_targets);
    let query = QueryScreen::new(&QueryMol::from_str("COCOC").unwrap());

    assert_eq!(query.required_path3_features.len(), 2);
    assert_eq!(query.required_path3_feature_counts.len(), 2);
    assert_eq!(index.candidate_ids(&query), alloc::vec![1, 2]);
}

#[test]
fn path4_feature_filter_rejects_missing_four_atom_context() {
    let prepared_targets = ["CCOCO", "CCOCC", "NCOCO", "CCOCOC"]
        .into_iter()
        .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
        .collect::<alloc::vec::Vec<_>>();
    let index = TargetCorpusIndex::new(&prepared_targets);
    let query = QueryScreen::new(&QueryMol::from_str("COCO").unwrap());

    assert_eq!(query.required_path4_features.len(), 1);
    assert_eq!(query.required_path4_feature_counts.len(), 1);
    assert_eq!(index.candidate_ids(&query), alloc::vec![0, 2, 3]);
}

#[test]
fn path4_feature_count_filter_respects_required_multiplicity() {
    let prepared_targets = ["CCCCC", "CCCCCC", "CCCCCCC"]
        .into_iter()
        .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
        .collect::<alloc::vec::Vec<_>>();
    let index = TargetCorpusIndex::new(&prepared_targets);
    let query = QueryScreen::new(&QueryMol::from_str("CCCCCC").unwrap());

    assert_eq!(query.required_path4_features.len(), 1);
    assert_eq!(query.required_path4_feature_counts.len(), 1);
    assert_eq!(index.candidate_ids(&query), alloc::vec![1, 2]);
}

#[test]
fn path4_and_star3_features_use_exact_local_atom_identity() {
    let path4 = QueryScreen::new(&QueryMol::from_str("[C;D1;H3]-[C;D2]-[C;D2]-[O;D1;H1]").unwrap());
    let path4_atoms = path4
        .required_path4_features
        .iter()
        .flat_map(|feature| {
            [
                feature.left,
                feature.left_middle,
                feature.right_middle,
                feature.right,
            ]
        })
        .collect::<alloc::vec::Vec<_>>();

    assert!(path4_atoms.iter().any(|atom| {
        atom.element == Some(elements_rs::Element::C)
            && atom.aromatic == Some(false)
            && atom.degree == Some(1)
            && atom.total_hydrogens == Some(3)
    }));
    assert!(path4_atoms.iter().any(|atom| {
        atom.element == Some(elements_rs::Element::O)
            && atom.aromatic == Some(false)
            && atom.degree == Some(1)
            && atom.total_hydrogens == Some(1)
    }));

    let star3 = QueryScreen::new(&QueryMol::from_str("[C;D3]([O;D1;H1])([N;D1])-[C;D1]").unwrap());
    let star3_feature = star3
        .required_star3_features
        .first()
        .expect("query should have one star3 feature");

    assert_eq!(star3_feature.center.element, Some(elements_rs::Element::C));
    assert_eq!(star3_feature.center.degree, Some(3));
    assert!(star3_feature.arms.iter().any(|arm| {
        arm.atom.element == Some(elements_rs::Element::O)
            && arm.atom.degree == Some(1)
            && arm.atom.total_hydrogens == Some(1)
    }));
}

#[test]
fn query_screen_plans_more_specific_local_filters_first() {
    let query = QueryScreen::new(&QueryMol::from_str("COCO").unwrap());

    assert!(matches!(
        query.planned_feature_filters.first(),
        Some(QueryFeatureFilter::Path4 { .. })
    ));
    assert!(query
        .planned_feature_filters
        .iter()
        .skip(1)
        .any(|filter| matches!(filter, QueryFeatureFilter::Edge { .. })));
}

#[test]
fn cached_feature_masks_can_be_reused_after_scalar_filters() {
    let prepared_targets = ["CCO", "CCOC", "ClCCO", "NCCO", "CCCC", "COC", "CCN", "OCCO"]
        .into_iter()
        .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
        .collect::<alloc::vec::Vec<_>>();
    let index = TargetCorpusIndex::new(&prepared_targets);
    let warmup_query = QueryScreen::new(&QueryMol::from_str("CO").unwrap());
    let narrow_query = QueryScreen::new(&QueryMol::from_str("[Cl].CO").unwrap());

    let mut scratch = TargetCorpusScratch::new();
    let mut warmed_candidates = alloc::vec::Vec::new();
    index.candidate_ids_with_scratch_into(&warmup_query, &mut scratch, &mut warmed_candidates);
    assert!(!scratch.edge_mask_cache.is_empty());

    let mut reused_scratch_candidates = alloc::vec::Vec::new();
    index.candidate_ids_with_scratch_into(
        &narrow_query,
        &mut scratch,
        &mut reused_scratch_candidates,
    );

    assert_eq!(
        index.candidate_ids(&narrow_query),
        reused_scratch_candidates
    );
    assert_eq!(reused_scratch_candidates, alloc::vec![2]);
}

#[test]
fn batched_candidate_sets_match_individual_candidate_sets() {
    let prepared_targets = [
        "CCO",
        "CCOC",
        "ClCCO",
        "NCCO",
        "CCCC",
        "COC",
        "CCN",
        "OCCO",
        "CC(O)(N)Cl",
        "CC(O)(N)F",
    ]
    .into_iter()
    .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
    .collect::<alloc::vec::Vec<_>>();
    let query_screens = ["CO", "[Cl].CO", "COCO", "C(O)(N)Cl", "CCO"]
        .into_iter()
        .map(|smarts| QueryScreen::new(&QueryMol::from_str(smarts).unwrap()))
        .collect::<alloc::vec::Vec<_>>();
    let index = TargetCorpusIndex::new(&prepared_targets);

    let mut scratch = TargetCorpusScratch::new();
    let batched = index.candidate_sets_with_scratch(&query_screens, &mut scratch);
    let individual = query_screens
        .iter()
        .map(|screen| index.candidate_set(screen))
        .collect::<alloc::vec::Vec<_>>();

    assert_eq!(batched, individual);
}

#[test]
fn batched_streaming_candidates_match_candidate_sets() {
    let prepared_targets = [
        "CCO",
        "CCOC",
        "ClCCO",
        "NCCO",
        "CCCC",
        "COC",
        "CCN",
        "OCCO",
        "CC(O)(N)Cl",
        "CC(O)(N)F",
    ]
    .into_iter()
    .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
    .collect::<alloc::vec::Vec<_>>();
    let query_screens = ["CO", "[Cl].CO", "COCO", "C(O)(N)Cl", "CCO"]
        .into_iter()
        .map(|smarts| QueryScreen::new(&QueryMol::from_str(smarts).unwrap()))
        .collect::<alloc::vec::Vec<_>>();
    let index = TargetCorpusIndex::new(&prepared_targets);

    let mut scratch = TargetCorpusScratch::new();
    let expected = index.candidate_sets_with_scratch(&query_screens, &mut scratch);
    let mut actual = alloc::vec![alloc::vec::Vec::new(); query_screens.len()];
    index.for_each_candidate_id_batch_with_scratch(
        &query_screens,
        &mut scratch,
        |query_id, target_id| actual[query_id].push(target_id),
    );

    for (actual, expected) in actual.iter().zip(&expected) {
        assert_eq!(actual.as_slice(), expected.target_ids());
    }
}

#[test]
fn star3_feature_filter_rejects_missing_branch_context() {
    let prepared_targets = ["CC(O)(N)Cl", "CC(O)(N)F", "CC(O)Cl", "CC(Cl)(N)O"]
        .into_iter()
        .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
        .collect::<alloc::vec::Vec<_>>();
    let index = TargetCorpusIndex::new(&prepared_targets);
    let query = QueryScreen::new(&QueryMol::from_str("C(O)(N)Cl").unwrap());

    assert_eq!(query.required_star3_features.len(), 1);
    assert_eq!(query.required_star3_feature_counts.len(), 1);
    assert_eq!(index.candidate_ids(&query), alloc::vec![0, 3]);
}

#[test]
fn star3_feature_filter_keeps_matches_with_permuted_wildcard_arms() {
    let prepared_targets = ["CC(N)O", "CC(O)O"]
        .into_iter()
        .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
        .collect::<alloc::vec::Vec<_>>();
    let index = TargetCorpusIndex::new(&prepared_targets);
    let query = QueryMol::from_str("C(*)(*)N").unwrap();
    let compiled = CompiledQuery::new(query.clone()).unwrap();
    let query_screen = QueryScreen::new(&query);

    let mut match_scratch = MatchScratch::new();
    let expected = prepared_targets
        .iter()
        .enumerate()
        .filter_map(|(target_id, target)| {
            compiled
                .matches_with_scratch(target, &mut match_scratch)
                .then_some(target_id)
        })
        .collect::<alloc::vec::Vec<_>>();

    assert_eq!(expected, alloc::vec![0]);
    assert_eq!(index.candidate_ids(&query_screen), expected);
}

#[test]
fn query_screen_extracts_atomic_number_isotope_and_nonpositive_ring_bounds_conservatively() {
    let atomic_number = QueryScreen::new(&QueryMol::from_str("[#8]").unwrap());
    let isotope = QueryScreen::new(&QueryMol::from_str("[18O]").unwrap());
    let ring_zero = QueryScreen::new(&QueryMol::from_str("[R0]").unwrap());
    let ring_range_zero = QueryScreen::new(&QueryMol::from_str("[r{0-2}]").unwrap());

    assert_eq!(
        atomic_number
            .required_element_counts
            .get(&elements_rs::Element::O),
        Some(&1)
    );
    assert_eq!(
        isotope
            .required_element_counts
            .get(&elements_rs::Element::O),
        Some(&1)
    );
    assert_eq!(ring_zero.min_ring_atom_count, 0);
    assert_eq!(ring_range_zero.min_ring_atom_count, 0);
}

#[test]
fn screen_never_filters_true_matches_from_frozen_fixtures() {
    for fixture in [
        include_str!("../../corpus/matching/single-atom-v0.rdkit.json"),
        include_str!("../../corpus/matching/connected-v0.rdkit.json"),
        include_str!("../../corpus/matching/ring-v0.rdkit.json"),
        include_str!("../../corpus/matching/counts-v0.rdkit.json"),
        include_str!("../../corpus/matching/disconnected-v0.rdkit.json"),
        include_str!("../../corpus/matching/recursive-v0.rdkit.json"),
        include_str!("../../corpus/matching/stereo-v0.rdkit.json"),
    ] {
        let cases: alloc::vec::Vec<ExpectedCase> =
            serde_json::from_str(fixture).expect("valid frozen fixture");
        for case in cases {
            if !case.expected_match {
                continue;
            }
            let query = QueryMol::from_str(&case.smarts).expect("valid SMARTS");
            let target = PreparedTarget::new(case.smiles.parse::<Smiles>().expect("valid SMILES"));
            let query_screen = QueryScreen::new(&query);
            let target_screen = TargetScreen::new(&target);
            let index = TargetCorpusIndex::new(alloc::slice::from_ref(&target));
            assert!(
                query_screen.may_match(&target_screen),
                "screen rejected known true match: SMARTS {:?} vs SMILES {:?}",
                case.smarts,
                case.smiles
            );
            assert_eq!(
                index.candidate_ids(&query_screen),
                alloc::vec![0],
                "index rejected known true match: SMARTS {:?} vs SMILES {:?}",
                case.smarts,
                case.smiles
            );
        }
    }
}

#[test]
fn index_never_filters_scalar_matches_from_frozen_fixtures() {
    for fixture in [
        include_str!("../../corpus/matching/single-atom-v0.rdkit.json"),
        include_str!("../../corpus/matching/connected-v0.rdkit.json"),
        include_str!("../../corpus/matching/ring-v0.rdkit.json"),
        include_str!("../../corpus/matching/counts-v0.rdkit.json"),
        include_str!("../../corpus/matching/disconnected-v0.rdkit.json"),
        include_str!("../../corpus/matching/recursive-v0.rdkit.json"),
        include_str!("../../corpus/matching/stereo-v0.rdkit.json"),
        include_str!("../../corpus/matching/stereo-gap-v1.rdkit.json"),
        include_str!("../../corpus/matching/stereo-gap-v2.rdkit.json"),
        include_str!("../../corpus/matching/stereo-gap-v3.rdkit.json"),
        include_str!("../../corpus/matching/stereo-gap-v4.rdkit.json"),
        include_str!("../../corpus/matching/stereo-gap-v5.rdkit.json"),
        include_str!("../../corpus/matching/stereo-gap-v6.rdkit.json"),
        include_str!("../../corpus/matching/stereo-gap-v7.rdkit.json"),
        include_str!("../../corpus/matching/benchmark-alerts-v0.rdkit.json"),
        include_str!("../../corpus/matching/benchmark-aromaticity-v0.rdkit.json"),
        include_str!("../../corpus/matching/benchmark-counted-v0.rdkit.json"),
        include_str!("../../corpus/matching/benchmark-extensions-v0.rdkit.json"),
    ] {
        let cases: alloc::vec::Vec<ExpectedCase> =
            serde_json::from_str(fixture).expect("valid frozen fixture");
        for case in cases {
            let query = QueryMol::from_str(&case.smarts).expect("valid SMARTS");
            let compiled = CompiledQuery::new(query.clone()).expect("supported SMARTS");
            let target = PreparedTarget::new(case.smiles.parse::<Smiles>().expect("valid SMILES"));
            let mut match_scratch = MatchScratch::new();
            if !compiled.matches_with_scratch(&target, &mut match_scratch) {
                continue;
            }

            let query_screen = QueryScreen::new(&query);
            let index = TargetCorpusIndex::new(alloc::slice::from_ref(&target));
            assert_eq!(
                index.candidate_ids(&query_screen),
                alloc::vec![0],
                "index rejected scalar match: SMARTS {:?} vs SMILES {:?}",
                case.smarts,
                case.smiles
            );
        }
    }
}

#[test]
fn sharded_corpus_index_error_displays_each_variant() {
    use alloc::string::ToString;

    let overflow = ShardedTargetCorpusIndexError::TargetIdOverflow {
        shard_index: 2,
        base_target_id: 9,
        shard_len: 4,
    };
    assert_eq!(
        overflow.to_string(),
        "target id overflow in shard 2: base=9, len=4"
    );

    assert_eq!(
        ShardedTargetCorpusIndexError::TargetCountOverflow.to_string(),
        "sharded target count overflow"
    );

    let overlap = ShardedTargetCorpusIndexError::OverlappingShard {
        shard_index: 1,
        previous_end_target_id: 5,
        shard_base_target_id: 3,
    };
    assert_eq!(
        overlap.to_string(),
        "shard 1 starts at 3, before previous shard end 5"
    );
}

#[test]
fn query_screen_feature_stats_count_required_signatures() {
    let plain = QueryScreen::new(&QueryMol::from_str("C").unwrap());
    let plain_stats = plain.feature_stats();
    assert_eq!(plain_stats.edge_features, 0);
    assert_eq!(plain_stats.path4_features, 0);
    assert_eq!(plain_stats.star3_features, 0);
    assert_eq!(plain_stats.alternative_screen_groups, 0);

    // A four-atom branched skeleton induces edge, path, and star signatures.
    let rich = QueryScreen::new(&QueryMol::from_str("CC(C)CO").unwrap());
    let stats = rich.feature_stats();
    assert!(stats.edge_features > 0);
    assert!(stats.path3_features > 0);
    assert!(stats.path4_features > 0 || stats.star3_features > 0);

    // A recursive alternative populates the alternative-screen group counts.
    let recursive = QueryScreen::new(&QueryMol::from_str("[$(CO),$(CN)]").unwrap());
    let recursive_stats = recursive.feature_stats();
    assert_eq!(recursive_stats.alternative_screen_groups, 1);
    assert_eq!(recursive_stats.alternative_screens, 2);
}

#[test]
fn corpus_index_matching_target_ids_wrappers_return_exact_hits() {
    let targets = ["CCO", "CC=O", "c1ccccc1", "CCN"]
        .into_iter()
        .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
        .collect::<alloc::vec::Vec<_>>();
    let index = TargetCorpusIndex::new(&targets);
    let query = CompiledQuery::new(QueryMol::from_str("[#6]=[#8]").unwrap()).unwrap();

    // Exercises the allocating wrapper and, transitively, matching_target_ids_into.
    assert_eq!(index.matching_target_ids(&query, &targets), alloc::vec![1]);

    let mut out = alloc::vec![999_usize];
    index.matching_target_ids_into(&query, &targets, &mut out);
    assert_eq!(out, alloc::vec![1]);
}

#[test]
fn candidate_id_iteration_spans_full_and_partial_bitset_words() {
    // More than 64 targets forces the candidate-bit iterator to walk at least
    // one full 64-bit word plus a trailing partial word. Even-indexed targets
    // carry oxygen; odd-indexed ones do not.
    let prepared_targets = (0..70usize)
        .map(|i| {
            let smiles = if i % 2 == 0 { "CCO" } else { "CC" };
            PreparedTarget::new(Smiles::from_str(smiles).unwrap())
        })
        .collect::<alloc::vec::Vec<_>>();
    let index = TargetCorpusIndex::new(&prepared_targets);

    let query = QueryScreen::new(&QueryMol::from_str("[#8]").unwrap());
    let candidates = index.candidate_ids(&query);

    let expected = (0..70usize)
        .filter(|i| i % 2 == 0)
        .collect::<alloc::vec::Vec<_>>();
    assert_eq!(candidates, expected);
    assert!(
        candidates.iter().any(|&id| id >= 64),
        "candidates must reach beyond the first 64-bit word"
    );
}

#[cfg(feature = "mem_dbg")]
#[test]
fn target_corpus_index_memory_stats_account_for_every_index_component() {
    use mem_dbg::{MemSize, SizeFlags};

    let prepared_targets = [
        "c1ccccc1",
        "C1CCCCC1",
        "CC(=O)O",
        "CCN",
        "C#N",
        "ClC(Cl)Cl",
        "c1ccncc1",
        "O=C(O)c1ccccc1",
        "CC(C)(C)O",
        "C1CC1",
    ]
    .into_iter()
    .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
    .collect::<alloc::vec::Vec<_>>();

    // `new` builds the full local-feature indexes but drops retained screens.
    let index = TargetCorpusIndex::new(&prepared_targets);
    let stats = index.memory_stats();

    assert_eq!(stats.struct_size, size_of::<TargetCorpusIndex>());
    assert!(stats.scalar_count_indexes > 0);
    assert!(stats.atom_property_count_indexes > 0);
    assert!(stats.edge_postings > 0);
    assert!(stats.edge_masks > 0);
    assert!(stats.path3_postings > 0);
    assert!(stats.path4_postings > 0);
    assert!(stats.star3_postings > 0);
    // `new` drops retained screens, so that component contributes nothing here.
    assert_eq!(stats.retained_screens, 0);
    assert!(stats.total() > stats.struct_size);
    // The custom `MemSize` impl reports the same aggregate as `memory_stats`.
    assert_eq!(index.mem_size(SizeFlags::default()), stats.total());

    // `from_screens` retains per-target screens, exercising the retained-screen
    // heap accounting and the BTree map size heuristic.
    let screens = prepared_targets
        .iter()
        .map(TargetScreen::new)
        .collect::<alloc::vec::Vec<_>>();
    let retained = TargetCorpusIndex::from_screens(screens);
    let retained_stats = retained.memory_stats();
    assert!(retained_stats.retained_screens > 0);
    assert_eq!(
        retained.mem_size(SizeFlags::default()),
        retained_stats.total()
    );

    // The sharded aggregate is the sum of its shards' accounted bytes.
    let sharded =
        ShardedTargetCorpusIndex::from_prepared_target_chunks(prepared_targets.chunks(4)).unwrap();
    let sharded_stats = sharded.memory_stats();
    let shard_total: usize = sharded
        .shards()
        .iter()
        .map(|shard| shard.index().memory_stats().total())
        .sum();
    // The sharded aggregate adds the wrapper struct and shard-array bytes on top
    // of every inner shard's accounted bytes.
    assert!(sharded_stats.total() > shard_total);
    assert_eq!(
        sharded.mem_size(SizeFlags::default()),
        sharded_stats.total()
    );
}
