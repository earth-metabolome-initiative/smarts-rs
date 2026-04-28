use alloc::string::String;
use core::str::FromStr;

use crate::QueryMol;
use serde::Deserialize;
use smiles_parser::Smiles;

use super::{
    AtomFeature, BondCountScreen, EdgeBondFeature, EdgeFeature, QueryFeatureFilter, QueryScreen,
    ShardedTargetCorpusIndex, ShardedTargetCorpusIndexError, TargetCandidateSet, TargetCorpusIndex,
    TargetCorpusIndexShard, TargetCorpusScratch, TargetScreen,
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
}

#[test]
fn query_screen_extracts_conservative_bond_bounds() {
    let double = QueryScreen::new(&QueryMol::from_str("C=C").unwrap());
    let triple = QueryScreen::new(&QueryMol::from_str("C#N").unwrap());
    let aromatic = QueryScreen::new(&QueryMol::from_str("c:c").unwrap());
    let ring = QueryScreen::new(&QueryMol::from_str("C@C").unwrap());

    assert_eq!(double.required_bond_counts.double, 1);
    assert_eq!(triple.required_bond_counts.triple, 1);
    assert_eq!(aromatic.required_bond_counts.aromatic, 1);
    assert_eq!(ring.required_bond_counts.ring, 1);
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

    let small_target = TargetScreen::new(&PreparedTarget::new(Smiles::from_str("CC").unwrap()));
    let aliphatic_target = TargetScreen::new(&PreparedTarget::new(Smiles::from_str("CC").unwrap()));
    let acyclic_target = TargetScreen::new(&PreparedTarget::new(Smiles::from_str("CCC").unwrap()));
    let oxygen_target = TargetScreen::new(&PreparedTarget::new(Smiles::from_str("O").unwrap()));
    let single_bond_target =
        TargetScreen::new(&PreparedTarget::new(Smiles::from_str("CC").unwrap()));

    assert!(!atom_count_query.may_match(&small_target));
    assert!(!aromatic_query.may_match(&aliphatic_target));
    assert!(!ring_query.may_match(&acyclic_target));
    assert!(!element_query.may_match(&oxygen_target));
    assert!(!double_bond_query.may_match(&single_bond_target));
    assert!(!ring_bond_query.may_match(&single_bond_target));
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
            (
                CompiledQuery::new(query.clone()).unwrap(),
                QueryScreen::new(&query),
            )
        })
        .collect::<alloc::vec::Vec<_>>();
    let targets = target_cases
        .into_iter()
        .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
        .collect::<alloc::vec::Vec<_>>();
    let index = TargetCorpusIndex::new(&targets);

    let mut match_scratch = MatchScratch::new();
    let mut naive = alloc::vec::Vec::new();
    for (query_id, (query, _)) in queries.iter().enumerate() {
        for (target_id, target) in targets.iter().enumerate() {
            if query.matches_with_scratch(target, &mut match_scratch) {
                naive.push((query_id, target_id));
            }
        }
    }

    let mut index_scratch = TargetCorpusScratch::new();
    let mut match_scratch = MatchScratch::new();
    let mut indexed = alloc::vec::Vec::new();
    for (query_id, (query, screen)) in queries.iter().enumerate() {
        let candidate_set = index.candidate_set_with_scratch(screen, &mut index_scratch);
        for &target_id in candidate_set.target_ids() {
            if query.matches_with_scratch(&targets[target_id], &mut match_scratch) {
                indexed.push((query_id, target_id));
            }
        }
    }

    assert_eq!(indexed, naive);
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
