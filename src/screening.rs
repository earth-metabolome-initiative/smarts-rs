//! Conservative screening for many-SMARTS × many-SMILES workloads.
//!
//! The screen never proves a match. It only rules out impossible pairs using
//! cheap lower bounds derived from the query and cheap summary counts derived
//! from the prepared target.
//!
//! The main workflow is:
//! - prepare the fixed target corpus once
//! - build one [`TargetCorpusIndex`] over those prepared targets
//! - build a [`QueryScreen`] for each new SMARTS query
//! - run full SMARTS matching only on the returned candidate target ids
//!
//! ```
//! use core::str::FromStr;
//!
//! use smarts_rs::{
//!     CompiledQuery, MatchScratch, PreparedTarget, QueryMol, QueryScreen, TargetCorpusIndex,
//!     TargetCorpusScratch,
//! };
//! use smiles_parser::Smiles;
//!
//! let targets = ["CCO", "CCCC", "CC=O", "CCN"]
//!     .into_iter()
//!     .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
//!     .collect::<Vec<_>>();
//!
//! let index = TargetCorpusIndex::new(&targets);
//! let query = CompiledQuery::new(QueryMol::from_str("[#6]=[#8]").unwrap()).unwrap();
//! let screen = QueryScreen::new(query.query());
//!
//! let mut corpus_scratch = TargetCorpusScratch::new();
//! let candidates = index.candidate_set_with_scratch(&screen, &mut corpus_scratch);
//!
//! let mut match_scratch = MatchScratch::new();
//! let indexed_hits = candidates
//!     .target_ids()
//!     .iter()
//!     .copied()
//!     .filter(|&target_id| {
//!         query.matches_with_scratch(&targets[target_id], &mut match_scratch)
//!     })
//!     .collect::<Vec<_>>();
//!
//! let naive_hits = targets
//!     .iter()
//!     .enumerate()
//!     .filter_map(|(target_id, target)| {
//!         query.matches_with_scratch(target, &mut match_scratch)
//!             .then_some(target_id)
//!     })
//!     .collect::<Vec<_>>();
//!
//! assert_eq!(indexed_hits, naive_hits);
//! assert_eq!(indexed_hits, vec![2]);
//! ```

use alloc::{boxed::Box, collections::BTreeMap, vec, vec::Vec};

use crate::{AtomExpr, ComponentGroupId, QueryMol};
use elements_rs::Element;

use crate::{prepared::PreparedTarget, target::BondLabel};

mod bitset;
mod features;
mod requirements;

use bitset::{
    bitset_word_count, ensure_zeroed_words, for_each_set_bit, intersect_source,
    intersect_source_with_population, set_bit, CachedFeatureMask, CountBitsetIndex,
    RequiredCountFilter,
};
use features::{
    AtomFeature, BondCountScreen, EdgeBondFeature, EdgeFeature, Path3Feature, Path4Feature,
    RequiredBondKind, Star3Arm, Star3Feature,
};
use requirements::{
    exact_atom_requirement, forced_atom_count_requirement, forced_atom_requirement,
    forced_bond_requirement,
};

/// Counts of graph-signature requirements extracted from one query screen.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QueryScreenFeatureStats {
    /// Number of distinct required 2-atom edge signatures.
    pub edge_features: usize,
    /// Number of distinct required 3-atom path signatures.
    pub path3_features: usize,
    /// Number of distinct required 4-atom path signatures.
    pub path4_features: usize,
    /// Number of distinct required 3-neighbor star signatures.
    pub star3_features: usize,
    /// Number of required edge signature/multiplicity masks.
    pub edge_feature_masks: usize,
    /// Number of required path3 signature/multiplicity masks.
    pub path3_feature_masks: usize,
    /// Number of required path4 signature/multiplicity masks.
    pub path4_feature_masks: usize,
    /// Number of required star3 signature/multiplicity masks.
    pub star3_feature_masks: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum QueryFeatureFilter {
    Edge {
        feature: EdgeFeature,
        required_count: u16,
    },
    Path3 {
        feature: Path3Feature,
        required_count: u16,
    },
    Path4 {
        feature: Path4Feature,
        required_count: u16,
    },
    Star3 {
        feature: Star3Feature,
        required_count: u16,
    },
}

/// Conservative summary of one compiled SMARTS query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryScreen {
    /// Minimum number of target atoms required by the query.
    min_atom_count: usize,
    /// Minimum number of target connected components required by grouped
    /// disconnected SMARTS.
    min_target_component_count: usize,
    /// Lower bounds for exact element occurrences the target must contain.
    required_element_counts: BTreeMap<Element, usize>,
    /// Lower bounds for exact SMARTS `D` topological degree occurrences.
    required_degree_counts: BTreeMap<u16, usize>,
    /// Lower bounds for exact SMARTS `H` total-hydrogen occurrences.
    required_total_hydrogen_counts: BTreeMap<u16, usize>,
    /// Minimum number of aromatic atoms required by the query.
    min_aromatic_atom_count: usize,
    /// Minimum number of ring-member atoms required by the query.
    min_ring_atom_count: usize,
    /// Lower bounds for required bond categories.
    required_bond_counts: BondCountScreen,
    /// Required exact local edge signatures.
    required_edge_features: Box<[EdgeFeature]>,
    /// Required exact local edge signature multiplicities.
    required_edge_feature_counts: Box<[(EdgeFeature, u16)]>,
    /// Required exact local 3-atom path signatures.
    required_path3_features: Box<[Path3Feature]>,
    /// Required exact local 3-atom path signature multiplicities.
    required_path3_feature_counts: Box<[(Path3Feature, u16)]>,
    /// Required exact local 4-atom path signatures.
    required_path4_features: Box<[Path4Feature]>,
    /// Required exact local 4-atom path signature multiplicities.
    required_path4_feature_counts: Box<[(Path4Feature, u16)]>,
    /// Required exact 3-neighbor star signatures.
    required_star3_features: Box<[Star3Feature]>,
    /// Required exact 3-neighbor star signature multiplicities.
    required_star3_feature_counts: Box<[(Star3Feature, u16)]>,
    /// Planned local-signature filter order.
    planned_feature_filters: Box<[QueryFeatureFilter]>,
}

impl QueryScreen {
    /// Builds a conservative screen summary from one parsed SMARTS query.
    #[must_use]
    pub fn new(query: &QueryMol) -> Self {
        let atom_requirements = collect_query_atom_requirements(query);
        let (incident_bonds, required_bond_counts) = collect_query_bond_requirements(query);
        let feature_counts = collect_query_feature_counts(query, &incident_bonds);
        let required_edge_feature_counts = feature_counts.edge.into_iter().collect::<Vec<_>>();
        let required_path3_feature_counts = feature_counts.path3.into_iter().collect::<Vec<_>>();
        let required_path4_feature_counts = feature_counts.path4.into_iter().collect::<Vec<_>>();
        let required_star3_feature_counts = feature_counts.star3.into_iter().collect::<Vec<_>>();
        let planned_feature_filters = plan_feature_filters(
            &required_edge_feature_counts,
            &required_path3_feature_counts,
            &required_path4_feature_counts,
            &required_star3_feature_counts,
        );

        Self {
            min_atom_count: query.atom_count(),
            min_target_component_count: grouped_component_count(query.component_groups()),
            required_element_counts: atom_requirements.element_counts,
            required_degree_counts: atom_requirements.degree_counts,
            required_total_hydrogen_counts: atom_requirements.total_hydrogen_counts,
            min_aromatic_atom_count: atom_requirements.min_aromatic_count,
            min_ring_atom_count: atom_requirements.min_ring_count,
            required_bond_counts,
            required_edge_features: required_edge_feature_counts
                .iter()
                .map(|(feature, _)| *feature)
                .collect(),
            required_edge_feature_counts: required_edge_feature_counts.into_boxed_slice(),
            required_path3_features: required_path3_feature_counts
                .iter()
                .map(|(feature, _)| *feature)
                .collect(),
            required_path3_feature_counts: required_path3_feature_counts.into_boxed_slice(),
            required_path4_features: required_path4_feature_counts
                .iter()
                .map(|(feature, _)| *feature)
                .collect(),
            required_path4_feature_counts: required_path4_feature_counts.into_boxed_slice(),
            required_star3_features: required_star3_feature_counts
                .iter()
                .map(|(feature, _)| *feature)
                .collect(),
            required_star3_feature_counts: required_star3_feature_counts.into_boxed_slice(),
            planned_feature_filters,
        }
    }

    /// Returns diagnostic counts for graph-signature requirements.
    ///
    /// These counts are useful for benchmark/index introspection without
    /// exposing the exact internal signature types.
    #[must_use]
    pub fn feature_stats(&self) -> QueryScreenFeatureStats {
        QueryScreenFeatureStats {
            edge_features: self.required_edge_features.len(),
            path3_features: self.required_path3_features.len(),
            path4_features: self.required_path4_features.len(),
            star3_features: self.required_star3_features.len(),
            edge_feature_masks: self.required_edge_feature_counts.len(),
            path3_feature_masks: self.required_path3_feature_counts.len(),
            path4_feature_masks: self.required_path4_feature_counts.len(),
            star3_feature_masks: self.required_star3_feature_counts.len(),
        }
    }

    /// Returns whether the target summary could still satisfy the query.
    #[must_use]
    pub fn may_match(&self, target: &TargetScreen) -> bool {
        if target.atom_count < self.min_atom_count {
            return false;
        }
        if target.connected_component_count < self.min_target_component_count {
            return false;
        }
        if target.aromatic_atom_count < self.min_aromatic_atom_count {
            return false;
        }
        if target.ring_atom_count < self.min_ring_atom_count {
            return false;
        }
        if target.bond_counts.single < self.required_bond_counts.single
            || target.bond_counts.double < self.required_bond_counts.double
            || target.bond_counts.triple < self.required_bond_counts.triple
            || target.bond_counts.aromatic < self.required_bond_counts.aromatic
            || target.bond_counts.ring < self.required_bond_counts.ring
        {
            return false;
        }
        self.required_element_counts
            .iter()
            .all(|(element, required)| {
                target
                    .element_counts
                    .get(element)
                    .copied()
                    .unwrap_or_default()
                    >= *required
            })
            && required_counts_may_match(&self.required_degree_counts, &target.degree_counts)
            && required_counts_may_match(
                &self.required_total_hydrogen_counts,
                &target.total_hydrogen_counts,
            )
    }
}

/// Conservative summary of one prepared target molecule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetScreen {
    /// Number of atoms in the target.
    atom_count: usize,
    /// Number of connected components in the target.
    connected_component_count: usize,
    /// Exact element occurrence counts in the target.
    element_counts: BTreeMap<Element, usize>,
    /// Exact SMARTS `D` topological degree occurrence counts in the target.
    degree_counts: BTreeMap<u16, usize>,
    /// Exact SMARTS `H` total-hydrogen occurrence counts in the target.
    total_hydrogen_counts: BTreeMap<u16, usize>,
    /// Number of aromatic atoms in the target.
    aromatic_atom_count: usize,
    /// Number of ring-member atoms in the target.
    ring_atom_count: usize,
    /// Exact bond-category counts in the target.
    bond_counts: BondCountScreen,
}

impl TargetScreen {
    /// Builds a conservative target summary from one prepared target.
    #[must_use]
    pub fn new(target: &PreparedTarget) -> Self {
        let mut element_counts = BTreeMap::new();
        let mut degree_counts = BTreeMap::new();
        let mut total_hydrogen_counts = BTreeMap::new();
        let mut aromatic_atom_count = 0usize;
        let mut ring_atom_count = 0usize;
        let mut bond_counts = BondCountScreen::default();
        let mut max_component = None::<usize>;

        for atom_id in 0..target.atom_count() {
            if let Some(element) = target
                .atom(atom_id)
                .and_then(smiles_parser::atom::Atom::element)
            {
                *element_counts.entry(element).or_insert(0) += 1;
            }
            if let Some(degree) = target.degree(atom_id) {
                *degree_counts
                    .entry(u16::try_from(degree).unwrap_or(u16::MAX))
                    .or_insert(0) += 1;
            }
            if let Some(total_hydrogens) = target.total_hydrogen_count(atom_id) {
                *total_hydrogen_counts
                    .entry(u16::from(total_hydrogens))
                    .or_insert(0) += 1;
            }
            if target.is_aromatic(atom_id) {
                aromatic_atom_count += 1;
            }
            if target.is_ring_atom(atom_id) {
                ring_atom_count += 1;
            }
            if let Some(component_id) = target.connected_component(atom_id) {
                max_component =
                    Some(max_component.map_or(component_id, |current| current.max(component_id)));
            }
            for (other_id, label) in target.neighbors(atom_id) {
                if atom_id >= other_id {
                    continue;
                }
                match label {
                    BondLabel::Single | BondLabel::Up | BondLabel::Down => {
                        bond_counts.single += 1;
                    }
                    BondLabel::Double => bond_counts.double += 1,
                    BondLabel::Triple => bond_counts.triple += 1,
                    BondLabel::Aromatic => bond_counts.aromatic += 1,
                    BondLabel::Any => {}
                }
                if target.is_ring_bond(atom_id, other_id) {
                    bond_counts.ring += 1;
                }
            }
        }

        Self {
            atom_count: target.atom_count(),
            connected_component_count: max_component.map_or(0, |value| value + 1),
            element_counts,
            degree_counts,
            total_hydrogen_counts,
            aromatic_atom_count,
            ring_atom_count,
            bond_counts,
        }
    }
}

type TargetId = u32;
type EdgeFeatureCountIndex = BTreeMap<EdgeFeature, Vec<(TargetId, u16)>>;
type Path3FeatureCountIndex = BTreeMap<Path3Feature, Vec<(TargetId, u16)>>;
type Path4FeatureCountIndex = BTreeMap<Path4Feature, Vec<(TargetId, u16)>>;
type Star3FeatureCountIndex = BTreeMap<Star3Feature, Vec<(TargetId, u16)>>;
type IndexedFeatureCountIndex<T> = Box<[(T, CountBitsetIndex)]>;
type SparseFeatureCounts = Box<[(TargetId, u16)]>;
type IndexedSparseFeatureCountIndex<T> = Box<[(T, SparseFeatureCounts)]>;
type FeatureIdMask = Box<[u64]>;
type IndexedFeatureIdMask<T> = Box<[(T, FeatureIdMask)]>;
type IndexedEdgeSparseIndexParts = (
    IndexedSparseFeatureCountIndex<EdgeFeature>,
    EdgeFeatureMaskIndex,
    Box<[AtomFeature]>,
    Box<[EdgeBondFeature]>,
);

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct EdgeFeatureMaskIndex {
    left_atoms: IndexedFeatureIdMask<AtomFeature>,
    bonds: IndexedFeatureIdMask<EdgeBondFeature>,
    right_atoms: IndexedFeatureIdMask<AtomFeature>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CandidateMaskState {
    has_active_source: bool,
    population: usize,
}

type IndexedPath3SparseIndexParts = (
    IndexedSparseFeatureCountIndex<Path3Feature>,
    Path3FeatureMaskIndex,
    Box<[AtomFeature]>,
    Box<[EdgeBondFeature]>,
);

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Path3FeatureMaskIndex {
    left_atoms: IndexedFeatureIdMask<AtomFeature>,
    left_bonds: IndexedFeatureIdMask<EdgeBondFeature>,
    center_atoms: IndexedFeatureIdMask<AtomFeature>,
    right_bonds: IndexedFeatureIdMask<EdgeBondFeature>,
    right_atoms: IndexedFeatureIdMask<AtomFeature>,
}

type IndexedPath4SparseIndexParts = (
    IndexedSparseFeatureCountIndex<Path4Feature>,
    Path4FeatureMaskIndex,
    Box<[AtomFeature]>,
    Box<[EdgeBondFeature]>,
);

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Path4FeatureMaskIndex {
    left_atoms: IndexedFeatureIdMask<AtomFeature>,
    left_bonds: IndexedFeatureIdMask<EdgeBondFeature>,
    left_middle_atoms: IndexedFeatureIdMask<AtomFeature>,
    center_bonds: IndexedFeatureIdMask<EdgeBondFeature>,
    right_middle_atoms: IndexedFeatureIdMask<AtomFeature>,
    right_bonds: IndexedFeatureIdMask<EdgeBondFeature>,
    right_atoms: IndexedFeatureIdMask<AtomFeature>,
}

type IndexedStar3SparseIndexParts = (
    IndexedSparseFeatureCountIndex<Star3Feature>,
    Star3FeatureMaskIndex,
    Box<[AtomFeature]>,
    Box<[EdgeBondFeature]>,
);

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Star3FeatureMaskIndex {
    center_atoms: IndexedFeatureIdMask<AtomFeature>,
    first_bonds: IndexedFeatureIdMask<EdgeBondFeature>,
    first_atoms: IndexedFeatureIdMask<AtomFeature>,
    second_bonds: IndexedFeatureIdMask<EdgeBondFeature>,
    second_atoms: IndexedFeatureIdMask<AtomFeature>,
    third_bonds: IndexedFeatureIdMask<EdgeBondFeature>,
    third_atoms: IndexedFeatureIdMask<AtomFeature>,
}

type TargetGraphFeatures = (
    BTreeMap<EdgeFeature, usize>,
    BTreeMap<Path3Feature, usize>,
    BTreeMap<Path4Feature, usize>,
    BTreeMap<Star3Feature, usize>,
);

/// Reusable scratch buffers for candidate generation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TargetCorpusScratch<'a> {
    candidate_mask: Vec<u64>,
    active_candidate_mask: Vec<u64>,
    filters: Vec<RequiredCountFilter<'a>>,
    feature_filters_to_prime: Vec<QueryFeatureFilter>,
    repeated_feature_filters: Vec<QueryFeatureFilter>,
    cache_owner: Option<usize>,
    edge_mask_cache: BTreeMap<(EdgeFeature, u16), CachedFeatureMask>,
    edge_count_by_target: Vec<u16>,
    edge_touched_targets: Vec<usize>,
    edge_candidate_mask: Vec<u64>,
    edge_left_atoms: Vec<AtomFeature>,
    edge_right_atoms: Vec<AtomFeature>,
    edge_bonds: Vec<EdgeBondFeature>,
    edge_feature_candidate_mask: Vec<u64>,
    edge_feature_reverse_mask: Vec<u64>,
    edge_feature_component_mask: Vec<u64>,
    path3_mask_cache: BTreeMap<(Path3Feature, u16), CachedFeatureMask>,
    path3_count_by_target: Vec<u16>,
    path3_touched_targets: Vec<usize>,
    path3_candidate_mask: Vec<u64>,
    path3_left_atoms: Vec<AtomFeature>,
    path3_center_atoms: Vec<AtomFeature>,
    path3_right_atoms: Vec<AtomFeature>,
    path3_left_bonds: Vec<EdgeBondFeature>,
    path3_right_bonds: Vec<EdgeBondFeature>,
    path3_feature_candidate_mask: Vec<u64>,
    path3_feature_reverse_mask: Vec<u64>,
    path3_feature_component_mask: Vec<u64>,
    path4_mask_cache: BTreeMap<(Path4Feature, u16), CachedFeatureMask>,
    path4_count_by_target: Vec<u16>,
    path4_touched_targets: Vec<usize>,
    path4_candidate_mask: Vec<u64>,
    path4_left_atoms: Vec<AtomFeature>,
    path4_left_middle_atoms: Vec<AtomFeature>,
    path4_right_middle_atoms: Vec<AtomFeature>,
    path4_right_atoms: Vec<AtomFeature>,
    path4_left_bonds: Vec<EdgeBondFeature>,
    path4_center_bonds: Vec<EdgeBondFeature>,
    path4_right_bonds: Vec<EdgeBondFeature>,
    path4_feature_candidate_mask: Vec<u64>,
    path4_feature_reverse_mask: Vec<u64>,
    path4_feature_component_mask: Vec<u64>,
    star3_mask_cache: BTreeMap<(Star3Feature, u16), CachedFeatureMask>,
    star3_count_by_target: Vec<u16>,
    star3_touched_targets: Vec<usize>,
    star3_candidate_mask: Vec<u64>,
    star3_center_atoms: Vec<AtomFeature>,
    star3_first_atoms: Vec<AtomFeature>,
    star3_second_atoms: Vec<AtomFeature>,
    star3_third_atoms: Vec<AtomFeature>,
    star3_first_bonds: Vec<EdgeBondFeature>,
    star3_second_bonds: Vec<EdgeBondFeature>,
    star3_third_bonds: Vec<EdgeBondFeature>,
    star3_feature_candidate_mask: Vec<u64>,
    star3_feature_permutation_mask: Vec<u64>,
    star3_feature_component_mask: Vec<u64>,
}

impl TargetCorpusScratch<'_> {
    /// Builds empty reusable scratch buffers for one target corpus index.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            candidate_mask: Vec::new(),
            active_candidate_mask: Vec::new(),
            filters: Vec::new(),
            feature_filters_to_prime: Vec::new(),
            repeated_feature_filters: Vec::new(),
            cache_owner: None,
            edge_mask_cache: BTreeMap::new(),
            edge_count_by_target: Vec::new(),
            edge_touched_targets: Vec::new(),
            edge_candidate_mask: Vec::new(),
            edge_left_atoms: Vec::new(),
            edge_right_atoms: Vec::new(),
            edge_bonds: Vec::new(),
            edge_feature_candidate_mask: Vec::new(),
            edge_feature_reverse_mask: Vec::new(),
            edge_feature_component_mask: Vec::new(),
            path3_mask_cache: BTreeMap::new(),
            path3_count_by_target: Vec::new(),
            path3_touched_targets: Vec::new(),
            path3_candidate_mask: Vec::new(),
            path3_left_atoms: Vec::new(),
            path3_center_atoms: Vec::new(),
            path3_right_atoms: Vec::new(),
            path3_left_bonds: Vec::new(),
            path3_right_bonds: Vec::new(),
            path3_feature_candidate_mask: Vec::new(),
            path3_feature_reverse_mask: Vec::new(),
            path3_feature_component_mask: Vec::new(),
            path4_mask_cache: BTreeMap::new(),
            path4_count_by_target: Vec::new(),
            path4_touched_targets: Vec::new(),
            path4_candidate_mask: Vec::new(),
            path4_left_atoms: Vec::new(),
            path4_left_middle_atoms: Vec::new(),
            path4_right_middle_atoms: Vec::new(),
            path4_right_atoms: Vec::new(),
            path4_left_bonds: Vec::new(),
            path4_center_bonds: Vec::new(),
            path4_right_bonds: Vec::new(),
            path4_feature_candidate_mask: Vec::new(),
            path4_feature_reverse_mask: Vec::new(),
            path4_feature_component_mask: Vec::new(),
            star3_mask_cache: BTreeMap::new(),
            star3_count_by_target: Vec::new(),
            star3_touched_targets: Vec::new(),
            star3_candidate_mask: Vec::new(),
            star3_center_atoms: Vec::new(),
            star3_first_atoms: Vec::new(),
            star3_second_atoms: Vec::new(),
            star3_third_atoms: Vec::new(),
            star3_first_bonds: Vec::new(),
            star3_second_bonds: Vec::new(),
            star3_third_bonds: Vec::new(),
            star3_feature_candidate_mask: Vec::new(),
            star3_feature_permutation_mask: Vec::new(),
            star3_feature_component_mask: Vec::new(),
        }
    }

    fn ensure_word_count(&mut self, word_count: usize) {
        if self.candidate_mask.len() != word_count {
            self.candidate_mask.resize(word_count, 0);
        }
    }

    fn ensure_cache_owner(&mut self, owner: usize) {
        if self.cache_owner == Some(owner) {
            return;
        }
        self.cache_owner = Some(owner);
        self.edge_mask_cache.clear();
        self.path3_mask_cache.clear();
        self.path4_mask_cache.clear();
        self.star3_mask_cache.clear();
    }
}

impl QueryFeatureFilter {
    fn populate_candidate_mask(
        self,
        index: &TargetCorpusIndex,
        active_candidate_mask: Option<&[u64]>,
        scratch: &mut TargetCorpusScratch<'_>,
    ) -> usize {
        match self {
            Self::Edge {
                feature,
                required_count,
            } => index.populate_edge_candidate_mask(
                feature,
                required_count,
                active_candidate_mask,
                scratch,
            ),
            Self::Path3 {
                feature,
                required_count,
            } => index.populate_path3_candidate_mask(
                feature,
                required_count,
                active_candidate_mask,
                scratch,
            ),
            Self::Path4 {
                feature,
                required_count,
            } => index.populate_path4_candidate_mask(
                feature,
                required_count,
                active_candidate_mask,
                scratch,
            ),
            Self::Star3 {
                feature,
                required_count,
            } => index.populate_star3_candidate_mask(
                feature,
                required_count,
                active_candidate_mask,
                scratch,
            ),
        }
    }

    fn is_cached(self, scratch: &TargetCorpusScratch<'_>) -> bool {
        match self {
            Self::Edge {
                feature,
                required_count,
            } => scratch
                .edge_mask_cache
                .contains_key(&(feature, required_count)),
            Self::Path3 {
                feature,
                required_count,
            } => scratch
                .path3_mask_cache
                .contains_key(&(feature, required_count)),
            Self::Path4 {
                feature,
                required_count,
            } => scratch
                .path4_mask_cache
                .contains_key(&(feature, required_count)),
            Self::Star3 {
                feature,
                required_count,
            } => scratch
                .star3_mask_cache
                .contains_key(&(feature, required_count)),
        }
    }

    fn intersect_active_candidate_mask(self, scratch: &mut TargetCorpusScratch<'_>) -> usize {
        let source = match self {
            Self::Edge { .. } => &scratch.edge_candidate_mask,
            Self::Path3 { .. } => &scratch.path3_candidate_mask,
            Self::Path4 { .. } => &scratch.path4_candidate_mask,
            Self::Star3 { .. } => &scratch.star3_candidate_mask,
        };

        let mut population = 0usize;
        for (candidate_word, &source_word) in scratch.candidate_mask.iter_mut().zip(source) {
            *candidate_word &= source_word;
            population += candidate_word.count_ones() as usize;
        }
        population
    }

    const fn move_candidate_mask_to_active(self, scratch: &mut TargetCorpusScratch<'_>) {
        match self {
            Self::Edge { .. } => {
                core::mem::swap(
                    &mut scratch.candidate_mask,
                    &mut scratch.edge_candidate_mask,
                );
            }
            Self::Path3 { .. } => {
                core::mem::swap(
                    &mut scratch.candidate_mask,
                    &mut scratch.path3_candidate_mask,
                );
            }
            Self::Path4 { .. } => {
                core::mem::swap(
                    &mut scratch.candidate_mask,
                    &mut scratch.path4_candidate_mask,
                );
            }
            Self::Star3 { .. } => {
                core::mem::swap(
                    &mut scratch.candidate_mask,
                    &mut scratch.star3_candidate_mask,
                );
            }
        }
    }
}

/// Precomputed candidate target ids for one query against one target corpus index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetCandidateSet {
    target_ids: Box<[usize]>,
}

impl TargetCandidateSet {
    #[must_use]
    fn new(target_ids: Vec<usize>) -> Self {
        Self {
            target_ids: target_ids.into_boxed_slice(),
        }
    }

    /// Returns candidate target ids that still require full SMARTS matching.
    #[inline]
    #[must_use]
    pub fn target_ids(&self) -> &[usize] {
        &self.target_ids
    }

    /// Returns the number of candidate targets.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.target_ids.len()
    }

    /// Returns whether this query has no candidate targets.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.target_ids.is_empty()
    }
}

/// Summary statistics for a [`TargetCorpusIndex`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TargetCorpusIndexStats {
    /// Number of indexed targets.
    pub target_count: usize,
    /// Number of exact edge-feature posting lists.
    pub edge_feature_count: usize,
    /// Total number of target postings across exact edge features.
    pub edge_posting_count: usize,
    /// Number of target atom-feature values used by edge expansion.
    pub edge_atom_domain_count: usize,
    /// Number of target bond-feature values used by edge expansion.
    pub edge_bond_domain_count: usize,
    /// Number of exact path3-feature posting lists.
    pub path3_feature_count: usize,
    /// Total number of target postings across exact path3 features.
    pub path3_posting_count: usize,
    /// Number of target atom-feature values used by path3 expansion.
    pub path3_atom_domain_count: usize,
    /// Number of target bond-feature values used by path3 expansion.
    pub path3_bond_domain_count: usize,
    /// Number of exact path4-feature posting lists.
    pub path4_feature_count: usize,
    /// Total number of target postings across exact path4 features.
    pub path4_posting_count: usize,
    /// Number of target atom-feature values used by path4 expansion.
    pub path4_atom_domain_count: usize,
    /// Number of target bond-feature values used by path4 expansion.
    pub path4_bond_domain_count: usize,
    /// Number of exact star3-feature posting lists.
    pub star3_feature_count: usize,
    /// Total number of target postings across exact star3 features.
    pub star3_posting_count: usize,
    /// Number of target atom-feature values used by star3 expansion.
    pub star3_atom_domain_count: usize,
    /// Number of target bond-feature values used by star3 expansion.
    pub star3_bond_domain_count: usize,
}

/// Persistent target-side index for repeated many-query screening.
///
/// This is intended for workloads where the target corpus stays fixed and many
/// new SMARTS queries are evaluated against it, such as one GA generation over
/// a static SMILES support set.
///
/// `TargetCorpusIndex` is a conservative prefilter. A target id returned by
/// [`TargetCorpusIndex::candidate_set`] or
/// [`TargetCorpusIndex::candidate_set_with_scratch`] may still be a false
/// positive, so callers must run exact matching before accepting a hit.
///
/// ```
/// use core::str::FromStr;
///
/// use smarts_rs::{
///     CompiledQuery, MatchScratch, PreparedTarget, QueryMol, QueryScreen, TargetCorpusIndex,
///     TargetCorpusScratch,
/// };
/// use smiles_parser::Smiles;
///
/// let targets = ["CCO", "CCCC", "CC=O", "CCN"]
///     .into_iter()
///     .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
///     .collect::<Vec<_>>();
///
/// let index = TargetCorpusIndex::new(&targets);
/// let query = CompiledQuery::new(QueryMol::from_str("[#6]=[#8]").unwrap()).unwrap();
/// let screen = QueryScreen::new(query.query());
///
/// let mut corpus_scratch = TargetCorpusScratch::new();
/// let candidates = index.candidate_set_with_scratch(&screen, &mut corpus_scratch);
///
/// let mut match_scratch = MatchScratch::new();
/// let hits = candidates
///     .target_ids()
///     .iter()
///     .copied()
///     .filter(|&target_id| {
///         query.matches_with_scratch(&targets[target_id], &mut match_scratch)
///     })
///     .collect::<Vec<_>>();
///
/// assert_eq!(hits, vec![2]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetCorpusIndex {
    target_count: usize,
    retained_screens: Box<[TargetScreen]>,
    atom_count_index: CountBitsetIndex,
    component_count_index: CountBitsetIndex,
    aromatic_atom_count_index: CountBitsetIndex,
    ring_atom_count_index: CountBitsetIndex,
    indexed_element_count_index: Box<[(Element, CountBitsetIndex)]>,
    indexed_degree_count_index: IndexedFeatureCountIndex<u16>,
    indexed_total_hydrogen_count_index: IndexedFeatureCountIndex<u16>,
    single_bond_count_index: CountBitsetIndex,
    double_bond_count_index: CountBitsetIndex,
    triple_bond_count_index: CountBitsetIndex,
    aromatic_bond_count_index: CountBitsetIndex,
    ring_bond_count_index: CountBitsetIndex,
    indexed_edge_feature_count_index: IndexedSparseFeatureCountIndex<EdgeFeature>,
    edge_feature_mask_index: EdgeFeatureMaskIndex,
    edge_atom_feature_domain: Box<[AtomFeature]>,
    edge_bond_feature_domain: Box<[EdgeBondFeature]>,
    indexed_path3_feature_count_index: IndexedSparseFeatureCountIndex<Path3Feature>,
    path3_feature_mask_index: Path3FeatureMaskIndex,
    path3_atom_feature_domain: Box<[AtomFeature]>,
    path3_bond_feature_domain: Box<[EdgeBondFeature]>,
    indexed_path4_feature_count_index: IndexedSparseFeatureCountIndex<Path4Feature>,
    path4_feature_mask_index: Path4FeatureMaskIndex,
    path4_atom_feature_domain: Box<[AtomFeature]>,
    path4_bond_feature_domain: Box<[EdgeBondFeature]>,
    indexed_star3_feature_count_index: IndexedSparseFeatureCountIndex<Star3Feature>,
    star3_feature_mask_index: Star3FeatureMaskIndex,
    star3_atom_feature_domain: Box<[AtomFeature]>,
    star3_bond_feature_domain: Box<[EdgeBondFeature]>,
}

#[cfg(feature = "mem_dbg")]
impl mem_dbg::FlatType for TargetCorpusIndex {
    type Flat = mem_dbg::False;
}

#[cfg(feature = "mem_dbg")]
impl mem_dbg::MemSize for TargetCorpusIndex {
    fn mem_size_rec(
        &self,
        _flags: mem_dbg::SizeFlags,
        _refs: &mut mem_dbg::HashMap<usize, usize>,
    ) -> usize {
        size_of::<Self>()
            + retained_target_screens_heap_size(&self.retained_screens)
            + count_bitset_index_heap_size(&self.atom_count_index)
            + count_bitset_index_heap_size(&self.component_count_index)
            + count_bitset_index_heap_size(&self.aromatic_atom_count_index)
            + count_bitset_index_heap_size(&self.ring_atom_count_index)
            + count_feature_index_heap_size(&self.indexed_element_count_index)
            + count_feature_index_heap_size(&self.indexed_degree_count_index)
            + count_feature_index_heap_size(&self.indexed_total_hydrogen_count_index)
            + count_bitset_index_heap_size(&self.single_bond_count_index)
            + count_bitset_index_heap_size(&self.double_bond_count_index)
            + count_bitset_index_heap_size(&self.triple_bond_count_index)
            + count_bitset_index_heap_size(&self.aromatic_bond_count_index)
            + count_bitset_index_heap_size(&self.ring_bond_count_index)
            + sparse_feature_count_index_heap_size(&self.indexed_edge_feature_count_index)
            + edge_feature_mask_index_heap_size(&self.edge_feature_mask_index)
            + box_slice_heap_size(&self.edge_atom_feature_domain)
            + box_slice_heap_size(&self.edge_bond_feature_domain)
            + sparse_feature_count_index_heap_size(&self.indexed_path3_feature_count_index)
            + path3_feature_mask_index_heap_size(&self.path3_feature_mask_index)
            + box_slice_heap_size(&self.path3_atom_feature_domain)
            + box_slice_heap_size(&self.path3_bond_feature_domain)
            + sparse_feature_count_index_heap_size(&self.indexed_path4_feature_count_index)
            + path4_feature_mask_index_heap_size(&self.path4_feature_mask_index)
            + box_slice_heap_size(&self.path4_atom_feature_domain)
            + box_slice_heap_size(&self.path4_bond_feature_domain)
            + sparse_feature_count_index_heap_size(&self.indexed_star3_feature_count_index)
            + star3_feature_mask_index_heap_size(&self.star3_feature_mask_index)
            + box_slice_heap_size(&self.star3_atom_feature_domain)
            + box_slice_heap_size(&self.star3_bond_feature_domain)
    }
}

#[cfg(feature = "mem_dbg")]
fn count_bitset_index_heap_size(index: &CountBitsetIndex) -> usize {
    index.heap_size()
}

#[cfg(feature = "mem_dbg")]
fn count_feature_index_heap_size<T>(index: &[(T, CountBitsetIndex)]) -> usize {
    size_of_val(index)
        + index
            .iter()
            .map(|(_, counts)| count_bitset_index_heap_size(counts))
            .sum::<usize>()
}

#[cfg(feature = "mem_dbg")]
fn sparse_feature_count_index_heap_size<T>(index: &[(T, SparseFeatureCounts)]) -> usize {
    size_of_val(index)
        + index
            .iter()
            .map(|(_, counts)| box_slice_heap_size(counts))
            .sum::<usize>()
}

#[cfg(feature = "mem_dbg")]
fn edge_feature_mask_index_heap_size(index: &EdgeFeatureMaskIndex) -> usize {
    feature_id_mask_index_heap_size(&index.left_atoms)
        + feature_id_mask_index_heap_size(&index.bonds)
        + feature_id_mask_index_heap_size(&index.right_atoms)
}

#[cfg(feature = "mem_dbg")]
fn path3_feature_mask_index_heap_size(index: &Path3FeatureMaskIndex) -> usize {
    feature_id_mask_index_heap_size(&index.left_atoms)
        + feature_id_mask_index_heap_size(&index.left_bonds)
        + feature_id_mask_index_heap_size(&index.center_atoms)
        + feature_id_mask_index_heap_size(&index.right_bonds)
        + feature_id_mask_index_heap_size(&index.right_atoms)
}

#[cfg(feature = "mem_dbg")]
fn path4_feature_mask_index_heap_size(index: &Path4FeatureMaskIndex) -> usize {
    feature_id_mask_index_heap_size(&index.left_atoms)
        + feature_id_mask_index_heap_size(&index.left_bonds)
        + feature_id_mask_index_heap_size(&index.left_middle_atoms)
        + feature_id_mask_index_heap_size(&index.center_bonds)
        + feature_id_mask_index_heap_size(&index.right_middle_atoms)
        + feature_id_mask_index_heap_size(&index.right_bonds)
        + feature_id_mask_index_heap_size(&index.right_atoms)
}

#[cfg(feature = "mem_dbg")]
fn star3_feature_mask_index_heap_size(index: &Star3FeatureMaskIndex) -> usize {
    feature_id_mask_index_heap_size(&index.center_atoms)
        + feature_id_mask_index_heap_size(&index.first_bonds)
        + feature_id_mask_index_heap_size(&index.first_atoms)
        + feature_id_mask_index_heap_size(&index.second_bonds)
        + feature_id_mask_index_heap_size(&index.second_atoms)
        + feature_id_mask_index_heap_size(&index.third_bonds)
        + feature_id_mask_index_heap_size(&index.third_atoms)
}

#[cfg(feature = "mem_dbg")]
fn feature_id_mask_index_heap_size<T>(index: &[(T, FeatureIdMask)]) -> usize {
    size_of_val(index)
        + index
            .iter()
            .map(|(_, mask)| box_slice_heap_size(mask))
            .sum::<usize>()
}

#[cfg(feature = "mem_dbg")]
const fn box_slice_heap_size<T>(slice: &[T]) -> usize {
    size_of_val(slice)
}

#[cfg(feature = "mem_dbg")]
fn retained_target_screens_heap_size(screens: &[TargetScreen]) -> usize {
    size_of_val(screens)
        + screens
            .iter()
            .map(target_screen_map_heap_size)
            .sum::<usize>()
}

#[cfg(feature = "mem_dbg")]
fn target_screen_map_heap_size(screen: &TargetScreen) -> usize {
    btree_map_heap_estimate::<Element, usize>(screen.element_counts.len())
        + btree_map_heap_estimate::<u16, usize>(screen.degree_counts.len())
        + btree_map_heap_estimate::<u16, usize>(screen.total_hydrogen_counts.len())
}

#[cfg(feature = "mem_dbg")]
const fn btree_map_heap_estimate<K, V>(len: usize) -> usize {
    // Match mem_dbg's BTree node heuristic for retained TargetScreen maps whose
    // Element keys cannot implement external traits in this crate.
    const B: usize = 6;
    const CAPACITY: usize = 2 * B - 1;

    if len == 0 {
        return 0;
    }

    let pointer_size = size_of::<usize>();
    let header_size = 2 * size_of::<usize>();
    let mut leaf_size = header_size;
    leaf_size = align_up(leaf_size, align_of::<K>());
    leaf_size += size_of::<K>() * CAPACITY;
    leaf_size = align_up(leaf_size, align_of::<V>());
    leaf_size += size_of::<V>() * CAPACITY;

    let mut internal_size = leaf_size;
    internal_size = align_up(internal_size, align_of::<usize>());
    internal_size += pointer_size * (CAPACITY + 1);

    if len <= CAPACITY {
        leaf_size
    } else {
        let average_node_size = (leaf_size * B + internal_size) / (B + 1);
        (len / B) * average_node_size
    }
}

#[cfg(feature = "mem_dbg")]
const fn align_up(size: usize, align: usize) -> usize {
    (size + align - 1) & !(align - 1)
}

impl TargetCorpusIndex {
    /// Builds one persistent target-side index from prepared targets.
    #[must_use]
    pub fn new(targets: &[PreparedTarget]) -> Self {
        let screens = targets.iter().map(TargetScreen::new).collect::<Vec<_>>();
        let (
            edge_feature_count_index,
            path3_feature_count_index,
            path4_feature_count_index,
            star3_feature_count_index,
        ) = build_target_feature_indexes(targets);
        let mut index = Self::from_screens_inner(screens, false);
        let (edge_index, edge_feature_mask_index, edge_atom_domain, edge_bond_domain) =
            build_edge_sparse_index(edge_feature_count_index);
        index.indexed_edge_feature_count_index = edge_index;
        index.edge_feature_mask_index = edge_feature_mask_index;
        index.edge_atom_feature_domain = edge_atom_domain;
        index.edge_bond_feature_domain = edge_bond_domain;
        let (path3_index, path3_feature_mask_index, atom_domain, bond_domain) =
            build_path3_sparse_index(path3_feature_count_index);
        index.indexed_path3_feature_count_index = path3_index;
        index.path3_feature_mask_index = path3_feature_mask_index;
        index.path3_atom_feature_domain = atom_domain;
        index.path3_bond_feature_domain = bond_domain;
        let (path4_index, path4_feature_mask_index, path4_atom_domain, path4_bond_domain) =
            build_path4_sparse_index(path4_feature_count_index);
        index.indexed_path4_feature_count_index = path4_index;
        index.path4_feature_mask_index = path4_feature_mask_index;
        index.path4_atom_feature_domain = path4_atom_domain;
        index.path4_bond_feature_domain = path4_bond_domain;
        let (star3_index, star3_feature_mask_index, star3_atom_domain, star3_bond_domain) =
            build_star3_sparse_index(star3_feature_count_index);
        index.indexed_star3_feature_count_index = star3_index;
        index.star3_feature_mask_index = star3_feature_mask_index;
        index.star3_atom_feature_domain = star3_atom_domain;
        index.star3_bond_feature_domain = star3_bond_domain;
        index
    }

    /// Builds one persistent target-side index from already prepared screens.
    #[must_use]
    pub fn from_screens(screens: Vec<TargetScreen>) -> Self {
        Self::from_screens_inner(screens, true)
    }

    fn from_screens_inner(screens: Vec<TargetScreen>, retain_screens: bool) -> Self {
        let target_count = screens.len();
        let atom_count_index = CountBitsetIndex::from_counts(
            target_count,
            screens.iter().map(|screen| screen.atom_count),
        );
        let component_count_index = CountBitsetIndex::from_counts(
            target_count,
            screens
                .iter()
                .map(|screen| screen.connected_component_count),
        );
        let aromatic_atom_count_index = CountBitsetIndex::from_counts(
            target_count,
            screens.iter().map(|screen| screen.aromatic_atom_count),
        );
        let ring_atom_count_index = CountBitsetIndex::from_counts(
            target_count,
            screens.iter().map(|screen| screen.ring_atom_count),
        );
        let single_bond_count_index = CountBitsetIndex::from_counts(
            target_count,
            screens.iter().map(|screen| screen.bond_counts.single),
        );
        let double_bond_count_index = CountBitsetIndex::from_counts(
            target_count,
            screens.iter().map(|screen| screen.bond_counts.double),
        );
        let triple_bond_count_index = CountBitsetIndex::from_counts(
            target_count,
            screens.iter().map(|screen| screen.bond_counts.triple),
        );
        let aromatic_bond_count_index = CountBitsetIndex::from_counts(
            target_count,
            screens.iter().map(|screen| screen.bond_counts.aromatic),
        );
        let ring_bond_count_index = CountBitsetIndex::from_counts(
            target_count,
            screens.iter().map(|screen| screen.bond_counts.ring),
        );

        let indexed_element_count_index =
            build_screen_count_index(&screens, |screen| &screen.element_counts);
        let indexed_degree_count_index =
            build_screen_count_index(&screens, |screen| &screen.degree_counts);
        let indexed_total_hydrogen_count_index =
            build_screen_count_index(&screens, |screen| &screen.total_hydrogen_counts);

        Self {
            target_count,
            retained_screens: if retain_screens {
                screens.into_boxed_slice()
            } else {
                Box::new([])
            },
            atom_count_index,
            component_count_index,
            aromatic_atom_count_index,
            ring_atom_count_index,
            indexed_element_count_index,
            indexed_degree_count_index,
            indexed_total_hydrogen_count_index,
            single_bond_count_index,
            double_bond_count_index,
            triple_bond_count_index,
            aromatic_bond_count_index,
            ring_bond_count_index,
            indexed_edge_feature_count_index: Box::new([]),
            edge_feature_mask_index: EdgeFeatureMaskIndex::default(),
            edge_atom_feature_domain: Box::new([]),
            edge_bond_feature_domain: Box::new([]),
            indexed_path3_feature_count_index: Box::new([]),
            path3_feature_mask_index: Path3FeatureMaskIndex::default(),
            path3_atom_feature_domain: Box::new([]),
            path3_bond_feature_domain: Box::new([]),
            indexed_path4_feature_count_index: Box::new([]),
            path4_feature_mask_index: Path4FeatureMaskIndex::default(),
            path4_atom_feature_domain: Box::new([]),
            path4_bond_feature_domain: Box::new([]),
            indexed_star3_feature_count_index: Box::new([]),
            star3_feature_mask_index: Star3FeatureMaskIndex::default(),
            star3_atom_feature_domain: Box::new([]),
            star3_bond_feature_domain: Box::new([]),
        }
    }

    /// Returns the number of indexed targets.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.target_count
    }

    /// Returns whether the index contains no targets.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.target_count == 0
    }

    /// Returns the retained target screen for one indexed target.
    ///
    /// Indexes built with [`TargetCorpusIndex::from_screens`] retain screens.
    /// Indexes built with [`TargetCorpusIndex::new`] drop screens after
    /// construction to avoid storing per-target summary maps at large corpus
    /// scale.
    #[inline]
    #[must_use]
    pub fn screen(&self, target_id: usize) -> Option<&TargetScreen> {
        self.retained_screens.get(target_id)
    }

    /// Returns aggregate index sizes for diagnostics and benchmark reporting.
    #[must_use]
    pub fn stats(&self) -> TargetCorpusIndexStats {
        TargetCorpusIndexStats {
            target_count: self.target_count,
            edge_feature_count: self.indexed_edge_feature_count_index.len(),
            edge_posting_count: sparse_posting_count(&self.indexed_edge_feature_count_index),
            edge_atom_domain_count: self.edge_atom_feature_domain.len(),
            edge_bond_domain_count: self.edge_bond_feature_domain.len(),
            path3_feature_count: self.indexed_path3_feature_count_index.len(),
            path3_posting_count: sparse_posting_count(&self.indexed_path3_feature_count_index),
            path3_atom_domain_count: self.path3_atom_feature_domain.len(),
            path3_bond_domain_count: self.path3_bond_feature_domain.len(),
            path4_feature_count: self.indexed_path4_feature_count_index.len(),
            path4_posting_count: sparse_posting_count(&self.indexed_path4_feature_count_index),
            path4_atom_domain_count: self.path4_atom_feature_domain.len(),
            path4_bond_domain_count: self.path4_bond_feature_domain.len(),
            star3_feature_count: self.indexed_star3_feature_count_index.len(),
            star3_posting_count: sparse_posting_count(&self.indexed_star3_feature_count_index),
            star3_atom_domain_count: self.star3_atom_feature_domain.len(),
            star3_bond_domain_count: self.star3_bond_feature_domain.len(),
        }
    }

    /// Collects candidate target ids that pass the indexed screening stage.
    ///
    /// The returned ids are conservative candidates only. Full SMARTS matching
    /// still needs to run afterward.
    pub fn candidate_ids_into(&self, query: &QueryScreen, out: &mut Vec<usize>) {
        let mut scratch = TargetCorpusScratch::new();
        self.candidate_ids_with_scratch_into(query, &mut scratch, out);
    }

    /// Collects candidate target ids using reusable scratch buffers.
    pub fn candidate_ids_with_scratch_into<'idx>(
        &'idx self,
        query: &QueryScreen,
        scratch: &mut TargetCorpusScratch<'idx>,
        out: &mut Vec<usize>,
    ) {
        out.clear();
        let Some(state) = self.populate_candidate_mask_with_scratch(query, scratch) else {
            return;
        };
        out.reserve(state.population);
        if !state.has_active_source {
            out.extend(0..self.target_count);
            return;
        }
        for_each_set_bit(&scratch.candidate_mask, self.target_count, |target_id| {
            out.push(target_id);
        });
    }

    /// Counts candidate targets that pass the indexed screening stage.
    ///
    /// This avoids materializing a candidate id vector when callers only need
    /// the size of the indexed exact-match workload.
    #[must_use]
    pub fn candidate_count(&self, query: &QueryScreen) -> usize {
        let mut scratch = TargetCorpusScratch::new();
        self.candidate_count_with_scratch(query, &mut scratch)
    }

    /// Counts candidate targets using reusable scratch buffers.
    pub fn candidate_count_with_scratch<'idx>(
        &'idx self,
        query: &QueryScreen,
        scratch: &mut TargetCorpusScratch<'idx>,
    ) -> usize {
        self.populate_candidate_mask_with_scratch(query, scratch)
            .map_or(0, |state| state.population)
    }

    /// Returns candidate target ids that pass the indexed screening stage.
    #[must_use]
    pub fn candidate_ids(&self, query: &QueryScreen) -> Vec<usize> {
        let mut out = Vec::new();
        self.candidate_ids_into(query, &mut out);
        out
    }

    /// Builds reusable candidate sets for a batch of queries.
    ///
    /// This primes repeated local-signature masks across the batch before
    /// screening individual queries. It is intended for generation-style
    /// workloads where many SMARTS queries are evaluated against the same
    /// fixed target corpus.
    #[must_use]
    pub fn candidate_sets(&self, queries: &[QueryScreen]) -> Vec<TargetCandidateSet> {
        let mut scratch = TargetCorpusScratch::new();
        self.candidate_sets_with_scratch(queries, &mut scratch)
    }

    /// Builds reusable candidate sets for a batch of queries using reusable
    /// scratch buffers.
    pub fn candidate_sets_with_scratch<'idx>(
        &'idx self,
        queries: &[QueryScreen],
        scratch: &mut TargetCorpusScratch<'idx>,
    ) -> Vec<TargetCandidateSet> {
        self.prime_repeated_feature_filter_cache(queries, scratch);
        queries
            .iter()
            .map(|query| self.candidate_set_with_scratch(query, scratch))
            .collect()
    }

    /// Streams candidate target ids for a batch of queries using reusable
    /// scratch buffers.
    ///
    /// This primes repeated local-signature masks across the batch, like
    /// [`TargetCorpusIndex::candidate_sets_with_scratch`], but avoids
    /// materializing [`TargetCandidateSet`] values when callers can consume
    /// `(query_id, target_id)` pairs immediately.
    pub fn for_each_candidate_id_batch_with_scratch<'idx, F>(
        &'idx self,
        queries: &[QueryScreen],
        scratch: &mut TargetCorpusScratch<'idx>,
        mut f: F,
    ) where
        F: FnMut(usize, usize),
    {
        self.prime_repeated_feature_filter_cache(queries, scratch);
        for (query_id, query) in queries.iter().enumerate() {
            self.for_each_candidate_id_with_scratch(query, scratch, |target_id| {
                f(query_id, target_id);
            });
        }
    }

    /// Builds a reusable candidate set for one query.
    ///
    /// This is useful when the same query/target candidate list is consumed by
    /// more than one matching pass, or when screening should be measured
    /// separately from full substructure matching.
    #[must_use]
    pub fn candidate_set(&self, query: &QueryScreen) -> TargetCandidateSet {
        let mut scratch = TargetCorpusScratch::new();
        self.candidate_set_with_scratch(query, &mut scratch)
    }

    /// Builds a reusable candidate set for one query using reusable scratch buffers.
    pub fn candidate_set_with_scratch<'idx>(
        &'idx self,
        query: &QueryScreen,
        scratch: &mut TargetCorpusScratch<'idx>,
    ) -> TargetCandidateSet {
        let mut target_ids = Vec::new();
        self.candidate_ids_with_scratch_into(query, scratch, &mut target_ids);
        TargetCandidateSet::new(target_ids)
    }

    /// Streams candidate target ids using reusable scratch buffers.
    ///
    /// This avoids materializing an intermediate `Vec<usize>` on the hot path.
    pub fn for_each_candidate_id_with_scratch<'idx, F>(
        &'idx self,
        query: &QueryScreen,
        scratch: &mut TargetCorpusScratch<'idx>,
        f: F,
    ) where
        F: FnMut(usize),
    {
        let Some(state) = self.populate_candidate_mask_with_scratch(query, scratch) else {
            return;
        };
        if !state.has_active_source {
            (0..self.target_count).for_each(f);
            return;
        }
        for_each_set_bit(&scratch.candidate_mask, self.target_count, f);
    }

    fn populate_candidate_mask_with_scratch<'idx>(
        &'idx self,
        query: &QueryScreen,
        scratch: &mut TargetCorpusScratch<'idx>,
    ) -> Option<CandidateMaskState> {
        if self.target_count == 0 {
            return None;
        }

        scratch.ensure_cache_owner(core::ptr::from_ref(self) as usize);
        scratch.filters.clear();
        scratch.filters.reserve(
            12 + query.required_element_counts.len()
                + query.required_degree_counts.len()
                + query.required_total_hydrogen_counts.len()
                + query.required_edge_feature_counts.len()
                + query.required_path3_feature_counts.len()
                + query.required_path4_feature_counts.len()
                + query.required_star3_feature_counts.len(),
        );
        if !self.collect_required_count_filters(query, &mut scratch.filters) {
            return None;
        }

        scratch.ensure_word_count(bitset_word_count(self.target_count));
        let mut has_active_source = false;
        scratch
            .filters
            .sort_unstable_by_key(|filter| filter.population);
        let mut candidate_population = self.target_count;
        for &filter in &scratch.filters {
            let population = intersect_source_with_population(
                &mut scratch.candidate_mask,
                &mut has_active_source,
                filter.source,
                filter.population,
            )?;
            candidate_population = population;
        }
        if !self.apply_feature_count_filters(
            query,
            scratch,
            &mut has_active_source,
            &mut candidate_population,
        ) {
            return None;
        }

        Some(CandidateMaskState {
            has_active_source,
            population: candidate_population,
        })
    }

    fn prime_repeated_feature_filter_cache<'idx>(
        &'idx self,
        queries: &[QueryScreen],
        scratch: &mut TargetCorpusScratch<'idx>,
    ) {
        if self.target_count == 0 || queries.is_empty() {
            return;
        }

        scratch.ensure_cache_owner(core::ptr::from_ref(self) as usize);

        let mut filters = core::mem::take(&mut scratch.feature_filters_to_prime);
        filters.clear();
        filters.extend(
            queries
                .iter()
                .flat_map(|query| query.planned_feature_filters.iter().copied()),
        );
        filters.sort_unstable();

        let mut repeated = core::mem::take(&mut scratch.repeated_feature_filters);
        repeated.clear();
        let mut start = 0usize;
        while start < filters.len() {
            let filter = filters[start];
            let mut end = start + 1;
            while end < filters.len() && filters[end] == filter {
                end += 1;
            }
            if end - start > 1 && !filter.is_cached(scratch) {
                repeated.push(filter);
            }
            start = end;
        }

        scratch.feature_filters_to_prime = filters;
        for filter in repeated.iter().copied() {
            filter.populate_candidate_mask(self, None, scratch);
        }
        scratch.repeated_feature_filters = repeated;
    }

    fn collect_required_count_filters<'idx>(
        &'idx self,
        query: &QueryScreen,
        filters: &mut Vec<RequiredCountFilter<'idx>>,
    ) -> bool {
        self.collect_basic_count_filters(query, filters)
            && self.collect_element_count_filters(query, filters)
            && self.collect_atom_property_count_filters(query, filters)
    }

    fn collect_basic_count_filters<'idx>(
        &'idx self,
        query: &QueryScreen,
        filters: &mut Vec<RequiredCountFilter<'idx>>,
    ) -> bool {
        let required_counts = [
            (&self.atom_count_index, query.min_atom_count),
            (
                &self.component_count_index,
                query.min_target_component_count,
            ),
            (
                &self.aromatic_atom_count_index,
                query.min_aromatic_atom_count,
            ),
            (&self.ring_atom_count_index, query.min_ring_atom_count),
            (
                &self.single_bond_count_index,
                query.required_bond_counts.single,
            ),
            (
                &self.double_bond_count_index,
                query.required_bond_counts.double,
            ),
            (
                &self.triple_bond_count_index,
                query.required_bond_counts.triple,
            ),
            (
                &self.aromatic_bond_count_index,
                query.required_bond_counts.aromatic,
            ),
            (&self.ring_bond_count_index, query.required_bond_counts.ring),
        ];

        required_counts
            .into_iter()
            .all(|(index, required)| push_required_filter(filters, index, required))
    }

    fn collect_element_count_filters<'idx>(
        &'idx self,
        query: &QueryScreen,
        filters: &mut Vec<RequiredCountFilter<'idx>>,
    ) -> bool {
        for (element, required) in &query.required_element_counts {
            let Some((_, index)) = find_count_index(&self.indexed_element_count_index, element)
            else {
                return false;
            };
            if !push_required_filter(filters, index, *required) {
                return false;
            }
        }

        true
    }

    fn collect_atom_property_count_filters<'idx>(
        &'idx self,
        query: &QueryScreen,
        filters: &mut Vec<RequiredCountFilter<'idx>>,
    ) -> bool {
        collect_indexed_required_count_filters(
            filters,
            &self.indexed_degree_count_index,
            &query.required_degree_counts,
        ) && collect_indexed_required_count_filters(
            filters,
            &self.indexed_total_hydrogen_count_index,
            &query.required_total_hydrogen_counts,
        )
    }

    fn apply_feature_count_filters<'idx>(
        &'idx self,
        query: &QueryScreen,
        scratch: &mut TargetCorpusScratch<'idx>,
        has_active_source: &mut bool,
        candidate_population: &mut usize,
    ) -> bool {
        for filter in query.planned_feature_filters.iter().copied() {
            let mut active_candidate_mask = self.copy_active_candidate_mask_if_sparse(
                scratch,
                *has_active_source,
                *candidate_population,
            );
            let used_active_candidate_mask = active_candidate_mask.is_some();
            let population =
                filter.populate_candidate_mask(self, active_candidate_mask.as_deref(), scratch);
            if let Some(mask) = active_candidate_mask.take() {
                scratch.active_candidate_mask = mask;
            }
            if population == 0 {
                return false;
            }

            if used_active_candidate_mask || !*has_active_source {
                filter.move_candidate_mask_to_active(scratch);
                *has_active_source = true;
                *candidate_population = population;
            } else {
                let population = filter.intersect_active_candidate_mask(scratch);
                if population == 0 {
                    return false;
                }
                *candidate_population = population;
            }
        }

        true
    }

    fn copy_active_candidate_mask_if_sparse(
        &self,
        scratch: &mut TargetCorpusScratch<'_>,
        has_active_source: bool,
        candidate_population: usize,
    ) -> Option<Vec<u64>> {
        if !should_filter_sparse_counts(has_active_source, candidate_population, self.target_count)
        {
            return None;
        }

        let mut mask = core::mem::take(&mut scratch.active_candidate_mask);
        mask.clear();
        mask.extend_from_slice(&scratch.candidate_mask);
        Some(mask)
    }

    fn populate_edge_candidate_mask(
        &self,
        query_feature: EdgeFeature,
        required_count: u16,
        active_candidate_mask: Option<&[u64]>,
        scratch: &mut TargetCorpusScratch<'_>,
    ) -> usize {
        if required_count == 0 {
            return self.target_count;
        }
        if let Some(cached) = scratch
            .edge_mask_cache
            .get(&(query_feature, required_count))
        {
            return load_cached_candidate_mask(
                &mut scratch.edge_candidate_mask,
                cached,
                active_candidate_mask,
            );
        }

        prepare_sparse_candidate_counts(
            self.target_count,
            &mut scratch.edge_count_by_target,
            &mut scratch.edge_touched_targets,
        );

        self.collect_edge_domain_matches(query_feature, scratch);
        self.accumulate_indexed_edge_matches(query_feature, scratch, active_candidate_mask);

        finalize_cached_sparse_candidate_mask(
            self.target_count,
            query_feature,
            required_count,
            active_candidate_mask,
            SparseCandidateMaskBuffers {
                cache: &mut scratch.edge_mask_cache,
                count_by_target: scratch.edge_count_by_target.as_mut_slice(),
                touched_targets: &scratch.edge_touched_targets,
                candidate_mask: &mut scratch.edge_candidate_mask,
            },
        )
    }

    fn collect_edge_domain_matches(
        &self,
        query_feature: EdgeFeature,
        scratch: &mut TargetCorpusScratch<'_>,
    ) {
        collect_matching_atom_features(
            &self.edge_atom_feature_domain,
            query_feature.left,
            &mut scratch.edge_left_atoms,
        );
        collect_matching_atom_features(
            &self.edge_atom_feature_domain,
            query_feature.right,
            &mut scratch.edge_right_atoms,
        );
        collect_matching_bond_features(
            &self.edge_bond_feature_domain,
            query_feature.bond,
            &mut scratch.edge_bonds,
        );
    }

    fn accumulate_indexed_edge_matches(
        &self,
        query_feature: EdgeFeature,
        scratch: &mut TargetCorpusScratch<'_>,
        active_candidate_mask: Option<&[u64]>,
    ) {
        let feature_count = self.indexed_edge_feature_count_index.len();
        let forward_has_candidates = build_edge_orientation_feature_mask(
            &self.edge_feature_mask_index,
            &EdgeOrientationFeatureMask {
                query_left: query_feature.left,
                left_atoms: &scratch.edge_left_atoms,
                query_bond: query_feature.bond,
                bonds: &scratch.edge_bonds,
                query_right: query_feature.right,
                right_atoms: &scratch.edge_right_atoms,
            },
            feature_count,
            &mut scratch.edge_feature_candidate_mask,
            &mut scratch.edge_feature_component_mask,
        );
        let reverse_has_candidates = build_edge_orientation_feature_mask(
            &self.edge_feature_mask_index,
            &EdgeOrientationFeatureMask {
                query_left: query_feature.right,
                left_atoms: &scratch.edge_right_atoms,
                query_bond: query_feature.bond,
                bonds: &scratch.edge_bonds,
                query_right: query_feature.left,
                right_atoms: &scratch.edge_left_atoms,
            },
            feature_count,
            &mut scratch.edge_feature_reverse_mask,
            &mut scratch.edge_feature_component_mask,
        );

        merge_reversible_feature_mask(
            forward_has_candidates,
            reverse_has_candidates,
            &mut scratch.edge_feature_candidate_mask,
            &scratch.edge_feature_reverse_mask,
        );

        accumulate_feature_id_mask_counts(
            &self.indexed_edge_feature_count_index,
            &scratch.edge_feature_candidate_mask,
            &mut scratch.edge_count_by_target,
            &mut scratch.edge_touched_targets,
            active_candidate_mask,
        );
    }

    fn populate_path3_candidate_mask(
        &self,
        query_feature: Path3Feature,
        required_count: u16,
        active_candidate_mask: Option<&[u64]>,
        scratch: &mut TargetCorpusScratch<'_>,
    ) -> usize {
        if required_count == 0 {
            return self.target_count;
        }
        if let Some(cached) = scratch
            .path3_mask_cache
            .get(&(query_feature, required_count))
        {
            return load_cached_candidate_mask(
                &mut scratch.path3_candidate_mask,
                cached,
                active_candidate_mask,
            );
        }

        prepare_sparse_candidate_counts(
            self.target_count,
            &mut scratch.path3_count_by_target,
            &mut scratch.path3_touched_targets,
        );

        self.collect_path3_domain_matches(query_feature, scratch);
        self.accumulate_indexed_path3_matches(query_feature, scratch, active_candidate_mask);

        finalize_cached_sparse_candidate_mask(
            self.target_count,
            query_feature,
            required_count,
            active_candidate_mask,
            SparseCandidateMaskBuffers {
                cache: &mut scratch.path3_mask_cache,
                count_by_target: scratch.path3_count_by_target.as_mut_slice(),
                touched_targets: &scratch.path3_touched_targets,
                candidate_mask: &mut scratch.path3_candidate_mask,
            },
        )
    }

    fn collect_path3_domain_matches(
        &self,
        query_feature: Path3Feature,
        scratch: &mut TargetCorpusScratch<'_>,
    ) {
        collect_matching_atom_features(
            &self.path3_atom_feature_domain,
            query_feature.left,
            &mut scratch.path3_left_atoms,
        );
        collect_matching_atom_features(
            &self.path3_atom_feature_domain,
            query_feature.center,
            &mut scratch.path3_center_atoms,
        );
        collect_matching_atom_features(
            &self.path3_atom_feature_domain,
            query_feature.right,
            &mut scratch.path3_right_atoms,
        );
        collect_matching_bond_features(
            &self.path3_bond_feature_domain,
            query_feature.left_bond,
            &mut scratch.path3_left_bonds,
        );
        collect_matching_bond_features(
            &self.path3_bond_feature_domain,
            query_feature.right_bond,
            &mut scratch.path3_right_bonds,
        );
    }

    fn accumulate_indexed_path3_matches(
        &self,
        query_feature: Path3Feature,
        scratch: &mut TargetCorpusScratch<'_>,
        active_candidate_mask: Option<&[u64]>,
    ) {
        let feature_count = self.indexed_path3_feature_count_index.len();
        let forward_has_candidates = build_path3_orientation_feature_mask(
            &self.path3_feature_mask_index,
            &Path3OrientationFeatureMask {
                query_left: query_feature.left,
                left_atoms: &scratch.path3_left_atoms,
                query_left_bond: query_feature.left_bond,
                left_bonds: &scratch.path3_left_bonds,
                query_center: query_feature.center,
                center_atoms: &scratch.path3_center_atoms,
                query_right_bond: query_feature.right_bond,
                right_bonds: &scratch.path3_right_bonds,
                query_right: query_feature.right,
                right_atoms: &scratch.path3_right_atoms,
            },
            feature_count,
            &mut scratch.path3_feature_candidate_mask,
            &mut scratch.path3_feature_component_mask,
        );
        let reverse_has_candidates = build_path3_orientation_feature_mask(
            &self.path3_feature_mask_index,
            &Path3OrientationFeatureMask {
                query_left: query_feature.right,
                left_atoms: &scratch.path3_right_atoms,
                query_left_bond: query_feature.right_bond,
                left_bonds: &scratch.path3_right_bonds,
                query_center: query_feature.center,
                center_atoms: &scratch.path3_center_atoms,
                query_right_bond: query_feature.left_bond,
                right_bonds: &scratch.path3_left_bonds,
                query_right: query_feature.left,
                right_atoms: &scratch.path3_left_atoms,
            },
            feature_count,
            &mut scratch.path3_feature_reverse_mask,
            &mut scratch.path3_feature_component_mask,
        );

        merge_reversible_feature_mask(
            forward_has_candidates,
            reverse_has_candidates,
            &mut scratch.path3_feature_candidate_mask,
            &scratch.path3_feature_reverse_mask,
        );

        accumulate_feature_id_mask_counts(
            &self.indexed_path3_feature_count_index,
            &scratch.path3_feature_candidate_mask,
            &mut scratch.path3_count_by_target,
            &mut scratch.path3_touched_targets,
            active_candidate_mask,
        );
    }

    fn populate_path4_candidate_mask(
        &self,
        query_feature: Path4Feature,
        required_count: u16,
        active_candidate_mask: Option<&[u64]>,
        scratch: &mut TargetCorpusScratch<'_>,
    ) -> usize {
        if required_count == 0 {
            return self.target_count;
        }
        if let Some(cached) = scratch
            .path4_mask_cache
            .get(&(query_feature, required_count))
        {
            return load_cached_candidate_mask(
                &mut scratch.path4_candidate_mask,
                cached,
                active_candidate_mask,
            );
        }

        prepare_sparse_candidate_counts(
            self.target_count,
            &mut scratch.path4_count_by_target,
            &mut scratch.path4_touched_targets,
        );

        self.collect_path4_domain_matches(query_feature, scratch);
        self.accumulate_indexed_path4_matches(query_feature, scratch, active_candidate_mask);

        finalize_cached_sparse_candidate_mask(
            self.target_count,
            query_feature,
            required_count,
            active_candidate_mask,
            SparseCandidateMaskBuffers {
                cache: &mut scratch.path4_mask_cache,
                count_by_target: scratch.path4_count_by_target.as_mut_slice(),
                touched_targets: &scratch.path4_touched_targets,
                candidate_mask: &mut scratch.path4_candidate_mask,
            },
        )
    }

    fn collect_path4_domain_matches(
        &self,
        query_feature: Path4Feature,
        scratch: &mut TargetCorpusScratch<'_>,
    ) {
        collect_matching_atom_features(
            &self.path4_atom_feature_domain,
            query_feature.left,
            &mut scratch.path4_left_atoms,
        );
        collect_matching_atom_features(
            &self.path4_atom_feature_domain,
            query_feature.left_middle,
            &mut scratch.path4_left_middle_atoms,
        );
        collect_matching_atom_features(
            &self.path4_atom_feature_domain,
            query_feature.right_middle,
            &mut scratch.path4_right_middle_atoms,
        );
        collect_matching_atom_features(
            &self.path4_atom_feature_domain,
            query_feature.right,
            &mut scratch.path4_right_atoms,
        );
        collect_matching_bond_features(
            &self.path4_bond_feature_domain,
            query_feature.left_bond,
            &mut scratch.path4_left_bonds,
        );
        collect_matching_bond_features(
            &self.path4_bond_feature_domain,
            query_feature.center_bond,
            &mut scratch.path4_center_bonds,
        );
        collect_matching_bond_features(
            &self.path4_bond_feature_domain,
            query_feature.right_bond,
            &mut scratch.path4_right_bonds,
        );
    }

    fn accumulate_indexed_path4_matches(
        &self,
        query_feature: Path4Feature,
        scratch: &mut TargetCorpusScratch<'_>,
        active_candidate_mask: Option<&[u64]>,
    ) {
        let feature_count = self.indexed_path4_feature_count_index.len();
        let forward_has_candidates = build_path4_orientation_feature_mask(
            &self.path4_feature_mask_index,
            &Path4OrientationFeatureMask {
                query_left: query_feature.left,
                left_atoms: &scratch.path4_left_atoms,
                query_left_bond: query_feature.left_bond,
                left_bonds: &scratch.path4_left_bonds,
                query_left_middle: query_feature.left_middle,
                left_middle_atoms: &scratch.path4_left_middle_atoms,
                query_center_bond: query_feature.center_bond,
                center_bonds: &scratch.path4_center_bonds,
                query_right_middle: query_feature.right_middle,
                right_middle_atoms: &scratch.path4_right_middle_atoms,
                query_right_bond: query_feature.right_bond,
                right_bonds: &scratch.path4_right_bonds,
                query_right: query_feature.right,
                right_atoms: &scratch.path4_right_atoms,
            },
            feature_count,
            &mut scratch.path4_feature_candidate_mask,
            &mut scratch.path4_feature_component_mask,
        );
        let reverse_has_candidates = build_path4_orientation_feature_mask(
            &self.path4_feature_mask_index,
            &Path4OrientationFeatureMask {
                query_left: query_feature.right,
                left_atoms: &scratch.path4_right_atoms,
                query_left_bond: query_feature.right_bond,
                left_bonds: &scratch.path4_right_bonds,
                query_left_middle: query_feature.right_middle,
                left_middle_atoms: &scratch.path4_right_middle_atoms,
                query_center_bond: query_feature.center_bond,
                center_bonds: &scratch.path4_center_bonds,
                query_right_middle: query_feature.left_middle,
                right_middle_atoms: &scratch.path4_left_middle_atoms,
                query_right_bond: query_feature.left_bond,
                right_bonds: &scratch.path4_left_bonds,
                query_right: query_feature.left,
                right_atoms: &scratch.path4_left_atoms,
            },
            feature_count,
            &mut scratch.path4_feature_reverse_mask,
            &mut scratch.path4_feature_component_mask,
        );

        merge_reversible_feature_mask(
            forward_has_candidates,
            reverse_has_candidates,
            &mut scratch.path4_feature_candidate_mask,
            &scratch.path4_feature_reverse_mask,
        );

        accumulate_feature_id_mask_counts(
            &self.indexed_path4_feature_count_index,
            &scratch.path4_feature_candidate_mask,
            &mut scratch.path4_count_by_target,
            &mut scratch.path4_touched_targets,
            active_candidate_mask,
        );
    }

    fn populate_star3_candidate_mask(
        &self,
        query_feature: Star3Feature,
        required_count: u16,
        active_candidate_mask: Option<&[u64]>,
        scratch: &mut TargetCorpusScratch<'_>,
    ) -> usize {
        if required_count == 0 {
            return self.target_count;
        }
        if let Some(cached) = scratch
            .star3_mask_cache
            .get(&(query_feature, required_count))
        {
            return load_cached_candidate_mask(
                &mut scratch.star3_candidate_mask,
                cached,
                active_candidate_mask,
            );
        }

        prepare_sparse_candidate_counts(
            self.target_count,
            &mut scratch.star3_count_by_target,
            &mut scratch.star3_touched_targets,
        );

        self.collect_star3_domain_matches(query_feature, scratch);
        self.accumulate_indexed_star3_matches(query_feature, scratch, active_candidate_mask);

        finalize_cached_sparse_candidate_mask(
            self.target_count,
            query_feature,
            required_count,
            active_candidate_mask,
            SparseCandidateMaskBuffers {
                cache: &mut scratch.star3_mask_cache,
                count_by_target: scratch.star3_count_by_target.as_mut_slice(),
                touched_targets: &scratch.star3_touched_targets,
                candidate_mask: &mut scratch.star3_candidate_mask,
            },
        )
    }

    fn collect_star3_domain_matches(
        &self,
        query_feature: Star3Feature,
        scratch: &mut TargetCorpusScratch<'_>,
    ) {
        collect_matching_atom_features(
            &self.star3_atom_feature_domain,
            query_feature.center,
            &mut scratch.star3_center_atoms,
        );
        collect_matching_atom_features(
            &self.star3_atom_feature_domain,
            query_feature.arms[0].atom,
            &mut scratch.star3_first_atoms,
        );
        collect_matching_atom_features(
            &self.star3_atom_feature_domain,
            query_feature.arms[1].atom,
            &mut scratch.star3_second_atoms,
        );
        collect_matching_atom_features(
            &self.star3_atom_feature_domain,
            query_feature.arms[2].atom,
            &mut scratch.star3_third_atoms,
        );
        collect_matching_bond_features(
            &self.star3_bond_feature_domain,
            query_feature.arms[0].bond,
            &mut scratch.star3_first_bonds,
        );
        collect_matching_bond_features(
            &self.star3_bond_feature_domain,
            query_feature.arms[1].bond,
            &mut scratch.star3_second_bonds,
        );
        collect_matching_bond_features(
            &self.star3_bond_feature_domain,
            query_feature.arms[2].bond,
            &mut scratch.star3_third_bonds,
        );
    }

    fn accumulate_indexed_star3_matches(
        &self,
        query_feature: Star3Feature,
        scratch: &mut TargetCorpusScratch<'_>,
        active_candidate_mask: Option<&[u64]>,
    ) {
        const ARM_ORDERS: [[usize; 3]; 6] = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];

        let feature_count = self.indexed_star3_feature_count_index.len();
        ensure_zeroed_words(
            &mut scratch.star3_feature_candidate_mask,
            bitset_word_count(feature_count),
        );

        let target_atom_matches = [
            scratch.star3_first_atoms.as_slice(),
            scratch.star3_second_atoms.as_slice(),
            scratch.star3_third_atoms.as_slice(),
        ];
        let target_bond_matches = [
            scratch.star3_first_bonds.as_slice(),
            scratch.star3_second_bonds.as_slice(),
            scratch.star3_third_bonds.as_slice(),
        ];
        let target_atom_indexes = [
            self.star3_feature_mask_index.first_atoms.as_ref(),
            self.star3_feature_mask_index.second_atoms.as_ref(),
            self.star3_feature_mask_index.third_atoms.as_ref(),
        ];
        let target_bond_indexes = [
            self.star3_feature_mask_index.first_bonds.as_ref(),
            self.star3_feature_mask_index.second_bonds.as_ref(),
            self.star3_feature_mask_index.third_bonds.as_ref(),
        ];

        for order in ARM_ORDERS {
            if build_star3_orientation_feature_mask(
                &Star3OrientationFeatureMask {
                    query_center: query_feature.center,
                    center_atoms: &scratch.star3_center_atoms,
                    center_atom_index: &self.star3_feature_mask_index.center_atoms,
                    arms: [
                        Star3OrientationArmFeatureMask {
                            query: query_feature.arms[0],
                            atoms: target_atom_matches[0],
                            atom_index: target_atom_indexes[order[0]],
                            bonds: target_bond_matches[0],
                            bond_index: target_bond_indexes[order[0]],
                        },
                        Star3OrientationArmFeatureMask {
                            query: query_feature.arms[1],
                            atoms: target_atom_matches[1],
                            atom_index: target_atom_indexes[order[1]],
                            bonds: target_bond_matches[1],
                            bond_index: target_bond_indexes[order[1]],
                        },
                        Star3OrientationArmFeatureMask {
                            query: query_feature.arms[2],
                            atoms: target_atom_matches[2],
                            atom_index: target_atom_indexes[order[2]],
                            bonds: target_bond_matches[2],
                            bond_index: target_bond_indexes[order[2]],
                        },
                    ],
                },
                feature_count,
                &mut scratch.star3_feature_permutation_mask,
                &mut scratch.star3_feature_component_mask,
            ) {
                for (dst, &src) in scratch
                    .star3_feature_candidate_mask
                    .iter_mut()
                    .zip(&scratch.star3_feature_permutation_mask)
                {
                    *dst |= src;
                }
            }
        }

        accumulate_feature_id_mask_counts(
            &self.indexed_star3_feature_count_index,
            &scratch.star3_feature_candidate_mask,
            &mut scratch.star3_count_by_target,
            &mut scratch.star3_touched_targets,
            active_candidate_mask,
        );
    }
}

fn required_counts_may_match<T: Ord>(
    required_counts: &BTreeMap<T, usize>,
    target_counts: &BTreeMap<T, usize>,
) -> bool {
    required_counts.iter().all(|(feature, required)| {
        target_counts.get(feature).copied().unwrap_or_default() >= *required
    })
}

fn build_screen_count_index<T: Ord + Copy, F>(
    screens: &[TargetScreen],
    count_map: F,
) -> IndexedFeatureCountIndex<T>
where
    F: Fn(&TargetScreen) -> &BTreeMap<T, usize>,
{
    let mut features = Vec::new();
    for screen in screens {
        for feature in count_map(screen).keys().copied() {
            if !features.contains(&feature) {
                features.push(feature);
            }
        }
    }
    features.sort_unstable();
    features
        .into_iter()
        .map(|feature| {
            let counts = screens
                .iter()
                .map(|screen| count_map(screen).get(&feature).copied().unwrap_or_default());
            (
                feature,
                CountBitsetIndex::from_counts(screens.len(), counts),
            )
        })
        .collect()
}

fn push_required_filter<'idx>(
    filters: &mut Vec<RequiredCountFilter<'idx>>,
    index: &'idx CountBitsetIndex,
    required: usize,
) -> bool {
    if required == 0 {
        return true;
    }
    let Some(filter) = index.filter_for_at_least(required) else {
        return false;
    };
    filters.push(filter);
    true
}

fn collect_indexed_required_count_filters<'idx, T: Ord>(
    filters: &mut Vec<RequiredCountFilter<'idx>>,
    entries: &'idx [(T, CountBitsetIndex)],
    required_counts: &BTreeMap<T, usize>,
) -> bool {
    for (feature, required_count) in required_counts {
        let Some((_, index)) = find_count_index(entries, feature) else {
            return false;
        };
        if !push_required_filter(filters, index, *required_count) {
            return false;
        }
    }
    true
}

fn compact_target_id(target_id: usize) -> TargetId {
    TargetId::try_from(target_id).expect("target index exceeds compact posting id capacity")
}

fn accumulate_sparse_counts(
    count_by_target: &mut [u16],
    touched_targets: &mut Vec<usize>,
    sparse_counts: &[(TargetId, u16)],
    active_candidate_mask: Option<&[u64]>,
) {
    match active_candidate_mask {
        Some(mask) => {
            for &(target_id, count) in sparse_counts {
                let target_id = target_id as usize;
                if !bitset_contains(mask, target_id) {
                    continue;
                }
                if count_by_target[target_id] == 0 {
                    touched_targets.push(target_id);
                }
                count_by_target[target_id] = count_by_target[target_id].saturating_add(count);
            }
        }
        None => {
            for &(target_id, count) in sparse_counts {
                let target_id = target_id as usize;
                if count_by_target[target_id] == 0 {
                    touched_targets.push(target_id);
                }
                count_by_target[target_id] = count_by_target[target_id].saturating_add(count);
            }
        }
    }
}

fn prepare_sparse_candidate_counts(
    target_count: usize,
    count_by_target: &mut Vec<u16>,
    touched_targets: &mut Vec<usize>,
) {
    touched_targets.clear();
    if count_by_target.len() != target_count {
        count_by_target.clear();
        count_by_target.resize(target_count, 0);
    }
}

fn finalize_sparse_candidate_mask(
    target_count: usize,
    required_count: u16,
    count_by_target: &mut [u16],
    touched_targets: &[usize],
    candidate_mask: &mut Vec<u64>,
) -> usize {
    let word_count = bitset_word_count(target_count);
    if candidate_mask.len() == word_count {
        candidate_mask.fill(0);
    } else {
        candidate_mask.resize(word_count, 0);
    }

    let mut population = 0usize;
    for &target_id in touched_targets {
        if count_by_target[target_id] >= required_count {
            set_bit(candidate_mask, target_id);
            population += 1;
        }
        count_by_target[target_id] = 0;
    }

    population
}

struct SparseCandidateMaskBuffers<'a, T> {
    cache: &'a mut BTreeMap<(T, u16), CachedFeatureMask>,
    count_by_target: &'a mut [u16],
    touched_targets: &'a [usize],
    candidate_mask: &'a mut Vec<u64>,
}

fn finalize_cached_sparse_candidate_mask<T: Copy + Ord>(
    target_count: usize,
    query_feature: T,
    required_count: u16,
    active_candidate_mask: Option<&[u64]>,
    buffers: SparseCandidateMaskBuffers<'_, T>,
) -> usize {
    let SparseCandidateMaskBuffers {
        cache,
        count_by_target,
        touched_targets,
        candidate_mask,
    } = buffers;
    let population = finalize_sparse_candidate_mask(
        target_count,
        required_count,
        count_by_target,
        touched_targets,
        candidate_mask,
    );
    if active_candidate_mask.is_none() {
        cache.insert(
            (query_feature, required_count),
            CachedFeatureMask {
                population,
                words: candidate_mask.clone().into_boxed_slice(),
            },
        );
    }

    population
}

fn load_cached_candidate_mask(
    out: &mut Vec<u64>,
    cached: &CachedFeatureMask,
    active_candidate_mask: Option<&[u64]>,
) -> usize {
    let Some(active_candidate_mask) = active_candidate_mask else {
        out.clear();
        out.extend_from_slice(&cached.words);
        return cached.population;
    };

    if out.len() == cached.words.len() {
        out.fill(0);
    } else {
        out.clear();
        out.resize(cached.words.len(), 0);
    }

    let mut population = 0usize;
    for ((dst, &cached_word), &active_word) in out
        .iter_mut()
        .zip(cached.words.iter())
        .zip(active_candidate_mask)
    {
        let word = cached_word & active_word;
        *dst = word;
        population += word.count_ones() as usize;
    }
    population
}

fn merge_reversible_feature_mask(
    forward_has_candidates: bool,
    reverse_has_candidates: bool,
    candidate_mask: &mut Vec<u64>,
    reverse_mask: &[u64],
) {
    if reverse_has_candidates {
        if forward_has_candidates {
            for (dst, src) in candidate_mask.iter_mut().zip(reverse_mask) {
                *dst |= *src;
            }
        } else {
            candidate_mask.clear();
            candidate_mask.extend_from_slice(reverse_mask);
        }
    } else if !forward_has_candidates {
        candidate_mask.clear();
    }
}

fn find_count_index<'a, T: Ord>(
    entries: &'a [(T, CountBitsetIndex)],
    feature: &T,
) -> Option<(usize, &'a CountBitsetIndex)> {
    let index = entries
        .binary_search_by(|(entry_feature, _)| entry_feature.cmp(feature))
        .ok()?;
    Some((index, &entries[index].1))
}

fn find_feature_id_mask<'a, T: Ord>(
    entries: &'a [(T, FeatureIdMask)],
    feature: &T,
) -> Option<&'a [u64]> {
    let index = entries
        .binary_search_by(|(entry_feature, _)| entry_feature.cmp(feature))
        .ok()?;
    Some(entries[index].1.as_ref())
}

fn sparse_posting_count<T>(entries: &[(T, SparseFeatureCounts)]) -> usize {
    entries
        .iter()
        .map(|(_, sparse_counts)| sparse_counts.len())
        .sum()
}

fn union_feature_id_masks<T: Ord>(
    entries: &[(T, FeatureIdMask)],
    matching_values: &[T],
    out: &mut Vec<u64>,
    word_count: usize,
) -> bool {
    ensure_zeroed_words(out, word_count);
    for value in matching_values {
        let Some(source) = find_feature_id_mask(entries, value) else {
            continue;
        };
        for (dst, src) in out.iter_mut().zip(source) {
            *dst |= *src;
        }
    }
    out.iter().any(|&word| word != 0)
}

struct FeatureMaskBuilder<'a> {
    candidate_mask: &'a mut Vec<u64>,
    component_mask: &'a mut Vec<u64>,
    word_count: usize,
    has_active_candidate: bool,
}

impl<'a> FeatureMaskBuilder<'a> {
    fn new(
        feature_count: usize,
        candidate_mask: &'a mut Vec<u64>,
        component_mask: &'a mut Vec<u64>,
    ) -> Self {
        let word_count = bitset_word_count(feature_count);
        ensure_zeroed_words(candidate_mask, word_count);
        Self {
            candidate_mask,
            component_mask,
            word_count,
            has_active_candidate: false,
        }
    }

    fn intersect<T: Ord>(&mut self, entries: &[(T, FeatureIdMask)], matching_values: &[T]) -> bool {
        if !union_feature_id_masks(
            entries,
            matching_values,
            self.component_mask,
            self.word_count,
        ) {
            return false;
        }
        intersect_source(
            self.candidate_mask,
            &mut self.has_active_candidate,
            self.component_mask,
        )
    }

    const fn has_candidates(&self) -> bool {
        self.has_active_candidate
    }
}

enum FeatureMaskConstraint<'a> {
    Atom {
        query: AtomFeature,
        entries: &'a [(AtomFeature, FeatureIdMask)],
        matching_values: &'a [AtomFeature],
    },
    Bond {
        query: EdgeBondFeature,
        entries: &'a [(EdgeBondFeature, FeatureIdMask)],
        matching_values: &'a [EdgeBondFeature],
    },
}

impl<'a> FeatureMaskConstraint<'a> {
    const fn atom(
        query: AtomFeature,
        entries: &'a [(AtomFeature, FeatureIdMask)],
        matching_values: &'a [AtomFeature],
    ) -> Self {
        Self::Atom {
            query,
            entries,
            matching_values,
        }
    }

    const fn bond(
        query: EdgeBondFeature,
        entries: &'a [(EdgeBondFeature, FeatureIdMask)],
        matching_values: &'a [EdgeBondFeature],
    ) -> Self {
        Self::Bond {
            query,
            entries,
            matching_values,
        }
    }

    fn intersects(self, builder: &mut FeatureMaskBuilder<'_>) -> bool {
        match self {
            Self::Atom {
                query,
                entries,
                matching_values,
            } => {
                atom_feature_is_unconstrained(query) || builder.intersect(entries, matching_values)
            }
            Self::Bond {
                query,
                entries,
                matching_values,
            } => {
                bond_feature_is_unconstrained(query) || builder.intersect(entries, matching_values)
            }
        }
    }
}

fn build_feature_mask<'a>(
    feature_count: usize,
    candidate_mask: &mut Vec<u64>,
    component_mask: &mut Vec<u64>,
    constraints: impl IntoIterator<Item = FeatureMaskConstraint<'a>>,
) -> bool {
    let mut builder = FeatureMaskBuilder::new(feature_count, candidate_mask, component_mask);
    for constraint in constraints {
        if !constraint.intersects(&mut builder) {
            return false;
        }
    }
    builder.has_candidates()
}

struct EdgeOrientationFeatureMask<'a> {
    query_left: AtomFeature,
    left_atoms: &'a [AtomFeature],
    query_bond: EdgeBondFeature,
    bonds: &'a [EdgeBondFeature],
    query_right: AtomFeature,
    right_atoms: &'a [AtomFeature],
}

fn build_edge_orientation_feature_mask(
    index: &EdgeFeatureMaskIndex,
    orientation: &EdgeOrientationFeatureMask<'_>,
    feature_count: usize,
    candidate_mask: &mut Vec<u64>,
    component_mask: &mut Vec<u64>,
) -> bool {
    build_feature_mask(
        feature_count,
        candidate_mask,
        component_mask,
        [
            FeatureMaskConstraint::atom(
                orientation.query_left,
                &index.left_atoms,
                orientation.left_atoms,
            ),
            FeatureMaskConstraint::bond(orientation.query_bond, &index.bonds, orientation.bonds),
            FeatureMaskConstraint::atom(
                orientation.query_right,
                &index.right_atoms,
                orientation.right_atoms,
            ),
        ],
    )
}

struct Path3OrientationFeatureMask<'a> {
    query_left: AtomFeature,
    left_atoms: &'a [AtomFeature],
    query_left_bond: EdgeBondFeature,
    left_bonds: &'a [EdgeBondFeature],
    query_center: AtomFeature,
    center_atoms: &'a [AtomFeature],
    query_right_bond: EdgeBondFeature,
    right_bonds: &'a [EdgeBondFeature],
    query_right: AtomFeature,
    right_atoms: &'a [AtomFeature],
}

fn build_path3_orientation_feature_mask(
    index: &Path3FeatureMaskIndex,
    orientation: &Path3OrientationFeatureMask<'_>,
    feature_count: usize,
    candidate_mask: &mut Vec<u64>,
    component_mask: &mut Vec<u64>,
) -> bool {
    build_feature_mask(
        feature_count,
        candidate_mask,
        component_mask,
        [
            FeatureMaskConstraint::atom(
                orientation.query_left,
                &index.left_atoms,
                orientation.left_atoms,
            ),
            FeatureMaskConstraint::bond(
                orientation.query_left_bond,
                &index.left_bonds,
                orientation.left_bonds,
            ),
            FeatureMaskConstraint::atom(
                orientation.query_center,
                &index.center_atoms,
                orientation.center_atoms,
            ),
            FeatureMaskConstraint::bond(
                orientation.query_right_bond,
                &index.right_bonds,
                orientation.right_bonds,
            ),
            FeatureMaskConstraint::atom(
                orientation.query_right,
                &index.right_atoms,
                orientation.right_atoms,
            ),
        ],
    )
}

struct Path4OrientationFeatureMask<'a> {
    query_left: AtomFeature,
    left_atoms: &'a [AtomFeature],
    query_left_bond: EdgeBondFeature,
    left_bonds: &'a [EdgeBondFeature],
    query_left_middle: AtomFeature,
    left_middle_atoms: &'a [AtomFeature],
    query_center_bond: EdgeBondFeature,
    center_bonds: &'a [EdgeBondFeature],
    query_right_middle: AtomFeature,
    right_middle_atoms: &'a [AtomFeature],
    query_right_bond: EdgeBondFeature,
    right_bonds: &'a [EdgeBondFeature],
    query_right: AtomFeature,
    right_atoms: &'a [AtomFeature],
}

fn build_path4_orientation_feature_mask(
    index: &Path4FeatureMaskIndex,
    orientation: &Path4OrientationFeatureMask<'_>,
    feature_count: usize,
    candidate_mask: &mut Vec<u64>,
    component_mask: &mut Vec<u64>,
) -> bool {
    build_feature_mask(
        feature_count,
        candidate_mask,
        component_mask,
        [
            FeatureMaskConstraint::atom(
                orientation.query_left,
                &index.left_atoms,
                orientation.left_atoms,
            ),
            FeatureMaskConstraint::bond(
                orientation.query_left_bond,
                &index.left_bonds,
                orientation.left_bonds,
            ),
            FeatureMaskConstraint::atom(
                orientation.query_left_middle,
                &index.left_middle_atoms,
                orientation.left_middle_atoms,
            ),
            FeatureMaskConstraint::bond(
                orientation.query_center_bond,
                &index.center_bonds,
                orientation.center_bonds,
            ),
            FeatureMaskConstraint::atom(
                orientation.query_right_middle,
                &index.right_middle_atoms,
                orientation.right_middle_atoms,
            ),
            FeatureMaskConstraint::bond(
                orientation.query_right_bond,
                &index.right_bonds,
                orientation.right_bonds,
            ),
            FeatureMaskConstraint::atom(
                orientation.query_right,
                &index.right_atoms,
                orientation.right_atoms,
            ),
        ],
    )
}

struct Star3OrientationArmFeatureMask<'a> {
    query: Star3Arm,
    atoms: &'a [AtomFeature],
    atom_index: &'a [(AtomFeature, FeatureIdMask)],
    bonds: &'a [EdgeBondFeature],
    bond_index: &'a [(EdgeBondFeature, FeatureIdMask)],
}

struct Star3OrientationFeatureMask<'a> {
    query_center: AtomFeature,
    center_atoms: &'a [AtomFeature],
    center_atom_index: &'a [(AtomFeature, FeatureIdMask)],
    arms: [Star3OrientationArmFeatureMask<'a>; 3],
}

fn build_star3_orientation_feature_mask(
    orientation: &Star3OrientationFeatureMask<'_>,
    feature_count: usize,
    candidate_mask: &mut Vec<u64>,
    component_mask: &mut Vec<u64>,
) -> bool {
    let [first, second, third] = &orientation.arms;
    build_feature_mask(
        feature_count,
        candidate_mask,
        component_mask,
        [
            FeatureMaskConstraint::atom(
                orientation.query_center,
                orientation.center_atom_index,
                orientation.center_atoms,
            ),
            FeatureMaskConstraint::bond(first.query.bond, first.bond_index, first.bonds),
            FeatureMaskConstraint::atom(first.query.atom, first.atom_index, first.atoms),
            FeatureMaskConstraint::bond(second.query.bond, second.bond_index, second.bonds),
            FeatureMaskConstraint::atom(second.query.atom, second.atom_index, second.atoms),
            FeatureMaskConstraint::bond(third.query.bond, third.bond_index, third.bonds),
            FeatureMaskConstraint::atom(third.query.atom, third.atom_index, third.atoms),
        ],
    )
}

fn accumulate_feature_id_mask_counts<T>(
    entries: &[(T, SparseFeatureCounts)],
    feature_mask: &[u64],
    count_by_target: &mut [u16],
    touched_targets: &mut Vec<usize>,
    active_candidate_mask: Option<&[u64]>,
) {
    for (word_index, mut word) in feature_mask.iter().copied().enumerate() {
        while word != 0 {
            let bit = word.trailing_zeros() as usize;
            let feature_id = word_index * u64::BITS as usize + bit;
            if let Some((_, sparse_counts)) = entries.get(feature_id) {
                accumulate_sparse_counts(
                    count_by_target,
                    touched_targets,
                    sparse_counts,
                    active_candidate_mask,
                );
            }
            word &= word - 1;
        }
    }
}

fn bitset_contains(words: &[u64], bit: usize) -> bool {
    let word = bit / u64::BITS as usize;
    let offset = bit % u64::BITS as usize;
    (words[word] & (1u64 << offset)) != 0
}

const fn should_filter_sparse_counts(
    has_active_source: bool,
    candidate_population: usize,
    target_count: usize,
) -> bool {
    has_active_source && candidate_population.saturating_mul(4) <= target_count
}

fn plan_feature_filters(
    edge_filters: &[(EdgeFeature, u16)],
    path3_filters: &[(Path3Feature, u16)],
    path4_filters: &[(Path4Feature, u16)],
    star3_filters: &[(Star3Feature, u16)],
) -> Box<[QueryFeatureFilter]> {
    let mut filters = Vec::with_capacity(
        edge_filters.len() + path3_filters.len() + path4_filters.len() + star3_filters.len(),
    );
    filters.extend(
        edge_filters
            .iter()
            .copied()
            .map(|(feature, required_count)| QueryFeatureFilter::Edge {
                feature,
                required_count,
            }),
    );
    filters.extend(
        path3_filters
            .iter()
            .copied()
            .map(|(feature, required_count)| QueryFeatureFilter::Path3 {
                feature,
                required_count,
            }),
    );
    filters.extend(
        path4_filters
            .iter()
            .copied()
            .map(|(feature, required_count)| QueryFeatureFilter::Path4 {
                feature,
                required_count,
            }),
    );
    filters.extend(
        star3_filters
            .iter()
            .copied()
            .map(|(feature, required_count)| QueryFeatureFilter::Star3 {
                feature,
                required_count,
            }),
    );
    filters.sort_unstable_by(|left, right| {
        feature_filter_sort_key(*right).cmp(&feature_filter_sort_key(*left))
    });
    filters.into_boxed_slice()
}

const fn feature_filter_sort_key(filter: QueryFeatureFilter) -> (usize, u16, usize) {
    (
        feature_filter_specificity(filter),
        feature_filter_required_count(filter),
        feature_filter_width(filter),
    )
}

const fn feature_filter_required_count(filter: QueryFeatureFilter) -> u16 {
    match filter {
        QueryFeatureFilter::Edge { required_count, .. }
        | QueryFeatureFilter::Path3 { required_count, .. }
        | QueryFeatureFilter::Path4 { required_count, .. }
        | QueryFeatureFilter::Star3 { required_count, .. } => required_count,
    }
}

const fn feature_filter_width(filter: QueryFeatureFilter) -> usize {
    match filter {
        QueryFeatureFilter::Edge { .. } => 3,
        QueryFeatureFilter::Path3 { .. } => 5,
        QueryFeatureFilter::Path4 { .. } | QueryFeatureFilter::Star3 { .. } => 7,
    }
}

const fn feature_filter_specificity(filter: QueryFeatureFilter) -> usize {
    match filter {
        QueryFeatureFilter::Edge { feature, .. } => {
            atom_feature_specificity(feature.left)
                + bond_feature_specificity(feature.bond)
                + atom_feature_specificity(feature.right)
        }
        QueryFeatureFilter::Path3 { feature, .. } => {
            atom_feature_specificity(feature.left)
                + bond_feature_specificity(feature.left_bond)
                + atom_feature_specificity(feature.center)
                + bond_feature_specificity(feature.right_bond)
                + atom_feature_specificity(feature.right)
        }
        QueryFeatureFilter::Path4 { feature, .. } => {
            atom_feature_specificity(feature.left)
                + bond_feature_specificity(feature.left_bond)
                + atom_feature_specificity(feature.left_middle)
                + bond_feature_specificity(feature.center_bond)
                + atom_feature_specificity(feature.right_middle)
                + bond_feature_specificity(feature.right_bond)
                + atom_feature_specificity(feature.right)
        }
        QueryFeatureFilter::Star3 { feature, .. } => {
            atom_feature_specificity(feature.center)
                + bond_feature_specificity(feature.arms[0].bond)
                + atom_feature_specificity(feature.arms[0].atom)
                + bond_feature_specificity(feature.arms[1].bond)
                + atom_feature_specificity(feature.arms[1].atom)
                + bond_feature_specificity(feature.arms[2].bond)
                + atom_feature_specificity(feature.arms[2].atom)
        }
    }
}

const fn atom_feature_specificity(feature: AtomFeature) -> usize {
    bool_score(feature.element.is_some(), 8)
        + bool_score(feature.aromatic.is_some(), 4)
        + bool_score(feature.requires_ring, 3)
        + bool_score(feature.degree.is_some(), 2)
        + bool_score(feature.total_hydrogens.is_some(), 2)
}

const fn bond_feature_specificity(feature: EdgeBondFeature) -> usize {
    bool_score(feature.kind.is_some(), 4) + bool_score(feature.requires_ring, 2)
}

const fn bool_score(enabled: bool, score: usize) -> usize {
    if enabled {
        score
    } else {
        0
    }
}

fn collect_matching_atom_features(
    domain: &[AtomFeature],
    query: AtomFeature,
    out: &mut Vec<AtomFeature>,
) {
    out.clear();
    if atom_feature_is_unconstrained(query) {
        return;
    }
    out.extend(
        domain
            .iter()
            .copied()
            .filter(|&feature| atom_feature_satisfies_query(feature, query)),
    );
}

fn collect_matching_bond_features(
    domain: &[EdgeBondFeature],
    query: EdgeBondFeature,
    out: &mut Vec<EdgeBondFeature>,
) {
    out.clear();
    if bond_feature_is_unconstrained(query) {
        return;
    }
    out.extend(
        domain
            .iter()
            .copied()
            .filter(|&feature| bond_feature_satisfies_query(feature, query)),
    );
}

fn build_target_feature_indexes(
    targets: &[PreparedTarget],
) -> (
    EdgeFeatureCountIndex,
    Path3FeatureCountIndex,
    Path4FeatureCountIndex,
    Star3FeatureCountIndex,
) {
    let mut edge_occurrences = EdgeFeatureCountIndex::new();
    let mut path3_occurrences = Path3FeatureCountIndex::new();
    let mut path4_occurrences = Path4FeatureCountIndex::new();
    let mut star3_occurrences = Star3FeatureCountIndex::new();

    for (target_id, target) in targets.iter().enumerate() {
        let target_id = compact_target_id(target_id);
        let (edge_counts, path3_features, path4_features, star3_features) =
            target_graph_features(target);
        for (feature, count) in edge_counts {
            edge_occurrences.entry(feature).or_default().push((
                target_id,
                u16::try_from(count).expect("target edge feature multiplicity must fit in u16"),
            ));
        }
        for (feature, count) in path3_features {
            path3_occurrences.entry(feature).or_default().push((
                target_id,
                u16::try_from(count).expect("target path3 feature multiplicity must fit in u16"),
            ));
        }
        for (feature, count) in path4_features {
            path4_occurrences.entry(feature).or_default().push((
                target_id,
                u16::try_from(count).expect("target path4 feature multiplicity must fit in u16"),
            ));
        }
        for (feature, count) in star3_features {
            star3_occurrences.entry(feature).or_default().push((
                target_id,
                u16::try_from(count).expect("target star3 feature multiplicity must fit in u16"),
            ));
        }
    }

    let edge_feature_count_index = edge_occurrences;
    let path3_feature_count_index = path3_occurrences;

    let path4_feature_count_index = path4_occurrences;
    let star3_feature_count_index = star3_occurrences;

    (
        edge_feature_count_index,
        path3_feature_count_index,
        path4_feature_count_index,
        star3_feature_count_index,
    )
}

fn build_feature_domains(
    mut atom_domain: Vec<AtomFeature>,
    mut bond_domain: Vec<EdgeBondFeature>,
) -> (Box<[AtomFeature]>, Box<[EdgeBondFeature]>) {
    atom_domain.sort_unstable();
    atom_domain.dedup();
    bond_domain.sort_unstable();
    bond_domain.dedup();
    (
        atom_domain.into_boxed_slice(),
        bond_domain.into_boxed_slice(),
    )
}

fn build_edge_sparse_index(edge_occurrences: EdgeFeatureCountIndex) -> IndexedEdgeSparseIndexParts {
    let mut atom_domain = Vec::new();
    let mut bond_domain = Vec::new();
    let entries = edge_occurrences
        .into_iter()
        .map(|(feature, sparse_counts)| {
            push_unique_sorted_domain_value(&mut atom_domain, feature.left);
            push_unique_sorted_domain_value(&mut atom_domain, feature.right);
            push_unique_sorted_domain_value(&mut bond_domain, feature.bond);
            (feature, sparse_counts.into_boxed_slice())
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();

    let mask_index = build_edge_feature_mask_index(&entries);
    let (atom_domain, bond_domain) = build_feature_domains(atom_domain, bond_domain);
    (entries, mask_index, atom_domain, bond_domain)
}

fn build_edge_feature_mask_index(
    entries: &[(EdgeFeature, SparseFeatureCounts)],
) -> EdgeFeatureMaskIndex {
    let word_count = bitset_word_count(entries.len());
    let mut left_atoms = BTreeMap::new();
    let mut bonds = BTreeMap::new();
    let mut right_atoms = BTreeMap::new();

    for (feature_id, &(feature, _)) in entries.iter().enumerate() {
        push_feature_id_mask_value(&mut left_atoms, feature.left, feature_id, word_count);
        push_feature_id_mask_value(&mut bonds, feature.bond, feature_id, word_count);
        push_feature_id_mask_value(&mut right_atoms, feature.right, feature_id, word_count);
    }

    EdgeFeatureMaskIndex {
        left_atoms: into_feature_id_mask_index(left_atoms),
        bonds: into_feature_id_mask_index(bonds),
        right_atoms: into_feature_id_mask_index(right_atoms),
    }
}

fn build_path3_sparse_index(
    path3_occurrences: Path3FeatureCountIndex,
) -> IndexedPath3SparseIndexParts {
    let mut atom_domain = Vec::new();
    let mut bond_domain = Vec::new();
    let entries = path3_occurrences
        .into_iter()
        .map(|(feature, sparse_counts)| {
            push_unique_sorted_domain_value(&mut atom_domain, feature.left);
            push_unique_sorted_domain_value(&mut atom_domain, feature.center);
            push_unique_sorted_domain_value(&mut atom_domain, feature.right);
            push_unique_sorted_domain_value(&mut bond_domain, feature.left_bond);
            push_unique_sorted_domain_value(&mut bond_domain, feature.right_bond);
            (feature, sparse_counts.into_boxed_slice())
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();

    let mask_index = build_path3_feature_mask_index(&entries);
    let (atom_domain, bond_domain) = build_feature_domains(atom_domain, bond_domain);
    (entries, mask_index, atom_domain, bond_domain)
}

fn build_path3_feature_mask_index(
    entries: &[(Path3Feature, SparseFeatureCounts)],
) -> Path3FeatureMaskIndex {
    let word_count = bitset_word_count(entries.len());
    let mut left_atoms = BTreeMap::new();
    let mut left_bonds = BTreeMap::new();
    let mut center_atoms = BTreeMap::new();
    let mut right_bonds = BTreeMap::new();
    let mut right_atoms = BTreeMap::new();

    for (feature_id, &(feature, _)) in entries.iter().enumerate() {
        push_feature_id_mask_value(&mut left_atoms, feature.left, feature_id, word_count);
        push_feature_id_mask_value(&mut left_bonds, feature.left_bond, feature_id, word_count);
        push_feature_id_mask_value(&mut center_atoms, feature.center, feature_id, word_count);
        push_feature_id_mask_value(&mut right_bonds, feature.right_bond, feature_id, word_count);
        push_feature_id_mask_value(&mut right_atoms, feature.right, feature_id, word_count);
    }

    Path3FeatureMaskIndex {
        left_atoms: into_feature_id_mask_index(left_atoms),
        left_bonds: into_feature_id_mask_index(left_bonds),
        center_atoms: into_feature_id_mask_index(center_atoms),
        right_bonds: into_feature_id_mask_index(right_bonds),
        right_atoms: into_feature_id_mask_index(right_atoms),
    }
}

fn push_feature_id_mask_value<T: Ord>(
    masks: &mut BTreeMap<T, Vec<u64>>,
    value: T,
    feature_id: usize,
    word_count: usize,
) {
    let words = masks.entry(value).or_insert_with(|| vec![0; word_count]);
    set_bit(words, feature_id);
}

fn into_feature_id_mask_index<T: Ord>(masks: BTreeMap<T, Vec<u64>>) -> IndexedFeatureIdMask<T> {
    masks
        .into_iter()
        .map(|(feature, words)| (feature, words.into_boxed_slice()))
        .collect()
}

fn build_path4_sparse_index(
    path4_occurrences: Path4FeatureCountIndex,
) -> IndexedPath4SparseIndexParts {
    let mut atom_domain = Vec::new();
    let mut bond_domain = Vec::new();
    let entries = path4_occurrences
        .into_iter()
        .map(|(feature, sparse_counts)| {
            push_unique_sorted_domain_value(&mut atom_domain, feature.left);
            push_unique_sorted_domain_value(&mut atom_domain, feature.left_middle);
            push_unique_sorted_domain_value(&mut atom_domain, feature.right_middle);
            push_unique_sorted_domain_value(&mut atom_domain, feature.right);
            push_unique_sorted_domain_value(&mut bond_domain, feature.left_bond);
            push_unique_sorted_domain_value(&mut bond_domain, feature.center_bond);
            push_unique_sorted_domain_value(&mut bond_domain, feature.right_bond);
            (feature, sparse_counts.into_boxed_slice())
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();

    let mask_index = build_path4_feature_mask_index(&entries);
    let (atom_domain, bond_domain) = build_feature_domains(atom_domain, bond_domain);
    (entries, mask_index, atom_domain, bond_domain)
}

fn build_path4_feature_mask_index(
    entries: &[(Path4Feature, SparseFeatureCounts)],
) -> Path4FeatureMaskIndex {
    let word_count = bitset_word_count(entries.len());
    let mut left_atoms = BTreeMap::new();
    let mut left_bonds = BTreeMap::new();
    let mut left_middle_atoms = BTreeMap::new();
    let mut center_bonds = BTreeMap::new();
    let mut right_middle_atoms = BTreeMap::new();
    let mut right_bonds = BTreeMap::new();
    let mut right_atoms = BTreeMap::new();

    for (feature_id, &(feature, _)) in entries.iter().enumerate() {
        push_feature_id_mask_value(&mut left_atoms, feature.left, feature_id, word_count);
        push_feature_id_mask_value(&mut left_bonds, feature.left_bond, feature_id, word_count);
        push_feature_id_mask_value(
            &mut left_middle_atoms,
            feature.left_middle,
            feature_id,
            word_count,
        );
        push_feature_id_mask_value(
            &mut center_bonds,
            feature.center_bond,
            feature_id,
            word_count,
        );
        push_feature_id_mask_value(
            &mut right_middle_atoms,
            feature.right_middle,
            feature_id,
            word_count,
        );
        push_feature_id_mask_value(&mut right_bonds, feature.right_bond, feature_id, word_count);
        push_feature_id_mask_value(&mut right_atoms, feature.right, feature_id, word_count);
    }

    Path4FeatureMaskIndex {
        left_atoms: into_feature_id_mask_index(left_atoms),
        left_bonds: into_feature_id_mask_index(left_bonds),
        left_middle_atoms: into_feature_id_mask_index(left_middle_atoms),
        center_bonds: into_feature_id_mask_index(center_bonds),
        right_middle_atoms: into_feature_id_mask_index(right_middle_atoms),
        right_bonds: into_feature_id_mask_index(right_bonds),
        right_atoms: into_feature_id_mask_index(right_atoms),
    }
}

fn build_star3_sparse_index(
    star3_occurrences: Star3FeatureCountIndex,
) -> IndexedStar3SparseIndexParts {
    let mut atom_domain = Vec::new();
    let mut bond_domain = Vec::new();
    let entries = star3_occurrences
        .into_iter()
        .map(|(feature, sparse_counts)| {
            push_unique_sorted_domain_value(&mut atom_domain, feature.center);
            for arm in feature.arms {
                push_unique_sorted_domain_value(&mut atom_domain, arm.atom);
                push_unique_sorted_domain_value(&mut bond_domain, arm.bond);
            }
            (feature, sparse_counts.into_boxed_slice())
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();

    let mask_index = build_star3_feature_mask_index(&entries);
    let (atom_domain, bond_domain) = build_feature_domains(atom_domain, bond_domain);
    (entries, mask_index, atom_domain, bond_domain)
}

fn build_star3_feature_mask_index(
    entries: &[(Star3Feature, SparseFeatureCounts)],
) -> Star3FeatureMaskIndex {
    let word_count = bitset_word_count(entries.len());
    let mut center_atoms = BTreeMap::new();
    let mut first_bonds = BTreeMap::new();
    let mut first_atoms = BTreeMap::new();
    let mut second_bonds = BTreeMap::new();
    let mut second_atoms = BTreeMap::new();
    let mut third_bonds = BTreeMap::new();
    let mut third_atoms = BTreeMap::new();

    for (feature_id, &(feature, _)) in entries.iter().enumerate() {
        push_feature_id_mask_value(&mut center_atoms, feature.center, feature_id, word_count);
        push_feature_id_mask_value(
            &mut first_bonds,
            feature.arms[0].bond,
            feature_id,
            word_count,
        );
        push_feature_id_mask_value(
            &mut first_atoms,
            feature.arms[0].atom,
            feature_id,
            word_count,
        );
        push_feature_id_mask_value(
            &mut second_bonds,
            feature.arms[1].bond,
            feature_id,
            word_count,
        );
        push_feature_id_mask_value(
            &mut second_atoms,
            feature.arms[1].atom,
            feature_id,
            word_count,
        );
        push_feature_id_mask_value(
            &mut third_bonds,
            feature.arms[2].bond,
            feature_id,
            word_count,
        );
        push_feature_id_mask_value(
            &mut third_atoms,
            feature.arms[2].atom,
            feature_id,
            word_count,
        );
    }

    Star3FeatureMaskIndex {
        center_atoms: into_feature_id_mask_index(center_atoms),
        first_bonds: into_feature_id_mask_index(first_bonds),
        first_atoms: into_feature_id_mask_index(first_atoms),
        second_bonds: into_feature_id_mask_index(second_bonds),
        second_atoms: into_feature_id_mask_index(second_atoms),
        third_bonds: into_feature_id_mask_index(third_bonds),
        third_atoms: into_feature_id_mask_index(third_atoms),
    }
}

fn push_unique_sorted_domain_value<T: Copy + PartialEq>(domain: &mut Vec<T>, value: T) {
    if !domain.contains(&value) {
        domain.push(value);
    }
}

#[derive(Debug, Clone, Copy)]
struct IncidentTargetFeature {
    atom_id: usize,
    atom: AtomFeature,
    bond: EdgeBondFeature,
}

fn target_graph_features(target: &PreparedTarget) -> TargetGraphFeatures {
    let mut edges = BTreeMap::new();
    let mut paths = BTreeMap::new();
    let mut path4s = BTreeMap::new();
    let mut stars = BTreeMap::new();
    let mut incident = Vec::new();

    for atom_id in 0..target.atom_count() {
        let Some(center) = target_atom_feature(target, atom_id) else {
            continue;
        };
        collect_target_incident_features(target, atom_id, center, &mut edges, &mut incident);
        count_target_path3_features(center, &incident, &mut paths);
        count_target_path4_features(target, atom_id, center, &incident, &mut path4s);
        count_target_star3_features(center, &incident, &mut stars);
    }

    (edges, paths, path4s, stars)
}

fn collect_target_incident_features(
    target: &PreparedTarget,
    center_id: usize,
    center: AtomFeature,
    edges: &mut BTreeMap<EdgeFeature, usize>,
    incident: &mut Vec<IncidentTargetFeature>,
) {
    incident.clear();
    for (other_id, label) in target.neighbors(center_id) {
        let Some(other) = target_atom_feature(target, other_id) else {
            continue;
        };
        let bond = target_bond_feature(target, center_id, other_id, label);
        incident.push(IncidentTargetFeature {
            atom_id: other_id,
            atom: other,
            bond,
        });
        if center_id < other_id {
            let feature = EdgeFeature::new(center, bond, other);
            if edge_feature_is_informative(feature) {
                *edges.entry(feature).or_insert(0) += 1;
            }
        }
    }
}

fn count_target_path3_features(
    center: AtomFeature,
    incident: &[IncidentTargetFeature],
    paths: &mut BTreeMap<Path3Feature, usize>,
) {
    for left_idx in 0..incident.len() {
        for right_idx in left_idx + 1..incident.len() {
            let left = incident[left_idx];
            let right = incident[right_idx];
            let feature = Path3Feature::new(left.atom, left.bond, center, right.bond, right.atom);
            if path3_feature_is_informative(feature) {
                *paths.entry(feature).or_insert(0) += 1;
            }
        }
    }
}

fn count_target_path4_features(
    target: &PreparedTarget,
    center_id: usize,
    center: AtomFeature,
    incident: &[IncidentTargetFeature],
    path4s: &mut BTreeMap<Path4Feature, usize>,
) {
    for &right_center in incident {
        if center_id >= right_center.atom_id {
            continue;
        }
        for &left in incident {
            if left.atom_id == right_center.atom_id {
                continue;
            }
            count_target_path4_right_extensions(
                target,
                center_id,
                center,
                left,
                right_center,
                path4s,
            );
        }
    }
}

fn count_target_path4_right_extensions(
    target: &PreparedTarget,
    center_id: usize,
    center: AtomFeature,
    left: IncidentTargetFeature,
    right_center: IncidentTargetFeature,
    path4s: &mut BTreeMap<Path4Feature, usize>,
) {
    for (right_id, right_label) in target.neighbors(right_center.atom_id) {
        if right_id == center_id || right_id == left.atom_id {
            continue;
        }
        let Some(right_atom) = target_atom_feature(target, right_id) else {
            continue;
        };
        let right_bond = target_bond_feature(target, right_center.atom_id, right_id, right_label);
        let feature = Path4Feature::new(
            left.atom,
            left.bond,
            center,
            right_center.bond,
            right_center.atom,
            right_bond,
            right_atom,
        );
        if path4_feature_is_informative(feature) {
            *path4s.entry(feature).or_insert(0) += 1;
        }
    }
}

fn count_target_star3_features(
    center: AtomFeature,
    incident: &[IncidentTargetFeature],
    stars: &mut BTreeMap<Star3Feature, usize>,
) {
    for first_idx in 0..incident.len() {
        for second_idx in first_idx + 1..incident.len() {
            for third_idx in second_idx + 1..incident.len() {
                let first = incident[first_idx];
                let second = incident[second_idx];
                let third = incident[third_idx];
                let feature = Star3Feature::new(
                    center,
                    [
                        Star3Arm {
                            bond: first.bond,
                            atom: first.atom,
                        },
                        Star3Arm {
                            bond: second.bond,
                            atom: second.atom,
                        },
                        Star3Arm {
                            bond: third.bond,
                            atom: third.atom,
                        },
                    ],
                );
                if star3_feature_is_informative(feature) {
                    *stars.entry(feature).or_insert(0) += 1;
                }
            }
        }
    }
}

fn grouped_component_count(component_groups: &[Option<ComponentGroupId>]) -> usize {
    let mut groups = Vec::new();
    for group in component_groups.iter().flatten().copied() {
        if !groups.contains(&group) {
            groups.push(group);
        }
    }
    groups.len()
}

#[derive(Debug, Default)]
struct QueryAtomRequirements {
    element_counts: BTreeMap<Element, usize>,
    degree_counts: BTreeMap<u16, usize>,
    total_hydrogen_counts: BTreeMap<u16, usize>,
    min_aromatic_count: usize,
    min_ring_count: usize,
}

#[derive(Debug, Default)]
struct QueryFeatureCounts {
    edge: BTreeMap<EdgeFeature, u16>,
    path3: BTreeMap<Path3Feature, u16>,
    path4: BTreeMap<Path4Feature, u16>,
    star3: BTreeMap<Star3Feature, u16>,
}

fn collect_query_atom_requirements(query: &QueryMol) -> QueryAtomRequirements {
    let mut requirements = QueryAtomRequirements::default();
    for atom in query.atoms() {
        let atom_requirement = forced_atom_requirement(&atom.expr);
        if let Some(element) = atom_requirement.element {
            *requirements.element_counts.entry(element).or_insert(0) += 1;
        }
        if atom_requirement.requires_aromatic {
            requirements.min_aromatic_count += 1;
        }
        if atom_requirement.requires_ring {
            requirements.min_ring_count += 1;
        }

        let count_requirement = forced_atom_count_requirement(&atom.expr);
        if let Some(degree) = count_requirement.degree {
            *requirements.degree_counts.entry(degree).or_insert(0) += 1;
        }
        if let Some(total_hydrogens) = count_requirement.total_hydrogens {
            *requirements
                .total_hydrogen_counts
                .entry(total_hydrogens)
                .or_insert(0) += 1;
        }
    }
    requirements
}

fn collect_query_bond_requirements(query: &QueryMol) -> (Vec<Vec<usize>>, BondCountScreen) {
    let mut incident_bonds = vec![Vec::new(); query.atom_count()];
    let mut required_bond_counts = BondCountScreen::default();

    for bond in query.bonds() {
        let requirement = forced_bond_requirement(&bond.expr);
        incident_bonds[bond.src].push(bond.id);
        incident_bonds[bond.dst].push(bond.id);
        count_required_bond_kind(&mut required_bond_counts, requirement.kind);
        if requirement.requires_ring {
            required_bond_counts.ring += 1;
        }
    }

    (incident_bonds, required_bond_counts)
}

const fn count_required_bond_kind(counts: &mut BondCountScreen, kind: Option<RequiredBondKind>) {
    match kind {
        Some(RequiredBondKind::Single) => counts.single += 1,
        Some(RequiredBondKind::Double) => counts.double += 1,
        Some(RequiredBondKind::Triple) => counts.triple += 1,
        Some(RequiredBondKind::Aromatic) => counts.aromatic += 1,
        None => {}
    }
}

fn collect_query_feature_counts(
    query: &QueryMol,
    incident_bonds: &[Vec<usize>],
) -> QueryFeatureCounts {
    let mut counts = QueryFeatureCounts::default();
    count_query_edge_features(query, &mut counts.edge);
    count_query_path3_and_star3_features(
        query,
        incident_bonds,
        &mut counts.path3,
        &mut counts.star3,
    );
    count_query_path4_features(query, incident_bonds, &mut counts.path4);
    counts
}

fn count_query_edge_features(query: &QueryMol, counts: &mut BTreeMap<EdgeFeature, u16>) {
    for bond in query.bonds() {
        if let Some(feature) = query_edge_feature(query, bond.id) {
            *counts.entry(feature).or_insert(0) += 1;
        }
    }
}

fn count_query_path3_and_star3_features(
    query: &QueryMol,
    incident_bonds: &[Vec<usize>],
    path3_counts: &mut BTreeMap<Path3Feature, u16>,
    star3_counts: &mut BTreeMap<Star3Feature, u16>,
) {
    for (center_id, bonds) in incident_bonds.iter().enumerate() {
        for left_idx in 0..bonds.len() {
            for right_idx in left_idx + 1..bonds.len() {
                if let Some(feature) =
                    query_path3_feature(query, center_id, bonds[left_idx], bonds[right_idx])
                {
                    *path3_counts.entry(feature).or_insert(0) += 1;
                }
            }
        }
        for first_idx in 0..bonds.len() {
            for second_idx in first_idx + 1..bonds.len() {
                for third_idx in second_idx + 1..bonds.len() {
                    if let Some(feature) = query_star3_feature(
                        query,
                        center_id,
                        bonds[first_idx],
                        bonds[second_idx],
                        bonds[third_idx],
                    ) {
                        *star3_counts.entry(feature).or_insert(0) += 1;
                    }
                }
            }
        }
    }
}

fn count_query_path4_features(
    query: &QueryMol,
    incident_bonds: &[Vec<usize>],
    counts: &mut BTreeMap<Path4Feature, u16>,
) {
    for center_bond in query.bonds() {
        for &left_bond_id in &incident_bonds[center_bond.src] {
            if left_bond_id == center_bond.id {
                continue;
            }
            for &right_bond_id in &incident_bonds[center_bond.dst] {
                if right_bond_id == center_bond.id {
                    continue;
                }
                if let Some(feature) =
                    query_path4_feature(query, left_bond_id, center_bond.id, right_bond_id)
                {
                    *counts.entry(feature).or_insert(0) += 1;
                }
            }
        }
    }
}

fn query_atom_feature(query: &QueryMol, atom_id: usize) -> Option<AtomFeature> {
    Some(query_local_atom_feature(&query.atom(atom_id)?.expr))
}

fn query_edge_feature(query: &QueryMol, bond_id: usize) -> Option<EdgeFeature> {
    let bond = query.bond(bond_id)?;
    let left = query_atom_feature(query, bond.src)?;
    let right = query_atom_feature(query, bond.dst)?;
    let requirement = forced_bond_requirement(&bond.expr);
    let feature = EdgeFeature::new(
        left,
        EdgeBondFeature {
            kind: requirement.kind,
            requires_ring: requirement.requires_ring,
        },
        right,
    );
    edge_feature_is_informative(feature).then_some(feature)
}

fn query_path3_feature(
    query: &QueryMol,
    center_id: usize,
    left_bond_id: usize,
    right_bond_id: usize,
) -> Option<Path3Feature> {
    let center = query_path3_atom_feature(&query.atom(center_id)?.expr);
    let left_bond = query.bond(left_bond_id)?;
    let right_bond = query.bond(right_bond_id)?;
    let left_neighbor = other_query_atom(left_bond, center_id)?;
    let right_neighbor = other_query_atom(right_bond, center_id)?;
    if left_neighbor == right_neighbor {
        return None;
    }
    let left = query_path3_atom_feature(&query.atom(left_neighbor)?.expr);
    let right = query_path3_atom_feature(&query.atom(right_neighbor)?.expr);
    let left_bond = forced_bond_requirement(&left_bond.expr);
    let right_bond = forced_bond_requirement(&right_bond.expr);
    let feature = Path3Feature::new(
        left,
        EdgeBondFeature {
            kind: left_bond.kind,
            requires_ring: false,
        },
        center,
        EdgeBondFeature {
            kind: right_bond.kind,
            requires_ring: false,
        },
        right,
    );
    path3_feature_is_informative(feature).then_some(feature)
}

fn query_path4_feature(
    query: &QueryMol,
    left_bond_id: usize,
    center_bond_id: usize,
    right_bond_id: usize,
) -> Option<Path4Feature> {
    let left_bond = query.bond(left_bond_id)?;
    let center_bond = query.bond(center_bond_id)?;
    let right_bond = query.bond(right_bond_id)?;

    let shared_atom = if left_bond.src == center_bond.src || left_bond.src == center_bond.dst {
        left_bond.src
    } else if left_bond.dst == center_bond.src || left_bond.dst == center_bond.dst {
        left_bond.dst
    } else {
        return None;
    };
    let other_center = other_query_atom(center_bond, shared_atom)?;
    let right_shared = if right_bond.src == other_center || right_bond.dst == other_center {
        other_center
    } else {
        return None;
    };
    let left = other_query_atom(left_bond, shared_atom)?;
    let right = other_query_atom(right_bond, right_shared)?;
    if left == other_center || left == right || shared_atom == right {
        return None;
    }

    let left_requirement = forced_bond_requirement(&left_bond.expr);
    let center_requirement = forced_bond_requirement(&center_bond.expr);
    let right_requirement = forced_bond_requirement(&right_bond.expr);

    let feature = Path4Feature::new(
        query_path4_atom_feature(&query.atom(left)?.expr),
        EdgeBondFeature {
            kind: left_requirement.kind,
            requires_ring: false,
        },
        query_path4_atom_feature(&query.atom(shared_atom)?.expr),
        EdgeBondFeature {
            kind: center_requirement.kind,
            requires_ring: false,
        },
        query_path4_atom_feature(&query.atom(other_center)?.expr),
        EdgeBondFeature {
            kind: right_requirement.kind,
            requires_ring: false,
        },
        query_path4_atom_feature(&query.atom(right)?.expr),
    );
    path4_feature_is_informative(feature).then_some(feature)
}

fn query_star3_feature(
    query: &QueryMol,
    center_id: usize,
    first_bond_id: usize,
    second_bond_id: usize,
    third_bond_id: usize,
) -> Option<Star3Feature> {
    let center = query_star3_atom_feature(&query.atom(center_id)?.expr);
    let first_bond = query.bond(first_bond_id)?;
    let second_bond = query.bond(second_bond_id)?;
    let third_bond = query.bond(third_bond_id)?;
    let first_requirement = forced_bond_requirement(&first_bond.expr);
    let second_requirement = forced_bond_requirement(&second_bond.expr);
    let third_requirement = forced_bond_requirement(&third_bond.expr);
    let first_neighbor = other_query_atom(first_bond, center_id)?;
    let second_neighbor = other_query_atom(second_bond, center_id)?;
    let third_neighbor = other_query_atom(third_bond, center_id)?;
    if first_neighbor == second_neighbor
        || first_neighbor == third_neighbor
        || second_neighbor == third_neighbor
    {
        return None;
    }
    let feature = Star3Feature::new(
        center,
        [
            Star3Arm {
                bond: EdgeBondFeature {
                    kind: first_requirement.kind,
                    requires_ring: false,
                },
                atom: query_star3_atom_feature(&query.atom(first_neighbor)?.expr),
            },
            Star3Arm {
                bond: EdgeBondFeature {
                    kind: second_requirement.kind,
                    requires_ring: false,
                },
                atom: query_star3_atom_feature(&query.atom(second_neighbor)?.expr),
            },
            Star3Arm {
                bond: EdgeBondFeature {
                    kind: third_requirement.kind,
                    requires_ring: false,
                },
                atom: query_star3_atom_feature(&query.atom(third_neighbor)?.expr),
            },
        ],
    );
    star3_feature_is_informative(feature).then_some(feature)
}

const fn other_query_atom(bond: &crate::QueryBond, atom_id: usize) -> Option<usize> {
    if bond.src == atom_id {
        Some(bond.dst)
    } else if bond.dst == atom_id {
        Some(bond.src)
    } else {
        None
    }
}

const fn atom_feature_is_unconstrained(feature: AtomFeature) -> bool {
    feature.element.is_none()
        && feature.aromatic.is_none()
        && !feature.requires_ring
        && feature.degree.is_none()
        && feature.total_hydrogens.is_none()
}

const fn bond_feature_is_unconstrained(feature: EdgeBondFeature) -> bool {
    feature.kind.is_none() && !feature.requires_ring
}

const fn edge_feature_is_informative(feature: EdgeFeature) -> bool {
    !(atom_feature_is_unconstrained(feature.left)
        && bond_feature_is_unconstrained(feature.bond)
        && atom_feature_is_unconstrained(feature.right))
}

const fn path3_feature_is_informative(feature: Path3Feature) -> bool {
    !(atom_feature_is_unconstrained(feature.left)
        && bond_feature_is_unconstrained(feature.left_bond)
        && atom_feature_is_unconstrained(feature.center)
        && bond_feature_is_unconstrained(feature.right_bond)
        && atom_feature_is_unconstrained(feature.right))
}

fn atom_feature_satisfies_query(target: AtomFeature, query: AtomFeature) -> bool {
    optional_requirement_satisfied(target.element, query.element)
        && optional_requirement_satisfied(target.aromatic, query.aromatic)
        && (!query.requires_ring || target.requires_ring)
        && optional_requirement_satisfied(target.degree, query.degree)
        && optional_requirement_satisfied(target.total_hydrogens, query.total_hydrogens)
}

fn bond_feature_satisfies_query(target: EdgeBondFeature, query: EdgeBondFeature) -> bool {
    optional_requirement_satisfied(target.kind, query.kind)
        && (!query.requires_ring || target.requires_ring)
}

fn optional_requirement_satisfied<T: PartialEq>(
    target_value: Option<T>,
    required_value: Option<T>,
) -> bool {
    required_value
        .is_none_or(|value| matches!(target_value, Some(target_value) if target_value == value))
}

const fn path4_feature_is_informative(feature: Path4Feature) -> bool {
    !(atom_feature_is_unconstrained(feature.left)
        && bond_feature_is_unconstrained(feature.left_bond)
        && atom_feature_is_unconstrained(feature.left_middle)
        && bond_feature_is_unconstrained(feature.center_bond)
        && atom_feature_is_unconstrained(feature.right_middle)
        && bond_feature_is_unconstrained(feature.right_bond)
        && atom_feature_is_unconstrained(feature.right))
}

const fn star3_feature_is_informative(feature: Star3Feature) -> bool {
    !(atom_feature_is_unconstrained(feature.center)
        && bond_feature_is_unconstrained(feature.arms[0].bond)
        && atom_feature_is_unconstrained(feature.arms[0].atom)
        && bond_feature_is_unconstrained(feature.arms[1].bond)
        && atom_feature_is_unconstrained(feature.arms[1].atom)
        && bond_feature_is_unconstrained(feature.arms[2].bond)
        && atom_feature_is_unconstrained(feature.arms[2].atom))
}

fn target_atom_feature(target: &PreparedTarget, atom_id: usize) -> Option<AtomFeature> {
    let element = target.atom(atom_id)?.element()?;
    Some(AtomFeature {
        element: Some(element),
        aromatic: Some(target.is_aromatic(atom_id)),
        requires_ring: target.is_ring_atom(atom_id),
        degree: target
            .degree(atom_id)
            .map(|degree| u16::try_from(degree).unwrap_or(u16::MAX)),
        total_hydrogens: target.total_hydrogen_count(atom_id).map(u16::from),
    })
}

fn query_path3_atom_feature(expr: &AtomExpr) -> AtomFeature {
    query_local_atom_feature(expr)
}

fn query_path4_atom_feature(expr: &AtomExpr) -> AtomFeature {
    query_local_atom_feature(expr)
}

fn query_star3_atom_feature(expr: &AtomExpr) -> AtomFeature {
    query_local_atom_feature(expr)
}

fn query_local_atom_feature(expr: &AtomExpr) -> AtomFeature {
    let requirement = exact_atom_requirement(expr);
    let count_requirement = forced_atom_count_requirement(expr);
    let (degree, total_hydrogens) = if requirement.element.is_some() {
        (count_requirement.degree, count_requirement.total_hydrogens)
    } else {
        (None, None)
    };
    AtomFeature {
        element: requirement.element,
        aromatic: requirement.aromatic,
        requires_ring: requirement.requires_ring,
        degree,
        total_hydrogens,
    }
}

const fn normalized_required_bond_kind(label: BondLabel) -> Option<RequiredBondKind> {
    match label {
        BondLabel::Single | BondLabel::Up | BondLabel::Down => Some(RequiredBondKind::Single),
        BondLabel::Double => Some(RequiredBondKind::Double),
        BondLabel::Triple => Some(RequiredBondKind::Triple),
        BondLabel::Aromatic => Some(RequiredBondKind::Aromatic),
        BondLabel::Any => None,
    }
}

fn target_bond_feature(
    target: &PreparedTarget,
    left_atom: usize,
    right_atom: usize,
    label: BondLabel,
) -> EdgeBondFeature {
    EdgeBondFeature {
        kind: normalized_required_bond_kind(label),
        requires_ring: target.is_ring_bond(left_atom, right_atom),
    }
}

#[cfg(test)]
mod tests;
