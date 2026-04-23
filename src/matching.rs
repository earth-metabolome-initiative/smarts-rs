use alloc::{boxed::Box, collections::BTreeMap, string::String, string::ToString};

use crate::{
    AtomExpr, AtomId as QueryAtomId, AtomPrimitive, BondExpr, BondExprTree, BondPrimitive,
    BracketExprTree, HydrogenKind, NumericQuery, QueryMol,
};
use elements_rs::{AtomicNumber, Element, ElementVariant, MassNumber};
use smiles_parser::{
    atom::{bracketed::chirality::Chirality, Atom},
    bond::Bond,
    DoubleBondStereoConfig, Smiles,
};

use crate::{error::SmartsMatchError, prepared::PreparedTarget, target::BondLabel};

type QueryNeighbors = alloc::vec::Vec<alloc::vec::Vec<(QueryAtomId, usize)>>;
type QueryAtomScores = alloc::vec::Vec<usize>;
type CompiledQueryParts = (
    QueryNeighbors,
    alloc::vec::Vec<usize>,
    QueryAtomScores,
    QueryStereoPlan,
);
type AtomStereoCache = BTreeMap<AtomStereoCacheKey, Option<Chirality>>;
type UniqueMatches = BTreeMap<Box<[usize]>, Box<[usize]>>;
type MatchKeys = alloc::vec::Vec<Box<[usize]>>;

struct SearchContext<'a> {
    compiled_query: &'a CompiledQuery,
    query_neighbors: &'a [alloc::vec::Vec<(QueryAtomId, usize)>],
    query_atom_scores: &'a [usize],
    query_atom_anchor_widths: &'a [usize],
    query_to_target: &'a mut [Option<usize>],
    used_target_atoms: &'a mut [u32],
    used_target_generation: u32,
    stereo_plan: &'a QueryStereoPlan,
    atom_stereo_cache: &'a mut AtomStereoCache,
    recursive_cache: &'a mut RecursiveMatchCache,
}

struct SearchScratchView<'a> {
    query_atom_anchor_widths: &'a [usize],
    query_to_target: &'a mut [Option<usize>],
    used_target_atoms: &'a mut [u32],
    used_target_generation: u32,
}

#[derive(Debug, Clone)]
struct FreshSearchBuffers {
    query_atom_anchor_widths: alloc::vec::Vec<usize>,
    query_to_target: alloc::vec::Vec<Option<usize>>,
    used_target_atoms: alloc::vec::Vec<u32>,
}

impl FreshSearchBuffers {
    fn new(query: &CompiledQuery, target: &PreparedTarget) -> Self {
        let mut query_atom_anchor_widths = alloc::vec::Vec::new();
        prepare_query_atom_anchor_widths(&mut query_atom_anchor_widths, query, target);
        Self {
            query_atom_anchor_widths,
            query_to_target: alloc::vec![None; query.query.atom_count()],
            used_target_atoms: alloc::vec![0; target.atom_count()],
        }
    }

    fn view(&mut self) -> SearchScratchView<'_> {
        SearchScratchView {
            query_atom_anchor_widths: &self.query_atom_anchor_widths,
            query_to_target: &mut self.query_to_target,
            used_target_atoms: &mut self.used_target_atoms,
            used_target_generation: 1,
        }
    }
}

struct AtomMatchContext<'a, 'cache> {
    target: &'a PreparedTarget,
    recursive_query_lookup: &'a BTreeMap<usize, usize>,
    recursive_queries: &'a [RecursiveQueryEntry],
    recursive_cache: &'cache mut RecursiveMatchCache,
}

impl<'a, 'cache> AtomMatchContext<'a, 'cache> {
    const fn new(
        query: &'a CompiledQuery,
        target: &'a PreparedTarget,
        recursive_cache: &'cache mut RecursiveMatchCache,
    ) -> Self {
        Self {
            target,
            recursive_query_lookup: &query.recursive_query_lookup,
            recursive_queries: &query.recursive_queries,
            recursive_cache,
        }
    }
}

impl<'a> SearchScratchView<'a> {
    fn into_context(
        self,
        query: &'a CompiledQuery,
        atom_stereo_cache: &'a mut AtomStereoCache,
        recursive_cache: &'a mut RecursiveMatchCache,
    ) -> SearchContext<'a> {
        SearchContext {
            compiled_query: query,
            query_neighbors: &query.query_neighbors,
            query_atom_scores: &query.query_atom_scores,
            query_atom_anchor_widths: self.query_atom_anchor_widths,
            query_to_target: self.query_to_target,
            used_target_atoms: self.used_target_atoms,
            used_target_generation: self.used_target_generation,
            stereo_plan: &query.stereo_plan,
            atom_stereo_cache,
            recursive_cache,
        }
    }
}

impl SearchContext<'_> {
    #[inline]
    fn target_atom_is_used(&self, target_atom: usize) -> bool {
        self.used_target_atoms[target_atom] == self.used_target_generation
    }

    #[inline]
    fn mark_target_atom_used(&mut self, target_atom: usize) {
        self.used_target_atoms[target_atom] = self.used_target_generation;
    }

    #[inline]
    fn unmark_target_atom_used(&mut self, target_atom: usize) {
        self.used_target_atoms[target_atom] = 0;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct QueryStereoPlan {
    directional_bond_ids: alloc::vec::Vec<bool>,
    double_bond_constraints: alloc::vec::Vec<QueryDoubleBondStereoConstraint>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QueryDoubleBondStereoConstraint {
    left_atom: QueryAtomId,
    right_atom: QueryAtomId,
    config: DoubleBondStereoConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QueryDirectionalSubstituent {
    bond_id: usize,
    direction: Bond,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QueryLocalSubstituent {
    bond_id: usize,
    direction: Option<Bond>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct AtomStereoCacheKey {
    query_atom_id: QueryAtomId,
    neighbor_tokens: alloc::vec::Vec<String>,
}

#[derive(Debug, Clone)]
struct RecursiveQueryEntry {
    query_key: usize,
    cache_slot: usize,
    compiled: Box<CompiledQuery>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CompiledBondMatcher {
    state_mask: u16,
    ring_sensitive: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CompiledAtomMatcher {
    predicates: Box<[AtomFastPredicate]>,
    needs_atom: bool,
    needs_aromatic: bool,
    complete: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AtomFastPredicate {
    HasElement,
    Element(Element),
    Aromatic(bool),
    Isotope(Option<u16>),
    AtomicNumber(u16),
    Degree(Option<NumericQuery>),
    Connectivity(Option<NumericQuery>),
    Valence(Option<NumericQuery>),
    Hybridization(NumericQuery),
    TotalHydrogen(Option<NumericQuery>),
    ImplicitHydrogen(Option<NumericQuery>),
    RingMembership(Option<NumericQuery>),
    RingSize(Option<NumericQuery>),
    RingConnectivity(Option<NumericQuery>),
    HeteroNeighbor(Option<NumericQuery>),
    AliphaticHeteroNeighbor(Option<NumericQuery>),
    Charge(i8),
}

#[derive(Debug, Clone)]
struct RecursiveMatchCache {
    target_atom_count: usize,
    values: alloc::vec::Vec<Option<bool>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ThreeAtomTreeLayout {
    center: QueryAtomId,
    leaf_a: QueryAtomId,
    leaf_b: QueryAtomId,
    bond_a: usize,
    bond_b: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct InitialAtomMapping {
    query_atom: Option<QueryAtomId>,
    target_atom: Option<usize>,
}

impl InitialAtomMapping {
    const fn new(query_atom: Option<QueryAtomId>, target_atom: Option<usize>) -> Self {
        Self {
            query_atom,
            target_atom,
        }
    }

    const fn pair(self) -> Option<(QueryAtomId, usize)> {
        match (self.query_atom, self.target_atom) {
            (Some(query_atom), Some(target_atom)) => Some((query_atom, target_atom)),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TwoAtomTargetMapping {
    left: usize,
    right: usize,
    bond: BondLabel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TwoAtomQueryMapping {
    left: QueryAtomId,
    right: QueryAtomId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MappedNeighborSeed {
    target_atom: usize,
    bond_id: usize,
    query_neighbor: QueryAtomId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ThreeAtomTargetMapping {
    center: usize,
    leaf_a: usize,
    bond_a: BondLabel,
    leaf_b: usize,
    bond_b: BondLabel,
}

enum SingleAtomCandidates<'a> {
    One(usize),
    Anchored(&'a [usize]),
    All(core::ops::Range<usize>),
}

impl SingleAtomCandidates<'_> {
    fn any(self, mut predicate: impl FnMut(usize) -> bool) -> bool {
        match self {
            Self::One(target_atom) => predicate(target_atom),
            Self::Anchored(target_atoms) => target_atoms.iter().copied().any(predicate),
            Self::All(mut target_atoms) => target_atoms.any(predicate),
        }
    }

    fn count(self, mut predicate: impl FnMut(usize) -> bool) -> usize {
        match self {
            Self::One(target_atom) => usize::from(predicate(target_atom)),
            Self::Anchored(target_atoms) => target_atoms
                .iter()
                .copied()
                .filter(|&target_atom| predicate(target_atom))
                .count(),
            Self::All(target_atoms) => target_atoms
                .filter(|&target_atom| predicate(target_atom))
                .count(),
        }
    }

    fn collect_matches(self, mut predicate: impl FnMut(usize) -> bool) -> Box<[Box<[usize]>]> {
        match self {
            Self::One(target_atom) if predicate(target_atom) => {
                alloc::vec![Box::<[usize]>::from([target_atom])].into_boxed_slice()
            }
            Self::One(_) => Box::default(),
            Self::Anchored(target_atoms) => target_atoms
                .iter()
                .copied()
                .filter(|&target_atom| predicate(target_atom))
                .map(|target_atom| Box::<[usize]>::from([target_atom]))
                .collect::<alloc::vec::Vec<_>>()
                .into_boxed_slice(),
            Self::All(target_atoms) => target_atoms
                .filter(|&target_atom| predicate(target_atom))
                .map(|target_atom| Box::<[usize]>::from([target_atom]))
                .collect::<alloc::vec::Vec<_>>()
                .into_boxed_slice(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct FixedThreeAtomLeaves {
    leaf_a: Option<(usize, BondLabel)>,
    leaf_b: Option<(usize, BondLabel)>,
}

struct ThreeAtomCenterSearch<'a, 'cache> {
    query: &'a CompiledQuery,
    target: &'a PreparedTarget,
    recursive_cache: &'cache mut RecursiveMatchCache,
    layout: ThreeAtomTreeLayout,
    target_center: usize,
    initial_mapping: InitialAtomMapping,
}

impl<'a, 'cache> ThreeAtomCenterSearch<'a, 'cache> {
    const fn new(
        query: &'a CompiledQuery,
        target: &'a PreparedTarget,
        recursive_cache: &'cache mut RecursiveMatchCache,
        layout: ThreeAtomTreeLayout,
        target_center: usize,
        initial_mapping: InitialAtomMapping,
    ) -> Self {
        Self {
            query,
            target,
            recursive_cache,
            layout,
            target_center,
            initial_mapping,
        }
    }

    fn center_matches(&mut self) -> bool {
        self.target
            .degree(self.target_center)
            .is_none_or(|degree| degree >= self.query.query_degrees[self.layout.center])
            && compiled_query_atom_matches(
                self.query,
                self.target,
                self.recursive_cache,
                self.layout.center,
                self.target_center,
            )
    }

    const fn target_mapping(
        &self,
        leaf_a: usize,
        bond_a: BondLabel,
        leaf_b: usize,
        bond_b: BondLabel,
    ) -> ThreeAtomTargetMapping {
        ThreeAtomTargetMapping {
            center: self.target_center,
            leaf_a,
            bond_a,
            leaf_b,
            bond_b,
        }
    }

    fn mapping_matches(&mut self, target_mapping: ThreeAtomTargetMapping) -> bool {
        three_atom_mapping_matches(
            self.query,
            self.target,
            self.recursive_cache,
            self.layout,
            target_mapping,
            self.initial_mapping,
        )
    }
}

impl RecursiveMatchCache {
    fn new(slot_count: usize, target_atom_count: usize) -> Self {
        Self {
            target_atom_count,
            values: alloc::vec![None; slot_count.saturating_mul(target_atom_count)],
        }
    }

    fn reset(&mut self, slot_count: usize, target_atom_count: usize) {
        self.target_atom_count = target_atom_count;
        let len = slot_count.saturating_mul(target_atom_count);
        if self.values.len() == len {
            self.values.fill(None);
        } else {
            self.values = alloc::vec![None; len];
        }
    }

    fn get(&self, slot: usize, atom_id: usize) -> Option<bool> {
        self.values
            .get(slot.checked_mul(self.target_atom_count)? + atom_id)
            .copied()
            .flatten()
    }

    fn insert(&mut self, slot: usize, atom_id: usize, value: bool) {
        if let Some(cell) = self.values.get_mut(
            slot.saturating_mul(self.target_atom_count)
                .saturating_add(atom_id),
        ) {
            *cell = Some(value);
        }
    }
}

/// Reusable per-thread workspace for repeated boolean matching.
#[derive(Debug, Clone)]
pub struct MatchScratch {
    query_atom_anchor_widths: alloc::vec::Vec<usize>,
    query_to_target: alloc::vec::Vec<Option<usize>>,
    used_target_atoms: alloc::vec::Vec<u32>,
    used_target_generation: u32,
    atom_stereo_cache: AtomStereoCache,
    recursive_cache: RecursiveMatchCache,
}

impl Default for MatchScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl MatchScratch {
    /// Builds empty reusable matcher scratch buffers.
    #[must_use]
    pub fn new() -> Self {
        Self {
            query_atom_anchor_widths: alloc::vec::Vec::new(),
            query_to_target: alloc::vec::Vec::new(),
            used_target_atoms: alloc::vec::Vec::new(),
            used_target_generation: 0,
            atom_stereo_cache: AtomStereoCache::new(),
            recursive_cache: RecursiveMatchCache::new(0, 0),
        }
    }

    fn prepare(&mut self, query: &CompiledQuery, target: &PreparedTarget) {
        let query_atom_count = query.query.atom_count();
        prepare_query_atom_anchor_widths(&mut self.query_atom_anchor_widths, query, target);
        if self.query_to_target.len() == query_atom_count {
            self.query_to_target.fill(None);
        } else {
            self.query_to_target.resize(query_atom_count, None);
        }

        let target_atom_count = target.atom_count();
        if self.used_target_atoms.len() != target_atom_count {
            self.used_target_atoms.resize(target_atom_count, 0);
        }
        self.used_target_generation = self.used_target_generation.wrapping_add(1);
        if self.used_target_generation == 0 {
            self.used_target_atoms.fill(0);
            self.used_target_generation = 1;
        }

        self.atom_stereo_cache.clear();
        self.recursive_cache
            .reset(query.recursive_cache_slots, target_atom_count);
    }
}

fn prepare_target_smiles(target: &str) -> Result<PreparedTarget, SmartsMatchError> {
    if target.is_empty() {
        return Err(SmartsMatchError::EmptyTarget);
    }

    let target =
        target
            .parse::<Smiles>()
            .map_err(|error| SmartsMatchError::InvalidTargetSmiles {
                source: error.smiles_error(),
                start: error.start(),
                end: error.end(),
            })?;
    Ok(PreparedTarget::new(target))
}

impl QueryMol {
    /// Match this SMARTS query against a target `SMILES` string.
    ///
    /// Target molecules are parsed through `smiles-parser` and prepared with
    /// RDKit-default aromaticity, degree, implicit-hydrogen,
    /// total-hydrogen, and effective bond-label caches before matching.
    ///
    /// # Errors
    ///
    /// Returns:
    /// - [`SmartsMatchError::EmptyTarget`] for empty target strings
    /// - [`SmartsMatchError::InvalidTargetSmiles`] when the target is not
    ///   valid `SMILES`
    /// - [`SmartsMatchError::UnsupportedAtomPrimitive`] when the query uses an
    ///   atom primitive outside the current matcher implementation
    /// - [`SmartsMatchError::UnsupportedBondPrimitive`] when the query uses a
    ///   bond primitive outside the current matcher implementation
    pub fn matches(&self, target: &str) -> Result<bool, SmartsMatchError> {
        let target = prepare_target_smiles(target)?;
        self.matches_prepared(&target)
    }

    /// Match this SMARTS query against a prepared target molecule.
    ///
    /// This convenience method derives reusable query-side state for each
    /// call. For repeated matching of one SMARTS against many targets, prefer
    /// [`CompiledQuery`] plus [`CompiledQuery::matches`].
    ///
    /// # Errors
    ///
    /// Returns:
    /// - [`SmartsMatchError::UnsupportedAtomPrimitive`] when the query uses an
    ///   atom primitive outside the current matcher implementation
    /// - [`SmartsMatchError::UnsupportedBondPrimitive`] when the query uses a
    ///   bond primitive outside the current matcher implementation
    pub fn matches_prepared(&self, target: &PreparedTarget) -> Result<bool, SmartsMatchError> {
        let compiled = CompiledQuery::new(self.clone())?;
        Ok(compiled.matches(target))
    }

    /// Collect all unique accepted substructure matches against a target
    /// `SMILES` string.
    ///
    /// Each inner match lists target atom ids in query atom order.
    ///
    /// # Errors
    ///
    /// Returns:
    /// - [`SmartsMatchError::EmptyTarget`] for empty target strings
    /// - [`SmartsMatchError::InvalidTargetSmiles`] when the target is not
    ///   valid `SMILES`
    /// - [`SmartsMatchError::UnsupportedAtomPrimitive`] when the query uses an
    ///   atom primitive outside the current matcher implementation
    /// - [`SmartsMatchError::UnsupportedBondPrimitive`] when the query uses a
    ///   bond primitive outside the current matcher implementation
    pub fn substructure_matches(
        &self,
        target: &str,
    ) -> Result<Box<[Box<[usize]>]>, SmartsMatchError> {
        let target = prepare_target_smiles(target)?;
        self.substructure_matches_prepared(&target)
    }

    /// Collect all unique accepted substructure matches against one prepared
    /// target molecule.
    ///
    /// Each inner match lists target atom ids in query atom order.
    ///
    /// # Errors
    ///
    /// Returns:
    /// - [`SmartsMatchError::UnsupportedAtomPrimitive`] when the query uses an
    ///   atom primitive outside the current matcher implementation
    /// - [`SmartsMatchError::UnsupportedBondPrimitive`] when the query uses a
    ///   bond primitive outside the current matcher implementation
    pub fn substructure_matches_prepared(
        &self,
        target: &PreparedTarget,
    ) -> Result<Box<[Box<[usize]>]>, SmartsMatchError> {
        let compiled = CompiledQuery::new(self.clone())?;
        Ok(compiled.substructure_matches(target))
    }

    /// Count all unique accepted substructure matches against a target
    /// `SMILES` string.
    ///
    /// # Errors
    ///
    /// Returns:
    /// - [`SmartsMatchError::EmptyTarget`] for empty target strings
    /// - [`SmartsMatchError::InvalidTargetSmiles`] when the target is not
    ///   valid `SMILES`
    /// - [`SmartsMatchError::UnsupportedAtomPrimitive`] when the query uses an
    ///   atom primitive outside the current matcher implementation
    /// - [`SmartsMatchError::UnsupportedBondPrimitive`] when the query uses a
    ///   bond primitive outside the current matcher implementation
    pub fn match_count(&self, target: &str) -> Result<usize, SmartsMatchError> {
        let target = prepare_target_smiles(target)?;
        self.match_count_prepared(&target)
    }

    /// Count all unique accepted substructure matches against one prepared
    /// target.
    ///
    /// # Errors
    ///
    /// Returns:
    /// - [`SmartsMatchError::UnsupportedAtomPrimitive`] when the query uses an
    ///   atom primitive outside the current matcher implementation
    /// - [`SmartsMatchError::UnsupportedBondPrimitive`] when the query uses a
    ///   bond primitive outside the current matcher implementation
    pub fn match_count_prepared(&self, target: &PreparedTarget) -> Result<usize, SmartsMatchError> {
        let compiled = CompiledQuery::new(self.clone())?;
        Ok(compiled.match_count(target))
    }
}

/// Reusable compiled SMARTS query state for repeated matching.
#[derive(Debug, Clone)]
pub struct CompiledQuery {
    query: QueryMol,
    query_neighbors: QueryNeighbors,
    query_degrees: alloc::vec::Vec<usize>,
    query_atom_scores: QueryAtomScores,
    stereo_plan: QueryStereoPlan,
    atom_matchers: Box<[CompiledAtomMatcher]>,
    bond_matchers: Box<[CompiledBondMatcher]>,
    has_component_constraints: bool,
    recursive_cache_slots: usize,
    recursive_query_lookup: BTreeMap<usize, usize>,
    recursive_queries: Box<[RecursiveQueryEntry]>,
}

impl CompiledQuery {
    /// Compile one parsed SMARTS query into reusable matcher state.
    ///
    /// # Errors
    ///
    /// Returns:
    /// - [`SmartsMatchError::UnsupportedAtomPrimitive`] when the query uses an
    ///   atom primitive outside the current matcher implementation
    /// - [`SmartsMatchError::UnsupportedBondPrimitive`] when the query uses a
    ///   bond primitive outside the current matcher implementation
    pub fn new(query: QueryMol) -> Result<Self, SmartsMatchError> {
        let mut next_recursive_cache_slot = 0usize;
        Self::new_with_recursive_slots(query, &mut next_recursive_cache_slot)
    }

    fn new_with_recursive_slots(
        query: QueryMol,
        next_recursive_cache_slot: &mut usize,
    ) -> Result<Self, SmartsMatchError> {
        let (query_neighbors, query_degrees, query_atom_scores, stereo_plan) =
            compile_query_parts(&query)?;
        let atom_matchers = compile_atom_matchers(&query);
        let bond_matchers = compile_bond_matchers(&query, &stereo_plan);
        let has_component_constraints = query.component_groups().iter().any(Option::is_some);
        let recursive_queries = compile_recursive_queries(&query, next_recursive_cache_slot)?;
        let recursive_query_lookup = recursive_queries
            .iter()
            .enumerate()
            .map(|(index, entry)| (entry.query_key, index))
            .collect();
        Ok(Self {
            query,
            query_neighbors,
            query_degrees,
            query_atom_scores,
            stereo_plan,
            atom_matchers,
            bond_matchers,
            has_component_constraints,
            recursive_cache_slots: *next_recursive_cache_slot,
            recursive_query_lookup,
            recursive_queries,
        })
    }

    /// Borrow the underlying parsed SMARTS query.
    #[inline]
    #[must_use]
    pub const fn query(&self) -> &QueryMol {
        &self.query
    }

    /// Match this compiled SMARTS query against a prepared target.
    #[inline]
    #[must_use]
    pub fn matches(&self, target: &PreparedTarget) -> bool {
        query_matches(self, target)
    }

    /// Match this compiled SMARTS query against a prepared target using
    /// reusable per-thread scratch buffers.
    #[inline]
    #[must_use]
    pub fn matches_with_scratch(
        &self,
        target: &PreparedTarget,
        scratch: &mut MatchScratch,
    ) -> bool {
        query_matches_with_scratch(self, target, scratch)
    }

    /// Collect all unique accepted substructure matches against one prepared
    /// target.
    ///
    /// Each inner match lists target atom ids in query atom order.
    #[must_use]
    pub fn substructure_matches(&self, target: &PreparedTarget) -> Box<[Box<[usize]>]> {
        query_substructure_matches(self, target)
    }

    /// Count all unique accepted substructure matches against one prepared
    /// target.
    #[inline]
    #[must_use]
    pub fn match_count(&self, target: &PreparedTarget) -> usize {
        query_match_count(self, target)
    }
}

fn ensure_supported_query(
    query: &QueryMol,
    stereo_plan: &QueryStereoPlan,
) -> Result<(), SmartsMatchError> {
    for atom in query.atoms() {
        ensure_supported_atom_expr(&atom.expr, stereo_plan)?;
    }
    for bond in query.bonds() {
        ensure_supported_bond_expr(bond.id, &bond.expr, stereo_plan)?;
    }
    Ok(())
}

fn ensure_supported_bond_expr(
    bond_id: usize,
    expr: &BondExpr,
    stereo_plan: &QueryStereoPlan,
) -> Result<(), SmartsMatchError> {
    if stereo_plan
        .directional_bond_ids
        .get(bond_id)
        .copied()
        .unwrap_or(false)
    {
        return Ok(());
    }
    match expr {
        BondExpr::Elided => Ok(()),
        BondExpr::Query(tree) => ensure_supported_bond_tree(tree),
    }
}

fn query_bond_is_directional(stereo_plan: &QueryStereoPlan, bond_id: usize) -> bool {
    stereo_plan
        .directional_bond_ids
        .get(bond_id)
        .copied()
        .unwrap_or(false)
}

fn ensure_supported_bond_tree(tree: &BondExprTree) -> Result<(), SmartsMatchError> {
    match tree {
        BondExprTree::Primitive(primitive) => ensure_supported_bond_primitive(*primitive),
        BondExprTree::Not(inner) => ensure_supported_bond_tree(inner),
        BondExprTree::HighAnd(items) | BondExprTree::Or(items) | BondExprTree::LowAnd(items) => {
            for item in items {
                ensure_supported_bond_tree(item)?;
            }
            Ok(())
        }
    }
}

const fn ensure_supported_bond_primitive(primitive: BondPrimitive) -> Result<(), SmartsMatchError> {
    match primitive {
        BondPrimitive::Bond(
            Bond::Single | Bond::Double | Bond::Triple | Bond::Aromatic | Bond::Up | Bond::Down,
        )
        | BondPrimitive::Any
        | BondPrimitive::Ring => Ok(()),
        BondPrimitive::Bond(Bond::Quadruple) => Err(unsupported_bond_primitive("$")),
    }
}

const fn unsupported_bond_primitive(primitive: &'static str) -> SmartsMatchError {
    SmartsMatchError::UnsupportedBondPrimitive { primitive }
}

fn ensure_supported_atom_expr(
    expr: &AtomExpr,
    stereo_plan: &QueryStereoPlan,
) -> Result<(), SmartsMatchError> {
    match expr {
        AtomExpr::Wildcard | AtomExpr::Bare { .. } => Ok(()),
        AtomExpr::Bracket(expr) => ensure_supported_bracket_tree(&expr.tree, stereo_plan),
    }
}

fn ensure_supported_bracket_tree(
    tree: &BracketExprTree,
    stereo_plan: &QueryStereoPlan,
) -> Result<(), SmartsMatchError> {
    match tree {
        BracketExprTree::Primitive(primitive) => {
            ensure_supported_atom_primitive(primitive, stereo_plan)
        }
        BracketExprTree::Not(inner) => ensure_supported_bracket_tree(inner, stereo_plan),
        BracketExprTree::HighAnd(items)
        | BracketExprTree::Or(items)
        | BracketExprTree::LowAnd(items) => {
            for item in items {
                ensure_supported_bracket_tree(item, stereo_plan)?;
            }
            Ok(())
        }
    }
}

fn ensure_supported_atom_primitive(
    primitive: &AtomPrimitive,
    stereo_plan: &QueryStereoPlan,
) -> Result<(), SmartsMatchError> {
    match primitive {
        AtomPrimitive::Wildcard
        | AtomPrimitive::AliphaticAny
        | AtomPrimitive::AromaticAny
        | AtomPrimitive::Symbol { .. }
        | AtomPrimitive::Isotope { .. }
        | AtomPrimitive::IsotopeWildcard(_)
        | AtomPrimitive::Degree(_)
        | AtomPrimitive::Connectivity(_)
        | AtomPrimitive::Valence(_)
        | AtomPrimitive::Hybridization(_)
        | AtomPrimitive::Hydrogen(_, _)
        | AtomPrimitive::RingMembership(_)
        | AtomPrimitive::RingSize(_)
        | AtomPrimitive::RingConnectivity(_)
        | AtomPrimitive::HeteroNeighbor(_)
        | AtomPrimitive::AliphaticHeteroNeighbor(_)
        | AtomPrimitive::AtomicNumber(_)
        | AtomPrimitive::Charge(_)
        | AtomPrimitive::Chirality(
            Chirality::At
            | Chirality::AtAt
            | Chirality::TH(1 | 2)
            | Chirality::AL(1 | 2)
            | Chirality::SP(1..=3)
            | Chirality::TB(1..=20)
            | Chirality::OH(1..=30),
        ) => Ok(()),
        AtomPrimitive::RecursiveQuery(query) => {
            let query_neighbors = build_query_neighbors(query);
            let recursive_stereo_plan = build_query_stereo_plan(query, &query_neighbors)?;
            ensure_supported_query(query, &recursive_stereo_plan)
        }
        AtomPrimitive::Chirality(_) => {
            let _ = stereo_plan;
            Err(unsupported_atom_primitive("@"))
        }
    }
}

const fn unsupported_atom_primitive(primitive: &'static str) -> SmartsMatchError {
    SmartsMatchError::UnsupportedAtomPrimitive { primitive }
}

fn compile_query_parts(query: &QueryMol) -> Result<CompiledQueryParts, SmartsMatchError> {
    let query_neighbors = build_query_neighbors(query);
    let stereo_plan = build_query_stereo_plan(query, &query_neighbors)?;
    ensure_supported_query(query, &stereo_plan)?;
    let query_degrees = query_neighbors
        .iter()
        .map(alloc::vec::Vec::len)
        .collect::<alloc::vec::Vec<_>>();
    let query_atom_scores = build_query_atom_scores(query, &query_neighbors);
    Ok((
        query_neighbors,
        query_degrees,
        query_atom_scores,
        stereo_plan,
    ))
}

fn compile_atom_matchers(query: &QueryMol) -> Box<[CompiledAtomMatcher]> {
    query
        .atoms()
        .iter()
        .map(|atom| compile_atom_matcher(&atom.expr))
        .collect()
}

fn compile_atom_matcher(expr: &AtomExpr) -> CompiledAtomMatcher {
    if let Some(predicates) = compile_complete_atom_predicates(expr) {
        return compile_predicate_atom_matcher(predicates, true);
    }

    compile_predicate_atom_matcher(compile_required_atom_predicates(expr), false)
}

fn compile_predicate_atom_matcher(
    predicates: alloc::vec::Vec<AtomFastPredicate>,
    complete: bool,
) -> CompiledAtomMatcher {
    let needs_atom = predicates.iter().any(|predicate| {
        matches!(
            predicate,
            AtomFastPredicate::HasElement
                | AtomFastPredicate::Element(_)
                | AtomFastPredicate::Isotope(_)
                | AtomFastPredicate::AtomicNumber(_)
        )
    });
    let needs_aromatic = predicates
        .iter()
        .any(|predicate| matches!(predicate, AtomFastPredicate::Aromatic(_)));
    CompiledAtomMatcher {
        predicates: predicates.into_boxed_slice(),
        needs_atom,
        needs_aromatic,
        complete,
    }
}

fn compile_complete_atom_predicates(expr: &AtomExpr) -> Option<alloc::vec::Vec<AtomFastPredicate>> {
    match expr {
        AtomExpr::Wildcard => Some(alloc::vec::Vec::new()),
        AtomExpr::Bare { element, aromatic } => Some(alloc::vec![
            AtomFastPredicate::Element(*element),
            AtomFastPredicate::Aromatic(*aromatic),
        ]),
        AtomExpr::Bracket(expr) => compile_complete_bracket_predicates(&expr.tree),
    }
}

fn compile_complete_bracket_predicates(
    tree: &BracketExprTree,
) -> Option<alloc::vec::Vec<AtomFastPredicate>> {
    match tree {
        BracketExprTree::Primitive(primitive) => compile_complete_primitive_predicates(primitive),
        BracketExprTree::Not(_) | BracketExprTree::Or(_) => None,
        BracketExprTree::HighAnd(items) | BracketExprTree::LowAnd(items) => {
            let mut predicates = alloc::vec::Vec::new();
            for item in items {
                predicates.extend(compile_complete_bracket_predicates(item)?);
            }
            Some(predicates)
        }
    }
}

fn compile_complete_primitive_predicates(
    primitive: &AtomPrimitive,
) -> Option<alloc::vec::Vec<AtomFastPredicate>> {
    let predicates = match primitive {
        AtomPrimitive::Wildcard | AtomPrimitive::Chirality(_) => alloc::vec::Vec::new(),
        AtomPrimitive::AliphaticAny => alloc::vec![
            AtomFastPredicate::HasElement,
            AtomFastPredicate::Aromatic(false),
        ],
        AtomPrimitive::AromaticAny => alloc::vec![
            AtomFastPredicate::HasElement,
            AtomFastPredicate::Aromatic(true),
        ],
        AtomPrimitive::Symbol { element, aromatic } => alloc::vec![
            AtomFastPredicate::Element(*element),
            AtomFastPredicate::Aromatic(*aromatic),
        ],
        AtomPrimitive::Isotope { isotope, aromatic } => alloc::vec![
            AtomFastPredicate::Element(isotope.element()),
            AtomFastPredicate::Isotope(Some(isotope.mass_number())),
            AtomFastPredicate::Aromatic(*aromatic),
        ],
        AtomPrimitive::IsotopeWildcard(mass_number) => alloc::vec![AtomFastPredicate::Isotope(
            (*mass_number != 0).then_some(*mass_number),
        )],
        AtomPrimitive::AtomicNumber(atomic_number) => {
            alloc::vec![AtomFastPredicate::AtomicNumber(*atomic_number)]
        }
        AtomPrimitive::Degree(expected) => alloc::vec![AtomFastPredicate::Degree(*expected)],
        AtomPrimitive::Connectivity(expected) => {
            alloc::vec![AtomFastPredicate::Connectivity(*expected)]
        }
        AtomPrimitive::Valence(expected) => alloc::vec![AtomFastPredicate::Valence(*expected)],
        AtomPrimitive::Hybridization(expected) => {
            alloc::vec![AtomFastPredicate::Hybridization(*expected)]
        }
        AtomPrimitive::Hydrogen(HydrogenKind::Total, expected) => {
            alloc::vec![AtomFastPredicate::TotalHydrogen(*expected)]
        }
        AtomPrimitive::Hydrogen(HydrogenKind::Implicit, expected) => {
            alloc::vec![AtomFastPredicate::ImplicitHydrogen(*expected)]
        }
        AtomPrimitive::RingMembership(expected) => {
            alloc::vec![AtomFastPredicate::RingMembership(*expected)]
        }
        AtomPrimitive::RingSize(expected) => alloc::vec![AtomFastPredicate::RingSize(*expected)],
        AtomPrimitive::RingConnectivity(expected) => {
            alloc::vec![AtomFastPredicate::RingConnectivity(*expected)]
        }
        AtomPrimitive::HeteroNeighbor(expected) => {
            alloc::vec![AtomFastPredicate::HeteroNeighbor(*expected)]
        }
        AtomPrimitive::AliphaticHeteroNeighbor(expected) => {
            alloc::vec![AtomFastPredicate::AliphaticHeteroNeighbor(*expected)]
        }
        AtomPrimitive::Charge(expected) => alloc::vec![AtomFastPredicate::Charge(*expected)],
        AtomPrimitive::RecursiveQuery(_) => return None,
    };
    Some(predicates)
}

fn compile_required_atom_predicates(expr: &AtomExpr) -> alloc::vec::Vec<AtomFastPredicate> {
    let mut predicates = alloc::vec::Vec::new();
    match expr {
        AtomExpr::Wildcard => {}
        AtomExpr::Bare { element, aromatic } => {
            predicates.push(AtomFastPredicate::Element(*element));
            predicates.push(AtomFastPredicate::Aromatic(*aromatic));
        }
        AtomExpr::Bracket(expr) => {
            collect_required_bracket_predicates(&expr.tree, &mut predicates);
        }
    }
    predicates
}

fn collect_required_bracket_predicates(
    tree: &BracketExprTree,
    predicates: &mut alloc::vec::Vec<AtomFastPredicate>,
) {
    match tree {
        BracketExprTree::Primitive(primitive) => {
            if let Some(required) = compile_complete_primitive_predicates(primitive) {
                predicates.extend(required);
            }
        }
        BracketExprTree::HighAnd(items) | BracketExprTree::LowAnd(items) => {
            for item in items {
                collect_required_bracket_predicates(item, predicates);
            }
        }
        BracketExprTree::Not(_) | BracketExprTree::Or(_) => {}
    }
}

fn compile_bond_matchers(
    query: &QueryMol,
    stereo_plan: &QueryStereoPlan,
) -> Box<[CompiledBondMatcher]> {
    query
        .bonds()
        .iter()
        .map(|bond| {
            let state_mask = if bond_expr_contains_negated_directional_primitive(&bond.expr) {
                0
            } else if query_bond_is_directional(stereo_plan, bond.id) {
                single_like_bond_state_mask()
            } else {
                bond_expr_state_mask(&bond.expr)
            };
            CompiledBondMatcher {
                state_mask,
                ring_sensitive: bond_state_mask_is_ring_sensitive(state_mask),
            }
        })
        .collect()
}

const BOND_LABEL_STATE_COUNT: usize = 7;
const BOND_STATE_MASK_ALL: u16 = (1u16 << (BOND_LABEL_STATE_COUNT * 2)) - 1;

const fn bond_label_state_index(label: BondLabel) -> usize {
    match label {
        BondLabel::Single => 0,
        BondLabel::Double => 1,
        BondLabel::Triple => 2,
        BondLabel::Aromatic => 3,
        BondLabel::Up => 4,
        BondLabel::Down => 5,
        BondLabel::Any => 6,
    }
}

const fn bond_state_bit(label: BondLabel, ring: bool) -> u16 {
    let offset = if ring { BOND_LABEL_STATE_COUNT } else { 0 };
    1u16 << (bond_label_state_index(label) + offset)
}

const fn bond_label_state_mask(label: BondLabel) -> u16 {
    bond_state_bit(label, false) | bond_state_bit(label, true)
}

const fn single_like_bond_state_mask() -> u16 {
    bond_label_state_mask(BondLabel::Single)
        | bond_label_state_mask(BondLabel::Up)
        | bond_label_state_mask(BondLabel::Down)
}

const fn ring_bond_state_mask() -> u16 {
    bond_state_bit(BondLabel::Single, true)
        | bond_state_bit(BondLabel::Double, true)
        | bond_state_bit(BondLabel::Triple, true)
        | bond_state_bit(BondLabel::Aromatic, true)
        | bond_state_bit(BondLabel::Up, true)
        | bond_state_bit(BondLabel::Down, true)
        | bond_state_bit(BondLabel::Any, true)
}

const fn bond_state_mask_is_ring_sensitive(mask: u16) -> bool {
    let mut index = 0usize;
    while index < BOND_LABEL_STATE_COUNT {
        let non_ring = (mask >> index) & 1;
        let ring = (mask >> (index + BOND_LABEL_STATE_COUNT)) & 1;
        if non_ring != ring {
            return true;
        }
        index += 1;
    }
    false
}

fn bond_expr_state_mask(expr: &BondExpr) -> u16 {
    match expr {
        BondExpr::Elided => {
            single_like_bond_state_mask() | bond_label_state_mask(BondLabel::Aromatic)
        }
        BondExpr::Query(tree) => bond_tree_state_mask(tree),
    }
}

fn bond_tree_state_mask(tree: &BondExprTree) -> u16 {
    match tree {
        BondExprTree::Primitive(primitive) => bond_primitive_state_mask(*primitive),
        BondExprTree::Not(inner) => !bond_tree_state_mask(inner) & BOND_STATE_MASK_ALL,
        BondExprTree::HighAnd(items) | BondExprTree::LowAnd(items) => {
            items.iter().fold(BOND_STATE_MASK_ALL, |mask, item| {
                mask & bond_tree_state_mask(item)
            })
        }
        BondExprTree::Or(items) => items
            .iter()
            .fold(0u16, |mask, item| mask | bond_tree_state_mask(item)),
    }
}

const fn bond_primitive_state_mask(primitive: BondPrimitive) -> u16 {
    match primitive {
        BondPrimitive::Bond(Bond::Single | Bond::Up | Bond::Down) => single_like_bond_state_mask(),
        BondPrimitive::Bond(Bond::Double) => bond_label_state_mask(BondLabel::Double),
        BondPrimitive::Bond(Bond::Triple) => bond_label_state_mask(BondLabel::Triple),
        BondPrimitive::Bond(Bond::Aromatic) => bond_label_state_mask(BondLabel::Aromatic),
        BondPrimitive::Bond(Bond::Quadruple) => 0,
        BondPrimitive::Any => BOND_STATE_MASK_ALL,
        BondPrimitive::Ring => ring_bond_state_mask(),
    }
}

fn build_query_atom_scores(
    query: &QueryMol,
    _query_neighbors: &[alloc::vec::Vec<(QueryAtomId, usize)>],
) -> QueryAtomScores {
    (0..query.atom_count())
        .map(|atom_id| {
            let atom = query
                .atom(atom_id)
                .expect("query atom id must stay in range");
            atom_expr_order_score(&atom.expr)
        })
        .collect()
}

fn atom_expr_order_score(expr: &AtomExpr) -> usize {
    match expr {
        AtomExpr::Wildcard => 0,
        AtomExpr::Bare { .. } => 12,
        AtomExpr::Bracket(expr) => bracket_expr_order_score(&expr.tree),
    }
}

fn bracket_expr_order_score(tree: &BracketExprTree) -> usize {
    match tree {
        BracketExprTree::Primitive(primitive) => atom_primitive_order_score(primitive),
        BracketExprTree::Not(_) => 0,
        BracketExprTree::Or(items) => items
            .iter()
            .map(bracket_expr_order_score)
            .min()
            .unwrap_or_default(),
        BracketExprTree::HighAnd(items) | BracketExprTree::LowAnd(items) => {
            items.iter().map(bracket_expr_order_score).sum()
        }
    }
}

const fn atom_primitive_order_score(primitive: &AtomPrimitive) -> usize {
    match primitive {
        AtomPrimitive::Symbol { .. }
        | AtomPrimitive::Isotope { .. }
        | AtomPrimitive::AtomicNumber(_) => 10,
        AtomPrimitive::AromaticAny => 4,
        AtomPrimitive::RingMembership(_)
        | AtomPrimitive::RingSize(_)
        | AtomPrimitive::RingConnectivity(_) => 3,
        AtomPrimitive::Hybridization(_)
        | AtomPrimitive::HeteroNeighbor(_)
        | AtomPrimitive::AliphaticHeteroNeighbor(_)
        | AtomPrimitive::Degree(_)
        | AtomPrimitive::Connectivity(_)
        | AtomPrimitive::Valence(_)
        | AtomPrimitive::Hydrogen(_, _)
        | AtomPrimitive::Chirality(_)
        | AtomPrimitive::Charge(_)
        | AtomPrimitive::RecursiveQuery(_) => 2,
        AtomPrimitive::AliphaticAny => 1,
        AtomPrimitive::Wildcard | AtomPrimitive::IsotopeWildcard(_) => 0,
    }
}

fn compile_recursive_queries(
    query: &QueryMol,
    next_recursive_cache_slot: &mut usize,
) -> Result<Box<[RecursiveQueryEntry]>, SmartsMatchError> {
    let mut entries = alloc::vec::Vec::new();
    for atom in query.atoms() {
        collect_recursive_queries_from_atom_expr(
            &atom.expr,
            &mut entries,
            next_recursive_cache_slot,
        )?;
    }
    Ok(entries.into_boxed_slice())
}

fn collect_recursive_queries_from_atom_expr(
    expr: &AtomExpr,
    entries: &mut alloc::vec::Vec<RecursiveQueryEntry>,
    next_recursive_cache_slot: &mut usize,
) -> Result<(), SmartsMatchError> {
    match expr {
        AtomExpr::Wildcard | AtomExpr::Bare { .. } => Ok(()),
        AtomExpr::Bracket(expr) => collect_recursive_queries_from_bracket_tree(
            &expr.tree,
            entries,
            next_recursive_cache_slot,
        ),
    }
}

fn collect_recursive_queries_from_bracket_tree(
    tree: &BracketExprTree,
    entries: &mut alloc::vec::Vec<RecursiveQueryEntry>,
    next_recursive_cache_slot: &mut usize,
) -> Result<(), SmartsMatchError> {
    match tree {
        BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(query)) => {
            let query_key = core::ptr::from_ref(query.as_ref()) as usize;
            if entries.iter().all(|entry| entry.query_key != query_key) {
                let cache_slot = *next_recursive_cache_slot;
                *next_recursive_cache_slot += 1;
                entries.push(RecursiveQueryEntry {
                    query_key,
                    cache_slot,
                    compiled: Box::new(CompiledQuery::new_with_recursive_slots(
                        query.as_ref().clone(),
                        next_recursive_cache_slot,
                    )?),
                });
            }
            Ok(())
        }
        BracketExprTree::Primitive(_) => Ok(()),
        BracketExprTree::Not(inner) => {
            collect_recursive_queries_from_bracket_tree(inner, entries, next_recursive_cache_slot)
        }
        BracketExprTree::HighAnd(items)
        | BracketExprTree::Or(items)
        | BracketExprTree::LowAnd(items) => {
            for item in items {
                collect_recursive_queries_from_bracket_tree(
                    item,
                    entries,
                    next_recursive_cache_slot,
                )?;
            }
            Ok(())
        }
    }
}

fn query_matches(query: &CompiledQuery, target: &PreparedTarget) -> bool {
    let mut atom_stereo_cache = AtomStereoCache::new();
    let mut recursive_cache =
        RecursiveMatchCache::new(query.recursive_cache_slots, target.atom_count());
    query_matches_with_mapping(
        query,
        target,
        &mut atom_stereo_cache,
        &mut recursive_cache,
        None,
        None,
    )
}

fn query_matches_with_scratch(
    query: &CompiledQuery,
    target: &PreparedTarget,
    scratch: &mut MatchScratch,
) -> bool {
    scratch.prepare(query, target);
    query_matches_with_mapping_state(
        query,
        target,
        &mut scratch.atom_stereo_cache,
        &mut scratch.recursive_cache,
        SearchScratchView {
            query_atom_anchor_widths: &scratch.query_atom_anchor_widths,
            query_to_target: &mut scratch.query_to_target,
            used_target_atoms: &mut scratch.used_target_atoms,
            used_target_generation: scratch.used_target_generation,
        },
        InitialAtomMapping::default(),
    )
}

fn query_substructure_matches(
    query: &CompiledQuery,
    target: &PreparedTarget,
) -> Box<[Box<[usize]>]> {
    let mut atom_stereo_cache = AtomStereoCache::new();
    let mut recursive_cache =
        RecursiveMatchCache::new(query.recursive_cache_slots, target.atom_count());
    query_substructure_matches_with_mapping(
        query,
        target,
        &mut atom_stereo_cache,
        &mut recursive_cache,
        None,
        None,
    )
}

fn query_match_count(query: &CompiledQuery, target: &PreparedTarget) -> usize {
    let mut atom_stereo_cache = AtomStereoCache::new();
    let mut recursive_cache =
        RecursiveMatchCache::new(query.recursive_cache_slots, target.atom_count());
    query_match_count_with_mapping(
        query,
        target,
        &mut atom_stereo_cache,
        &mut recursive_cache,
        None,
        None,
    )
}

fn query_matches_with_mapping(
    query: &CompiledQuery,
    target: &PreparedTarget,
    atom_stereo_cache: &mut AtomStereoCache,
    recursive_cache: &mut RecursiveMatchCache,
    initial_query_atom: Option<QueryAtomId>,
    initial_target_atom: Option<usize>,
) -> bool {
    let initial_mapping = InitialAtomMapping::new(initial_query_atom, initial_target_atom);
    if query.query.atom_count() == 1 {
        return single_atom_query_matches(
            query,
            target,
            recursive_cache,
            initial_mapping.target_atom,
        );
    }
    if is_two_atom_edge_query(&query.query) {
        return two_atom_query_matches(query, target, recursive_cache, initial_mapping);
    }
    if let Some(layout) = three_atom_tree_layout(query) {
        return three_atom_query_matches(
            query,
            target,
            atom_stereo_cache,
            recursive_cache,
            layout,
            initial_mapping,
        );
    }

    let mut scratch = FreshSearchBuffers::new(query, target);
    query_matches_with_mapping_state(
        query,
        target,
        atom_stereo_cache,
        recursive_cache,
        scratch.view(),
        initial_mapping,
    )
}

fn query_matches_with_mapping_state(
    query: &CompiledQuery,
    target: &PreparedTarget,
    atom_stereo_cache: &mut AtomStereoCache,
    recursive_cache: &mut RecursiveMatchCache,
    scratch: SearchScratchView<'_>,
    initial_mapping: InitialAtomMapping,
) -> bool {
    let mut context = scratch.into_context(query, atom_stereo_cache, recursive_cache);
    let Some(mapped_count) = bind_initial_mapping(query, target, &mut context, initial_mapping)
    else {
        return false;
    };

    search_mapping(&query.query, target, &mut context, mapped_count)
}

fn with_fresh_bound_search_context<R>(
    query: &CompiledQuery,
    target: &PreparedTarget,
    atom_stereo_cache: &mut AtomStereoCache,
    recursive_cache: &mut RecursiveMatchCache,
    initial_mapping: InitialAtomMapping,
    run: impl FnOnce(&mut SearchContext<'_>, usize) -> R,
) -> Option<R> {
    let mut scratch = FreshSearchBuffers::new(query, target);
    let mut context = scratch
        .view()
        .into_context(query, atom_stereo_cache, recursive_cache);
    let mapped_count = bind_initial_mapping(query, target, &mut context, initial_mapping)?;
    Some(run(&mut context, mapped_count))
}

fn bind_initial_mapping(
    query: &CompiledQuery,
    target: &PreparedTarget,
    context: &mut SearchContext<'_>,
    initial_mapping: InitialAtomMapping,
) -> Option<usize> {
    let Some((query_atom, target_atom)) = initial_mapping.pair() else {
        return Some(0);
    };
    if target
        .degree(target_atom)
        .is_some_and(|degree| degree < query.query_degrees[query_atom])
    {
        return None;
    }
    if query.has_component_constraints
        && !component_constraints_match(
            query_atom,
            target_atom,
            &query.query,
            target,
            context.query_to_target,
        )
    {
        return None;
    }
    if !compiled_query_atom_matches(
        query,
        target,
        context.recursive_cache,
        query_atom,
        target_atom,
    ) {
        return None;
    }
    context.query_to_target[query_atom] = Some(target_atom);
    context.mark_target_atom_used(target_atom);
    Some(1)
}

fn compiled_query_atom_matches(
    query: &CompiledQuery,
    target: &PreparedTarget,
    recursive_cache: &mut RecursiveMatchCache,
    query_atom: QueryAtomId,
    target_atom: usize,
) -> bool {
    let mut context = AtomMatchContext::new(query, target, recursive_cache);
    cached_query_atom_matches(
        &query.query.atoms()[query_atom].expr,
        &query.atom_matchers[query_atom],
        query_atom,
        target_atom,
        &mut context,
    )
}

fn query_substructure_matches_with_mapping(
    query: &CompiledQuery,
    target: &PreparedTarget,
    atom_stereo_cache: &mut AtomStereoCache,
    recursive_cache: &mut RecursiveMatchCache,
    initial_query_atom: Option<QueryAtomId>,
    initial_target_atom: Option<usize>,
) -> Box<[Box<[usize]>]> {
    let initial_mapping = InitialAtomMapping::new(initial_query_atom, initial_target_atom);
    if query.query.atom_count() == 1 {
        return single_atom_query_substructure_matches(
            query,
            target,
            recursive_cache,
            initial_mapping.target_atom,
        );
    }
    if is_two_atom_edge_query(&query.query) {
        return two_atom_query_substructure_matches(
            query,
            target,
            recursive_cache,
            initial_mapping,
        );
    }
    if let Some(layout) = three_atom_tree_layout(query) {
        return three_atom_query_substructure_matches(
            query,
            target,
            atom_stereo_cache,
            recursive_cache,
            layout,
            initial_mapping,
        );
    }

    with_fresh_bound_search_context(
        query,
        target,
        atom_stereo_cache,
        recursive_cache,
        initial_mapping,
        |context, mapped_count| {
            let mut matches = UniqueMatches::new();
            collect_mappings(&query.query, target, context, mapped_count, &mut matches);
            matches
                .into_values()
                .collect::<alloc::vec::Vec<_>>()
                .into_boxed_slice()
        },
    )
    .unwrap_or_default()
}

fn query_match_count_with_mapping(
    query: &CompiledQuery,
    target: &PreparedTarget,
    atom_stereo_cache: &mut AtomStereoCache,
    recursive_cache: &mut RecursiveMatchCache,
    initial_query_atom: Option<QueryAtomId>,
    initial_target_atom: Option<usize>,
) -> usize {
    let initial_mapping = InitialAtomMapping::new(initial_query_atom, initial_target_atom);
    if query.query.atom_count() == 1 {
        return single_atom_query_match_count(
            query,
            target,
            recursive_cache,
            initial_mapping.target_atom,
        );
    }
    if is_two_atom_edge_query(&query.query) {
        return two_atom_query_match_count(query, target, recursive_cache, initial_mapping);
    }
    if let Some(layout) = three_atom_tree_layout(query) {
        return three_atom_query_match_count(
            query,
            target,
            atom_stereo_cache,
            recursive_cache,
            layout,
            initial_mapping,
        );
    }

    with_fresh_bound_search_context(
        query,
        target,
        atom_stereo_cache,
        recursive_cache,
        initial_mapping,
        |context, mapped_count| {
            let mut matches = MatchKeys::new();
            collect_match_keys(&query.query, target, context, mapped_count, &mut matches);
            matches.sort_unstable();
            matches.dedup();
            matches.len()
        },
    )
    .unwrap_or(0)
}

fn single_atom_query_matches(
    query: &CompiledQuery,
    target: &PreparedTarget,
    recursive_cache: &mut RecursiveMatchCache,
    initial_target_atom: Option<usize>,
) -> bool {
    let mut anchored_target_atoms = alloc::vec::Vec::new();
    single_atom_candidates(
        &query.atom_matchers[0],
        target,
        initial_target_atom,
        &mut anchored_target_atoms,
    )
    .any(|target_atom| compiled_query_atom_matches(query, target, recursive_cache, 0, target_atom))
}

fn single_atom_query_substructure_matches(
    query: &CompiledQuery,
    target: &PreparedTarget,
    recursive_cache: &mut RecursiveMatchCache,
    initial_target_atom: Option<usize>,
) -> Box<[Box<[usize]>]> {
    let mut anchored_target_atoms = alloc::vec::Vec::new();
    single_atom_candidates(
        &query.atom_matchers[0],
        target,
        initial_target_atom,
        &mut anchored_target_atoms,
    )
    .collect_matches(|target_atom| {
        compiled_query_atom_matches(query, target, recursive_cache, 0, target_atom)
    })
}

fn single_atom_query_match_count(
    query: &CompiledQuery,
    target: &PreparedTarget,
    recursive_cache: &mut RecursiveMatchCache,
    initial_target_atom: Option<usize>,
) -> usize {
    let mut anchored_target_atoms = alloc::vec::Vec::new();
    single_atom_candidates(
        &query.atom_matchers[0],
        target,
        initial_target_atom,
        &mut anchored_target_atoms,
    )
    .count(|target_atom| {
        compiled_query_atom_matches(query, target, recursive_cache, 0, target_atom)
    })
}

fn is_two_atom_edge_query(query: &QueryMol) -> bool {
    query.atom_count() == 2
        && query.bond_count() == 1
        && query.atoms()[query.bonds()[0].src].component
            == query.atoms()[query.bonds()[0].dst].component
}

fn three_atom_tree_layout(query: &CompiledQuery) -> Option<ThreeAtomTreeLayout> {
    if query.query.atom_count() != 3 || query.query.bond_count() != 2 {
        return None;
    }

    let mut center = None;
    for (atom_id, degree) in query.query_degrees.iter().copied().enumerate() {
        match degree {
            2 => {
                if center.is_some() {
                    return None;
                }
                center = Some(atom_id);
            }
            1 => {}
            _ => return None,
        }
    }
    let center = center?;
    let component = query.query.atoms()[center].component;
    if query
        .query
        .atoms()
        .iter()
        .any(|atom| atom.component != component)
    {
        return None;
    }

    let neighbors = &query.query_neighbors[center];
    if neighbors.len() != 2 {
        return None;
    }

    Some(ThreeAtomTreeLayout {
        center,
        leaf_a: neighbors[0].0,
        leaf_b: neighbors[1].0,
        bond_a: neighbors[0].1,
        bond_b: neighbors[1].1,
    })
}

fn two_atom_query_matches(
    query: &CompiledQuery,
    target: &PreparedTarget,
    recursive_cache: &mut RecursiveMatchCache,
    initial_mapping: InitialAtomMapping,
) -> bool {
    let bond = &query.query.bonds()[0];
    let query_mapping = TwoAtomQueryMapping {
        left: bond.src,
        right: bond.dst,
    };
    for_each_target_bond(target, |left_atom, right_atom, bond_label| {
        mapping_matches_two_atom_query(
            query,
            target,
            recursive_cache,
            query_mapping,
            TwoAtomTargetMapping {
                left: left_atom,
                right: right_atom,
                bond: bond_label,
            },
            initial_mapping,
        ) || mapping_matches_two_atom_query(
            query,
            target,
            recursive_cache,
            query_mapping,
            TwoAtomTargetMapping {
                left: right_atom,
                right: left_atom,
                bond: bond_label,
            },
            initial_mapping,
        )
    })
}

fn two_atom_query_substructure_matches(
    query: &CompiledQuery,
    target: &PreparedTarget,
    recursive_cache: &mut RecursiveMatchCache,
    initial_mapping: InitialAtomMapping,
) -> Box<[Box<[usize]>]> {
    let bond = &query.query.bonds()[0];
    let query_mapping = TwoAtomQueryMapping {
        left: bond.src,
        right: bond.dst,
    };
    let mut matches = UniqueMatches::new();
    for_each_target_bond(target, |left_atom, right_atom, bond_label| {
        collect_two_atom_mapping(
            &mut matches,
            query,
            target,
            recursive_cache,
            query_mapping,
            TwoAtomTargetMapping {
                left: left_atom,
                right: right_atom,
                bond: bond_label,
            },
            initial_mapping,
        );
        collect_two_atom_mapping(
            &mut matches,
            query,
            target,
            recursive_cache,
            query_mapping,
            TwoAtomTargetMapping {
                left: right_atom,
                right: left_atom,
                bond: bond_label,
            },
            initial_mapping,
        );
        false
    });
    matches
        .into_values()
        .collect::<alloc::vec::Vec<_>>()
        .into_boxed_slice()
}

fn two_atom_query_match_count(
    query: &CompiledQuery,
    target: &PreparedTarget,
    recursive_cache: &mut RecursiveMatchCache,
    initial_mapping: InitialAtomMapping,
) -> usize {
    let bond = &query.query.bonds()[0];
    let query_mapping = TwoAtomQueryMapping {
        left: bond.src,
        right: bond.dst,
    };
    let mut matches = MatchKeys::new();
    for_each_target_bond(target, |left_atom, right_atom, bond_label| {
        collect_two_atom_match_key(
            &mut matches,
            query,
            target,
            recursive_cache,
            query_mapping,
            TwoAtomTargetMapping {
                left: left_atom,
                right: right_atom,
                bond: bond_label,
            },
            initial_mapping,
        );
        collect_two_atom_match_key(
            &mut matches,
            query,
            target,
            recursive_cache,
            query_mapping,
            TwoAtomTargetMapping {
                left: right_atom,
                right: left_atom,
                bond: bond_label,
            },
            initial_mapping,
        );
        false
    });
    matches.sort_unstable();
    matches.dedup();
    matches.len()
}

fn three_atom_query_matches(
    query: &CompiledQuery,
    target: &PreparedTarget,
    atom_stereo_cache: &mut AtomStereoCache,
    recursive_cache: &mut RecursiveMatchCache,
    layout: ThreeAtomTreeLayout,
    initial_mapping: InitialAtomMapping,
) -> bool {
    for_each_three_atom_mapping(
        query,
        target,
        recursive_cache,
        layout,
        initial_mapping,
        |mapping| {
            stereo_constraints_match(
                &query.query,
                &query.stereo_plan,
                &query.query_neighbors,
                target,
                mapping,
                atom_stereo_cache,
            )
        },
    )
}

fn three_atom_query_substructure_matches(
    query: &CompiledQuery,
    target: &PreparedTarget,
    atom_stereo_cache: &mut AtomStereoCache,
    recursive_cache: &mut RecursiveMatchCache,
    layout: ThreeAtomTreeLayout,
    initial_mapping: InitialAtomMapping,
) -> Box<[Box<[usize]>]> {
    let mut matches = UniqueMatches::new();
    for_each_three_atom_mapping(
        query,
        target,
        recursive_cache,
        layout,
        initial_mapping,
        |mapping| {
            if !stereo_constraints_match(
                &query.query,
                &query.stereo_plan,
                &query.query_neighbors,
                target,
                mapping,
                atom_stereo_cache,
            ) {
                return false;
            }

            let ordered = current_query_order_mapping(mapping);
            matches
                .entry(three_atom_canonical_key(
                    mapping[0].expect("three-atom fast path must bind every query atom"),
                    mapping[1].expect("three-atom fast path must bind every query atom"),
                    mapping[2].expect("three-atom fast path must bind every query atom"),
                ))
                .and_modify(|existing| {
                    if ordered < *existing {
                        existing.clone_from(&ordered);
                    }
                })
                .or_insert(ordered);
            false
        },
    );
    matches
        .into_values()
        .collect::<alloc::vec::Vec<_>>()
        .into_boxed_slice()
}

fn three_atom_query_match_count(
    query: &CompiledQuery,
    target: &PreparedTarget,
    atom_stereo_cache: &mut AtomStereoCache,
    recursive_cache: &mut RecursiveMatchCache,
    layout: ThreeAtomTreeLayout,
    initial_mapping: InitialAtomMapping,
) -> usize {
    let mut matches = MatchKeys::new();
    for_each_three_atom_mapping(
        query,
        target,
        recursive_cache,
        layout,
        initial_mapping,
        |mapping| {
            if !stereo_constraints_match(
                &query.query,
                &query.stereo_plan,
                &query.query_neighbors,
                target,
                mapping,
                atom_stereo_cache,
            ) {
                return false;
            }
            matches.push(three_atom_canonical_key(
                mapping[0].expect("three-atom fast path must bind every query atom"),
                mapping[1].expect("three-atom fast path must bind every query atom"),
                mapping[2].expect("three-atom fast path must bind every query atom"),
            ));
            false
        },
    );
    matches.sort_unstable();
    matches.dedup();
    matches.len()
}

fn for_each_three_atom_mapping(
    query: &CompiledQuery,
    target: &PreparedTarget,
    recursive_cache: &mut RecursiveMatchCache,
    layout: ThreeAtomTreeLayout,
    initial_mapping: InitialAtomMapping,
    mut visit: impl FnMut(&[Option<usize>]) -> bool,
) -> bool {
    if let Some((query_atom, target_atom)) = initial_mapping.pair() {
        if query_atom == layout.center {
            return for_each_three_atom_center_mapping(
                ThreeAtomCenterSearch::new(
                    query,
                    target,
                    recursive_cache,
                    layout,
                    target_atom,
                    initial_mapping,
                ),
                FixedThreeAtomLeaves::default(),
                visit,
            );
        }
        if query_atom == layout.leaf_a {
            for (target_center, target_bond_a) in target.neighbors(target_atom) {
                if for_each_three_atom_center_mapping(
                    ThreeAtomCenterSearch::new(
                        query,
                        target,
                        recursive_cache,
                        layout,
                        target_center,
                        initial_mapping,
                    ),
                    FixedThreeAtomLeaves {
                        leaf_a: Some((target_atom, target_bond_a)),
                        leaf_b: None,
                    },
                    &mut visit,
                ) {
                    return true;
                }
            }
            return false;
        }
        if query_atom == layout.leaf_b {
            for (target_center, target_bond_b) in target.neighbors(target_atom) {
                if for_each_three_atom_center_mapping(
                    ThreeAtomCenterSearch::new(
                        query,
                        target,
                        recursive_cache,
                        layout,
                        target_center,
                        initial_mapping,
                    ),
                    FixedThreeAtomLeaves {
                        leaf_a: None,
                        leaf_b: Some((target_atom, target_bond_b)),
                    },
                    &mut visit,
                ) {
                    return true;
                }
            }
            return false;
        }
    }

    for target_center in 0..target.atom_count() {
        if for_each_three_atom_center_mapping(
            ThreeAtomCenterSearch::new(
                query,
                target,
                recursive_cache,
                layout,
                target_center,
                initial_mapping,
            ),
            FixedThreeAtomLeaves::default(),
            &mut visit,
        ) {
            return true;
        }
    }
    false
}

fn for_each_three_atom_center_mapping(
    mut search: ThreeAtomCenterSearch<'_, '_>,
    fixed_leaves: FixedThreeAtomLeaves,
    mut visit: impl FnMut(&[Option<usize>]) -> bool,
) -> bool {
    if !search.center_matches() {
        return false;
    }

    match (fixed_leaves.leaf_a, fixed_leaves.leaf_b) {
        (Some(leaf_a), None) => for_each_three_atom_fixed_leaf_a(&mut search, leaf_a, &mut visit),
        (None, Some(leaf_b)) => for_each_three_atom_fixed_leaf_b(&mut search, leaf_b, &mut visit),
        (None, None) => for_each_three_atom_unfixed_leaves(&mut search, &mut visit),
        (Some(_), Some(_)) => false,
    }
}

fn for_each_three_atom_fixed_leaf_a(
    search: &mut ThreeAtomCenterSearch<'_, '_>,
    (target_leaf_a, target_bond_a): (usize, BondLabel),
    visit: &mut impl FnMut(&[Option<usize>]) -> bool,
) -> bool {
    for (target_leaf_b, target_bond_b) in search.target.neighbors(search.target_center) {
        if target_leaf_b == target_leaf_a {
            continue;
        }
        let target_mapping =
            search.target_mapping(target_leaf_a, target_bond_a, target_leaf_b, target_bond_b);
        if visit_three_atom_mapping(search, target_mapping, visit) {
            return true;
        }
    }
    false
}

fn for_each_three_atom_fixed_leaf_b(
    search: &mut ThreeAtomCenterSearch<'_, '_>,
    (target_leaf_b, target_bond_b): (usize, BondLabel),
    visit: &mut impl FnMut(&[Option<usize>]) -> bool,
) -> bool {
    for (target_leaf_a, target_bond_a) in search.target.neighbors(search.target_center) {
        if target_leaf_a == target_leaf_b {
            continue;
        }
        let target_mapping =
            search.target_mapping(target_leaf_a, target_bond_a, target_leaf_b, target_bond_b);
        if visit_three_atom_mapping(search, target_mapping, visit) {
            return true;
        }
    }
    false
}

fn for_each_three_atom_unfixed_leaves(
    search: &mut ThreeAtomCenterSearch<'_, '_>,
    visit: &mut impl FnMut(&[Option<usize>]) -> bool,
) -> bool {
    let neighbors = search
        .target
        .neighbors(search.target_center)
        .collect::<alloc::vec::Vec<_>>();
    let mut left_index = 0usize;
    while left_index < neighbors.len() {
        let (target_left, left_bond) = neighbors[left_index];
        let mut right_index = left_index + 1;
        while right_index < neighbors.len() {
            let (target_right, right_bond) = neighbors[right_index];
            if visit_three_atom_mapping(
                search,
                search.target_mapping(target_left, left_bond, target_right, right_bond),
                visit,
            ) || visit_three_atom_mapping(
                search,
                search.target_mapping(target_right, right_bond, target_left, left_bond),
                visit,
            ) {
                return true;
            }
            right_index += 1;
        }
        left_index += 1;
    }
    false
}

fn visit_three_atom_mapping(
    search: &mut ThreeAtomCenterSearch<'_, '_>,
    target_mapping: ThreeAtomTargetMapping,
    visit: &mut impl FnMut(&[Option<usize>]) -> bool,
) -> bool {
    search.mapping_matches(target_mapping)
        && visit(&three_atom_query_mapping(
            search.layout,
            target_mapping.center,
            target_mapping.leaf_a,
            target_mapping.leaf_b,
        ))
}

fn three_atom_mapping_matches(
    query: &CompiledQuery,
    target: &PreparedTarget,
    recursive_cache: &mut RecursiveMatchCache,
    layout: ThreeAtomTreeLayout,
    target_mapping: ThreeAtomTargetMapping,
    initial_mapping: InitialAtomMapping,
) -> bool {
    if initial_mapping
        .pair()
        .is_some_and(|(query_atom, target_atom)| {
            (query_atom == layout.center && target_atom != target_mapping.center)
                || (query_atom == layout.leaf_a && target_atom != target_mapping.leaf_a)
                || (query_atom == layout.leaf_b && target_atom != target_mapping.leaf_b)
        })
    {
        return false;
    }

    if target
        .degree(target_mapping.leaf_a)
        .is_some_and(|degree| degree < query.query_degrees[layout.leaf_a])
        || target
            .degree(target_mapping.leaf_b)
            .is_some_and(|degree| degree < query.query_degrees[layout.leaf_b])
    {
        return false;
    }

    if !query_bond_matches(
        query.bond_matchers[layout.bond_a],
        target,
        target_mapping.center,
        target_mapping.leaf_a,
        target_mapping.bond_a,
    ) || !query_bond_matches(
        query.bond_matchers[layout.bond_b],
        target,
        target_mapping.center,
        target_mapping.leaf_b,
        target_mapping.bond_b,
    ) {
        return false;
    }

    if !compiled_query_atom_matches(
        query,
        target,
        recursive_cache,
        layout.leaf_a,
        target_mapping.leaf_a,
    ) {
        return false;
    }
    compiled_query_atom_matches(
        query,
        target,
        recursive_cache,
        layout.leaf_b,
        target_mapping.leaf_b,
    )
}

const fn three_atom_query_mapping(
    layout: ThreeAtomTreeLayout,
    target_center: usize,
    target_leaf_a: usize,
    target_leaf_b: usize,
) -> [Option<usize>; 3] {
    let mut mapping = [None; 3];
    mapping[layout.center] = Some(target_center);
    mapping[layout.leaf_a] = Some(target_leaf_a);
    mapping[layout.leaf_b] = Some(target_leaf_b);
    mapping
}

fn for_each_target_bond(
    target: &PreparedTarget,
    mut visit: impl FnMut(usize, usize, BondLabel) -> bool,
) -> bool {
    for left_atom in 0..target.atom_count() {
        for (right_atom, bond_label) in target.neighbors(left_atom) {
            if right_atom <= left_atom {
                continue;
            }
            if visit(left_atom, right_atom, bond_label) {
                return true;
            }
        }
    }
    false
}

fn mapping_matches_two_atom_query(
    query: &CompiledQuery,
    target: &PreparedTarget,
    recursive_cache: &mut RecursiveMatchCache,
    query_mapping: TwoAtomQueryMapping,
    target_mapping: TwoAtomTargetMapping,
    initial_mapping: InitialAtomMapping,
) -> bool {
    if initial_mapping
        .pair()
        .is_some_and(|(query_atom, target_atom)| {
            (query_atom == query_mapping.left && target_atom != target_mapping.left)
                || (query_atom == query_mapping.right && target_atom != target_mapping.right)
        })
    {
        return false;
    }

    if target
        .degree(target_mapping.left)
        .is_some_and(|degree| degree < query.query_degrees[query_mapping.left])
        || target
            .degree(target_mapping.right)
            .is_some_and(|degree| degree < query.query_degrees[query_mapping.right])
    {
        return false;
    }

    if !query_bond_matches(
        query.bond_matchers[0],
        target,
        target_mapping.left,
        target_mapping.right,
        target_mapping.bond,
    ) {
        return false;
    }

    if !compiled_query_atom_matches(
        query,
        target,
        recursive_cache,
        query_mapping.left,
        target_mapping.left,
    ) {
        return false;
    }
    compiled_query_atom_matches(
        query,
        target,
        recursive_cache,
        query_mapping.right,
        target_mapping.right,
    )
}

fn collect_two_atom_mapping(
    matches: &mut UniqueMatches,
    query: &CompiledQuery,
    target: &PreparedTarget,
    recursive_cache: &mut RecursiveMatchCache,
    query_mapping: TwoAtomQueryMapping,
    target_mapping: TwoAtomTargetMapping,
    initial_mapping: InitialAtomMapping,
) {
    if mapping_matches_two_atom_query(
        query,
        target,
        recursive_cache,
        query_mapping,
        target_mapping,
        initial_mapping,
    ) {
        let mapping = if query_mapping.left == 0 {
            Box::<[usize]>::from([target_mapping.left, target_mapping.right])
        } else {
            Box::<[usize]>::from([target_mapping.right, target_mapping.left])
        };
        matches
            .entry(two_atom_canonical_key(
                target_mapping.left,
                target_mapping.right,
            ))
            .and_modify(|existing| {
                if mapping < *existing {
                    existing.clone_from(&mapping);
                }
            })
            .or_insert(mapping);
    }
}

fn collect_two_atom_match_key(
    matches: &mut MatchKeys,
    query: &CompiledQuery,
    target: &PreparedTarget,
    recursive_cache: &mut RecursiveMatchCache,
    query_mapping: TwoAtomQueryMapping,
    target_mapping: TwoAtomTargetMapping,
    initial_mapping: InitialAtomMapping,
) {
    if mapping_matches_two_atom_query(
        query,
        target,
        recursive_cache,
        query_mapping,
        target_mapping,
        initial_mapping,
    ) {
        matches.push(two_atom_canonical_key(
            target_mapping.left,
            target_mapping.right,
        ));
    }
}

fn build_query_neighbors(
    query: &QueryMol,
) -> alloc::vec::Vec<alloc::vec::Vec<(QueryAtomId, usize)>> {
    let mut neighbors = alloc::vec![alloc::vec::Vec::new(); query.atom_count()];
    for (bond_index, bond) in query.bonds().iter().enumerate() {
        neighbors[bond.src].push((bond.dst, bond_index));
        neighbors[bond.dst].push((bond.src, bond_index));
    }
    neighbors
}

fn build_query_stereo_plan(
    query: &QueryMol,
    query_neighbors: &[alloc::vec::Vec<(QueryAtomId, usize)>],
) -> Result<QueryStereoPlan, SmartsMatchError> {
    let mut directional_bond_ids = alloc::vec![false; query.bond_count()];
    let mut double_bond_constraints = alloc::vec::Vec::new();

    for bond in query.bonds() {
        if !is_simple_double_bond_expr(&bond.expr) {
            continue;
        }

        let left_directional =
            directional_substituents(query, query_neighbors, bond.src, bond.id, bond.dst)?;
        let right_directional =
            directional_substituents(query, query_neighbors, bond.dst, bond.id, bond.src)?;

        if left_directional.is_empty() || right_directional.is_empty() {
            continue;
        }

        let left_local = local_substituents(query, query_neighbors, bond.src, bond.id, bond.dst)?;
        let right_local = local_substituents(query, query_neighbors, bond.dst, bond.id, bond.src)?;
        let (fragment, left_atom_id, right_atom_id) =
            build_local_double_bond_fragment(&left_local, &right_local)?;
        let local_smiles = fragment
            .parse::<Smiles>()
            .map_err(|_| unsupported_bond_primitive("/"))?;
        let config = local_smiles
            .double_bond_stereo_config(left_atom_id, right_atom_id)
            .ok_or_else(|| unsupported_bond_primitive("/"))?;

        for substituent in left_directional.iter().chain(right_directional.iter()) {
            directional_bond_ids[substituent.bond_id] = true;
        }
        double_bond_constraints.push(QueryDoubleBondStereoConstraint {
            left_atom: bond.src,
            right_atom: bond.dst,
            config,
        });
    }

    Ok(QueryStereoPlan {
        directional_bond_ids,
        double_bond_constraints,
    })
}

const fn is_supported_query_chirality(chirality: Chirality) -> bool {
    matches!(
        chirality,
        Chirality::At
            | Chirality::AtAt
            | Chirality::TH(1 | 2)
            | Chirality::AL(1 | 2)
            | Chirality::SP(1..=3)
            | Chirality::TB(1..=20)
            | Chirality::OH(1..=30)
    )
}

const fn query_chirality_requires_tetrahedral_match(chirality: Chirality) -> bool {
    matches!(
        chirality,
        Chirality::At | Chirality::AtAt | Chirality::TH(1 | 2)
    )
}

fn extract_query_chirality(expr: &AtomExpr) -> Option<Chirality> {
    match expr {
        AtomExpr::Wildcard | AtomExpr::Bare { .. } => None,
        AtomExpr::Bracket(expr) => extract_chirality_from_bracket_tree(&expr.tree),
    }
}

fn extract_chirality_from_bracket_tree(tree: &BracketExprTree) -> Option<Chirality> {
    match tree {
        BracketExprTree::Primitive(AtomPrimitive::Chirality(chirality)) => Some(*chirality),
        BracketExprTree::Primitive(_) | BracketExprTree::Not(_) => None,
        BracketExprTree::HighAnd(items)
        | BracketExprTree::Or(items)
        | BracketExprTree::LowAnd(items) => {
            items.iter().find_map(extract_chirality_from_bracket_tree)
        }
    }
}

fn build_local_stereo_fragment(
    query: &QueryMol,
    query_neighbors: &[alloc::vec::Vec<(QueryAtomId, usize)>],
    atom_id: QueryAtomId,
    neighbor_tokens: &[String],
) -> Result<(String, usize), SmartsMatchError> {
    let incident = &query_neighbors[atom_id];
    let Some(query_chirality) = extract_query_chirality(&query.atoms()[atom_id].expr) else {
        return Err(unsupported_atom_primitive("@"));
    };
    let preserve_incoming_neighbor = query_has_total_hydrogen(&query.atoms()[atom_id].expr);
    let prefix_neighbor = preserve_incoming_neighbor
        .then(|| {
            incident
                .iter()
                .enumerate()
                .find_map(|(index, (neighbor_atom, _))| (*neighbor_atom < atom_id).then_some(index))
        })
        .flatten();
    let remaining_count = incident
        .len()
        .saturating_sub(usize::from(prefix_neighbor.is_some()));

    let mut fragment = prefix_neighbor.map_or_else(
        || render_local_stereo_center(query_chirality, preserve_incoming_neighbor),
        |prefix_index| {
            let (_, bond_id) = incident[prefix_index];
            let mut fragment = neighbor_tokens[prefix_index].clone();
            fragment.push_str(local_stereo_bond_text(&query.bonds()[bond_id].expr));
            fragment.push_str(&render_local_stereo_center(
                query_chirality,
                preserve_incoming_neighbor,
            ));
            fragment
        },
    );

    let mut rendered_remaining = 0usize;
    for (index, (_, bond_id)) in incident.iter().enumerate() {
        if Some(index) == prefix_neighbor {
            continue;
        }
        rendered_remaining += 1;
        let atom_text = &neighbor_tokens[index];
        let bond_text = local_stereo_bond_text(&query.bonds()[*bond_id].expr);
        if rendered_remaining == remaining_count {
            fragment.push_str(bond_text);
            fragment.push_str(atom_text);
        } else {
            fragment.push('(');
            fragment.push_str(bond_text);
            fragment.push_str(atom_text);
            fragment.push(')');
        }
    }
    Ok((fragment, usize::from(prefix_neighbor.is_some())))
}

fn local_neighbor_tokens_for_mapping(
    incident: &[(QueryAtomId, usize)],
    query_to_target: &[Option<usize>],
    target: &PreparedTarget,
) -> Result<alloc::vec::Vec<String>, SmartsMatchError> {
    const PLACEHOLDER_ATOMS: [&str; 8] = ["F", "Cl", "Br", "I", "N", "O", "P", "S"];

    let target_atoms_in_incident_order = incident
        .iter()
        .map(|(neighbor_atom, _)| {
            query_to_target[*neighbor_atom].ok_or_else(|| unsupported_atom_primitive("@"))
        })
        .collect::<Result<alloc::vec::Vec<_>, _>>()?;
    let mut target_atoms = target_atoms_in_incident_order.clone();
    target_atoms.sort_unstable();
    target_atoms.dedup();

    if target_atoms.len() > PLACEHOLDER_ATOMS.len() {
        return Err(unsupported_atom_primitive("@"));
    }

    let rendered_atoms = target_atoms
        .iter()
        .map(|target_atom_id| render_target_stereo_atom(target, *target_atom_id))
        .collect::<alloc::vec::Vec<_>>();
    let mut assigned_tokens = alloc::vec::Vec::with_capacity(target_atoms.len());
    let mut placeholder_index = 0usize;

    for (index, target_atom_id) in target_atoms.into_iter().enumerate() {
        let rendered_atom = rendered_atoms[index].clone();
        let rendered_atom_is_unique = rendered_atom.as_ref().is_some_and(|rendered_atom| {
            rendered_atoms
                .iter()
                .filter(|other_rendered_atom| other_rendered_atom.as_ref() == Some(rendered_atom))
                .count()
                == 1
        });

        if rendered_atom_is_unique {
            assigned_tokens.push((
                target_atom_id,
                rendered_atom.expect("checked that the rendered atom exists"),
            ));
            continue;
        }

        while PLACEHOLDER_ATOMS
            .get(placeholder_index)
            .is_some_and(|placeholder_atom| {
                assigned_tokens
                    .iter()
                    .any(|(_, assigned_token)| assigned_token == placeholder_atom)
            })
        {
            placeholder_index += 1;
        }
        let Some(placeholder_atom) = PLACEHOLDER_ATOMS.get(placeholder_index) else {
            return Err(unsupported_atom_primitive("@"));
        };
        assigned_tokens.push((target_atom_id, String::from(*placeholder_atom)));
        placeholder_index += 1;
    }

    target_atoms_in_incident_order
        .into_iter()
        .map(|target_atom_id| {
            assigned_tokens
                .iter()
                .find_map(|(mapped_target_atom_id, placeholder_atom)| {
                    (*mapped_target_atom_id == target_atom_id).then_some(placeholder_atom.clone())
                })
                .ok_or_else(|| unsupported_atom_primitive("@"))
        })
        .collect()
}

fn render_target_stereo_atom(target: &PreparedTarget, atom_id: usize) -> Option<String> {
    let atom = target.atom(atom_id)?;
    let mut symbol = atom.element()?.to_string();
    if target.is_aromatic(atom_id) {
        symbol.make_ascii_lowercase();
    }

    if atom.isotope_mass_number().is_none()
        && atom.charge_value() == 0
        && atom.element() != Some(Element::H)
    {
        return Some(symbol);
    }

    let mut rendered = String::from("[");
    if let Some(isotope_mass_number) = atom.isotope_mass_number() {
        rendered.push_str(&isotope_mass_number.to_string());
    }
    rendered.push_str(&symbol);
    if atom.charge_value() != 0 {
        rendered.push_str(&render_charge(atom.charge_value()));
    }
    rendered.push(']');
    Some(rendered)
}

fn query_has_total_hydrogen(expr: &AtomExpr) -> bool {
    match expr {
        AtomExpr::Wildcard | AtomExpr::Bare { .. } => false,
        AtomExpr::Bracket(expr) => bracket_tree_has_total_hydrogen(&expr.tree),
    }
}

fn bracket_tree_has_total_hydrogen(tree: &BracketExprTree) -> bool {
    match tree {
        BracketExprTree::Primitive(AtomPrimitive::Hydrogen(HydrogenKind::Total, _)) => true,
        BracketExprTree::Primitive(_) | BracketExprTree::Not(_) => false,
        BracketExprTree::HighAnd(items)
        | BracketExprTree::Or(items)
        | BracketExprTree::LowAnd(items) => items.iter().any(bracket_tree_has_total_hydrogen),
    }
}

fn render_local_stereo_center(chirality: Chirality, has_hydrogen: bool) -> String {
    let mut rendered = String::from("[C");
    rendered.push_str(&render_local_chirality(chirality, has_hydrogen));
    if has_hydrogen {
        rendered.push('H');
    }
    rendered.push(']');
    rendered
}

fn render_charge(charge: i8) -> String {
    match charge.cmp(&0) {
        core::cmp::Ordering::Less => {
            let magnitude = charge.unsigned_abs();
            if magnitude == 1 {
                String::from("-")
            } else {
                alloc::format!("-{magnitude}")
            }
        }
        core::cmp::Ordering::Equal => String::new(),
        core::cmp::Ordering::Greater => {
            let magnitude = u8::try_from(charge).unwrap_or(u8::MAX);
            if magnitude == 1 {
                String::from("+")
            } else {
                alloc::format!("+{magnitude}")
            }
        }
    }
}

const fn local_stereo_bond_text(expr: &BondExpr) -> &'static str {
    match expr {
        BondExpr::Elided => "",
        BondExpr::Query(_) => "-",
    }
}

fn numeric_query_matches_u16(query: NumericQuery, actual: u16) -> bool {
    match query {
        NumericQuery::Exact(expected) => actual == expected,
        NumericQuery::Range(range) => numeric_range_matches(range, actual),
    }
}

fn numeric_range_matches(range: crate::NumericRange, actual: u16) -> bool {
    range.min.is_none_or(|min| actual >= min) && range.max.is_none_or(|max| actual <= max)
}

fn count_query_matches_u16(query: Option<NumericQuery>, actual: u16, omitted_default: u16) -> bool {
    query.map_or(actual == omitted_default, |query| {
        numeric_query_matches_u16(query, actual)
    })
}

fn ring_query_matches_u16(query: Option<NumericQuery>, actual: u16) -> bool {
    query.map_or(actual > 0, |query| numeric_query_matches_u16(query, actual))
}

fn render_local_chirality(chirality: Chirality, has_hydrogen: bool) -> String {
    match (chirality, has_hydrogen) {
        (Chirality::At, true) => String::from("@"),
        (Chirality::AtAt, true) => String::from("@@"),
        (Chirality::At, false) => Chirality::TH(1).to_string(),
        (Chirality::AtAt, false) => Chirality::TH(2).to_string(),
        _ => chirality.to_string(),
    }
}

fn directional_substituents(
    query: &QueryMol,
    query_neighbors: &[alloc::vec::Vec<(QueryAtomId, usize)>],
    endpoint: QueryAtomId,
    double_bond_id: usize,
    opposite_endpoint: QueryAtomId,
) -> Result<alloc::vec::Vec<QueryDirectionalSubstituent>, SmartsMatchError> {
    let mut directional = alloc::vec::Vec::new();
    for (neighbor_atom, bond_id) in &query_neighbors[endpoint] {
        if *bond_id == double_bond_id || *neighbor_atom == opposite_endpoint {
            continue;
        }
        let Some(direction) = first_supported_directional_bond(&query.bonds()[*bond_id].expr)?
        else {
            continue;
        };
        directional.push(QueryDirectionalSubstituent {
            bond_id: *bond_id,
            direction,
        });
    }
    Ok(directional)
}

fn local_substituents(
    query: &QueryMol,
    query_neighbors: &[alloc::vec::Vec<(QueryAtomId, usize)>],
    endpoint: QueryAtomId,
    double_bond_id: usize,
    opposite_endpoint: QueryAtomId,
) -> Result<alloc::vec::Vec<QueryLocalSubstituent>, SmartsMatchError> {
    let mut substituents = alloc::vec::Vec::new();
    for (neighbor_atom, bond_id) in &query_neighbors[endpoint] {
        if *bond_id == double_bond_id || *neighbor_atom == opposite_endpoint {
            continue;
        }
        let expr = &query.bonds()[*bond_id].expr;
        let direction = first_supported_directional_bond(expr)?;
        substituents.push(QueryLocalSubstituent {
            bond_id: *bond_id,
            direction,
        });
    }
    Ok(substituents)
}

fn build_local_double_bond_fragment(
    left_substituents: &[QueryLocalSubstituent],
    right_substituents: &[QueryLocalSubstituent],
) -> Result<(String, usize, usize), SmartsMatchError> {
    const PLACEHOLDER_ATOMS: [&str; 8] = ["F", "Cl", "Br", "I", "N", "O", "P", "S"];

    let total_substituents = left_substituents.len() + right_substituents.len();
    if total_substituents > PLACEHOLDER_ATOMS.len() {
        return Err(unsupported_bond_primitive("/"));
    }

    let mut next_placeholder = 0usize;
    let mut next_atom_id = 0usize;
    let mut fragment = String::new();

    let left_prefix_index = left_substituents
        .iter()
        .position(|substituent| substituent.direction.is_some())
        .or_else(|| (!left_substituents.is_empty()).then_some(0));

    if let Some(prefix_index) = left_prefix_index {
        fragment.push_str(PLACEHOLDER_ATOMS[next_placeholder]);
        fragment.push_str(local_double_bond_direction_text(
            left_substituents[prefix_index].direction,
        ));
        next_placeholder += 1;
        next_atom_id += 1;
    }

    let left_center_atom_id = next_atom_id;
    fragment.push('C');
    next_atom_id += 1;

    for (index, substituent) in left_substituents.iter().enumerate() {
        if Some(index) == left_prefix_index {
            continue;
        }
        fragment.push('(');
        fragment.push_str(local_double_bond_direction_text(substituent.direction));
        fragment.push_str(PLACEHOLDER_ATOMS[next_placeholder]);
        fragment.push(')');
        next_placeholder += 1;
        next_atom_id += 1;
    }

    fragment.push('=');
    let right_center_atom_id = next_atom_id;
    fragment.push('C');
    next_atom_id += 1;

    if let Some(continuation_index) = right_continuation_index(right_substituents) {
        for (index, substituent) in right_substituents.iter().enumerate() {
            if index == continuation_index {
                continue;
            }
            fragment.push('(');
            fragment.push_str(local_double_bond_direction_text(substituent.direction));
            fragment.push_str(PLACEHOLDER_ATOMS[next_placeholder]);
            fragment.push(')');
            next_placeholder += 1;
            next_atom_id += 1;
        }

        let continuation = right_substituents[continuation_index];
        fragment.push_str(local_double_bond_direction_text(continuation.direction));
        fragment.push_str(PLACEHOLDER_ATOMS[next_placeholder]);
        let _ = next_atom_id;
    }

    Ok((fragment, left_center_atom_id, right_center_atom_id))
}

fn right_continuation_index(substituents: &[QueryLocalSubstituent]) -> Option<usize> {
    substituents
        .iter()
        .position(|substituent| substituent.direction.is_none())
        .or_else(|| (!substituents.is_empty()).then(|| substituents.len() - 1))
}

const fn local_double_bond_direction_text(direction: Option<Bond>) -> &'static str {
    match direction {
        Some(Bond::Up) => "/",
        Some(Bond::Down) => "\\",
        _ => "",
    }
}

fn first_supported_directional_bond(expr: &BondExpr) -> Result<Option<Bond>, SmartsMatchError> {
    match expr {
        BondExpr::Elided => Ok(None),
        BondExpr::Query(tree) => first_supported_directional_bond_tree(tree),
    }
}

fn bond_expr_contains_negated_directional_primitive(expr: &BondExpr) -> bool {
    match expr {
        BondExpr::Elided => false,
        BondExpr::Query(tree) => bond_tree_contains_negated_directional_primitive(tree),
    }
}

fn bond_tree_contains_negated_directional_primitive(tree: &BondExprTree) -> bool {
    match tree {
        BondExprTree::Primitive(_) => false,
        BondExprTree::Not(inner) => bond_tree_contains_directional_primitive(inner),
        BondExprTree::HighAnd(items) | BondExprTree::Or(items) | BondExprTree::LowAnd(items) => {
            let mut index = 0usize;
            while index < items.len() {
                if bond_tree_contains_negated_directional_primitive(&items[index]) {
                    return true;
                }
                index += 1;
            }
            false
        }
    }
}

fn first_supported_directional_bond_tree(
    tree: &BondExprTree,
) -> Result<Option<Bond>, SmartsMatchError> {
    match tree {
        BondExprTree::Primitive(BondPrimitive::Bond(Bond::Up | Bond::Down)) => match tree {
            BondExprTree::Primitive(BondPrimitive::Bond(bond)) => Ok(Some(*bond)),
            _ => Ok(None),
        },
        BondExprTree::Primitive(_) | BondExprTree::Not(_) => Ok(None),
        BondExprTree::HighAnd(items) | BondExprTree::Or(items) | BondExprTree::LowAnd(items) => {
            for item in items {
                if let Some(direction) = first_supported_directional_bond_tree(item)? {
                    return Ok(Some(direction));
                }
            }
            Ok(None)
        }
    }
}

const fn is_simple_double_bond_expr(expr: &BondExpr) -> bool {
    matches!(
        expr,
        BondExpr::Query(BondExprTree::Primitive(BondPrimitive::Bond(Bond::Double)))
    )
}

fn bond_tree_contains_directional_primitive(tree: &BondExprTree) -> bool {
    match tree {
        BondExprTree::Primitive(BondPrimitive::Bond(Bond::Up | Bond::Down)) => true,
        BondExprTree::Primitive(_) => false,
        BondExprTree::Not(inner) => bond_tree_contains_directional_primitive(inner),
        BondExprTree::HighAnd(items) | BondExprTree::Or(items) | BondExprTree::LowAnd(items) => {
            items.iter().any(bond_tree_contains_directional_primitive)
        }
    }
}

fn search_mapping(
    query: &QueryMol,
    target: &PreparedTarget,
    context: &mut SearchContext<'_>,
    mapped_count: usize,
) -> bool {
    if mapped_count == query.atom_count() {
        return stereo_constraints_match(
            query,
            context.stereo_plan,
            context.query_neighbors,
            target,
            context.query_to_target,
            context.atom_stereo_cache,
        );
    }

    let next_query_atom = select_next_query_atom_for_target(context);
    search_mapping_candidates(query, target, context, mapped_count, next_query_atom)
}

fn search_mapping_candidates(
    query: &QueryMol,
    target: &PreparedTarget,
    context: &mut SearchContext<'_>,
    mapped_count: usize,
    query_atom: QueryAtomId,
) -> bool {
    let query_degree = context.query_neighbors[query_atom].len();
    let Some(MappedNeighborSeed {
        target_atom: seed_target_atom,
        bond_id: seed_bond_id,
        query_neighbor: seed_query_neighbor,
    }) = best_mapped_neighbor_seed(query_atom, target, context)
    else {
        let mut target_atoms = alloc::vec::Vec::new();
        if collect_atom_matcher_anchor_candidates(
            &context.compiled_query.atom_matchers[query_atom],
            target,
            &mut target_atoms,
        ) {
            return search_mapping_unanchored_candidates(
                query,
                target,
                context,
                mapped_count,
                query_atom,
                target_atoms.into_iter(),
            );
        }
        return search_mapping_unanchored_candidates(
            query,
            target,
            context,
            mapped_count,
            query_atom,
            0..target.atom_count(),
        );
    };
    for (neighbor_atom, bond_label) in target.neighbors(seed_target_atom) {
        if context.target_atom_is_used(neighbor_atom)
            || target_degree_too_small(target, neighbor_atom, query_degree)
            || !query_bond_matches(
                context.compiled_query.bond_matchers[seed_bond_id],
                target,
                seed_target_atom,
                neighbor_atom,
                bond_label,
            )
        {
            continue;
        }
        let all_mapped_bonds_match = context.query_neighbors[query_atom]
            .iter()
            .filter(|(mapped_neighbor, _)| *mapped_neighbor != seed_query_neighbor)
            .all(|(mapped_neighbor, query_bond_id)| {
                let Some(mapped_target_atom) = context.query_to_target[*mapped_neighbor] else {
                    return true;
                };
                target
                    .bond(neighbor_atom, mapped_target_atom)
                    .is_some_and(|target_bond| {
                        query_bond_matches(
                            context.compiled_query.bond_matchers[*query_bond_id],
                            target,
                            neighbor_atom,
                            mapped_target_atom,
                            target_bond,
                        )
                    })
            });
        if all_mapped_bonds_match
            && search_mapping_candidate(
                query,
                target,
                context,
                mapped_count,
                query_atom,
                neighbor_atom,
            )
        {
            return true;
        }
    }

    false
}

fn search_mapping_unanchored_candidates(
    query: &QueryMol,
    target: &PreparedTarget,
    context: &mut SearchContext<'_>,
    mapped_count: usize,
    query_atom: QueryAtomId,
    target_atoms: impl Iterator<Item = usize>,
) -> bool {
    let query_degree = context.query_neighbors[query_atom].len();
    for target_atom in target_atoms {
        if context.target_atom_is_used(target_atom)
            || target_degree_too_small(target, target_atom, query_degree)
        {
            continue;
        }
        if search_mapping_candidate(
            query,
            target,
            context,
            mapped_count,
            query_atom,
            target_atom,
        ) {
            return true;
        }
    }
    false
}

fn search_mapping_candidate(
    query: &QueryMol,
    target: &PreparedTarget,
    context: &mut SearchContext<'_>,
    mapped_count: usize,
    query_atom: QueryAtomId,
    target_atom: usize,
) -> bool {
    if context.compiled_query.has_component_constraints
        && !component_constraints_match(
            query_atom,
            target_atom,
            query,
            target,
            context.query_to_target,
        )
    {
        return false;
    }
    if !compiled_query_atom_matches(
        context.compiled_query,
        target,
        context.recursive_cache,
        query_atom,
        target_atom,
    ) {
        return false;
    }

    context.query_to_target[query_atom] = Some(target_atom);
    context.mark_target_atom_used(target_atom);
    let matched = search_mapping(query, target, context, mapped_count + 1);
    context.unmark_target_atom_used(target_atom);
    context.query_to_target[query_atom] = None;
    matched
}

fn collect_mappings(
    query: &QueryMol,
    target: &PreparedTarget,
    context: &mut SearchContext<'_>,
    mapped_count: usize,
    matches: &mut UniqueMatches,
) {
    if mapped_count == query.atom_count() {
        if stereo_constraints_match(
            query,
            context.stereo_plan,
            context.query_neighbors,
            target,
            context.query_to_target,
            context.atom_stereo_cache,
        ) {
            let mapping = current_query_order_mapping(context.query_to_target);
            matches
                .entry(canonical_match_key(&mapping))
                .and_modify(|existing| {
                    if mapping < *existing {
                        existing.clone_from(&mapping);
                    }
                })
                .or_insert(mapping);
        }
        return;
    }

    let next_query_atom = select_next_query_atom_for_target(context);
    for target_atom in candidate_target_atoms(next_query_atom, target, context) {
        if context.compiled_query.has_component_constraints
            && !component_constraints_match(
                next_query_atom,
                target_atom,
                query,
                target,
                context.query_to_target,
            )
        {
            continue;
        }
        if !compiled_query_atom_matches(
            context.compiled_query,
            target,
            context.recursive_cache,
            next_query_atom,
            target_atom,
        ) {
            continue;
        }

        context.query_to_target[next_query_atom] = Some(target_atom);
        context.mark_target_atom_used(target_atom);
        collect_mappings(query, target, context, mapped_count + 1, matches);
        context.unmark_target_atom_used(target_atom);
        context.query_to_target[next_query_atom] = None;
    }
}

fn collect_match_keys(
    query: &QueryMol,
    target: &PreparedTarget,
    context: &mut SearchContext<'_>,
    mapped_count: usize,
    matches: &mut MatchKeys,
) {
    if mapped_count == query.atom_count() {
        if stereo_constraints_match(
            query,
            context.stereo_plan,
            context.query_neighbors,
            target,
            context.query_to_target,
            context.atom_stereo_cache,
        ) {
            matches.push(current_canonical_match_key(context.query_to_target));
        }
        return;
    }

    let next_query_atom = select_next_query_atom_for_target(context);
    for target_atom in candidate_target_atoms(next_query_atom, target, context) {
        if context.compiled_query.has_component_constraints
            && !component_constraints_match(
                next_query_atom,
                target_atom,
                query,
                target,
                context.query_to_target,
            )
        {
            continue;
        }
        if !compiled_query_atom_matches(
            context.compiled_query,
            target,
            context.recursive_cache,
            next_query_atom,
            target_atom,
        ) {
            continue;
        }

        context.query_to_target[next_query_atom] = Some(target_atom);
        context.mark_target_atom_used(target_atom);
        collect_match_keys(query, target, context, mapped_count + 1, matches);
        context.unmark_target_atom_used(target_atom);
        context.query_to_target[next_query_atom] = None;
    }
}

fn current_query_order_mapping(query_to_target: &[Option<usize>]) -> Box<[usize]> {
    query_to_target
        .iter()
        .map(|target_atom| target_atom.expect("complete mappings must bind every query atom"))
        .collect::<alloc::vec::Vec<_>>()
        .into_boxed_slice()
}

fn canonical_match_key(mapping: &[usize]) -> Box<[usize]> {
    let mut canonical = mapping.to_vec();
    canonical.sort_unstable();
    canonical.into_boxed_slice()
}

fn two_atom_canonical_key(left: usize, right: usize) -> Box<[usize]> {
    if left < right {
        Box::<[usize]>::from([left, right])
    } else {
        Box::<[usize]>::from([right, left])
    }
}

fn three_atom_canonical_key(first: usize, second: usize, third: usize) -> Box<[usize]> {
    let mut canonical = [first, second, third];
    canonical.sort_unstable();
    Box::<[usize]>::from(canonical)
}

fn current_canonical_match_key(query_to_target: &[Option<usize>]) -> Box<[usize]> {
    let mut canonical = query_to_target
        .iter()
        .map(|target_atom| target_atom.expect("complete mappings must bind every query atom"))
        .collect::<alloc::vec::Vec<_>>();
    canonical.sort_unstable();
    canonical.into_boxed_slice()
}

fn component_constraints_match(
    query_atom: QueryAtomId,
    target_atom: usize,
    query: &QueryMol,
    target: &PreparedTarget,
    query_to_target: &[Option<usize>],
) -> bool {
    let query_component = query.atoms()[query_atom].component;
    let Some(query_group) = query.component_group(query_component) else {
        return true;
    };
    let Some(target_component) = target.connected_component(target_atom) else {
        return false;
    };

    for (other_query_atom, mapped_target_atom) in query_to_target.iter().enumerate() {
        let Some(mapped_target_atom) = mapped_target_atom else {
            continue;
        };
        let other_component = query.atoms()[other_query_atom].component;
        if other_component == query_component {
            continue;
        }

        let Some(other_target_component) = target.connected_component(*mapped_target_atom) else {
            return false;
        };

        match query.component_group(other_component) {
            Some(other_group)
                if other_group == query_group && other_target_component != target_component =>
            {
                return false;
            }
            Some(other_group)
                if other_group != query_group && other_target_component == target_component =>
            {
                return false;
            }
            _ => {}
        }
    }

    true
}

#[cfg(test)]
fn select_next_query_atom(
    query_neighbors: &[alloc::vec::Vec<(QueryAtomId, usize)>],
    query_atom_scores: &[usize],
    query_to_target: &[Option<usize>],
) -> QueryAtomId {
    let mut best_atom = None;
    let mut best_key = (0usize, 0usize, 0usize, core::cmp::Reverse(usize::MAX));

    for (query_atom, neighbors) in query_neighbors.iter().enumerate() {
        if query_to_target[query_atom].is_some() {
            continue;
        }
        let mapped_neighbors = neighbors
            .iter()
            .filter(|(neighbor, _)| query_to_target[*neighbor].is_some())
            .count();
        let key = (
            mapped_neighbors,
            query_atom_scores[query_atom],
            neighbors.len(),
            core::cmp::Reverse(query_atom),
        );
        if best_atom.is_none() || key > best_key {
            best_atom = Some(query_atom);
            best_key = key;
        }
    }

    best_atom.expect("at least one unmapped query atom must remain")
}

fn select_next_query_atom_for_target(context: &SearchContext<'_>) -> QueryAtomId {
    let mut best_atom = None;
    let mut best_key = (
        0usize,
        core::cmp::Reverse(usize::MAX),
        0usize,
        0usize,
        core::cmp::Reverse(usize::MAX),
    );

    for (query_atom, neighbors) in context.query_neighbors.iter().enumerate() {
        if context.query_to_target[query_atom].is_some() {
            continue;
        }
        let mapped_neighbors = neighbors
            .iter()
            .filter(|(neighbor, _)| context.query_to_target[*neighbor].is_some())
            .count();
        let search_width_estimate = if mapped_neighbors == 0 {
            context.query_atom_anchor_widths[query_atom]
        } else {
            usize::MAX
        };
        let key = (
            mapped_neighbors,
            core::cmp::Reverse(search_width_estimate),
            context.query_atom_scores[query_atom],
            neighbors.len(),
            core::cmp::Reverse(query_atom),
        );
        if best_atom.is_none() || key > best_key {
            best_atom = Some(query_atom);
            best_key = key;
        }
    }

    best_atom.expect("at least one unmapped query atom must remain")
}

fn best_mapped_neighbor_seed(
    query_atom: QueryAtomId,
    target: &PreparedTarget,
    context: &SearchContext<'_>,
) -> Option<MappedNeighborSeed> {
    let mut best = None;

    for (query_neighbor, bond_id) in &context.query_neighbors[query_atom] {
        let Some(target_atom) = context.query_to_target[*query_neighbor] else {
            continue;
        };
        let degree = target.degree(target_atom).unwrap_or(usize::MAX);
        let seed = MappedNeighborSeed {
            target_atom,
            bond_id: *bond_id,
            query_neighbor: *query_neighbor,
        };
        if best.is_none_or(|(best_degree, _): (usize, MappedNeighborSeed)| degree < best_degree) {
            best = Some((degree, seed));
        }
    }

    best.map(|(_, seed)| seed)
}

fn prepare_query_atom_anchor_widths(
    widths: &mut alloc::vec::Vec<usize>,
    query: &CompiledQuery,
    target: &PreparedTarget,
) {
    let fallback = target.atom_count();
    widths.clear();
    widths.extend(query.atom_matchers.iter().map(|atom_matcher| {
        atom_matcher_anchor_candidates(atom_matcher, target).map_or(fallback, <[usize]>::len)
    }));
}

fn candidate_target_atoms(
    query_atom: QueryAtomId,
    target: &PreparedTarget,
    context: &SearchContext<'_>,
) -> alloc::vec::Vec<usize> {
    let query_degree = context.query_neighbors[query_atom].len();
    let Some(MappedNeighborSeed {
        target_atom: seed_target_atom,
        bond_id: seed_bond_id,
        query_neighbor: seed_query_neighbor,
    }) = best_mapped_neighbor_seed(query_atom, target, context)
    else {
        let mut target_atoms = alloc::vec::Vec::new();
        if collect_atom_matcher_anchor_candidates(
            &context.compiled_query.atom_matchers[query_atom],
            target,
            &mut target_atoms,
        ) {
            target_atoms.retain(|&target_atom| {
                !context.target_atom_is_used(target_atom)
                    && !target_degree_too_small(target, target_atom, query_degree)
            });
            return target_atoms;
        }
        return (0..target.atom_count())
            .filter(|&target_atom| {
                !context.target_atom_is_used(target_atom)
                    && !target_degree_too_small(target, target_atom, query_degree)
            })
            .collect();
    };
    let mut candidates = alloc::vec::Vec::new();
    for (neighbor_atom, bond_label) in target.neighbors(seed_target_atom) {
        if context.target_atom_is_used(neighbor_atom)
            || target_degree_too_small(target, neighbor_atom, query_degree)
            || !query_bond_matches(
                context.compiled_query.bond_matchers[seed_bond_id],
                target,
                seed_target_atom,
                neighbor_atom,
                bond_label,
            )
        {
            continue;
        }
        let all_mapped_bonds_match = context.query_neighbors[query_atom]
            .iter()
            .filter(|(mapped_neighbor, _)| *mapped_neighbor != seed_query_neighbor)
            .all(|(mapped_neighbor, query_bond_id)| {
                let Some(mapped_target_atom) = context.query_to_target[*mapped_neighbor] else {
                    return true;
                };
                target
                    .bond(neighbor_atom, mapped_target_atom)
                    .is_some_and(|target_bond| {
                        query_bond_matches(
                            context.compiled_query.bond_matchers[*query_bond_id],
                            target,
                            neighbor_atom,
                            mapped_target_atom,
                            target_bond,
                        )
                    })
            });
        if all_mapped_bonds_match {
            candidates.push(neighbor_atom);
        }
    }
    candidates
}

fn atom_matcher_anchor_candidates<'a>(
    atom_matcher: &CompiledAtomMatcher,
    target: &'a PreparedTarget,
) -> Option<&'a [usize]> {
    let mut best = None;
    for &predicate in &atom_matcher.predicates {
        let Some(candidates) = atom_fast_predicate_anchor_candidates(predicate, target) else {
            continue;
        };
        if candidates.is_empty() {
            return Some(candidates);
        }
        if best.is_none_or(|best_candidates: &[usize]| candidates.len() < best_candidates.len()) {
            best = Some(candidates);
        }
    }
    best
}

fn single_atom_candidates<'a>(
    atom_matcher: &CompiledAtomMatcher,
    target: &PreparedTarget,
    initial_target_atom: Option<usize>,
    anchored_target_atoms: &'a mut alloc::vec::Vec<usize>,
) -> SingleAtomCandidates<'a> {
    if let Some(target_atom) = initial_target_atom {
        return SingleAtomCandidates::One(target_atom);
    }
    if collect_atom_matcher_anchor_candidates(atom_matcher, target, anchored_target_atoms) {
        SingleAtomCandidates::Anchored(anchored_target_atoms)
    } else {
        SingleAtomCandidates::All(0..target.atom_count())
    }
}

fn collect_atom_matcher_anchor_candidates(
    atom_matcher: &CompiledAtomMatcher,
    target: &PreparedTarget,
    out: &mut alloc::vec::Vec<usize>,
) -> bool {
    let Some(best_candidates) = atom_matcher_anchor_candidates(atom_matcher, target) else {
        out.clear();
        return false;
    };

    out.clear();
    out.extend_from_slice(best_candidates);
    if out.len() <= 1 {
        return true;
    }

    for &predicate in &atom_matcher.predicates {
        let Some(candidates) = atom_fast_predicate_anchor_candidates(predicate, target) else {
            continue;
        };
        if core::ptr::eq(candidates, best_candidates) {
            continue;
        }
        out.retain(|atom_id| candidates.binary_search(atom_id).is_ok());
        if out.is_empty() {
            break;
        }
    }

    true
}

fn atom_fast_predicate_anchor_candidates(
    predicate: AtomFastPredicate,
    target: &PreparedTarget,
) -> Option<&[usize]> {
    const EMPTY_ATOM_IDS: &[usize] = &[];

    match predicate {
        AtomFastPredicate::Isotope(Some(mass_number)) => {
            Some(target.atom_ids_with_isotope_mass_number(mass_number))
        }
        AtomFastPredicate::Element(element) => {
            Some(target.atom_ids_with_atomic_number(u16::from(element.atomic_number())))
        }
        AtomFastPredicate::AtomicNumber(atomic_number) => {
            Some(target.atom_ids_with_atomic_number(atomic_number))
        }
        AtomFastPredicate::Aromatic(true) => Some(target.aromatic_atom_ids()),
        AtomFastPredicate::Aromatic(false) => Some(target.aliphatic_atom_ids()),
        AtomFastPredicate::Degree(query) => exact_count_requirement(query, 1)
            .map(|degree| target.atom_ids_with_degree(usize::from(degree))),
        AtomFastPredicate::Connectivity(query) => exact_count_requirement(query, 1)
            .map(|count| target.atom_ids_with_connectivity(u8::try_from(count).unwrap_or(u8::MAX))),
        AtomFastPredicate::Valence(query) => exact_count_requirement(query, 1).map(|count| {
            target.atom_ids_with_total_valence(u8::try_from(count).unwrap_or(u8::MAX))
        }),
        AtomFastPredicate::Hybridization(query) => exact_count_requirement(Some(query), 1)
            .map(|code| target.atom_ids_with_hybridization(u8::try_from(code).unwrap_or(u8::MAX))),
        AtomFastPredicate::ImplicitHydrogen(query) => {
            exact_count_requirement(query, 1).map(|count| {
                u8::try_from(count).map_or(EMPTY_ATOM_IDS, |count| {
                    target.atom_ids_with_implicit_hydrogen(count)
                })
            })
        }
        AtomFastPredicate::TotalHydrogen(query) => {
            exact_count_requirement(query, 1).map(|hydrogens| {
                u8::try_from(hydrogens).map_or(EMPTY_ATOM_IDS, |hydrogens| {
                    target.atom_ids_with_total_hydrogen(hydrogens)
                })
            })
        }
        AtomFastPredicate::RingMembership(query) => {
            exact_positive_ring_requirement(query).map(|count| {
                u8::try_from(count).map_or(EMPTY_ATOM_IDS, |count| {
                    target.atom_ids_with_ring_membership_count(count)
                })
            })
        }
        AtomFastPredicate::RingSize(query) => exact_positive_ring_requirement(query).map(|size| {
            u8::try_from(size).map_or(EMPTY_ATOM_IDS, |size| {
                target.atom_ids_with_smallest_ring_size(size)
            })
        }),
        AtomFastPredicate::RingConnectivity(query) => {
            exact_positive_ring_requirement(query).map(|count| {
                u8::try_from(count).map_or(EMPTY_ATOM_IDS, |count| {
                    target.atom_ids_with_ring_bond_count(count)
                })
            })
        }
        AtomFastPredicate::HeteroNeighbor(query) => {
            exact_count_requirement(query, 1).map(|count| {
                u8::try_from(count).map_or(EMPTY_ATOM_IDS, |count| {
                    target.atom_ids_with_hetero_neighbor_count(count)
                })
            })
        }
        AtomFastPredicate::AliphaticHeteroNeighbor(query) => {
            exact_count_requirement(query, 1).map(|count| {
                u8::try_from(count).map_or(EMPTY_ATOM_IDS, |count| {
                    target.atom_ids_with_aliphatic_hetero_neighbor_count(count)
                })
            })
        }
        AtomFastPredicate::Charge(charge) => Some(target.atom_ids_with_formal_charge(charge)),
        AtomFastPredicate::HasElement | AtomFastPredicate::Isotope(None) => None,
    }
}

const fn exact_count_requirement(query: Option<NumericQuery>, omitted_default: u16) -> Option<u16> {
    match query {
        None => Some(omitted_default),
        Some(NumericQuery::Exact(expected)) => Some(expected),
        Some(NumericQuery::Range(range)) => match (range.min, range.max) {
            (Some(min), Some(max)) if min == max => Some(min),
            _ => None,
        },
    }
}

const fn exact_positive_ring_requirement(query: Option<NumericQuery>) -> Option<u16> {
    match query {
        Some(NumericQuery::Exact(expected)) if expected > 0 => Some(expected),
        Some(NumericQuery::Range(range)) => match (range.min, range.max) {
            (Some(min), Some(max)) if min == max && min > 0 => Some(min),
            _ => None,
        },
        _ => None,
    }
}

fn target_degree_too_small(
    target: &PreparedTarget,
    target_atom: usize,
    query_degree: usize,
) -> bool {
    target
        .degree(target_atom)
        .is_some_and(|degree| degree < query_degree)
}

fn query_bond_matches(
    matcher: CompiledBondMatcher,
    target: &PreparedTarget,
    left_atom: usize,
    right_atom: usize,
    target_bond: BondLabel,
) -> bool {
    let ring = matcher.ring_sensitive && target.is_ring_bond(left_atom, right_atom);
    matcher.state_mask & bond_state_bit(target_bond, ring) != 0
}

fn atom_expr_matches(
    expr: &AtomExpr,
    query_atom_id: QueryAtomId,
    atom_id: usize,
    context: &mut AtomMatchContext<'_, '_>,
) -> bool {
    match expr {
        AtomExpr::Wildcard => true,
        AtomExpr::Bare { element, aromatic } => atom_matches_symbol(
            context.target.atom(atom_id),
            context.target,
            atom_id,
            context.target.is_aromatic(atom_id),
            *element,
            *aromatic,
        ),
        AtomExpr::Bracket(bracket_tree_expr) => {
            bracket_tree_matches(&bracket_tree_expr.tree, query_atom_id, atom_id, context)
        }
    }
}

fn cached_query_atom_matches(
    expr: &AtomExpr,
    atom_matcher: &CompiledAtomMatcher,
    query_atom_id: QueryAtomId,
    atom_id: usize,
    context: &mut AtomMatchContext<'_, '_>,
) -> bool {
    if is_hidden_attached_hydrogen(context.target, atom_id) {
        return false;
    }
    if !atom_fast_predicates_match(atom_matcher, context.target, atom_id) {
        return false;
    }
    if atom_matcher.complete {
        return true;
    }
    atom_expr_matches(expr, query_atom_id, atom_id, context)
}

fn atom_fast_predicates_match(
    atom_matcher: &CompiledAtomMatcher,
    target: &PreparedTarget,
    atom_id: usize,
) -> bool {
    let atom = atom_matcher
        .needs_atom
        .then(|| target.atom(atom_id))
        .flatten();
    let aromatic = atom_matcher.needs_aromatic && target.is_aromatic(atom_id);
    for &predicate in &atom_matcher.predicates {
        if !atom_fast_predicate_matches(predicate, target, atom_id, atom, aromatic) {
            return false;
        }
    }
    true
}

fn atom_fast_predicate_matches(
    predicate: AtomFastPredicate,
    target: &PreparedTarget,
    atom_id: usize,
    atom: Option<&Atom>,
    aromatic: bool,
) -> bool {
    match predicate {
        AtomFastPredicate::HasElement => atom.and_then(Atom::element).is_some(),
        AtomFastPredicate::Element(expected) => atom.and_then(Atom::element) == Some(expected),
        AtomFastPredicate::Aromatic(expected) => aromatic == expected,
        AtomFastPredicate::Isotope(expected) => {
            atom.is_some_and(|atom| atom.isotope_mass_number() == expected)
        }
        AtomFastPredicate::AtomicNumber(expected) => atom
            .and_then(Atom::element)
            .is_some_and(|element| u16::from(element.atomic_number()) == expected),
        AtomFastPredicate::Degree(expected) => target.degree(atom_id).is_some_and(|degree| {
            let actual = u16::try_from(degree).unwrap_or(u16::MAX);
            count_query_matches_u16(expected, actual, 1)
        }),
        AtomFastPredicate::Connectivity(expected) => target
            .connectivity(atom_id)
            .is_some_and(|count| count_query_matches_u16(expected, u16::from(count), 1)),
        AtomFastPredicate::Valence(expected) => target
            .total_valence(atom_id)
            .is_some_and(|count| count_query_matches_u16(expected, u16::from(count), 1)),
        AtomFastPredicate::Hybridization(expected) => target
            .hybridization(atom_id)
            .is_some_and(|code| numeric_query_matches_u16(expected, u16::from(code))),
        AtomFastPredicate::TotalHydrogen(expected) => target
            .total_hydrogen_count(atom_id)
            .is_some_and(|count| count_query_matches_u16(expected, u16::from(count), 1)),
        AtomFastPredicate::ImplicitHydrogen(expected) => target
            .implicit_hydrogen_count(atom_id)
            .is_some_and(|count| count_query_matches_u16(expected, u16::from(count), 1)),
        AtomFastPredicate::RingMembership(expected) => target
            .ring_membership_count(atom_id)
            .is_some_and(|count| ring_query_matches_u16(expected, u16::from(count))),
        AtomFastPredicate::RingSize(expected) => target
            .smallest_ring_size(atom_id)
            .is_some_and(|size| ring_query_matches_u16(expected, u16::from(size))),
        AtomFastPredicate::RingConnectivity(expected) => target
            .ring_bond_count(atom_id)
            .is_some_and(|count| ring_query_matches_u16(expected, u16::from(count))),
        AtomFastPredicate::HeteroNeighbor(expected) => target
            .hetero_neighbor_count(atom_id)
            .is_some_and(|count| count_query_matches_u16(expected, u16::from(count), 1)),
        AtomFastPredicate::AliphaticHeteroNeighbor(expected) => target
            .aliphatic_hetero_neighbor_count(atom_id)
            .is_some_and(|count| count_query_matches_u16(expected, u16::from(count), 1)),
        AtomFastPredicate::Charge(expected) => target.formal_charge(atom_id) == Some(expected),
    }
}

fn bracket_tree_matches(
    tree: &BracketExprTree,
    query_atom_id: QueryAtomId,
    atom_id: usize,
    context: &mut AtomMatchContext<'_, '_>,
) -> bool {
    match tree {
        BracketExprTree::Primitive(primitive) => {
            atom_primitive_matches(primitive, query_atom_id, atom_id, context)
        }
        BracketExprTree::Not(inner) => {
            !bracket_tree_matches(inner, query_atom_id, atom_id, context)
        }
        BracketExprTree::HighAnd(items) | BracketExprTree::LowAnd(items) => items
            .iter()
            .all(|item| bracket_tree_matches(item, query_atom_id, atom_id, context)),
        BracketExprTree::Or(items) => items
            .iter()
            .any(|item| bracket_tree_matches(item, query_atom_id, atom_id, context)),
    }
}

fn atom_primitive_matches(
    primitive: &AtomPrimitive,
    _query_atom_id: QueryAtomId,
    atom_id: usize,
    context: &mut AtomMatchContext<'_, '_>,
) -> bool {
    let target = context.target;
    let atom = target.atom(atom_id);
    let aromatic = target.is_aromatic(atom_id);

    match primitive {
        AtomPrimitive::Wildcard | AtomPrimitive::Chirality(_) => true,
        AtomPrimitive::AliphaticAny => atom.and_then(Atom::element).is_some() && !aromatic,
        AtomPrimitive::AromaticAny => atom.and_then(Atom::element).is_some() && aromatic,
        AtomPrimitive::Symbol { element, aromatic } => atom_matches_symbol(
            atom,
            target,
            atom_id,
            target.is_aromatic(atom_id),
            *element,
            *aromatic,
        ),
        AtomPrimitive::Isotope { isotope, aromatic } => {
            atom_matches_isotope(atom, target.is_aromatic(atom_id), *isotope, *aromatic)
        }
        AtomPrimitive::IsotopeWildcard(mass_number) => atom.is_some_and(|atom| {
            if *mass_number == 0 {
                atom.isotope_mass_number().is_none()
            } else {
                atom.isotope_mass_number() == Some(*mass_number)
            }
        }),
        AtomPrimitive::AtomicNumber(atomic_number) => {
            if *atomic_number == 1 && is_hidden_attached_hydrogen(target, atom_id) {
                false
            } else {
                atom.and_then(Atom::element)
                    .is_some_and(|element| u16::from(element.atomic_number()) == *atomic_number)
            }
        }
        AtomPrimitive::Degree(expected) => target.degree(atom_id).is_some_and(|degree| {
            let actual = u16::try_from(degree).unwrap_or(u16::MAX);
            count_query_matches_u16(*expected, actual, 1)
        }),
        AtomPrimitive::Connectivity(expected) => target
            .connectivity(atom_id)
            .is_some_and(|count| count_query_matches_u16(*expected, u16::from(count), 1)),
        AtomPrimitive::Valence(expected) => target
            .total_valence(atom_id)
            .is_some_and(|count| count_query_matches_u16(*expected, u16::from(count), 1)),
        AtomPrimitive::Hybridization(expected) => target
            .hybridization(atom_id)
            .is_some_and(|code| numeric_query_matches_u16(*expected, u16::from(code))),
        AtomPrimitive::Hydrogen(HydrogenKind::Total, expected) => target
            .total_hydrogen_count(atom_id)
            .is_some_and(|count| count_query_matches_u16(*expected, u16::from(count), 1)),
        AtomPrimitive::Hydrogen(HydrogenKind::Implicit, expected) => target
            .implicit_hydrogen_count(atom_id)
            .is_some_and(|count| count_query_matches_u16(*expected, u16::from(count), 1)),
        AtomPrimitive::RingMembership(expected) => target
            .ring_membership_count(atom_id)
            .is_some_and(|count| ring_query_matches_u16(*expected, u16::from(count))),
        AtomPrimitive::RingSize(expected) => target
            .smallest_ring_size(atom_id)
            .is_some_and(|size| ring_query_matches_u16(*expected, u16::from(size))),
        AtomPrimitive::RingConnectivity(expected) => target
            .ring_bond_count(atom_id)
            .is_some_and(|count| ring_query_matches_u16(*expected, u16::from(count))),
        AtomPrimitive::HeteroNeighbor(expected) => target
            .hetero_neighbor_count(atom_id)
            .is_some_and(|count| count_query_matches_u16(*expected, u16::from(count), 1)),
        AtomPrimitive::AliphaticHeteroNeighbor(expected) => target
            .aliphatic_hetero_neighbor_count(atom_id)
            .is_some_and(|count| count_query_matches_u16(*expected, u16::from(count), 1)),
        AtomPrimitive::Charge(expected) => target.formal_charge(atom_id) == Some(*expected),
        AtomPrimitive::RecursiveQuery(query) => recursive_query_matches(
            query,
            target,
            atom_id,
            context.recursive_query_lookup,
            context.recursive_queries,
            context.recursive_cache,
        ),
    }
}

fn atom_chirality_matches(
    query: &QueryMol,
    query_neighbors: &[alloc::vec::Vec<(QueryAtomId, usize)>],
    query_atom_id: QueryAtomId,
    target: &PreparedTarget,
    query_to_target: &[Option<usize>],
    atom_stereo_cache: &mut AtomStereoCache,
) -> bool {
    let Some(query_chirality) = extract_query_chirality(&query.atoms()[query_atom_id].expr) else {
        return true;
    };
    if !is_supported_query_chirality(query_chirality) {
        return false;
    }
    if !query_chirality_requires_tetrahedral_match(query_chirality) {
        return true;
    }

    let incident = &query_neighbors[query_atom_id];
    if incident.len() < 3 || incident.len() > 4 {
        return true;
    }

    let Some(target_atom_id) = query_to_target[query_atom_id] else {
        return false;
    };
    let Some(actual_target_chirality) = target.tetrahedral_chirality(target_atom_id) else {
        return false;
    };
    let Ok(neighbor_tokens) = local_neighbor_tokens_for_mapping(incident, query_to_target, target)
    else {
        return false;
    };
    let cache_key = AtomStereoCacheKey {
        query_atom_id,
        neighbor_tokens,
    };
    let expected_chirality = atom_stereo_cache.get(&cache_key).copied();
    let expected_chirality = expected_chirality.unwrap_or_else(|| {
        let computed = build_local_stereo_fragment(
            query,
            query_neighbors,
            query_atom_id,
            &cache_key.neighbor_tokens,
        )
        .ok()
        .and_then(|(local_fragment, local_center_atom_id)| {
            local_fragment
                .parse::<Smiles>()
                .ok()
                .and_then(|local_smiles| {
                    local_smiles.smarts_tetrahedral_chirality(local_center_atom_id)
                })
        });
        atom_stereo_cache.insert(cache_key.clone(), computed);
        computed
    });
    expected_chirality == Some(actual_target_chirality)
}

fn stereo_constraints_match(
    query: &QueryMol,
    stereo_plan: &QueryStereoPlan,
    query_neighbors: &[alloc::vec::Vec<(QueryAtomId, usize)>],
    target: &PreparedTarget,
    query_to_target: &[Option<usize>],
    atom_stereo_cache: &mut AtomStereoCache,
) -> bool {
    stereo_plan
        .double_bond_constraints
        .iter()
        .all(|constraint| {
            let Some(left_target) = query_to_target[constraint.left_atom] else {
                return false;
            };
            let Some(right_target) = query_to_target[constraint.right_atom] else {
                return false;
            };
            target.double_bond_stereo_config(left_target, right_target) == Some(constraint.config)
        })
        && query.atoms().iter().all(|atom| {
            atom_chirality_matches(
                query,
                query_neighbors,
                atom.id,
                target,
                query_to_target,
                atom_stereo_cache,
            )
        })
}

fn recursive_query_matches(
    query: &QueryMol,
    target: &PreparedTarget,
    atom_id: usize,
    recursive_query_lookup: &BTreeMap<usize, usize>,
    recursive_queries: &[RecursiveQueryEntry],
    recursive_cache: &mut RecursiveMatchCache,
) -> bool {
    let query_key = core::ptr::from_ref(query) as usize;
    let Some(entry_index) = recursive_query_lookup.get(&query_key).copied() else {
        return false;
    };
    let entry = &recursive_queries[entry_index];

    if let Some(cached) = recursive_cache.get(entry.cache_slot, atom_id) {
        return cached;
    }

    if query.is_empty() {
        recursive_cache.insert(entry.cache_slot, atom_id, false);
        return false;
    }

    let mut atom_stereo_cache = AtomStereoCache::new();
    let anchored = query_matches_with_mapping(
        entry.compiled.as_ref(),
        target,
        &mut atom_stereo_cache,
        recursive_cache,
        Some(0),
        Some(atom_id),
    );
    recursive_cache.insert(entry.cache_slot, atom_id, anchored);
    anchored
}

fn atom_matches_symbol(
    atom: Option<&Atom>,
    target: &PreparedTarget,
    atom_id: usize,
    target_is_aromatic: bool,
    expected_element: Element,
    expected_aromatic: bool,
) -> bool {
    if expected_element == Element::H
        && !expected_aromatic
        && is_hidden_attached_hydrogen(target, atom_id)
    {
        return false;
    }
    atom.and_then(Atom::element).is_some_and(|element| {
        element == expected_element && target_is_aromatic == expected_aromatic
    })
}

fn atom_matches_isotope(
    atom: Option<&Atom>,
    target_is_aromatic: bool,
    expected_isotope: elements_rs::Isotope,
    expected_aromatic: bool,
) -> bool {
    atom.is_some_and(|atom| {
        atom.element().is_some_and(|element| {
            element == expected_isotope.element()
                && atom.isotope_mass_number() == Some(expected_isotope.mass_number())
                && target_is_aromatic == expected_aromatic
        })
    })
}

fn is_hidden_attached_hydrogen(target: &PreparedTarget, atom_id: usize) -> bool {
    target.is_hidden_attached_hydrogen(atom_id)
}

#[cfg(test)]
mod tests {
    use core::str::FromStr;

    use smiles_parser::Smiles;

    use super::{
        component_constraints_match, select_next_query_atom, select_next_query_atom_for_target,
        AtomStereoCache, CompiledQuery, FreshSearchBuffers, MatchScratch, RecursiveMatchCache,
    };
    use crate::error::SmartsMatchError;
    use crate::prepared::PreparedTarget;
    use crate::QueryMol;

    fn query_matches_smiles(smarts: &str, smiles: &str) -> bool {
        QueryMol::from_str(smarts).unwrap().matches(smiles).unwrap()
    }

    #[test]
    fn rejects_empty_target() {
        let query = QueryMol::from_str("C").unwrap();
        let error = query.matches("").unwrap_err();
        assert!(matches!(error, SmartsMatchError::EmptyTarget));
    }

    #[test]
    fn rejects_invalid_target_smiles() {
        let query = QueryMol::from_str("C").unwrap();
        let error = query.matches("C)").unwrap_err();
        assert!(matches!(
            error,
            SmartsMatchError::InvalidTargetSmiles { .. }
        ));
    }

    #[test]
    fn negated_directional_bond_forms_match_nothing_in_current_rdkit_slice() {
        assert!(!QueryMol::from_str("F/!\\C=C/F")
            .unwrap()
            .matches("F/C=C/F")
            .unwrap());
        assert!(!QueryMol::from_str("F/!\\C=C/F")
            .unwrap()
            .matches("F\\C=C\\F")
            .unwrap());
        assert!(!QueryMol::from_str("F\\!/C=C/F")
            .unwrap()
            .matches("F/C=C\\F")
            .unwrap());
        assert!(!QueryMol::from_str("F\\!/C=C/F")
            .unwrap()
            .matches("FC=CF")
            .unwrap());
    }

    #[test]
    fn single_atom_supported_query_matches() {
        let query = QueryMol::from_str("[O;H1]").unwrap();
        assert!(query.matches("CCO").unwrap());
    }

    #[test]
    fn compiled_query_matches_existing_prepared_path() {
        let compiled = CompiledQuery::new(QueryMol::from_str("F/C=C/F").unwrap()).unwrap();
        let prepared = PreparedTarget::new(Smiles::from_str("F/C=C/F").unwrap());
        assert!(compiled.matches(&prepared));
        assert!(compiled.query().matches("F/C=C/F").unwrap());
    }

    #[test]
    fn compiled_query_matches_with_reusable_scratch() {
        let compiled = CompiledQuery::new(QueryMol::from_str("COCOC").unwrap()).unwrap();
        let good = PreparedTarget::new(Smiles::from_str("CCOCOCC").unwrap());
        let bad = PreparedTarget::new(Smiles::from_str("CCCCCCC").unwrap());
        let mut scratch = MatchScratch::new();

        assert!(compiled.matches_with_scratch(&good, &mut scratch));
        assert!(!compiled.matches_with_scratch(&bad, &mut scratch));
        assert_eq!(
            compiled.matches(&good),
            compiled.matches_with_scratch(&good, &mut scratch)
        );
    }

    #[test]
    fn counted_matches_follow_rdkit_uniquify_for_symmetric_queries() {
        let query = QueryMol::from_str("CC").unwrap();
        assert_eq!(query.match_count("CC").unwrap(), 1);
        assert_eq!(
            query.substructure_matches("CC").unwrap().as_ref(),
            &[alloc::boxed::Box::<[usize]>::from([0, 1])]
        );
    }

    #[test]
    fn counted_matches_include_overlapping_embeddings() {
        let query = QueryMol::from_str("CC").unwrap();
        assert_eq!(query.match_count("CCC").unwrap(), 2);
        assert_eq!(
            query.substructure_matches("CCC").unwrap().as_ref(),
            &[
                alloc::boxed::Box::<[usize]>::from([0, 1]),
                alloc::boxed::Box::<[usize]>::from([1, 2]),
            ]
        );
    }

    #[test]
    fn counted_matches_choose_a_canonical_query_order_representative() {
        let query = QueryMol::from_str("C~C~C~C").unwrap();
        assert_eq!(query.match_count("C1CCC=1").unwrap(), 1);
        assert_eq!(
            query.substructure_matches("C1CCC=1").unwrap().as_ref(),
            &[alloc::boxed::Box::<[usize]>::from([0, 1, 2, 3])]
        );
    }

    #[test]
    fn three_atom_tree_fast_path_preserves_counting_and_materialization() {
        let symmetric = CompiledQuery::new(QueryMol::from_str("C(C)C").unwrap()).unwrap();
        let branched_target = PreparedTarget::new(Smiles::from_str("CC(C)C").unwrap());
        assert_eq!(symmetric.match_count(&branched_target), 3);
        assert_eq!(
            symmetric.substructure_matches(&branched_target).as_ref(),
            &[
                alloc::boxed::Box::<[usize]>::from([1, 0, 2]),
                alloc::boxed::Box::<[usize]>::from([1, 0, 3]),
                alloc::boxed::Box::<[usize]>::from([1, 2, 3]),
            ]
        );

        let asymmetric = CompiledQuery::new(QueryMol::from_str("C(=O)O").unwrap()).unwrap();
        let anhydride_like = PreparedTarget::new(Smiles::from_str("OC(=O)OC(=O)O").unwrap());
        assert_eq!(asymmetric.match_count(&anhydride_like), 4);
        assert_eq!(
            asymmetric.substructure_matches(&anhydride_like).as_ref(),
            &[
                alloc::boxed::Box::<[usize]>::from([1, 2, 0]),
                alloc::boxed::Box::<[usize]>::from([1, 2, 3]),
                alloc::boxed::Box::<[usize]>::from([4, 5, 3]),
                alloc::boxed::Box::<[usize]>::from([4, 5, 6]),
            ]
        );
    }

    #[test]
    fn counted_matches_handle_simple_single_atom_cases() {
        assert_eq!(
            QueryMol::from_str("C").unwrap().match_count("CCC").unwrap(),
            3
        );
        assert_eq!(
            QueryMol::from_str("[#8]")
                .unwrap()
                .match_count("O=CO")
                .unwrap(),
            2
        );
        assert_eq!(
            QueryMol::from_str("[#8]")
                .unwrap()
                .match_count("CC.O")
                .unwrap(),
            1
        );
    }

    #[test]
    fn counted_matches_agree_with_boolean_matching() {
        let query = QueryMol::from_str("C.C").unwrap();
        let prepared = PreparedTarget::new(Smiles::from_str("CC").unwrap());
        let compiled = CompiledQuery::new(query).unwrap();
        assert_eq!(
            compiled.matches(&prepared),
            compiled.match_count(&prepared) > 0
        );
    }

    #[test]
    fn target_aware_root_selection_prefers_more_selective_unanchored_component() {
        let compiled = CompiledQuery::new(QueryMol::from_str("C.[Cl-].[Na+]").unwrap()).unwrap();
        let target = PreparedTarget::new(Smiles::from_str("CC.[Cl-].[Na+]").unwrap());
        let mut scratch = FreshSearchBuffers::new(&compiled, &target);
        let mut atom_stereo_cache = AtomStereoCache::new();
        let mut recursive_cache =
            RecursiveMatchCache::new(compiled.recursive_cache_slots, target.atom_count());
        let context =
            scratch
                .view()
                .into_context(&compiled, &mut atom_stereo_cache, &mut recursive_cache);

        assert_eq!(
            select_next_query_atom(
                context.query_neighbors,
                context.query_atom_scores,
                context.query_to_target,
            ),
            0
        );
        assert_eq!(select_next_query_atom_for_target(&context), 1);
    }

    #[test]
    fn anchored_complete_matchers_only_skip_rechecks_when_every_predicate_is_covered() {
        let query = QueryMol::from_str("[A;X1-2].O").unwrap();
        assert!(!query.matches("CC(C)C.O").unwrap());
        assert_eq!(query.match_count("CC(C)C.O").unwrap(), 0);
    }

    #[test]
    fn benchmark_screened_pair_does_not_regress_to_false_positive_match() {
        let query =
            QueryMol::from_str("[#6;D1;H3]-[#6;D2]-[#6;D3]=[#6;D3]-[#6;D3](=[#8;D1])-[#8;D1;H1]")
                .unwrap();
        let target = "C[C@@H]1C2[C@H](C(=O)N2C(=C1OC)C(=O)O)[C@@H](C)O";

        assert!(!query.matches(target).unwrap());
        assert_eq!(query.match_count(target).unwrap(), 0);
    }

    #[test]
    fn low_level_boolean_tree_is_respected() {
        let query = QueryMol::from_str("[!O]").unwrap();
        assert!(query.matches("C").unwrap());
        assert!(!query.matches("O").unwrap());
    }

    #[test]
    fn bracket_or_is_respected() {
        let query = QueryMol::from_str("[N,O]").unwrap();
        assert!(query.matches("CCO").unwrap());
    }

    #[test]
    fn connected_query_does_not_reuse_target_atoms() {
        let query = QueryMol::from_str("CCC").unwrap();
        assert!(!query.matches("CC").unwrap());
    }

    #[test]
    fn connected_query_matches_simple_cycle() {
        let query = QueryMol::from_str("C1CC1").unwrap();
        assert!(query.matches("C1CC1").unwrap());
        assert!(!query.matches("CCC").unwrap());
    }

    #[test]
    fn disconnected_query_matches_without_grouping() {
        let query = QueryMol::from_str("C.C").unwrap();
        assert!(query.matches("CC").unwrap());
        assert!(query.matches("C.C").unwrap());
        assert!(!query.matches("C").unwrap());
    }

    #[test]
    fn bond_boolean_logic_is_respected() {
        let query = QueryMol::from_str("C!#N").unwrap();
        assert!(query.matches("CN").unwrap());
        assert!(!query.matches("CC#N").unwrap());
    }

    #[test]
    fn connectivity_and_valence_primitives_are_respected() {
        assert!(QueryMol::from_str("[X4]").unwrap().matches("C").unwrap());
        assert!(!QueryMol::from_str("[X3]").unwrap().matches("C").unwrap());
        assert!(QueryMol::from_str("[N&X]").unwrap().matches("C#N").unwrap());

        assert!(QueryMol::from_str("[v4]")
            .unwrap()
            .matches("c1ccccc1")
            .unwrap());
        assert!(!QueryMol::from_str("[v3]")
            .unwrap()
            .matches("c1ccccc1")
            .unwrap());
        assert!(QueryMol::from_str("[Cl&v]")
            .unwrap()
            .matches("CCCl")
            .unwrap());
    }

    #[test]
    fn hybridization_primitives_are_respected() {
        assert_eq!(
            QueryMol::from_str("[^1]")
                .unwrap()
                .match_count("CC#N")
                .unwrap(),
            2
        );
        assert_eq!(
            QueryMol::from_str("[^2]")
                .unwrap()
                .match_count("CC=CF")
                .unwrap(),
            2
        );
        assert_eq!(
            QueryMol::from_str("[^2]")
                .unwrap()
                .match_count("c1ccccc1")
                .unwrap(),
            6
        );
        assert_eq!(
            QueryMol::from_str("[^2]")
                .unwrap()
                .match_count("CC(=O)NC")
                .unwrap(),
            3
        );
        assert_eq!(
            QueryMol::from_str("[^3]")
                .unwrap()
                .match_count("CC")
                .unwrap(),
            2
        );
        assert_eq!(
            QueryMol::from_str("[^3]")
                .unwrap()
                .match_count("CC=CF")
                .unwrap(),
            2
        );
    }

    #[test]
    fn ring_atom_primitives_are_respected() {
        assert!(QueryMol::from_str("[R]").unwrap().matches("C1CC1").unwrap());
        assert!(!QueryMol::from_str("[R]").unwrap().matches("CCC").unwrap());
        assert!(QueryMol::from_str("[R0]").unwrap().matches("CCC").unwrap());
        assert!(QueryMol::from_str("[r3]")
            .unwrap()
            .matches("C1CC1")
            .unwrap());
        assert!(QueryMol::from_str("[x2]")
            .unwrap()
            .matches("C1CC1")
            .unwrap());
    }

    #[test]
    fn ring_bond_primitives_are_respected() {
        assert!(QueryMol::from_str("C@C").unwrap().matches("C1CC1").unwrap());
        assert!(!QueryMol::from_str("C@C").unwrap().matches("CC").unwrap());
        assert!(QueryMol::from_str("C!@C").unwrap().matches("CC").unwrap());
        assert!(!QueryMol::from_str("C!@C")
            .unwrap()
            .matches("C1CC1")
            .unwrap());
    }

    #[test]
    fn ordinary_attached_hydrogens_are_not_matchable_as_hydrogen_atoms() {
        assert!(!QueryMol::from_str("[H]").unwrap().matches("[H]Cl").unwrap());
        assert!(!QueryMol::from_str("[#1]")
            .unwrap()
            .matches("[H]Cl")
            .unwrap());
        assert!(!QueryMol::from_str("[H]Cl")
            .unwrap()
            .matches("[H]Cl")
            .unwrap());
        assert!(QueryMol::from_str("[H][H]")
            .unwrap()
            .matches("[H][H]")
            .unwrap());
        assert!(QueryMol::from_str("[2H]")
            .unwrap()
            .matches("[2H]Cl")
            .unwrap());
    }

    #[test]
    fn zero_level_grouping_requires_same_target_component() {
        let query = QueryMol::from_str("(C.C)").unwrap();
        let target = PreparedTarget::new(Smiles::from_str("CC").unwrap());
        let compiled = CompiledQuery::new(query.clone()).unwrap();
        let mapping = [Some(0), None];
        assert_eq!(query.component_groups(), &[Some(0), Some(0)]);
        assert_eq!(target.connected_component(0), Some(0));
        assert_eq!(target.connected_component(1), Some(0));
        assert!(component_constraints_match(1, 1, &query, &target, &mapping));
        assert!(compiled.matches(&target));
        assert!(query.matches("CC").unwrap());
        assert!(QueryMol::from_str("(C.C)")
            .unwrap()
            .matches("CC.C")
            .unwrap());
        assert!(!QueryMol::from_str("(C.C)").unwrap().matches("C.C").unwrap());
    }

    #[test]
    fn separate_groups_require_different_target_components() {
        assert!(!QueryMol::from_str("(C).(C)")
            .unwrap()
            .matches("CC")
            .unwrap());
        assert!(QueryMol::from_str("(C).(C)")
            .unwrap()
            .matches("C.C")
            .unwrap());
        assert!(QueryMol::from_str("(C).(C)")
            .unwrap()
            .matches("CC.C")
            .unwrap());
    }

    #[test]
    fn grouped_and_ungrouped_components_interact_correctly() {
        assert!(QueryMol::from_str("(C).C").unwrap().matches("CC").unwrap());
        assert!(QueryMol::from_str("(C).C").unwrap().matches("C.C").unwrap());
        assert!(QueryMol::from_str("(C).(C).C")
            .unwrap()
            .matches("CC.C")
            .unwrap());
        assert!(!QueryMol::from_str("(C).(C.C)")
            .unwrap()
            .matches("CCC")
            .unwrap());
        assert!(QueryMol::from_str("(C).(C.C)")
            .unwrap()
            .matches("CC.C")
            .unwrap());
    }

    #[test]
    fn recursive_queries_are_respected() {
        assert!(QueryMol::from_str("[$(C)]").unwrap().matches("C").unwrap());
        assert!(!QueryMol::from_str("[$(O)]").unwrap().matches("C").unwrap());
        assert!(QueryMol::from_str("[$(CO)]")
            .unwrap()
            .matches("CO")
            .unwrap());
        assert!(!QueryMol::from_str("[$(CO)]")
            .unwrap()
            .matches("CC")
            .unwrap());
        assert!(QueryMol::from_str("[C&$(*O)]")
            .unwrap()
            .matches("CO")
            .unwrap());
        assert!(!QueryMol::from_str("[C&$(*O)]")
            .unwrap()
            .matches("CC")
            .unwrap());
    }

    #[test]
    fn explicit_tetrahedral_chirality_is_respected() {
        assert!(query_matches_smiles("F[C@](Cl)Br", "F[C@H](Cl)Br"));
        assert!(!query_matches_smiles("F[C@](Cl)Br", "F[C@@H](Cl)Br"));
        assert!(query_matches_smiles("F[C@@](Cl)Br", "F[C@@H](Cl)Br"));
        assert!(!query_matches_smiles("F[C@@](Cl)Br", "F[C@H](Cl)Br"));
        assert!(query_matches_smiles("F[C@H](Cl)Br", "F[C@H](Cl)Br"));
        assert!(!query_matches_smiles("F[C@H](Cl)Br", "F[C@@H](Cl)Br"));
        assert!(query_matches_smiles("F[C@@H](Cl)Br", "F[C@@H](Cl)Br"));
        assert!(!query_matches_smiles("F[C@@H](Cl)Br", "F[C@H](Cl)Br"));
        assert!(query_matches_smiles("F[C@TH1](Cl)Br", "F[C@H](Cl)Br"));
        assert!(!query_matches_smiles("F[C@TH1](Cl)Br", "F[C@@H](Cl)Br"));
        assert!(query_matches_smiles("F[C@TH2](Cl)Br", "F[C@@H](Cl)Br"));
        assert!(!query_matches_smiles("F[C@TH2](Cl)Br", "FC(Cl)Br"));
        assert!(query_matches_smiles("Br[C@TH1](Cl)F", "F[C@@H](Cl)Br"));
        assert!(!query_matches_smiles("Br[C@TH1](Cl)F", "F[C@H](Cl)Br"));
        assert!(query_matches_smiles("[C@TH2](Br)(Cl)F", "F[C@H](Cl)Br"));
        assert!(!query_matches_smiles("[C@TH2](Br)(Cl)F", "F[C@@H](Cl)Br"));
        assert!(query_matches_smiles("Br[C@](Cl)F", "F[C@@H](Cl)Br"));
        assert!(query_matches_smiles("Br[C@@](Cl)F", "F[C@H](Cl)Br"));
        assert!(query_matches_smiles("Br[C@H](Cl)F", "F[C@@H](Cl)Br"));
        assert!(!query_matches_smiles("Br[C@H](Cl)F", "F[C@H](Cl)Br"));
        assert!(query_matches_smiles("[C@H](F)(Cl)Br", "F[C@@H](Cl)Br"));
        assert!(!query_matches_smiles("[C@H](F)(Cl)Br", "F[C@H](Cl)Br"));
        assert!(query_matches_smiles("[C@@H](F)(Cl)Br", "F[C@H](Cl)Br"));
        assert!(!query_matches_smiles("[C@@H](F)(Cl)Br", "F[C@@H](Cl)Br"));
    }

    #[test]
    fn underconstrained_tetrahedral_queries_do_not_overconstrain() {
        assert!(QueryMol::from_str("[C@](F)Cl")
            .unwrap()
            .matches("F[C@H](Cl)Br")
            .unwrap());
        assert!(QueryMol::from_str("[C@](F)Cl")
            .unwrap()
            .matches("F[C@@H](Cl)Br")
            .unwrap());
        assert!(QueryMol::from_str("[C@@](F)Cl")
            .unwrap()
            .matches("F[C@H](Cl)Br")
            .unwrap());
        assert!(QueryMol::from_str("[C@@](F)Cl")
            .unwrap()
            .matches("F[C@@H](Cl)Br")
            .unwrap());
        assert!(QueryMol::from_str("[C@H](F)Cl")
            .unwrap()
            .matches("F[C@H](Cl)Br")
            .unwrap());
        assert!(QueryMol::from_str("[C@H](F)Cl")
            .unwrap()
            .matches("F[C@@H](Cl)Br")
            .unwrap());
        assert!(QueryMol::from_str("[C@@H](F)Cl")
            .unwrap()
            .matches("F[C@H](Cl)Br")
            .unwrap());
        assert!(QueryMol::from_str("[C@@H](F)Cl")
            .unwrap()
            .matches("F[C@@H](Cl)Br")
            .unwrap());
    }

    #[test]
    fn supported_nontetrahedral_query_classes_do_not_constrain_matches() {
        for smarts in [
            "[C@AL1](F)(Cl)Br",
            "[C@AL2](F)(Cl)Br",
            "[C@SP1](F)(Cl)Br",
            "[C@SP2](F)(Cl)Br",
            "[C@SP3](F)(Cl)Br",
            "[C@TB1](F)(Cl)Br",
            "[C@TB20](F)(Cl)Br",
            "[C@OH1](F)(Cl)Br",
            "[C@OH30](F)(Cl)Br",
        ] {
            let query = QueryMol::from_str(smarts).unwrap();
            assert!(query.matches("F[C@H](Cl)Br").unwrap(), "{smarts}");
            assert!(query.matches("F[C@@H](Cl)Br").unwrap(), "{smarts}");
            assert!(query.matches("FC(Cl)Br").unwrap(), "{smarts}");
        }
    }

    #[test]
    fn multidirectional_endpoint_queries_are_respected() {
        assert!(query_matches_smiles("F/C=C(/Cl)\\Br", "F/C=C(/Cl)Br"));
        assert!(!query_matches_smiles("F/C=C(/Cl)\\Br", "F/C=C(\\Cl)Br"));
        assert!(query_matches_smiles("F/,\\C=C/F", "F/C=C/F"));
        assert!(query_matches_smiles("F/,\\C=C/F", "F\\C=C\\F"));
        assert!(!query_matches_smiles("F/,\\C=C/F", "F/C=C\\F"));
        assert!(query_matches_smiles("F/;\\C=C/F", "F/C=C/F"));
        assert!(query_matches_smiles("F\\,/C=C/F", "F/C=C\\F"));
        assert!(query_matches_smiles("F/&\\C=C/F", "F/C=C/F"));
        assert!(query_matches_smiles("F\\&/C=C/F", "F/C=C\\F"));
    }

    #[test]
    fn semantic_double_bond_stereo_is_respected() {
        assert!(QueryMol::from_str("C/C").unwrap().matches("CC").unwrap());
        assert!(QueryMol::from_str("F/C=C")
            .unwrap()
            .matches("F/C=C/F")
            .unwrap());
        assert!(QueryMol::from_str("F/C=C")
            .unwrap()
            .matches("F/C=C\\F")
            .unwrap());
        assert!(QueryMol::from_str("F/C=C")
            .unwrap()
            .matches("FC=CF")
            .unwrap());
        assert!(QueryMol::from_str("C=C/F")
            .unwrap()
            .matches("F/C=C/F")
            .unwrap());
        assert!(QueryMol::from_str("C=C/F")
            .unwrap()
            .matches("FC=CF")
            .unwrap());
        assert!(QueryMol::from_str("F/C=C/F")
            .unwrap()
            .matches("F/C=C/F")
            .unwrap());
        assert!(QueryMol::from_str("F/C=C/F")
            .unwrap()
            .matches("F\\C=C\\F")
            .unwrap());
        assert!(!QueryMol::from_str("F/C=C/F")
            .unwrap()
            .matches("F/C=C\\F")
            .unwrap());
        assert!(!QueryMol::from_str("F/C=C/F")
            .unwrap()
            .matches("FC=CF")
            .unwrap());
        assert!(QueryMol::from_str("F/C=C\\F")
            .unwrap()
            .matches("F\\C=C/F")
            .unwrap());
        assert!(QueryMol::from_str("C/C=C/C")
            .unwrap()
            .matches("C/C=C/C")
            .unwrap());
        assert!(!QueryMol::from_str("C/C=C/C")
            .unwrap()
            .matches("C/C=C\\C")
            .unwrap());
        assert!(!QueryMol::from_str("C/C=C/C")
            .unwrap()
            .matches("CC=CC")
            .unwrap());
    }
}
