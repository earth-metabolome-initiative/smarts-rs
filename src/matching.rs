use alloc::{
    boxed::Box,
    collections::{BTreeMap, BTreeSet},
    string::String,
    string::ToString,
};
use core::cell::{Cell, RefCell};
#[cfg(all(
    feature = "std",
    not(all(target_family = "wasm", target_os = "unknown"))
))]
use std::time::{Duration, Instant};

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

struct SearchContext<'a, L: MatchLimiter> {
    compiled_query: &'a CompiledQuery,
    query_neighbors: &'a [alloc::vec::Vec<(QueryAtomId, usize)>],
    query_atom_scores: &'a [usize],
    query_atom_anchor_widths: &'a [usize],
    query_to_target: &'a mut [Option<usize>],
    used_target_atoms: &'a mut [u32],
    used_target_generation: u32,
    bondless_query_order: &'a mut alloc::vec::Vec<QueryAtomId>,
    bondless_candidate_offsets: &'a mut alloc::vec::Vec<usize>,
    bondless_candidates: &'a mut alloc::vec::Vec<usize>,
    bondless_target_assignments: &'a mut alloc::vec::Vec<Option<QueryAtomId>>,
    bondless_seen_targets: &'a mut alloc::vec::Vec<bool>,
    atom_stereo_cache: &'a mut AtomStereoCache,
    recursive_cache: &'a mut RecursiveMatchCache,
    limit: &'a L,
}

struct SearchScratchView<'a, L: MatchLimiter> {
    query_atom_anchor_widths: &'a [usize],
    query_to_target: &'a mut [Option<usize>],
    used_target_atoms: &'a mut [u32],
    used_target_generation: u32,
    bondless_query_order: &'a mut alloc::vec::Vec<QueryAtomId>,
    bondless_candidate_offsets: &'a mut alloc::vec::Vec<usize>,
    bondless_candidates: &'a mut alloc::vec::Vec<usize>,
    bondless_target_assignments: &'a mut alloc::vec::Vec<Option<QueryAtomId>>,
    bondless_seen_targets: &'a mut alloc::vec::Vec<bool>,
    limit: &'a L,
}

struct SearchScratchParts<'a, L: MatchLimiter> {
    view: SearchScratchView<'a, L>,
    atom_stereo_cache: &'a mut AtomStereoCache,
    recursive_cache: &'a mut RecursiveMatchCache,
}

#[derive(Debug, Clone)]
struct FreshSearchBuffers {
    query_atom_anchor_widths: alloc::vec::Vec<usize>,
    query_to_target: alloc::vec::Vec<Option<usize>>,
    used_target_atoms: alloc::vec::Vec<u32>,
    bondless_query_order: alloc::vec::Vec<QueryAtomId>,
    bondless_candidate_offsets: alloc::vec::Vec<usize>,
    bondless_candidates: alloc::vec::Vec<usize>,
    bondless_target_assignments: alloc::vec::Vec<Option<QueryAtomId>>,
    bondless_seen_targets: alloc::vec::Vec<bool>,
}

impl FreshSearchBuffers {
    fn new(
        query: &CompiledQuery,
        target: &PreparedTarget,
        recursive_cache: &mut RecursiveMatchCache,
    ) -> Self {
        Self::new_with_limit(query, target, recursive_cache, &NO_MATCH_LIMIT)
    }

    fn new_with_limit<L: MatchLimiter>(
        query: &CompiledQuery,
        target: &PreparedTarget,
        recursive_cache: &mut RecursiveMatchCache,
        limit: &L,
    ) -> Self {
        let mut query_atom_anchor_widths = alloc::vec::Vec::new();
        prepare_query_atom_search_widths(
            &mut query_atom_anchor_widths,
            query,
            target,
            recursive_cache,
            limit,
        );
        Self {
            query_atom_anchor_widths,
            query_to_target: alloc::vec![None; query.query.atom_count()],
            used_target_atoms: alloc::vec![0; target.atom_count()],
            bondless_query_order: alloc::vec::Vec::new(),
            bondless_candidate_offsets: alloc::vec::Vec::new(),
            bondless_candidates: alloc::vec::Vec::new(),
            bondless_target_assignments: alloc::vec::Vec::new(),
            bondless_seen_targets: alloc::vec::Vec::new(),
        }
    }

    fn view(&mut self) -> SearchScratchView<'_, NoMatchLimit> {
        self.view_with_limit(&NO_MATCH_LIMIT)
    }

    fn view_with_limit<'a, L: MatchLimiter>(
        &'a mut self,
        limit: &'a L,
    ) -> SearchScratchView<'a, L> {
        SearchScratchView {
            query_atom_anchor_widths: &self.query_atom_anchor_widths,
            query_to_target: &mut self.query_to_target,
            used_target_atoms: &mut self.used_target_atoms,
            used_target_generation: 1,
            bondless_query_order: &mut self.bondless_query_order,
            bondless_candidate_offsets: &mut self.bondless_candidate_offsets,
            bondless_candidates: &mut self.bondless_candidates,
            bondless_target_assignments: &mut self.bondless_target_assignments,
            bondless_seen_targets: &mut self.bondless_seen_targets,
            limit,
        }
    }
}

struct AtomMatchContext<'a, 'cache, L: MatchLimiter> {
    target: &'a PreparedTarget,
    recursive_query_lookup: &'a BTreeMap<usize, usize>,
    recursive_queries: &'a [RecursiveQueryEntry],
    recursive_cache: &'cache mut RecursiveMatchCache,
    limit: &'a L,
}

impl<'a, 'cache, L: MatchLimiter> AtomMatchContext<'a, 'cache, L> {
    const fn new(
        query: &'a CompiledQuery,
        target: &'a PreparedTarget,
        recursive_cache: &'cache mut RecursiveMatchCache,
        limit: &'a L,
    ) -> Self {
        Self {
            target,
            recursive_query_lookup: &query.recursive_query_lookup,
            recursive_queries: &query.recursive_queries,
            recursive_cache,
            limit,
        }
    }
}

impl<'a, L: MatchLimiter> SearchScratchView<'a, L> {
    fn into_context(
        self,
        query: &'a CompiledQuery,
        atom_stereo_cache: &'a mut AtomStereoCache,
        recursive_cache: &'a mut RecursiveMatchCache,
    ) -> SearchContext<'a, L> {
        SearchContext {
            compiled_query: query,
            query_neighbors: &query.query_neighbors,
            query_atom_scores: &query.query_atom_scores,
            query_atom_anchor_widths: self.query_atom_anchor_widths,
            query_to_target: self.query_to_target,
            used_target_atoms: self.used_target_atoms,
            used_target_generation: self.used_target_generation,
            bondless_query_order: self.bondless_query_order,
            bondless_candidate_offsets: self.bondless_candidate_offsets,
            bondless_candidates: self.bondless_candidates,
            bondless_target_assignments: self.bondless_target_assignments,
            bondless_seen_targets: self.bondless_seen_targets,
            atom_stereo_cache,
            recursive_cache,
            limit: self.limit,
        }
    }
}

impl<L: MatchLimiter> SearchContext<'_, L> {
    #[inline]
    fn continue_search(&self) -> bool {
        self.limit.continue_search()
    }

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

#[derive(Debug, Clone)]
struct ComponentPlanEntry {
    component_id: usize,
    group: Option<usize>,
    atom_count: usize,
    precheckable: bool,
    matcher: ComponentMatcher,
}

#[derive(Debug, Clone)]
enum ComponentMatcher {
    Compiled(Box<CompiledQuery>),
    SingleAtom(QueryAtomId),
    RecursiveComponent,
}

impl ComponentPlanEntry {
    const fn supports_precheck(&self) -> bool {
        self.precheckable
    }

    const fn supports_component_search(&self) -> bool {
        match &self.matcher {
            ComponentMatcher::Compiled(_)
            | ComponentMatcher::SingleAtom(_)
            | ComponentMatcher::RecursiveComponent => true,
        }
    }
}

struct ComponentEmbedding {
    target_atoms: Box<[usize]>,
    target_component: Option<usize>,
}

struct ComponentEmbeddingSet<'a> {
    entry: &'a ComponentPlanEntry,
    embeddings: alloc::vec::Vec<ComponentEmbedding>,
}

struct ComponentAssignmentFrame {
    next_embedding: usize,
    selected_embedding: Option<usize>,
    previous_group_target: Option<usize>,
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

trait MatchLimiter {
    fn continue_search(&self) -> bool;

    fn exceeded(&self) -> bool;
}

const MATCH_LIMIT_POLL_INTERVAL: usize = 4096;

struct NoMatchLimit;

static NO_MATCH_LIMIT: NoMatchLimit = NoMatchLimit;

impl MatchLimiter for NoMatchLimit {
    #[inline]
    fn continue_search(&self) -> bool {
        true
    }

    #[inline]
    fn exceeded(&self) -> bool {
        false
    }
}

struct MatchInterrupt<F> {
    checks_until_poll: Cell<usize>,
    exceeded: Cell<bool>,
    should_stop: RefCell<F>,
}

impl<F> MatchInterrupt<F> {
    const fn new(should_stop: F) -> Self {
        Self {
            checks_until_poll: Cell::new(0),
            exceeded: Cell::new(false),
            should_stop: RefCell::new(should_stop),
        }
    }
}

impl<F> MatchLimiter for MatchInterrupt<F>
where
    F: FnMut() -> bool,
{
    #[inline]
    fn continue_search(&self) -> bool {
        if self.exceeded.get() {
            return false;
        }

        let checks_until_poll = self.checks_until_poll.get();
        if checks_until_poll > 0 {
            self.checks_until_poll.set(checks_until_poll - 1);
            return true;
        }
        self.checks_until_poll.set(MATCH_LIMIT_POLL_INTERVAL);

        if (self.should_stop.borrow_mut())() {
            self.exceeded.set(true);
            return false;
        }
        true
    }

    #[inline]
    fn exceeded(&self) -> bool {
        self.exceeded.get()
    }
}

#[cfg(all(
    feature = "std",
    not(all(target_family = "wasm", target_os = "unknown"))
))]
struct MatchTimeLimit {
    deadline: Instant,
    checks_until_poll: Cell<usize>,
    exceeded: Cell<bool>,
}

#[cfg(all(
    feature = "std",
    not(all(target_family = "wasm", target_os = "unknown"))
))]
impl MatchTimeLimit {
    fn new(max_elapsed: Duration) -> Self {
        let started = Instant::now();
        Self {
            deadline: started.checked_add(max_elapsed).unwrap_or(started),
            checks_until_poll: Cell::new(0),
            exceeded: Cell::new(false),
        }
    }
}

#[cfg(all(
    feature = "std",
    not(all(target_family = "wasm", target_os = "unknown"))
))]
impl MatchLimiter for MatchTimeLimit {
    #[inline]
    fn continue_search(&self) -> bool {
        if self.exceeded.get() {
            return false;
        }

        let checks_until_poll = self.checks_until_poll.get();
        if checks_until_poll > 0 {
            self.checks_until_poll.set(checks_until_poll - 1);
            return true;
        }
        self.checks_until_poll.set(MATCH_LIMIT_POLL_INTERVAL);

        if Instant::now() >= self.deadline {
            self.exceeded.set(true);
            return false;
        }
        true
    }

    #[inline]
    fn exceeded(&self) -> bool {
        self.exceeded.get()
    }
}

/// Result of a limited boolean SMARTS match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchLimitResult {
    /// Matching completed within the requested limit.
    Complete(bool),
    /// Matching stopped because the limit was exceeded.
    Exceeded,
}

impl MatchLimitResult {
    /// Converts the result to an optional boolean.
    ///
    /// Returns `None` when the limit was exceeded.
    #[inline]
    #[must_use]
    pub const fn into_option(self) -> Option<bool> {
        match self {
            Self::Complete(matched) => Some(matched),
            Self::Exceeded => None,
        }
    }
}

/// Result of a limited SMARTS match outcome.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatchOutcomeLimitResult {
    /// Matching and coverage accumulation completed within the requested limit.
    Complete(MatchOutcome),
    /// Matching stopped because the limit was exceeded.
    Exceeded,
}

impl MatchOutcomeLimitResult {
    /// Converts the result to an optional match outcome.
    ///
    /// Returns `None` when the limit was exceeded.
    #[inline]
    #[must_use]
    pub const fn into_option(self) -> Option<MatchOutcome> {
        match self {
            Self::Complete(outcome) => Some(outcome),
            Self::Exceeded => None,
        }
    }
}

/// Boolean SMARTS match result plus target-topology coverage.
///
/// `coverage` is the fraction of target atoms and target bonds covered by the
/// union of all accepted full embeddings:
///
/// `(covered target atoms + covered target bonds) / (target atoms + target bonds)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatchOutcome {
    /// Whether at least one full SMARTS embedding matched the target.
    pub matched: bool,
    /// Combined target atom/bond coverage for the accepted full embeddings.
    pub coverage: f64,
}

impl MatchOutcome {
    /// Builds a match outcome from a boolean match flag and combined coverage.
    #[inline]
    #[must_use]
    pub const fn new(matched: bool, coverage: f64) -> Self {
        Self { matched, coverage }
    }
}

#[derive(Debug, Clone)]
struct MatchCoverageAccumulator {
    matched: bool,
    covered_atoms: alloc::vec::Vec<bool>,
    covered_bonds: BTreeSet<(usize, usize)>,
}

impl MatchCoverageAccumulator {
    fn new(target: &PreparedTarget) -> Self {
        Self {
            matched: false,
            covered_atoms: alloc::vec![false; target.atom_count()],
            covered_bonds: BTreeSet::new(),
        }
    }

    fn cover_mapping(
        &mut self,
        query: &QueryMol,
        target: &PreparedTarget,
        query_to_target: &[Option<usize>],
    ) {
        self.matched = true;
        for target_atom in query_to_target
            .iter()
            .copied()
            .map(|target_atom| target_atom.expect("complete mappings must bind every query atom"))
        {
            self.covered_atoms[target_atom] = true;
        }

        for query_bond in query.bonds() {
            let left = query_to_target[query_bond.src]
                .expect("complete mappings must bind every query bond source");
            let right = query_to_target[query_bond.dst]
                .expect("complete mappings must bind every query bond destination");
            if target.bond(left, right).is_some() {
                self.covered_bonds
                    .insert(canonical_target_bond_key(left, right));
            }
        }
    }

    fn into_outcome(self, target: &PreparedTarget) -> MatchOutcome {
        if !self.matched {
            return MatchOutcome::new(false, 0.0);
        }

        let denominator = target.atom_count() + target.bond_count();
        if denominator == 0 {
            return MatchOutcome::new(true, 0.0);
        }

        let covered_atom_count = self
            .covered_atoms
            .iter()
            .filter(|&&covered| covered)
            .count();
        let numerator = covered_atom_count + self.covered_bonds.len();
        MatchOutcome::new(true, coverage_ratio(numerator, denominator))
    }
}

#[allow(clippy::cast_precision_loss)]
fn coverage_ratio(numerator: usize, denominator: usize) -> f64 {
    numerator as f64 / denominator as f64
}

#[inline]
const fn canonical_target_bond_key(left: usize, right: usize) -> (usize, usize) {
    if left < right {
        (left, right)
    } else {
        (right, left)
    }
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
    bondless_query_order: alloc::vec::Vec<QueryAtomId>,
    bondless_candidate_offsets: alloc::vec::Vec<usize>,
    bondless_candidates: alloc::vec::Vec<usize>,
    bondless_target_assignments: alloc::vec::Vec<Option<QueryAtomId>>,
    bondless_seen_targets: alloc::vec::Vec<bool>,
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
            bondless_query_order: alloc::vec::Vec::new(),
            bondless_candidate_offsets: alloc::vec::Vec::new(),
            bondless_candidates: alloc::vec::Vec::new(),
            bondless_target_assignments: alloc::vec::Vec::new(),
            bondless_seen_targets: alloc::vec::Vec::new(),
            atom_stereo_cache: AtomStereoCache::new(),
            recursive_cache: RecursiveMatchCache::new(0, 0),
        }
    }

    fn prepare(&mut self, query: &CompiledQuery, target: &PreparedTarget) {
        self.prepare_caches(query, target);
        self.prepare_search_buffers(query, target, &NO_MATCH_LIMIT);
    }

    fn prepare_with_limit<L: MatchLimiter>(
        &mut self,
        query: &CompiledQuery,
        target: &PreparedTarget,
        limit: &L,
    ) {
        self.prepare_caches(query, target);
        self.prepare_search_buffers(query, target, limit);
    }

    fn search_parts(&mut self) -> SearchScratchParts<'_, NoMatchLimit> {
        self.search_parts_with_limit(&NO_MATCH_LIMIT)
    }

    fn search_parts_with_limit<'a, L: MatchLimiter>(
        &'a mut self,
        limit: &'a L,
    ) -> SearchScratchParts<'a, L> {
        let Self {
            query_atom_anchor_widths,
            query_to_target,
            used_target_atoms,
            used_target_generation,
            bondless_query_order,
            bondless_candidate_offsets,
            bondless_candidates,
            bondless_target_assignments,
            bondless_seen_targets,
            atom_stereo_cache,
            recursive_cache,
        } = self;
        SearchScratchParts {
            view: SearchScratchView {
                query_atom_anchor_widths,
                query_to_target,
                used_target_atoms,
                used_target_generation: *used_target_generation,
                bondless_query_order,
                bondless_candidate_offsets,
                bondless_candidates,
                bondless_target_assignments,
                bondless_seen_targets,
                limit,
            },
            atom_stereo_cache,
            recursive_cache,
        }
    }

    fn prepare_search_buffers<L: MatchLimiter>(
        &mut self,
        query: &CompiledQuery,
        target: &PreparedTarget,
        limit: &L,
    ) {
        let query_atom_count = query.query.atom_count();
        prepare_query_atom_search_widths(
            &mut self.query_atom_anchor_widths,
            query,
            target,
            &mut self.recursive_cache,
            limit,
        );
        self.prepare_mapping_buffers(query_atom_count, target);
    }

    fn prepare_mapping_buffers(&mut self, query_atom_count: usize, target: &PreparedTarget) {
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
    }

    fn prepare_caches(&mut self, query: &CompiledQuery, target: &PreparedTarget) {
        let target_atom_count = target.atom_count();
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

    /// Match this SMARTS query against a target `SMILES` string and return
    /// both the boolean result and combined target atom/bond coverage.
    ///
    /// Coverage is computed from the union of target atoms and bonds covered by
    /// accepted full embeddings. If the SMARTS does not match, coverage is
    /// `0.0`.
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
    pub fn match_outcome(&self, target: &str) -> Result<MatchOutcome, SmartsMatchError> {
        let target = prepare_target_smiles(target)?;
        self.match_outcome_prepared(&target)
    }

    /// Match this SMARTS query against a prepared target and return both the
    /// boolean result and combined target atom/bond coverage.
    ///
    /// This convenience method derives reusable query-side state for each
    /// call. For repeated matching of one SMARTS against many targets, prefer
    /// [`CompiledQuery`] plus [`CompiledQuery::match_outcome`].
    ///
    /// # Errors
    ///
    /// Returns:
    /// - [`SmartsMatchError::UnsupportedAtomPrimitive`] when the query uses an
    ///   atom primitive outside the current matcher implementation
    /// - [`SmartsMatchError::UnsupportedBondPrimitive`] when the query uses a
    ///   bond primitive outside the current matcher implementation
    pub fn match_outcome_prepared(
        &self,
        target: &PreparedTarget,
    ) -> Result<MatchOutcome, SmartsMatchError> {
        let compiled = CompiledQuery::new(self.clone())?;
        Ok(compiled.match_outcome(target))
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
    has_stereo_constraints: bool,
    atom_matchers: Box<[CompiledAtomMatcher]>,
    bond_matchers: Box<[CompiledBondMatcher]>,
    has_component_constraints: bool,
    recursive_cache_slots: usize,
    recursive_query_lookup: BTreeMap<usize, usize>,
    recursive_queries: Box<[RecursiveQueryEntry]>,
    component_plan: Box<[ComponentPlanEntry]>,
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
        let has_stereo_constraints = query_has_stereo_constraints(&query, &stereo_plan);
        let has_component_constraints = query.component_groups().iter().any(Option::is_some);
        let recursive_queries = compile_recursive_queries(&query, next_recursive_cache_slot)?;
        let recursive_query_lookup = recursive_queries
            .iter()
            .enumerate()
            .map(|(index, entry)| (entry.query_key, index))
            .collect();
        let component_plan = compile_component_plan(&query)?;
        Ok(Self {
            query,
            query_neighbors,
            query_degrees,
            query_atom_scores,
            stereo_plan,
            has_stereo_constraints,
            atom_matchers,
            bond_matchers,
            has_component_constraints,
            recursive_cache_slots: *next_recursive_cache_slot,
            recursive_query_lookup,
            recursive_queries,
            component_plan,
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

    /// Match this compiled SMARTS query against a prepared target and return
    /// both the boolean result and combined target atom/bond coverage.
    ///
    /// This enumerates accepted full embeddings once and accumulates the union
    /// of covered target atoms and bonds. Use [`Self::matches`] when only a
    /// boolean early-exit answer is needed.
    #[inline]
    #[must_use]
    pub fn match_outcome(&self, target: &PreparedTarget) -> MatchOutcome {
        query_match_outcome(self, target)
    }

    /// Match this compiled SMARTS query against a prepared target using
    /// reusable per-thread scratch buffers, returning both the boolean result
    /// and combined target atom/bond coverage.
    #[inline]
    #[must_use]
    pub fn match_outcome_with_scratch(
        &self,
        target: &PreparedTarget,
        scratch: &mut MatchScratch,
    ) -> MatchOutcome {
        query_match_outcome_with_scratch(self, target, scratch)
    }

    /// Match this compiled SMARTS query and accumulate coverage with a
    /// cooperative caller-provided interrupt predicate.
    ///
    /// The matcher calls `should_stop` periodically from its search loop. A
    /// result of [`MatchOutcomeLimitResult::Exceeded`] means the boolean result
    /// and coverage are unknown.
    #[inline]
    #[must_use]
    pub fn match_outcome_with_interrupt<F>(
        &self,
        target: &PreparedTarget,
        should_stop: F,
    ) -> MatchOutcomeLimitResult
    where
        F: FnMut() -> bool,
    {
        let mut scratch = MatchScratch::new();
        self.match_outcome_with_scratch_and_interrupt(target, &mut scratch, should_stop)
    }

    /// Match this compiled SMARTS query using reusable scratch and accumulate
    /// coverage with a cooperative caller-provided interrupt predicate.
    ///
    /// Use this for wasm by closing over a host-provided clock, for example a
    /// JavaScript deadline check. The predicate is polled periodically; it is
    /// not hard preemption.
    #[inline]
    #[must_use]
    pub fn match_outcome_with_scratch_and_interrupt<F>(
        &self,
        target: &PreparedTarget,
        scratch: &mut MatchScratch,
        should_stop: F,
    ) -> MatchOutcomeLimitResult
    where
        F: FnMut() -> bool,
    {
        query_match_outcome_with_scratch_and_interrupt(self, target, scratch, should_stop)
    }

    /// Match this compiled SMARTS query and accumulate coverage with a
    /// cooperative wall-clock limit.
    ///
    /// The matcher checks the clock periodically from its search loop. This is
    /// a safety fuse, not hard preemption: a result of
    /// [`MatchOutcomeLimitResult::Exceeded`] means the boolean result and
    /// coverage are unknown.
    #[cfg(all(
        feature = "std",
        not(all(target_family = "wasm", target_os = "unknown"))
    ))]
    #[inline]
    #[must_use]
    pub fn match_outcome_with_time_limit(
        &self,
        target: &PreparedTarget,
        max_elapsed: Duration,
    ) -> MatchOutcomeLimitResult {
        let mut scratch = MatchScratch::new();
        self.match_outcome_with_scratch_and_time_limit(target, &mut scratch, max_elapsed)
    }

    /// Match this compiled SMARTS query using reusable scratch and accumulate
    /// coverage with a cooperative wall-clock limit.
    ///
    /// Returns [`MatchOutcomeLimitResult::Exceeded`] if the limit is exceeded
    /// before proving either a non-match or the complete coverage for all
    /// accepted embeddings.
    #[cfg(all(
        feature = "std",
        not(all(target_family = "wasm", target_os = "unknown"))
    ))]
    #[inline]
    #[must_use]
    pub fn match_outcome_with_scratch_and_time_limit(
        &self,
        target: &PreparedTarget,
        scratch: &mut MatchScratch,
        max_elapsed: Duration,
    ) -> MatchOutcomeLimitResult {
        query_match_outcome_with_scratch_and_time_limit(self, target, scratch, max_elapsed)
    }

    /// Match this compiled SMARTS query with a cooperative caller-provided
    /// interrupt predicate.
    ///
    /// The matcher calls `should_stop` periodically from its search loop. This
    /// is a portable safety fuse for targets without a Rust `Instant`, including
    /// `wasm32-unknown-unknown`. A result of [`MatchLimitResult::Exceeded`]
    /// means the truth value is unknown.
    #[inline]
    #[must_use]
    pub fn matches_with_interrupt<F>(
        &self,
        target: &PreparedTarget,
        should_stop: F,
    ) -> MatchLimitResult
    where
        F: FnMut() -> bool,
    {
        let mut scratch = MatchScratch::new();
        self.matches_with_scratch_and_interrupt(target, &mut scratch, should_stop)
    }

    /// Match this compiled SMARTS query with reusable scratch and a cooperative
    /// caller-provided interrupt predicate.
    ///
    /// Use this for wasm by closing over a host-provided clock, for example a
    /// JavaScript deadline check. The predicate is polled periodically; it is
    /// not hard preemption.
    #[inline]
    #[must_use]
    pub fn matches_with_scratch_and_interrupt<F>(
        &self,
        target: &PreparedTarget,
        scratch: &mut MatchScratch,
        should_stop: F,
    ) -> MatchLimitResult
    where
        F: FnMut() -> bool,
    {
        query_matches_with_scratch_and_interrupt(self, target, scratch, should_stop)
    }

    /// Match this compiled SMARTS query with a cooperative wall-clock limit.
    ///
    /// The matcher checks the clock periodically from its search loop. This is
    /// a safety fuse, not hard preemption: a result of
    /// [`MatchLimitResult::Exceeded`] means the truth value is unknown.
    #[cfg(all(
        feature = "std",
        not(all(target_family = "wasm", target_os = "unknown"))
    ))]
    #[inline]
    #[must_use]
    pub fn matches_with_time_limit(
        &self,
        target: &PreparedTarget,
        max_elapsed: Duration,
    ) -> MatchLimitResult {
        let mut scratch = MatchScratch::new();
        self.matches_with_scratch_and_time_limit(target, &mut scratch, max_elapsed)
    }

    /// Match this compiled SMARTS query with reusable scratch and a cooperative
    /// wall-clock limit.
    ///
    /// Returns [`MatchLimitResult::Exceeded`] if the limit is exceeded before
    /// proving either a match or a non-match.
    #[cfg(all(
        feature = "std",
        not(all(target_family = "wasm", target_os = "unknown"))
    ))]
    #[inline]
    #[must_use]
    pub fn matches_with_scratch_and_time_limit(
        &self,
        target: &PreparedTarget,
        scratch: &mut MatchScratch,
        max_elapsed: Duration,
    ) -> MatchLimitResult {
        query_matches_with_scratch_and_time_limit(self, target, scratch, max_elapsed)
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
        BracketExprTree::Not(_) => {
            compile_complete_bracket_predicates(double_negated_bracket_tree(tree)?)
        }
        BracketExprTree::Or(_) => None,
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
        AtomPrimitive::RecursiveQuery(query) => {
            return single_atom_recursive_query_expr(query)
                .and_then(compile_complete_atom_predicates);
        }
    };
    Some(predicates)
}

fn single_atom_recursive_query_expr(query: &QueryMol) -> Option<&AtomExpr> {
    if query.atom_count() != 1 || query.bond_count() != 0 {
        return None;
    }
    match &query.atoms()[0].expr {
        AtomExpr::Bracket(expr) if expr.atom_map.is_some() => None,
        expr => Some(expr),
    }
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
        BracketExprTree::Not(_) => {
            if let Some(inner) = double_negated_bracket_tree(tree) {
                collect_required_bracket_predicates(inner, predicates);
            }
        }
        BracketExprTree::HighAnd(items) | BracketExprTree::LowAnd(items) => {
            for item in items {
                collect_required_bracket_predicates(item, predicates);
            }
        }
        BracketExprTree::Or(_) => {}
    }
}

fn double_negated_bracket_tree(tree: &BracketExprTree) -> Option<&BracketExprTree> {
    let BracketExprTree::Not(inner) = tree else {
        return None;
    };
    let BracketExprTree::Not(grandchild) = inner.as_ref() else {
        return None;
    };
    Some(grandchild)
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

fn query_has_stereo_constraints(query: &QueryMol, stereo_plan: &QueryStereoPlan) -> bool {
    !stereo_plan.double_bond_constraints.is_empty()
        || query
            .atoms()
            .iter()
            .any(|atom| extract_query_chirality(&atom.expr).is_some())
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
        BracketExprTree::Not(_) => double_negated_bracket_tree(tree)
            .map(bracket_expr_order_score)
            .unwrap_or_default(),
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

fn compile_component_plan(query: &QueryMol) -> Result<Box<[ComponentPlanEntry]>, SmartsMatchError> {
    if query.component_count() <= 1 {
        return Ok(Box::default());
    }

    let mut entries = alloc::vec::Vec::new();
    for component_id in 0..query.component_count() {
        let component_atoms = query.component_atoms(component_id);
        let precheckable = !component_contains_recursive_predicate(query, component_id);
        let matcher = if let [query_atom] = component_atoms {
            ComponentMatcher::SingleAtom(*query_atom)
        } else if !precheckable {
            ComponentMatcher::RecursiveComponent
        } else {
            ComponentMatcher::Compiled(Box::new(CompiledQuery::new(
                extract_single_component_query(query, component_id),
            )?))
        };
        let entry = ComponentPlanEntry {
            component_id,
            group: query.component_group(component_id),
            atom_count: component_atoms.len(),
            precheckable,
            matcher,
        };
        entries.push(entry);
    }
    Ok(entries.into_boxed_slice())
}

fn extract_single_component_query(query: &QueryMol, component_id: usize) -> QueryMol {
    let mut old_to_new = alloc::vec![None; query.atom_count()];
    let atoms = query
        .component_atoms(component_id)
        .iter()
        .copied()
        .enumerate()
        .map(|(new_atom_id, old_atom_id)| {
            old_to_new[old_atom_id] = Some(new_atom_id);
            let mut atom = query.atoms()[old_atom_id].clone();
            atom.id = new_atom_id;
            atom.component = 0;
            atom
        })
        .collect::<alloc::vec::Vec<_>>();

    let bonds = query
        .component_bonds(component_id)
        .iter()
        .copied()
        .enumerate()
        .map(|(new_bond_id, old_bond_id)| {
            let mut bond = query.bonds()[old_bond_id].clone();
            bond.id = new_bond_id;
            bond.src = old_to_new[bond.src].expect("component bond source must be remapped");
            bond.dst = old_to_new[bond.dst].expect("component bond destination must be remapped");
            bond
        })
        .collect::<alloc::vec::Vec<_>>();

    QueryMol::from_parts(atoms, bonds, 1, alloc::vec![None])
}

fn component_contains_recursive_predicate(query: &QueryMol, component_id: usize) -> bool {
    query
        .component_atoms(component_id)
        .iter()
        .copied()
        .any(|atom_id| atom_expr_contains_recursive_predicate(&query.atoms()[atom_id].expr))
}

fn atom_expr_contains_recursive_predicate(expr: &AtomExpr) -> bool {
    match expr {
        AtomExpr::Wildcard | AtomExpr::Bare { .. } => false,
        AtomExpr::Bracket(expr) => bracket_tree_contains_recursive_predicate(&expr.tree),
    }
}

fn bracket_tree_contains_recursive_predicate(tree: &BracketExprTree) -> bool {
    match tree {
        BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(_)) => true,
        BracketExprTree::Primitive(_) => false,
        BracketExprTree::Not(inner) => bracket_tree_contains_recursive_predicate(inner),
        BracketExprTree::HighAnd(items)
        | BracketExprTree::Or(items)
        | BracketExprTree::LowAnd(items) => {
            items.iter().any(bracket_tree_contains_recursive_predicate)
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
    let initial_mapping = InitialAtomMapping::default();
    if query.query.atom_count() == 1 {
        scratch.prepare_caches(query, target);
        return single_atom_query_matches(query, target, &mut scratch.recursive_cache, None);
    }
    if is_two_atom_edge_query(&query.query) {
        scratch.prepare_caches(query, target);
        return two_atom_query_matches(
            query,
            target,
            &mut scratch.recursive_cache,
            initial_mapping,
        );
    }
    if let Some(layout) = three_atom_tree_layout(query) {
        scratch.prepare_caches(query, target);
        return three_atom_query_matches(
            query,
            target,
            &mut scratch.atom_stereo_cache,
            &mut scratch.recursive_cache,
            layout,
            initial_mapping,
        );
    }
    scratch.prepare(query, target);
    let SearchScratchParts {
        view,
        atom_stereo_cache,
        recursive_cache,
    } = scratch.search_parts();
    query_matches_with_mapping_state(
        query,
        target,
        atom_stereo_cache,
        recursive_cache,
        view,
        initial_mapping,
    )
}

fn query_matches_with_scratch_and_interrupt<F>(
    query: &CompiledQuery,
    target: &PreparedTarget,
    scratch: &mut MatchScratch,
    should_stop: F,
) -> MatchLimitResult
where
    F: FnMut() -> bool,
{
    let limit = MatchInterrupt::new(should_stop);
    let matched = query_matches_with_scratch_limit(query, target, scratch, &limit);
    if limit.exceeded() {
        MatchLimitResult::Exceeded
    } else {
        MatchLimitResult::Complete(matched)
    }
}

#[cfg(all(
    feature = "std",
    not(all(target_family = "wasm", target_os = "unknown"))
))]
fn query_matches_with_scratch_and_time_limit(
    query: &CompiledQuery,
    target: &PreparedTarget,
    scratch: &mut MatchScratch,
    max_elapsed: Duration,
) -> MatchLimitResult {
    let limit = MatchTimeLimit::new(max_elapsed);
    let matched = query_matches_with_scratch_limit(query, target, scratch, &limit);
    if limit.exceeded() {
        MatchLimitResult::Exceeded
    } else {
        MatchLimitResult::Complete(matched)
    }
}

fn query_matches_with_scratch_limit<L: MatchLimiter>(
    query: &CompiledQuery,
    target: &PreparedTarget,
    scratch: &mut MatchScratch,
    limit: &L,
) -> bool {
    scratch.prepare_with_limit(query, target, limit);
    let SearchScratchParts {
        view,
        atom_stereo_cache,
        recursive_cache,
    } = scratch.search_parts_with_limit(limit);
    query_matches_with_mapping_state(
        query,
        target,
        atom_stereo_cache,
        recursive_cache,
        view,
        InitialAtomMapping::default(),
    )
}

fn query_match_outcome(query: &CompiledQuery, target: &PreparedTarget) -> MatchOutcome {
    let mut atom_stereo_cache = AtomStereoCache::new();
    let mut recursive_cache =
        RecursiveMatchCache::new(query.recursive_cache_slots, target.atom_count());
    query_match_outcome_with_mapping(
        query,
        target,
        &mut atom_stereo_cache,
        &mut recursive_cache,
        None,
        None,
    )
}

fn query_match_outcome_with_scratch(
    query: &CompiledQuery,
    target: &PreparedTarget,
    scratch: &mut MatchScratch,
) -> MatchOutcome {
    let mut coverage = MatchCoverageAccumulator::new(target);
    let _ = with_scratch_bound_search_context(query, target, scratch, |context, mapped_count| {
        accumulate_match_outcome(query, target, context, mapped_count, &mut coverage);
    });
    coverage.into_outcome(target)
}

fn query_match_outcome_with_scratch_and_interrupt<F>(
    query: &CompiledQuery,
    target: &PreparedTarget,
    scratch: &mut MatchScratch,
    should_stop: F,
) -> MatchOutcomeLimitResult
where
    F: FnMut() -> bool,
{
    let limit = MatchInterrupt::new(should_stop);
    let outcome = query_match_outcome_with_scratch_limit(query, target, scratch, &limit);
    if limit.exceeded() {
        MatchOutcomeLimitResult::Exceeded
    } else {
        MatchOutcomeLimitResult::Complete(outcome)
    }
}

#[cfg(all(
    feature = "std",
    not(all(target_family = "wasm", target_os = "unknown"))
))]
fn query_match_outcome_with_scratch_and_time_limit(
    query: &CompiledQuery,
    target: &PreparedTarget,
    scratch: &mut MatchScratch,
    max_elapsed: Duration,
) -> MatchOutcomeLimitResult {
    let limit = MatchTimeLimit::new(max_elapsed);
    let outcome = query_match_outcome_with_scratch_limit(query, target, scratch, &limit);
    if limit.exceeded() {
        MatchOutcomeLimitResult::Exceeded
    } else {
        MatchOutcomeLimitResult::Complete(outcome)
    }
}

fn query_match_outcome_with_scratch_limit<L: MatchLimiter>(
    query: &CompiledQuery,
    target: &PreparedTarget,
    scratch: &mut MatchScratch,
    limit: &L,
) -> MatchOutcome {
    scratch.prepare_with_limit(query, target, limit);
    let SearchScratchParts {
        view,
        atom_stereo_cache,
        recursive_cache,
    } = scratch.search_parts_with_limit(limit);
    let mut context = view.into_context(query, atom_stereo_cache, recursive_cache);
    let mut coverage = MatchCoverageAccumulator::new(target);
    if let Some(mapped_count) =
        bind_initial_mapping(query, target, &mut context, InitialAtomMapping::default())
    {
        accumulate_match_outcome(query, target, &mut context, mapped_count, &mut coverage);
    }
    coverage.into_outcome(target)
}

fn accumulate_match_outcome<L: MatchLimiter>(
    query: &CompiledQuery,
    target: &PreparedTarget,
    context: &mut SearchContext<'_, L>,
    mapped_count: usize,
    coverage: &mut MatchCoverageAccumulator,
) {
    for_each_mapping(
        &query.query,
        target,
        context,
        mapped_count,
        &mut |query_to_target| {
            coverage.cover_mapping(&query.query, target, query_to_target);
            false
        },
    );
}

fn with_scratch_bound_search_context<R>(
    query: &CompiledQuery,
    target: &PreparedTarget,
    scratch: &mut MatchScratch,
    run: impl FnOnce(&mut SearchContext<'_, NoMatchLimit>, usize) -> R,
) -> Option<R> {
    scratch.prepare(query, target);
    let SearchScratchParts {
        view,
        atom_stereo_cache,
        recursive_cache,
    } = scratch.search_parts();
    let mut context = view.into_context(query, atom_stereo_cache, recursive_cache);
    let mapped_count =
        bind_initial_mapping(query, target, &mut context, InitialAtomMapping::default())?;
    Some(run(&mut context, mapped_count))
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

    let mut scratch = FreshSearchBuffers::new(query, target, recursive_cache);
    query_matches_with_mapping_state(
        query,
        target,
        atom_stereo_cache,
        recursive_cache,
        scratch.view(),
        initial_mapping,
    )
}

fn query_matches_with_mapping_state<L: MatchLimiter>(
    query: &CompiledQuery,
    target: &PreparedTarget,
    atom_stereo_cache: &mut AtomStereoCache,
    recursive_cache: &mut RecursiveMatchCache,
    scratch: SearchScratchView<'_, L>,
    initial_mapping: InitialAtomMapping,
) -> bool {
    let mut context = scratch.into_context(query, atom_stereo_cache, recursive_cache);
    let Some(mapped_count) = bind_initial_mapping(query, target, &mut context, initial_mapping)
    else {
        return false;
    };
    if mapped_count == 0 && !component_prechecks_match(query, target, &mut context) {
        return false;
    }

    if query.query.bond_count() == 0 {
        if can_solve_bondless_by_assignment(query) {
            return search_bondless_assignment(&query.query, target, &mut context, mapped_count);
        }
        if should_precompute_bondless_candidates(target, &context) {
            return search_bondless_mapping(&query.query, target, &mut context, mapped_count);
        }
    }

    if mapped_count == 0 && should_use_disconnected_component_search(query) {
        return disconnected_component_search_matches(query, target, &mut context);
    }

    search_mapping(&query.query, target, &mut context, mapped_count)
}

fn query_match_outcome_with_mapping(
    query: &CompiledQuery,
    target: &PreparedTarget,
    atom_stereo_cache: &mut AtomStereoCache,
    recursive_cache: &mut RecursiveMatchCache,
    initial_query_atom: Option<QueryAtomId>,
    initial_target_atom: Option<usize>,
) -> MatchOutcome {
    let initial_mapping = InitialAtomMapping::new(initial_query_atom, initial_target_atom);
    let mut coverage = MatchCoverageAccumulator::new(target);
    let _ = with_fresh_bound_search_context(
        query,
        target,
        atom_stereo_cache,
        recursive_cache,
        initial_mapping,
        |context, mapped_count| {
            accumulate_match_outcome(query, target, context, mapped_count, &mut coverage);
        },
    );
    coverage.into_outcome(target)
}

fn component_prechecks_match<L: MatchLimiter>(
    query: &CompiledQuery,
    target: &PreparedTarget,
    context: &mut SearchContext<'_, L>,
) -> bool {
    if query.component_plan.is_empty() {
        return true;
    }

    let mut scratch = MatchScratch::new();
    query
        .component_plan
        .iter()
        .filter(|component| component.supports_precheck())
        .all(|component| match &component.matcher {
            ComponentMatcher::Compiled(component_query) => query_matches_with_scratch_limit(
                component_query,
                target,
                &mut scratch,
                context.limit,
            ),
            ComponentMatcher::SingleAtom(query_atom) => {
                single_atom_component_matches(query, target, context, *query_atom)
            }
            ComponentMatcher::RecursiveComponent => true,
        })
}

fn component_search_supported(query: &CompiledQuery) -> bool {
    !query.component_plan.is_empty()
        && query
            .component_plan
            .iter()
            .all(ComponentPlanEntry::supports_component_search)
}

fn should_use_disconnected_component_search(query: &CompiledQuery) -> bool {
    if !component_search_supported(query) {
        return false;
    }
    query.has_component_constraints || has_small_ungrouped_component_tail(query)
}

fn has_small_ungrouped_component_tail(query: &CompiledQuery) -> bool {
    const MAX_UNGROUPED_MULTI_ATOM_COMPONENT: usize = 4;

    if query.has_stereo_constraints {
        return false;
    }

    let mut has_multi_atom_component = false;
    for entry in &query.component_plan {
        if entry.group.is_some() {
            return false;
        }
        if entry.atom_count <= 1 {
            continue;
        }
        if entry.atom_count > MAX_UNGROUPED_MULTI_ATOM_COMPONENT {
            return false;
        }
        has_multi_atom_component = true;
    }

    has_multi_atom_component
}

fn with_fresh_bound_search_context<R>(
    query: &CompiledQuery,
    target: &PreparedTarget,
    atom_stereo_cache: &mut AtomStereoCache,
    recursive_cache: &mut RecursiveMatchCache,
    initial_mapping: InitialAtomMapping,
    run: impl FnOnce(&mut SearchContext<'_, NoMatchLimit>, usize) -> R,
) -> Option<R> {
    let mut scratch = FreshSearchBuffers::new(query, target, recursive_cache);
    let mut context = scratch
        .view()
        .into_context(query, atom_stereo_cache, recursive_cache);
    let mapped_count = bind_initial_mapping(query, target, &mut context, initial_mapping)?;
    Some(run(&mut context, mapped_count))
}

fn bind_initial_mapping<L: MatchLimiter>(
    query: &CompiledQuery,
    target: &PreparedTarget,
    context: &mut SearchContext<'_, L>,
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
    if !compiled_query_atom_matches_with_limit(
        query,
        target,
        context.recursive_cache,
        query_atom,
        target_atom,
        context.limit,
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
    compiled_query_atom_matches_with_limit(
        query,
        target,
        recursive_cache,
        query_atom,
        target_atom,
        &NO_MATCH_LIMIT,
    )
}

fn compiled_query_atom_matches_with_limit<L: MatchLimiter>(
    query: &CompiledQuery,
    target: &PreparedTarget,
    recursive_cache: &mut RecursiveMatchCache,
    query_atom: QueryAtomId,
    target_atom: usize,
    limit: &L,
) -> bool {
    if !limit.continue_search() {
        return false;
    }
    let atom_matcher = &query.atom_matchers[query_atom];
    if is_hidden_attached_hydrogen(target, target_atom) {
        return false;
    }
    if !atom_fast_predicates_match(atom_matcher, target, target_atom) {
        return false;
    }
    if atom_matcher.complete {
        return true;
    }

    let mut context = AtomMatchContext::new(query, target, recursive_cache, limit);
    atom_expr_matches(
        &query.query.atoms()[query_atom].expr,
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
            for_each_mapping(
                &query.query,
                target,
                context,
                mapped_count,
                &mut |query_to_target| {
                    let mapping = current_query_order_mapping(query_to_target);
                    matches
                        .entry(canonical_match_key(&mapping))
                        .and_modify(|existing| {
                            if mapping < *existing {
                                existing.clone_from(&mapping);
                            }
                        })
                        .or_insert(mapping);
                    false
                },
            );
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
            for_each_mapping(
                &query.query,
                target,
                context,
                mapped_count,
                &mut |query_to_target| {
                    matches.push(current_canonical_match_key(query_to_target));
                    false
                },
            );
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
    for_each_two_atom_mapping(query, target, recursive_cache, initial_mapping, |_| true)
}

fn two_atom_query_substructure_matches(
    query: &CompiledQuery,
    target: &PreparedTarget,
    recursive_cache: &mut RecursiveMatchCache,
    initial_mapping: InitialAtomMapping,
) -> Box<[Box<[usize]>]> {
    let mut matches = UniqueMatches::new();
    for_each_two_atom_mapping(query, target, recursive_cache, initial_mapping, |mapping| {
        let ordered = current_query_order_mapping(mapping);
        matches
            .entry(canonical_match_key(&ordered))
            .and_modify(|existing| {
                if ordered < *existing {
                    existing.clone_from(&ordered);
                }
            })
            .or_insert(ordered);
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
    let mut matches = MatchKeys::new();
    for_each_two_atom_mapping(query, target, recursive_cache, initial_mapping, |mapping| {
        matches.push(current_canonical_match_key(mapping));
        false
    });
    matches.sort_unstable();
    matches.dedup();
    matches.len()
}

fn for_each_two_atom_mapping(
    query: &CompiledQuery,
    target: &PreparedTarget,
    recursive_cache: &mut RecursiveMatchCache,
    initial_mapping: InitialAtomMapping,
    mut visit: impl FnMut(&[Option<usize>]) -> bool,
) -> bool {
    let bond = &query.query.bonds()[0];
    let query_mapping = TwoAtomQueryMapping {
        left: bond.src,
        right: bond.dst,
    };
    for_each_target_bond(target, |left_atom, right_atom, bond_label| {
        visit_two_atom_mapping(
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
            &mut visit,
        ) || visit_two_atom_mapping(
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
            &mut visit,
        )
    })
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
            compiled_query_stereo_constraints_match(query, target, mapping, atom_stereo_cache)
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
            if !compiled_query_stereo_constraints_match(query, target, mapping, atom_stereo_cache) {
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
            if !compiled_query_stereo_constraints_match(query, target, mapping, atom_stereo_cache) {
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
    let neighbors = search.target.neighbor_slice(search.target_center);
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

fn visit_two_atom_mapping(
    query: &CompiledQuery,
    target: &PreparedTarget,
    recursive_cache: &mut RecursiveMatchCache,
    query_mapping: TwoAtomQueryMapping,
    target_mapping: TwoAtomTargetMapping,
    initial_mapping: InitialAtomMapping,
    visit: &mut impl FnMut(&[Option<usize>]) -> bool,
) -> bool {
    mapping_matches_two_atom_query(
        query,
        target,
        recursive_cache,
        query_mapping,
        target_mapping,
        initial_mapping,
    ) && visit(&two_atom_query_order_mapping(query_mapping, target_mapping))
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

const fn two_atom_query_order_mapping(
    query_mapping: TwoAtomQueryMapping,
    target_mapping: TwoAtomTargetMapping,
) -> [Option<usize>; 2] {
    let mut mapping = [None; 2];
    mapping[query_mapping.left] = Some(target_mapping.left);
    mapping[query_mapping.right] = Some(target_mapping.right);
    mapping
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
    context: &mut SearchContext<'_, impl MatchLimiter>,
    mapped_count: usize,
) -> bool {
    if !context.continue_search() {
        return false;
    }
    if mapped_count == query.atom_count() {
        return compiled_query_stereo_constraints_match(
            context.compiled_query,
            target,
            context.query_to_target,
            context.atom_stereo_cache,
        );
    }

    let next_query_atom = select_next_query_atom_for_target(context);
    search_mapping_candidates(query, target, context, mapped_count, next_query_atom)
}

fn search_bondless_mapping(
    query: &QueryMol,
    target: &PreparedTarget,
    context: &mut SearchContext<'_, impl MatchLimiter>,
    mapped_count: usize,
) -> bool {
    if mapped_count == query.atom_count() {
        return compiled_query_stereo_constraints_match(
            context.compiled_query,
            target,
            context.query_to_target,
            context.atom_stereo_cache,
        );
    }

    if !prepare_bondless_query_candidates(query, target, context) {
        return false;
    }

    search_bondless_mapping_candidates(query, target, context, mapped_count, 0)
}

fn search_bondless_assignment(
    query: &QueryMol,
    target: &PreparedTarget,
    context: &mut SearchContext<'_, impl MatchLimiter>,
    mapped_count: usize,
) -> bool {
    if mapped_count == query.atom_count() {
        return true;
    }
    if !prepare_bondless_query_candidates(query, target, context) {
        return false;
    }

    context.bondless_target_assignments.clear();
    context
        .bondless_target_assignments
        .resize(target.atom_count(), None);
    context.bondless_seen_targets.clear();
    context
        .bondless_seen_targets
        .resize(target.atom_count(), false);
    bondless_assignment_exists(
        context.bondless_query_order,
        context.bondless_candidate_offsets,
        context.bondless_candidates,
        context.bondless_target_assignments,
        context.bondless_seen_targets,
        context.limit,
    )
}

fn disconnected_component_search_matches(
    query: &CompiledQuery,
    target: &PreparedTarget,
    context: &mut SearchContext<'_, impl MatchLimiter>,
) -> bool {
    let mut component_sets = alloc::vec::Vec::new();
    for entry in &query.component_plan {
        let embeddings = collect_component_embeddings(query, target, context, entry);
        if embeddings.is_empty() {
            return false;
        }
        component_sets.push(ComponentEmbeddingSet { entry, embeddings });
    }
    component_sets.sort_by_key(|set| {
        (
            set.embeddings.len(),
            core::cmp::Reverse(set.entry.atom_count),
            set.entry.component_id,
        )
    });

    let mut used_target_atoms = alloc::vec![false; target.atom_count()];
    let group_count = component_sets
        .iter()
        .filter_map(|set| set.entry.group)
        .max()
        .map_or(0, |group| group + 1);
    let mut group_targets = alloc::vec![None; group_count];
    component_embedding_assignment_exists(
        &component_sets,
        &mut used_target_atoms,
        &mut group_targets,
        context.limit,
    )
}

fn collect_component_embeddings(
    query: &CompiledQuery,
    target: &PreparedTarget,
    context: &mut SearchContext<'_, impl MatchLimiter>,
    entry: &ComponentPlanEntry,
) -> alloc::vec::Vec<ComponentEmbedding> {
    match &entry.matcher {
        ComponentMatcher::Compiled(component) => {
            collect_compiled_component_embeddings(component, target, context.limit)
        }
        ComponentMatcher::SingleAtom(query_atom) => {
            collect_single_atom_component_embeddings(query, target, context, *query_atom)
        }
        ComponentMatcher::RecursiveComponent => {
            collect_recursive_component_embeddings(query, target, context, entry.component_id)
        }
    }
}

fn single_atom_component_matches(
    query: &CompiledQuery,
    target: &PreparedTarget,
    context: &mut SearchContext<'_, impl MatchLimiter>,
    query_atom: QueryAtomId,
) -> bool {
    let mut anchored_target_atoms = alloc::vec::Vec::new();
    single_atom_candidates(
        &query.atom_matchers[query_atom],
        target,
        None,
        &mut anchored_target_atoms,
    )
    .any(|target_atom| {
        single_atom_component_target_matches(query, target, context, query_atom, target_atom)
    })
}

fn collect_single_atom_component_embeddings(
    query: &CompiledQuery,
    target: &PreparedTarget,
    context: &mut SearchContext<'_, impl MatchLimiter>,
    query_atom: QueryAtomId,
) -> alloc::vec::Vec<ComponentEmbedding> {
    let mut anchored_target_atoms = alloc::vec::Vec::new();
    single_atom_candidates(
        &query.atom_matchers[query_atom],
        target,
        None,
        &mut anchored_target_atoms,
    )
    .collect_matches(|target_atom| {
        single_atom_component_target_matches(query, target, context, query_atom, target_atom)
    })
    .into_vec()
    .into_iter()
    .map(|target_atoms| component_embedding(target, target_atoms))
    .collect()
}

fn collect_recursive_component_embeddings(
    query: &CompiledQuery,
    target: &PreparedTarget,
    context: &mut SearchContext<'_, impl MatchLimiter>,
    component_id: usize,
) -> alloc::vec::Vec<ComponentEmbedding> {
    let mut seen = BTreeSet::new();
    let mut embeddings = alloc::vec::Vec::new();
    for_each_component_mapping(
        query,
        target,
        context,
        component_id,
        0,
        &mut |query_to_target| {
            let target_atoms =
                current_component_canonical_match_key(&query.query, component_id, query_to_target);
            if seen.insert(target_atoms.clone()) {
                embeddings.push(component_embedding(target, target_atoms));
            }
            false
        },
    );
    embeddings
}

fn single_atom_component_target_matches(
    query: &CompiledQuery,
    target: &PreparedTarget,
    context: &mut SearchContext<'_, impl MatchLimiter>,
    query_atom: QueryAtomId,
    target_atom: usize,
) -> bool {
    compiled_query_atom_matches_with_limit(
        query,
        target,
        context.recursive_cache,
        query_atom,
        target_atom,
        context.limit,
    )
}

fn for_each_component_mapping(
    query: &CompiledQuery,
    target: &PreparedTarget,
    context: &mut SearchContext<'_, impl MatchLimiter>,
    component_id: usize,
    mapped_count: usize,
    visit: &mut impl FnMut(&[Option<usize>]) -> bool,
) -> bool {
    if !context.continue_search() {
        return true;
    }
    if mapped_count == query.query.component_atoms(component_id).len() {
        return component_mapping_stereo_constraints_match(
            query,
            component_id,
            target,
            context.query_to_target,
            context.atom_stereo_cache,
        ) && visit(context.query_to_target);
    }

    let next_query_atom =
        select_next_component_query_atom_for_target(&query.query, context, component_id);
    for target_atom in candidate_target_atoms(next_query_atom, target, context) {
        if !compiled_query_atom_matches_with_limit(
            context.compiled_query,
            target,
            context.recursive_cache,
            next_query_atom,
            target_atom,
            context.limit,
        ) {
            continue;
        }

        context.query_to_target[next_query_atom] = Some(target_atom);
        context.mark_target_atom_used(target_atom);
        let stop = for_each_component_mapping(
            query,
            target,
            context,
            component_id,
            mapped_count + 1,
            visit,
        );
        context.unmark_target_atom_used(target_atom);
        context.query_to_target[next_query_atom] = None;
        if stop {
            return true;
        }
    }
    false
}

fn select_next_component_query_atom_for_target(
    query: &QueryMol,
    context: &SearchContext<'_, impl MatchLimiter>,
    component_id: usize,
) -> QueryAtomId {
    let mut best_atom = None;
    let mut best_key = (
        0usize,
        core::cmp::Reverse(usize::MAX),
        0usize,
        0usize,
        core::cmp::Reverse(usize::MAX),
    );

    for &query_atom in query.component_atoms(component_id) {
        if context.query_to_target[query_atom].is_some() {
            continue;
        }
        let neighbors = &context.query_neighbors[query_atom];
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

    best_atom.expect("at least one unmapped component query atom must remain")
}

fn component_mapping_stereo_constraints_match(
    query: &CompiledQuery,
    component_id: usize,
    target: &PreparedTarget,
    query_to_target: &[Option<usize>],
    atom_stereo_cache: &mut AtomStereoCache,
) -> bool {
    if !query.has_stereo_constraints {
        return true;
    }

    let query_mol = &query.query;
    query
        .stereo_plan
        .double_bond_constraints
        .iter()
        .filter(|constraint| query_mol.atoms()[constraint.left_atom].component == component_id)
        .all(|constraint| {
            let Some(left_target) = query_to_target[constraint.left_atom] else {
                return false;
            };
            let Some(right_target) = query_to_target[constraint.right_atom] else {
                return false;
            };
            target.double_bond_stereo_config(left_target, right_target) == Some(constraint.config)
        })
        && query_mol
            .component_atoms(component_id)
            .iter()
            .copied()
            .all(|query_atom_id| {
                atom_chirality_matches(
                    query_mol,
                    &query.query_neighbors,
                    query_atom_id,
                    target,
                    query_to_target,
                    atom_stereo_cache,
                )
            })
}

fn collect_compiled_component_embeddings(
    component: &CompiledQuery,
    target: &PreparedTarget,
    limit: &impl MatchLimiter,
) -> alloc::vec::Vec<ComponentEmbedding> {
    let mut atom_stereo_cache = AtomStereoCache::new();
    let mut recursive_cache =
        RecursiveMatchCache::new(component.recursive_cache_slots, target.atom_count());
    let mut scratch =
        FreshSearchBuffers::new_with_limit(component, target, &mut recursive_cache, limit);
    let mut context = scratch.view_with_limit(limit).into_context(
        component,
        &mut atom_stereo_cache,
        &mut recursive_cache,
    );
    let mut seen = BTreeSet::new();
    let mut embeddings = alloc::vec::Vec::new();
    for_each_mapping(
        &component.query,
        target,
        &mut context,
        0,
        &mut |query_to_target| {
            let target_atoms = current_canonical_match_key(query_to_target);
            if seen.insert(target_atoms.clone()) {
                embeddings.push(component_embedding(target, target_atoms));
            }
            false
        },
    );
    embeddings
}

fn component_embedding(target: &PreparedTarget, target_atoms: Box<[usize]>) -> ComponentEmbedding {
    let target_component = target_atoms
        .first()
        .and_then(|&target_atom| target.connected_component(target_atom));
    ComponentEmbedding {
        target_atoms,
        target_component,
    }
}

fn component_embedding_assignment_exists(
    component_sets: &[ComponentEmbeddingSet<'_>],
    used_target_atoms: &mut [bool],
    group_targets: &mut [Option<usize>],
    limit: &impl MatchLimiter,
) -> bool {
    if component_sets.is_empty() {
        return true;
    }

    let mut frames = alloc::vec![ComponentAssignmentFrame {
        next_embedding: 0,
        selected_embedding: None,
        previous_group_target: None,
    }];

    loop {
        if !limit.continue_search() {
            return false;
        }

        let set_index = frames.len() - 1;
        if let Some(embedding_index) = frames[set_index].selected_embedding.take() {
            let component_set = &component_sets[set_index];
            let embedding = &component_set.embeddings[embedding_index];
            set_component_embedding_used(embedding, used_target_atoms, false);
            unbind_component_embedding_group(
                component_set.entry.group,
                frames[set_index].previous_group_target.take(),
                group_targets,
            );
        }

        let component_set = &component_sets[set_index];
        let mut descended = false;
        while frames[set_index].next_embedding < component_set.embeddings.len() {
            let embedding_index = frames[set_index].next_embedding;
            frames[set_index].next_embedding += 1;
            let embedding = &component_set.embeddings[embedding_index];
            if component_embedding_overlaps(embedding, used_target_atoms)
                || !component_embedding_group_matches(
                    component_set.entry.group,
                    embedding,
                    group_targets,
                )
            {
                continue;
            }

            let previous_group_target =
                bind_component_embedding_group(component_set.entry.group, embedding, group_targets);
            set_component_embedding_used(embedding, used_target_atoms, true);
            if set_index + 1 == component_sets.len() {
                return true;
            }

            frames[set_index].selected_embedding = Some(embedding_index);
            frames[set_index].previous_group_target = previous_group_target;
            frames.push(ComponentAssignmentFrame {
                next_embedding: 0,
                selected_embedding: None,
                previous_group_target: None,
            });
            descended = true;
            break;
        }

        if descended {
            continue;
        }

        if set_index == 0 {
            return false;
        }

        frames.pop();
    }
}

fn component_embedding_overlaps(
    embedding: &ComponentEmbedding,
    used_target_atoms: &[bool],
) -> bool {
    embedding
        .target_atoms
        .iter()
        .any(|&target_atom| used_target_atoms[target_atom])
}

fn set_component_embedding_used(
    embedding: &ComponentEmbedding,
    used_target_atoms: &mut [bool],
    used: bool,
) {
    for &target_atom in &embedding.target_atoms {
        used_target_atoms[target_atom] = used;
    }
}

fn component_embedding_group_matches(
    group: Option<usize>,
    embedding: &ComponentEmbedding,
    group_targets: &[Option<usize>],
) -> bool {
    let Some(group) = group else {
        return true;
    };
    let Some(target_component) = embedding.target_component else {
        return false;
    };
    group_targets
        .iter()
        .enumerate()
        .all(|(other_group, existing_target_component)| {
            existing_target_component.is_none_or(|existing_target_component| {
                if other_group == group {
                    existing_target_component == target_component
                } else {
                    existing_target_component != target_component
                }
            })
        })
}

fn bind_component_embedding_group(
    group: Option<usize>,
    embedding: &ComponentEmbedding,
    group_targets: &mut [Option<usize>],
) -> Option<usize> {
    let group = group?;
    let previous = group_targets[group];
    group_targets[group] = embedding.target_component;
    previous
}

fn unbind_component_embedding_group(
    group: Option<usize>,
    previous_target_component: Option<usize>,
    group_targets: &mut [Option<usize>],
) {
    if let Some(group) = group {
        group_targets[group] = previous_target_component;
    }
}

const fn can_solve_bondless_by_assignment(query: &CompiledQuery) -> bool {
    !query.has_component_constraints && !query.has_stereo_constraints
}

fn bondless_assignment_exists(
    query_order: &[QueryAtomId],
    candidate_offsets: &[usize],
    candidates: &[usize],
    target_assignments: &mut [Option<QueryAtomId>],
    seen_targets: &mut [bool],
    limit: &impl MatchLimiter,
) -> bool {
    for &query_atom in query_order {
        seen_targets.fill(false);
        if !assign_bondless_query_atom(
            query_atom,
            candidate_offsets,
            candidates,
            target_assignments,
            seen_targets,
            limit,
        ) {
            return false;
        }
    }

    true
}

fn assign_bondless_query_atom(
    query_atom: QueryAtomId,
    candidate_offsets: &[usize],
    candidates: &[usize],
    target_assignments: &mut [Option<QueryAtomId>],
    seen_targets: &mut [bool],
    limit: &impl MatchLimiter,
) -> bool {
    let start = candidate_offsets[query_atom];
    let end = candidate_offsets[query_atom + 1];
    for candidate_index in start..end {
        let target_atom = candidates[candidate_index];
        if seen_targets[target_atom] {
            continue;
        }
        if !limit.continue_search() {
            return false;
        }
        seen_targets[target_atom] = true;

        let previous_query_atom = target_assignments[target_atom];
        if previous_query_atom.is_none_or(|previous_query_atom| {
            assign_bondless_query_atom(
                previous_query_atom,
                candidate_offsets,
                candidates,
                target_assignments,
                seen_targets,
                limit,
            )
        }) {
            target_assignments[target_atom] = Some(query_atom);
            return true;
        }
    }

    false
}

fn should_precompute_bondless_candidates(
    target: &PreparedTarget,
    context: &SearchContext<'_, impl MatchLimiter>,
) -> bool {
    const MIN_TARGET_ATOMS: usize = 64;
    const MAX_ANCHORED_FRACTION: usize = 4;

    if target.atom_count() < MIN_TARGET_ATOMS {
        return false;
    }

    (0..context.compiled_query.query.atom_count())
        .filter(|&query_atom| context.query_to_target[query_atom].is_none())
        .all(|query_atom| {
            atom_matcher_anchor_candidates(
                &context.compiled_query.atom_matchers[query_atom],
                target,
            )
            .is_some_and(|candidates| {
                candidates.len().saturating_mul(MAX_ANCHORED_FRACTION) <= target.atom_count()
            })
        })
}

fn prepare_bondless_query_candidates(
    query: &QueryMol,
    target: &PreparedTarget,
    context: &mut SearchContext<'_, impl MatchLimiter>,
) -> bool {
    context.bondless_query_order.clear();
    context.bondless_candidate_offsets.clear();
    context.bondless_candidates.clear();
    context
        .bondless_candidate_offsets
        .resize(query.atom_count() + 1, 0);

    for query_atom in 0..query.atom_count() {
        if context.query_to_target[query_atom].is_some() {
            context.bondless_candidate_offsets[query_atom + 1] = context.bondless_candidates.len();
            continue;
        }

        context.bondless_query_order.push(query_atom);
        let start = context.bondless_candidates.len();
        push_bondless_atom_candidates(query, target, context, query_atom);
        if context.bondless_candidates.len() == start {
            return false;
        }
        context.bondless_candidate_offsets[query_atom + 1] = context.bondless_candidates.len();
    }

    let offsets = &context.bondless_candidate_offsets;
    let scores = context.query_atom_scores;
    context
        .bondless_query_order
        .sort_unstable_by_key(|&query_atom| {
            (
                offsets[query_atom + 1] - offsets[query_atom],
                core::cmp::Reverse(scores[query_atom]),
                query_atom,
            )
        });

    true
}

fn push_bondless_atom_candidates(
    query: &QueryMol,
    target: &PreparedTarget,
    context: &mut SearchContext<'_, impl MatchLimiter>,
    query_atom: QueryAtomId,
) {
    if let Some(target_atoms) =
        atom_matcher_anchor_candidates(&context.compiled_query.atom_matchers[query_atom], target)
    {
        for target_atom in target_atoms.iter().copied() {
            push_bondless_atom_candidate(query, target, context, query_atom, target_atom);
        }
        return;
    }

    for target_atom in 0..target.atom_count() {
        push_bondless_atom_candidate(query, target, context, query_atom, target_atom);
    }
}

fn push_bondless_atom_candidate(
    query: &QueryMol,
    target: &PreparedTarget,
    context: &mut SearchContext<'_, impl MatchLimiter>,
    query_atom: QueryAtomId,
    target_atom: usize,
) {
    if context.target_atom_is_used(target_atom) {
        return;
    }
    if context.compiled_query.has_component_constraints
        && !component_constraints_match(
            query_atom,
            target_atom,
            query,
            target,
            context.query_to_target,
        )
    {
        return;
    }
    if compiled_query_atom_matches_with_limit(
        context.compiled_query,
        target,
        context.recursive_cache,
        query_atom,
        target_atom,
        context.limit,
    ) {
        context.bondless_candidates.push(target_atom);
    }
}

fn search_bondless_mapping_candidates(
    query: &QueryMol,
    target: &PreparedTarget,
    context: &mut SearchContext<'_, impl MatchLimiter>,
    mapped_count: usize,
    order_index: usize,
) -> bool {
    if mapped_count == query.atom_count() {
        return compiled_query_stereo_constraints_match(
            context.compiled_query,
            target,
            context.query_to_target,
            context.atom_stereo_cache,
        );
    }

    let Some((next_order_index, query_atom)) =
        next_unmapped_bondless_query_atom(context, order_index)
    else {
        return false;
    };
    let start = context.bondless_candidate_offsets[query_atom];
    let end = context.bondless_candidate_offsets[query_atom + 1];

    for candidate_index in start..end {
        let target_atom = context.bondless_candidates[candidate_index];
        if context.target_atom_is_used(target_atom) {
            continue;
        }
        if !context.continue_search() {
            return false;
        }
        if context.compiled_query.has_component_constraints
            && !component_constraints_match(
                query_atom,
                target_atom,
                query,
                target,
                context.query_to_target,
            )
        {
            continue;
        }

        context.query_to_target[query_atom] = Some(target_atom);
        context.mark_target_atom_used(target_atom);
        let matched = search_bondless_mapping_candidates(
            query,
            target,
            context,
            mapped_count + 1,
            next_order_index + 1,
        );
        context.unmark_target_atom_used(target_atom);
        context.query_to_target[query_atom] = None;
        if matched {
            return true;
        }
    }

    false
}

fn next_unmapped_bondless_query_atom(
    context: &SearchContext<'_, impl MatchLimiter>,
    mut order_index: usize,
) -> Option<(usize, QueryAtomId)> {
    while order_index < context.bondless_query_order.len() {
        let query_atom = context.bondless_query_order[order_index];
        if context.query_to_target[query_atom].is_none() {
            return Some((order_index, query_atom));
        }
        order_index += 1;
    }

    None
}

fn search_mapping_candidates(
    query: &QueryMol,
    target: &PreparedTarget,
    context: &mut SearchContext<'_, impl MatchLimiter>,
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
        if let Some(target_atoms) = atom_matcher_anchor_candidates(
            &context.compiled_query.atom_matchers[query_atom],
            target,
        ) {
            return search_mapping_unanchored_candidates(
                query,
                target,
                context,
                mapped_count,
                query_atom,
                target_atoms.iter().copied(),
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
    context: &mut SearchContext<'_, impl MatchLimiter>,
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
    context: &mut SearchContext<'_, impl MatchLimiter>,
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
    if !compiled_query_atom_matches_with_limit(
        context.compiled_query,
        target,
        context.recursive_cache,
        query_atom,
        target_atom,
        context.limit,
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

fn for_each_mapping(
    query: &QueryMol,
    target: &PreparedTarget,
    context: &mut SearchContext<'_, impl MatchLimiter>,
    mapped_count: usize,
    visit: &mut impl FnMut(&[Option<usize>]) -> bool,
) -> bool {
    if !context.continue_search() {
        return true;
    }
    if mapped_count == query.atom_count() {
        return compiled_query_stereo_constraints_match(
            context.compiled_query,
            target,
            context.query_to_target,
            context.atom_stereo_cache,
        ) && visit(context.query_to_target);
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
        if !compiled_query_atom_matches_with_limit(
            context.compiled_query,
            target,
            context.recursive_cache,
            next_query_atom,
            target_atom,
            context.limit,
        ) {
            continue;
        }

        context.query_to_target[next_query_atom] = Some(target_atom);
        context.mark_target_atom_used(target_atom);
        let stop = for_each_mapping(query, target, context, mapped_count + 1, visit);
        context.unmark_target_atom_used(target_atom);
        context.query_to_target[next_query_atom] = None;
        if stop {
            return true;
        }
    }
    false
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

fn current_component_canonical_match_key(
    query: &QueryMol,
    component_id: usize,
    query_to_target: &[Option<usize>],
) -> Box<[usize]> {
    let mut canonical = query
        .component_atoms(component_id)
        .iter()
        .map(|&query_atom| {
            query_to_target[query_atom].expect("complete component mappings must bind every atom")
        })
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

fn select_next_query_atom_for_target(
    context: &SearchContext<'_, impl MatchLimiter>,
) -> QueryAtomId {
    let mut best_atom = None;
    let mut best_key = (
        0usize,
        core::cmp::Reverse(usize::MAX),
        0usize,
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
        let component_size = context
            .compiled_query
            .query
            .component_atoms(context.compiled_query.query.atoms()[query_atom].component)
            .len();
        let key = (
            mapped_neighbors,
            core::cmp::Reverse(search_width_estimate),
            component_size,
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
    context: &SearchContext<'_, impl MatchLimiter>,
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

fn prepare_query_atom_search_widths<L: MatchLimiter>(
    widths: &mut alloc::vec::Vec<usize>,
    query: &CompiledQuery,
    target: &PreparedTarget,
    recursive_cache: &mut RecursiveMatchCache,
    limit: &L,
) {
    let fallback = target.atom_count();
    widths.clear();
    widths.extend((0..query.query.atom_count()).map(|query_atom| {
        atom_matcher_anchor_candidates(&query.atom_matchers[query_atom], target).map_or_else(
            || {
                if should_scan_query_atom_for_ordering(query, query_atom) {
                    count_query_atom_target_matches_with_limit(
                        query,
                        target,
                        recursive_cache,
                        query_atom,
                        limit,
                    )
                } else {
                    fallback
                }
            },
            <[usize]>::len,
        )
    }));
}

fn should_scan_query_atom_for_ordering(query: &CompiledQuery, query_atom: QueryAtomId) -> bool {
    query.query.atom_count() > 1
        && !query.atom_matchers[query_atom].complete
        && atom_expr_needs_target_scan_for_ordering(&query.query.atoms()[query_atom].expr)
}

fn atom_expr_needs_target_scan_for_ordering(expr: &AtomExpr) -> bool {
    match expr {
        AtomExpr::Wildcard | AtomExpr::Bare { .. } => false,
        AtomExpr::Bracket(expr) => bracket_tree_needs_target_scan_for_ordering(&expr.tree),
    }
}

fn bracket_tree_needs_target_scan_for_ordering(tree: &BracketExprTree) -> bool {
    match tree {
        BracketExprTree::Primitive(AtomPrimitive::RecursiveQuery(_)) => true,
        BracketExprTree::Primitive(_) => false,
        BracketExprTree::Not(inner) => double_negated_bracket_tree(tree).map_or_else(
            || {
                bracket_tree_may_filter_without_fast_anchor(inner)
                    || bracket_tree_needs_target_scan_for_ordering(inner)
            },
            bracket_tree_needs_target_scan_for_ordering,
        ),
        BracketExprTree::HighAnd(items)
        | BracketExprTree::LowAnd(items)
        | BracketExprTree::Or(items) => items
            .iter()
            .any(bracket_tree_needs_target_scan_for_ordering),
    }
}

fn bracket_tree_may_filter_without_fast_anchor(tree: &BracketExprTree) -> bool {
    match tree {
        BracketExprTree::Primitive(AtomPrimitive::Wildcard | AtomPrimitive::Chirality(_)) => false,
        BracketExprTree::Primitive(_) => true,
        BracketExprTree::Not(inner) => bracket_tree_may_filter_without_fast_anchor(inner),
        BracketExprTree::HighAnd(items)
        | BracketExprTree::LowAnd(items)
        | BracketExprTree::Or(items) => items
            .iter()
            .any(bracket_tree_may_filter_without_fast_anchor),
    }
}

fn count_query_atom_target_matches_with_limit<L: MatchLimiter>(
    query: &CompiledQuery,
    target: &PreparedTarget,
    recursive_cache: &mut RecursiveMatchCache,
    query_atom: QueryAtomId,
    limit: &L,
) -> usize {
    (0..target.atom_count())
        .filter(|&target_atom| {
            compiled_query_atom_matches_with_limit(
                query,
                target,
                recursive_cache,
                query_atom,
                target_atom,
                limit,
            )
        })
        .count()
}

fn candidate_target_atoms(
    query_atom: QueryAtomId,
    target: &PreparedTarget,
    context: &SearchContext<'_, impl MatchLimiter>,
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

fn atom_expr_matches<L: MatchLimiter>(
    expr: &AtomExpr,
    query_atom_id: QueryAtomId,
    atom_id: usize,
    context: &mut AtomMatchContext<'_, '_, L>,
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

fn bracket_tree_matches<L: MatchLimiter>(
    tree: &BracketExprTree,
    query_atom_id: QueryAtomId,
    atom_id: usize,
    context: &mut AtomMatchContext<'_, '_, L>,
) -> bool {
    if !context.limit.continue_search() {
        return false;
    }
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

fn atom_primitive_matches<L: MatchLimiter>(
    primitive: &AtomPrimitive,
    _query_atom_id: QueryAtomId,
    atom_id: usize,
    context: &mut AtomMatchContext<'_, '_, L>,
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
            context.limit,
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

fn compiled_query_stereo_constraints_match(
    query: &CompiledQuery,
    target: &PreparedTarget,
    query_to_target: &[Option<usize>],
    atom_stereo_cache: &mut AtomStereoCache,
) -> bool {
    !query.has_stereo_constraints
        || stereo_constraints_match(
            &query.query,
            &query.stereo_plan,
            &query.query_neighbors,
            target,
            query_to_target,
            atom_stereo_cache,
        )
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

fn recursive_query_matches<L: MatchLimiter>(
    query: &QueryMol,
    target: &PreparedTarget,
    atom_id: usize,
    recursive_query_lookup: &BTreeMap<usize, usize>,
    recursive_queries: &[RecursiveQueryEntry],
    recursive_cache: &mut RecursiveMatchCache,
    limit: &L,
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
    let mut scratch =
        FreshSearchBuffers::new_with_limit(entry.compiled.as_ref(), target, recursive_cache, limit);
    let anchored = query_matches_with_mapping_state(
        entry.compiled.as_ref(),
        target,
        &mut atom_stereo_cache,
        recursive_cache,
        scratch.view_with_limit(limit),
        InitialAtomMapping::new(Some(0), Some(atom_id)),
    );
    if !limit.exceeded() {
        recursive_cache.insert(entry.cache_slot, atom_id, anchored);
    }
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
    use alloc::{format, string::String, vec, vec::Vec};
    use core::str::FromStr;

    use smiles_parser::Smiles;

    use super::{
        component_constraints_match, component_embedding_assignment_exists, select_next_query_atom,
        select_next_query_atom_for_target, should_use_disconnected_component_search,
        AtomStereoCache, CompiledQuery, ComponentEmbedding, ComponentEmbeddingSet,
        ComponentMatcher, ComponentPlanEntry, FreshSearchBuffers, MatchScratch,
        RecursiveMatchCache, NO_MATCH_LIMIT,
    };
    use super::{AtomFastPredicate, MatchLimitResult, MatchOutcomeLimitResult};
    use crate::error::SmartsMatchError;
    use crate::prepared::PreparedTarget;
    use crate::QueryMol;

    fn query_matches_smiles(smarts: &str, smiles: &str) -> bool {
        QueryMol::from_str(smarts).unwrap().matches(smiles).unwrap()
    }

    fn recursive_component_chain(depth: usize) -> String {
        let mut nested = String::from("CsO");
        for _ in 0..depth {
            nested = format!("Cs[$({nested})]O.OO");
        }
        format!("[$({nested})]O.O")
    }

    fn assert_coverage_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-12,
            "coverage {actual} != expected {expected}"
        );
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
    fn compiled_query_interrupt_reports_exhaustion_without_poisoning_scratch() {
        let compiled =
            CompiledQuery::new(QueryMol::from_str("[$([#6]-[#6])].[#7]").unwrap()).unwrap();
        let target = PreparedTarget::new(Smiles::from_str("CCC.N").unwrap());
        let mut scratch = MatchScratch::new();

        assert_eq!(
            compiled.matches_with_scratch_and_interrupt(&target, &mut scratch, || true),
            MatchLimitResult::Exceeded
        );
        assert_eq!(
            compiled.matches_with_scratch_and_interrupt(&target, &mut scratch, || false),
            MatchLimitResult::Complete(compiled.matches(&target))
        );
        assert!(compiled.matches(&target));
    }

    #[test]
    #[cfg(all(
        feature = "std",
        not(all(target_family = "wasm", target_os = "unknown"))
    ))]
    fn compiled_query_time_limit_reports_exhaustion_without_poisoning_scratch() {
        use std::time::Duration;

        let compiled =
            CompiledQuery::new(QueryMol::from_str("[$([#6]-[#6])].[#7]").unwrap()).unwrap();
        let target = PreparedTarget::new(Smiles::from_str("CCC.N").unwrap());
        let mut scratch = MatchScratch::new();

        assert_eq!(
            compiled.matches_with_scratch_and_time_limit(&target, &mut scratch, Duration::ZERO),
            MatchLimitResult::Exceeded
        );
        assert_eq!(
            compiled.matches_with_scratch_and_time_limit(
                &target,
                &mut scratch,
                Duration::from_mins(1)
            ),
            MatchLimitResult::Complete(compiled.matches(&target))
        );
        assert!(compiled.matches(&target));
    }

    #[test]
    fn compiled_query_outcome_interrupt_reports_exhaustion_without_poisoning_scratch() {
        let compiled = CompiledQuery::new(QueryMol::from_str("CC").unwrap()).unwrap();
        let target = PreparedTarget::new(Smiles::from_str("CCC").unwrap());
        let mut scratch = MatchScratch::new();
        let expected = compiled.match_outcome(&target);

        assert_eq!(
            compiled.match_outcome_with_scratch_and_interrupt(&target, &mut scratch, || true),
            MatchOutcomeLimitResult::Exceeded
        );
        assert_eq!(
            compiled.match_outcome_with_scratch_and_interrupt(&target, &mut scratch, || false),
            MatchOutcomeLimitResult::Complete(expected)
        );
        assert_eq!(
            compiled.match_outcome_with_interrupt(&target, || false),
            MatchOutcomeLimitResult::Complete(expected)
        );
        assert!(expected.matched);
        assert_coverage_close(expected.coverage, 1.0);
    }

    #[test]
    #[cfg(all(
        feature = "std",
        not(all(target_family = "wasm", target_os = "unknown"))
    ))]
    fn compiled_query_outcome_time_limit_reports_exhaustion_without_poisoning_scratch() {
        use std::time::Duration;

        let compiled = CompiledQuery::new(QueryMol::from_str("CC").unwrap()).unwrap();
        let target = PreparedTarget::new(Smiles::from_str("CCC").unwrap());
        let mut scratch = MatchScratch::new();
        let expected = compiled.match_outcome(&target);

        assert_eq!(
            compiled.match_outcome_with_scratch_and_time_limit(
                &target,
                &mut scratch,
                Duration::ZERO
            ),
            MatchOutcomeLimitResult::Exceeded
        );
        assert_eq!(
            compiled.match_outcome_with_scratch_and_time_limit(
                &target,
                &mut scratch,
                Duration::from_mins(1)
            ),
            MatchOutcomeLimitResult::Complete(expected)
        );
        assert_eq!(
            compiled.match_outcome_with_time_limit(&target, Duration::from_mins(1)),
            MatchOutcomeLimitResult::Complete(expected)
        );
        assert!(expected.matched);
        assert_coverage_close(expected.coverage, 1.0);
    }

    #[test]
    fn reusable_scratch_matches_fast_path_query_shapes() {
        let cases = [
            ("O", "CCO", true),
            ("O", "CCC", false),
            ("C=O", "CC(=O)O", true),
            ("C=O", "CCO", false),
            ("COC", "CCOC", true),
            ("COC", "CCO", false),
            ("COCO", "CCOCO", true),
            ("COCO", "CCOCC", false),
            ("[#6]-[#6]-[#8]-[#6]-[#8]", "CCOCO", true),
            ("[#6]-[#6]-[#8]-[#6]-[#8]", "CCOCC", false),
        ];
        let mut scratch = MatchScratch::new();

        for (smarts, smiles, expected) in cases {
            let compiled = CompiledQuery::new(QueryMol::from_str(smarts).unwrap()).unwrap();
            let target = PreparedTarget::new(Smiles::from_str(smiles).unwrap());

            assert_eq!(compiled.matches(&target), expected);
            assert_eq!(
                compiled.matches_with_scratch(&target, &mut scratch),
                expected,
                "scratch mismatch for {smarts} against {smiles}"
            );
        }
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
    fn match_outcome_reports_combined_atom_and_bond_coverage() {
        let ring = QueryMol::from_str("c1ccccc1").unwrap();
        let outcome = ring.match_outcome("Cc1ccccc1").unwrap();
        assert!(outcome.matched);
        assert_coverage_close(outcome.coverage, 12.0 / 14.0);

        let atom_only = QueryMol::from_str("[#6]").unwrap();
        let outcome = atom_only.match_outcome("c1ccccc1").unwrap();
        assert!(outcome.matched);
        assert_coverage_close(outcome.coverage, 6.0 / 12.0);

        let no_match = QueryMol::from_str("[#7]")
            .unwrap()
            .match_outcome("CC")
            .unwrap();
        assert!(!no_match.matched);
        assert_coverage_close(no_match.coverage, 0.0);
    }

    #[test]
    fn match_outcome_unions_coverage_across_full_embeddings() {
        let query = CompiledQuery::new(QueryMol::from_str("CC").unwrap()).unwrap();
        let target = PreparedTarget::new(Smiles::from_str("CCC").unwrap());
        let mut scratch = MatchScratch::new();

        let outcome = query.match_outcome_with_scratch(&target, &mut scratch);
        assert!(outcome.matched);
        assert_coverage_close(outcome.coverage, 1.0);
    }

    #[test]
    fn match_outcome_counts_bonds_from_all_valid_embeddings() {
        let query = CompiledQuery::new(QueryMol::from_str("CCC").unwrap()).unwrap();
        let target = PreparedTarget::new(Smiles::from_str("C1CC1").unwrap());

        let outcome = query.match_outcome(&target);
        assert!(outcome.matched);
        assert_coverage_close(outcome.coverage, 1.0);
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
    fn bondless_boolean_fast_path_matches_counting_reference() {
        let cases = [
            ("C.C", "CC"),
            ("C.C", "C"),
            ("(C.C)", "CC"),
            ("(C.C)", "C.C"),
            ("(C).(C)", "CC"),
            ("(C).(C)", "C.C"),
            ("[$([#6,#7])].[R]", "c1ccccc1"),
            (
                "[!!12C,16O,D,D11,h,v14].([!r6&$([#6,#7]):57773].[R:30837])",
                "c1ccccc1",
            ),
        ];
        let mut scratch = MatchScratch::new();

        for (smarts, smiles) in cases {
            let query = QueryMol::from_str(smarts).unwrap();
            let target = PreparedTarget::new(Smiles::from_str(smiles).unwrap());
            let compiled = CompiledQuery::new(query).unwrap();
            let expected = compiled.match_count(&target) > 0;

            assert_eq!(
                compiled.matches(&target),
                expected,
                "compiled boolean mismatch for {smarts} against {smiles}"
            );
            assert_eq!(
                compiled.matches_with_scratch(&target, &mut scratch),
                expected,
                "scratch boolean mismatch for {smarts} against {smiles}"
            );
        }

        let long_anchored_target = format!("{}.Cl.[Na]", "C".repeat(64));
        let query = QueryMol::from_str("[Cl].[Na]").unwrap();
        let target = PreparedTarget::new(Smiles::from_str(&long_anchored_target).unwrap());
        let compiled = CompiledQuery::new(query).unwrap();
        let expected = compiled.match_count(&target) > 0;
        assert_eq!(compiled.matches(&target), expected);
        assert_eq!(
            compiled.matches_with_scratch(&target, &mut scratch),
            expected
        );
    }

    #[test]
    fn target_aware_root_selection_prefers_more_selective_unanchored_component() {
        let compiled = CompiledQuery::new(QueryMol::from_str("C.[Cl-].[Na+]").unwrap()).unwrap();
        let target = PreparedTarget::new(Smiles::from_str("CC.[Cl-].[Na+]").unwrap());
        let mut atom_stereo_cache = AtomStereoCache::new();
        let mut recursive_cache =
            RecursiveMatchCache::new(compiled.recursive_cache_slots, target.atom_count());
        let mut scratch = FreshSearchBuffers::new(&compiled, &target, &mut recursive_cache);
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
    fn target_aware_root_selection_scans_unanchored_recursive_misses() {
        let compiled = CompiledQuery::new(QueryMol::from_str("C.[!$([!#1])]").unwrap()).unwrap();
        let target = PreparedTarget::new(Smiles::from_str("CCCC").unwrap());
        let mut atom_stereo_cache = AtomStereoCache::new();
        let mut recursive_cache =
            RecursiveMatchCache::new(compiled.recursive_cache_slots, target.atom_count());
        let mut scratch = FreshSearchBuffers::new(&compiled, &target, &mut recursive_cache);
        let context =
            scratch
                .view()
                .into_context(&compiled, &mut atom_stereo_cache, &mut recursive_cache);

        assert_eq!(context.query_atom_anchor_widths[1], 0);
        assert_eq!(select_next_query_atom_for_target(&context), 1);
        assert!(!compiled.matches(&target));
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
    fn disconnected_boolean_precheck_matches_counting_reference() {
        let cases = [
            (
                "*@[B,X{16-}].[!R,b].([!#16]~[!#8].[!107*&r6]-[#6&H]=&@[D3])",
                "C1CCCCC1O.CC",
            ),
            (
                "*-,:,=C.(*(!=[!-]-&@1)!-[#6]-&@1~[!#6&H]-[!Cl].[!#6&H]-[!Cl]-,/&@[#6&R])",
                "ClCC1=CCCCC1.CC",
            ),
            ("C.O.N", "CCO"),
            ("(C.O).N", "CCO.N"),
        ];
        let mut scratch = MatchScratch::new();

        for (smarts, smiles) in cases {
            let query = QueryMol::from_str(smarts).unwrap();
            let target = PreparedTarget::new(Smiles::from_str(smiles).unwrap());
            let compiled = CompiledQuery::new(query).unwrap();
            let expected = compiled.match_count(&target) > 0;

            assert_eq!(
                compiled.matches(&target),
                expected,
                "compiled boolean mismatch for {smarts} against {smiles}"
            );
            assert_eq!(
                compiled.matches_with_scratch(&target, &mut scratch),
                expected,
                "scratch boolean mismatch for {smarts} against {smiles}"
            );
        }
    }

    #[test]
    fn disconnected_prechecks_skip_recursive_components() {
        let compiled = CompiledQuery::new(QueryMol::from_str("[$(C.O)].N").unwrap()).unwrap();
        let precheck_count = compiled
            .component_plan
            .iter()
            .filter(|component| component.supports_precheck())
            .count();

        assert_eq!(precheck_count, 1);
    }

    #[test]
    fn disconnected_component_search_handles_repeated_component_fuzz_artifact() {
        let smarts = concat!(
            "[!!12C,16O,D,D11,h,v14].",
            "([!r6&$([#6,#7]):57773].[R:30837]).",
            "[+0,z11,+0&+0,+0]-;@[+0&+0,+0,z11,+0](!:[+0,z11,+0&+0,+0]).",
            "[+0&+0,+0,z11,+0]!:[+0,z11,+0&+0,+0](-;@[+0&+0,+0,z11,+0]).",
            "[z11,+0&+0,+0,z11]!:[+0,+0,z11,+0](!:[+0,z11,+0&+0,+0]).",
            "[+0,+0,z11,+0]!:[+0,z11,+0&+0,+0](-;@[+0&+0,+0,z11,+0])",
        );
        let target = PreparedTarget::new(
            Smiles::from_str("CCN1C(=O)C(=CNC2=CC=C(C=C2)CCN3CCN(CC3)C)SC1=C(C#N)C(=O)OCC")
                .unwrap(),
        );
        let compiled = CompiledQuery::new(QueryMol::from_str(smarts).unwrap()).unwrap();
        let mut scratch = MatchScratch::new();

        let mut polls = 0usize;
        assert_eq!(
            compiled.matches_with_scratch_and_interrupt(&target, &mut scratch, || {
                polls += 1;
                polls > 1_000
            }),
            MatchLimitResult::Complete(false)
        );
    }

    #[test]
    fn disconnected_component_search_handles_recursive_component_timeout_artifact() {
        let smarts = concat!(
            "*~[!!#6&!!#6&!!#6&!!#6]",
            "(~[!!#6&!!#6&!!#6&!!#6]~[!!#6&!!#6&!!#6&!!#6]).",
            "[!!#6&!!#6&!!#6&!!#6].",
            "[!!#6&!#6&$([#6,#7])&r]~[!!R&!!#6&!!#6].",
            "[!!#6&!!#6&!!#6&!!#6].",
            "[!!#6&!!#6&!!#6&!!#6]",
        );
        let target = PreparedTarget::new(
            Smiles::from_str(
                "C1CCN(C1)CCOC2=CC=C(C=C2)CC3=C(SC4=CC=CC=C43)C5=CC=C(C=C5)CCNCC6=CN=CC=C6",
            )
            .unwrap(),
        );
        let compiled = CompiledQuery::new(QueryMol::from_str(smarts).unwrap()).unwrap();
        let mut scratch = MatchScratch::new();
        let mut polls = 0usize;

        assert!(compiled
            .component_plan
            .iter()
            .all(ComponentPlanEntry::supports_component_search));
        assert_eq!(
            compiled.matches_with_scratch_and_interrupt(&target, &mut scratch, || {
                polls += 1;
                polls > 1_000
            }),
            MatchLimitResult::Complete(false)
        );
    }

    #[test]
    fn double_negated_bracket_atoms_keep_fast_anchors() {
        let compiled =
            CompiledQuery::new(QueryMol::from_str("[!!#6&!!#6&!!#6&!!#6]").unwrap()).unwrap();
        let matcher = &compiled.atom_matchers[0];

        assert!(matcher.complete);
        assert!(matcher
            .predicates
            .iter()
            .any(|predicate| { matches!(predicate, AtomFastPredicate::AtomicNumber(6)) }));
        assert!(compiled.matches(&PreparedTarget::new(Smiles::from_str("C").unwrap())));
        assert!(!compiled.matches(&PreparedTarget::new(Smiles::from_str("O").unwrap())));
    }

    #[test]
    fn bracket_expression_matching_polls_interrupt_limit() {
        let mut smarts = String::from("[");
        for index in 0..5000 {
            if index > 0 {
                smarts.push('&');
            }
            smarts.push_str("!#7");
        }
        smarts.push(']');

        let compiled = CompiledQuery::new(QueryMol::from_str(&smarts).unwrap()).unwrap();
        let target = PreparedTarget::new(Smiles::from_str("C").unwrap());
        let mut scratch = MatchScratch::new();
        let mut polls = 0usize;

        assert_eq!(
            compiled.matches_with_scratch_and_interrupt(&target, &mut scratch, || {
                polls += 1;
                polls > 1
            }),
            MatchLimitResult::Exceeded
        );
        assert!(polls > 1);
    }

    #[test]
    fn small_disconnected_tail_uses_component_assignment_fuzz_artifacts() {
        let cases = [
            concat!(
                "*[$([#6])&!!#6](~[!!#6&!!#6&!!#6&!!#6]~[!!#6&!!#6&!!#6&!!#6]).",
                "[!!#6&!!#6&!!#6&!!#6].",
                "[!!#6&!$([#6])&!!#6&!!#6].",
                "[!!#6&!!#6&!!#6&!!#6].",
                "[!!#6&!!#6&!!#6&!!#6]",
            ),
            concat!(
                "*~[D3&!D3&!a,$([#6,#7]),c:0]([!*:0]~[!R,!!R,!!R&!!#6]).",
                "[!!#6&!!#6&!!#6&!!#6].",
                "[!!#6&!!#6&!!#6&!!#6].",
                "[!!#6&!!#6&!!#6&!!#6].",
                "[!!#6&!!#6&!!#6&!!#6]",
            ),
            concat!(
                "[!!#6&!!#6&!!#6&!!#6]~[!!#6&!!#6&!!#6&!!#6].",
                "[!!#6&!!#6&*&!!#6].",
                "[!!#6&!!#6&!!#6,!*:0][!*:0]([!*:0][!$([#7])]).",
                "[$([#7])].",
                "[!!#6&!!#6&!!#6&!!#6]",
            ),
        ];
        let target = PreparedTarget::new(
            Smiles::from_str(
                "C1CCN(C1)CCOC2=CC=C(C=C2)CC3=C(SC4=CC=CC=C43)C5=CC=C(C=C5)CCNCC6=CN=CC=C6",
            )
            .unwrap(),
        );

        for smarts in cases {
            let compiled = CompiledQuery::new(QueryMol::from_str(smarts).unwrap()).unwrap();
            let mut scratch = MatchScratch::new();
            let mut polls = 0usize;

            assert!(
                should_use_disconnected_component_search(&compiled),
                "{smarts}"
            );
            assert!(matches!(
                compiled.matches_with_scratch_and_interrupt(&target, &mut scratch, || {
                    polls += 1;
                    polls > 1_000
                }),
                MatchLimitResult::Complete(_)
            ));
        }
    }

    #[test]
    fn large_ungrouped_wildcard_components_stay_on_general_search_path() {
        let compiled = CompiledQuery::new(
            QueryMol::from_str("*-***.*.*-***.*.***.***-*.****-*******").unwrap(),
        )
        .unwrap();

        assert!(!should_use_disconnected_component_search(&compiled));
    }

    #[test]
    fn ungrouped_disconnected_wildcard_query_does_not_enumerate_all_component_embeddings() {
        let query = QueryMol::from_str("*-***.*.*-***.*.***.***-*.****-*******").unwrap();
        let target = PreparedTarget::new(
            Smiles::from_str(
                "C1CCN(C1)CCOC2=CC=C(C=C2)CC3=C(SC4=CC=CC=C43)C5=CC=C(C=C5)CCNCC6=CN=CC=C6",
            )
            .unwrap(),
        );
        let compiled = CompiledQuery::new(query).unwrap();
        let mut scratch = MatchScratch::new();
        let mut polls = 0usize;

        assert!(matches!(
            compiled.matches_with_scratch_and_interrupt(&target, &mut scratch, || {
                polls += 1;
                polls > 1_000
            }),
            MatchLimitResult::Complete(_)
        ));
    }

    #[test]
    fn disconnected_component_search_respects_zero_level_groups() {
        let compiled = CompiledQuery::new(QueryMol::from_str("(C-C.N)").unwrap()).unwrap();
        let same_component = PreparedTarget::new(Smiles::from_str("CCN").unwrap());
        let separate_components = PreparedTarget::new(Smiles::from_str("CC.N").unwrap());

        assert!(compiled.matches(&same_component));
        assert!(!compiled.matches(&separate_components));
    }

    #[test]
    fn disconnected_component_assignment_handles_deep_component_lists_without_recursing() {
        let component_count = 50_000usize;
        let entries = (0..component_count)
            .map(|component_id| ComponentPlanEntry {
                component_id,
                group: None,
                atom_count: 1,
                precheckable: true,
                matcher: ComponentMatcher::SingleAtom(0),
            })
            .collect::<Vec<_>>();
        let component_sets = entries
            .iter()
            .enumerate()
            .map(|(target_atom, entry)| ComponentEmbeddingSet {
                entry,
                embeddings: vec![ComponentEmbedding {
                    target_atoms: alloc::boxed::Box::<[usize]>::from([target_atom]),
                    target_component: Some(0),
                }],
            })
            .collect::<Vec<_>>();
        let mut used_target_atoms = vec![false; component_count];
        let mut group_targets = Vec::new();

        assert!(component_embedding_assignment_exists(
            &component_sets,
            &mut used_target_atoms,
            &mut group_targets,
            &NO_MATCH_LIMIT,
        ));
    }

    #[test]
    fn compiled_query_handles_deep_recursive_component_chain_regression() {
        let smarts = recursive_component_chain(12);
        let compiled = CompiledQuery::new(QueryMol::from_str(&smarts).unwrap()).unwrap();
        let precheck_count = compiled
            .component_plan
            .iter()
            .filter(|component| component.supports_precheck())
            .count();

        assert_eq!(compiled.query().component_count(), 2);
        assert_eq!(precheck_count, 1);
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
        assert!(QueryMol::from_str("[$([#6;R])]")
            .unwrap()
            .matches("C1CCCCC1")
            .unwrap());
        assert!(!QueryMol::from_str("[$([#6;R])]")
            .unwrap()
            .matches("CC")
            .unwrap());
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
