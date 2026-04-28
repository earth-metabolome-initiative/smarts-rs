//! Epserde-backed persisted screening-index components.
//!
//! This module is feature-gated because epserde deserialization is unsafe and
//! because persisted indexes require `std` for file-backed use. The persisted
//! layouts here are explicit flat formats, not direct dumps of private runtime
//! structs.

use alloc::{boxed::Box, vec, vec::Vec};
use core::{marker::PhantomData, ops::Range};

use epserde::{deser, prelude::Epserde, ser};

use super::{
    atom_feature_is_unconstrained, atom_feature_satisfies_query,
    bitset::{
        bitset_word_count, ensure_zeroed_words, for_each_set_bit, intersect_source,
        intersect_source_with_population, CountBitsetIndex, RequiredCountFilter,
    },
    bond_feature_is_unconstrained, bond_feature_satisfies_query,
    features::{
        AtomFeature, EdgeBondFeature, EdgeFeature, Path3Feature, Path4Feature, RequiredBondKind,
        Star3Arm, Star3Feature,
    },
    finalize_cached_sparse_candidate_mask, load_cached_candidate_mask,
    merge_reversible_feature_mask, prepare_sparse_candidate_counts, should_filter_sparse_counts,
    CandidateMaskState, IndexedFeatureCountIndex, IndexedFeatureIdMask,
    IndexedSparseFeatureCountIndex, QueryFeatureFilter, QueryScreen, SparseCandidateMaskBuffers,
    TargetCandidateSet, TargetCorpusIndex, TargetCorpusIndexShard, TargetCorpusIndexStats,
    TargetCorpusScratch,
};

const FORMAT_VERSION: u32 = 1;
const ATOM_FEATURE_WIDTH: u16 = 5;
const BOND_FEATURE_WIDTH: u16 = 2;
const ATOM_FEATURE_WIDTH_USIZE: usize = ATOM_FEATURE_WIDTH as usize;
const BOND_FEATURE_WIDTH_USIZE: usize = BOND_FEATURE_WIDTH as usize;
const EDGE_FEATURE_WIDTH: u16 = ATOM_FEATURE_WIDTH * 2 + BOND_FEATURE_WIDTH;
const PATH3_FEATURE_WIDTH: u16 = ATOM_FEATURE_WIDTH * 3 + BOND_FEATURE_WIDTH * 2;
const PATH4_FEATURE_WIDTH: u16 = ATOM_FEATURE_WIDTH * 4 + BOND_FEATURE_WIDTH * 3;
const STAR3_FEATURE_WIDTH: u16 = ATOM_FEATURE_WIDTH * 4 + BOND_FEATURE_WIDTH * 3;

/// A count-bitset filter loaded from a persisted flat count index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PersistedRequiredCountFilter<'a> {
    /// Number of targets selected by `source`.
    pub population: usize,
    /// Bitset words for targets whose count is at least the requested value.
    pub source: &'a [u64],
}

/// Flat persisted representation of a [`CountBitsetIndex`].
///
/// The runtime index stores one boxed bitset per threshold. This persisted form
/// flattens those bitsets into one word array plus offsets, which is a better
/// fit for epserde and mmap-backed loading.
#[derive(Debug, Clone, PartialEq, Eq, Epserde)]
pub struct PersistedCountBitsetIndex<
    Thresholds = Vec<u32>,
    Populations = Vec<u64>,
    BitsetOffsets = Vec<u64>,
    BitsetWords = Vec<u64>,
> {
    thresholds: Thresholds,
    populations: Populations,
    bitset_offsets: BitsetOffsets,
    bitset_words: BitsetWords,
}

/// Persisted aggregate statistics for a target corpus index shard.
#[allow(clippy::struct_field_names)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Epserde)]
#[epserde(deep_copy)]
pub struct PersistedTargetCorpusIndexStats {
    target_count: u64,
    edge_feature_count: u64,
    edge_posting_count: u64,
    edge_atom_domain_count: u64,
    edge_bond_domain_count: u64,
    path3_feature_count: u64,
    path3_posting_count: u64,
    path3_atom_domain_count: u64,
    path3_bond_domain_count: u64,
    path4_feature_count: u64,
    path4_posting_count: u64,
    path4_atom_domain_count: u64,
    path4_bond_domain_count: u64,
    star3_feature_count: u64,
    star3_posting_count: u64,
    star3_atom_domain_count: u64,
    star3_bond_domain_count: u64,
}

/// Persisted keyed count indexes such as element, degree, and hydrogen counts.
#[derive(Debug, Clone, PartialEq, Eq, Epserde)]
pub struct PersistedKeyedCountIndexes<
    Keys = Vec<u16>,
    CountIndex = PersistedCountBitsetIndex,
    Indexes = Vec<CountIndex>,
> {
    keys: Keys,
    indexes: Indexes,
    count_index: PhantomData<CountIndex>,
}

/// Persisted flat form of the scalar count indexes in a target corpus index.
///
/// This covers atom/component/aromatic/ring atom counts and the basic bond
/// count screens. It is the first shard-persistence building block because
/// these count-bitset indexes are large, mmap-friendly, and independent of the
/// local feature-key encoding.
#[derive(Debug, Clone, PartialEq, Eq, Epserde)]
pub struct PersistedScalarCountIndexes<CountIndex = PersistedCountBitsetIndex> {
    atom: CountIndex,
    component: CountIndex,
    aromatic_atom: CountIndex,
    ring_atom: CountIndex,
    single_bond: CountIndex,
    double_bond: CountIndex,
    triple_bond: CountIndex,
    aromatic_bond: CountIndex,
    ring_bond: CountIndex,
}

/// Persisted flat form of element, degree, and total-hydrogen count indexes.
#[derive(Debug, Clone, PartialEq, Eq, Epserde)]
pub struct PersistedAtomPropertyCountIndexes<
    ElementCounts = PersistedKeyedCountIndexes,
    DegreeCounts = PersistedKeyedCountIndexes,
    TotalHydrogenCounts = PersistedKeyedCountIndexes,
> {
    element: ElementCounts,
    degree: DegreeCounts,
    total_hydrogen: TotalHydrogenCounts,
}

/// Persisted fixed-width feature domain.
#[derive(Debug, Clone, PartialEq, Eq, Epserde)]
pub struct PersistedFeatureDomain<Features = Vec<u16>> {
    feature_width: u16,
    features: Features,
}

/// Persisted feature-id mask index.
#[derive(Debug, Clone, PartialEq, Eq, Epserde)]
pub struct PersistedFeatureMaskIndex<
    Features = Vec<u16>,
    MaskOffsets = Vec<u64>,
    MaskWords = Vec<u64>,
> {
    feature_width: u16,
    features: Features,
    mask_offsets: MaskOffsets,
    mask_words: MaskWords,
}

/// Persisted sparse local-feature posting index.
#[derive(Debug, Clone, PartialEq, Eq, Epserde)]
pub struct PersistedSparseFeatureIndex<
    Features = Vec<u16>,
    SingletonOffsets = Vec<u64>,
    SingletonTargetIds = Vec<u32>,
    CountedOffsets = Vec<u64>,
    CountedTargetIds = Vec<u32>,
    CountedValues = Vec<u16>,
> {
    feature_width: u16,
    features: Features,
    singleton_offsets: SingletonOffsets,
    singleton_target_ids: SingletonTargetIds,
    counted_offsets: CountedOffsets,
    counted_target_ids: CountedTargetIds,
    counted_values: CountedValues,
}

/// Persisted local-feature family: postings, masks, and feature domains.
#[derive(Debug, Clone, PartialEq, Eq, Epserde)]
pub struct PersistedLocalFeatureFamily<
    Postings = PersistedSparseFeatureIndex,
    MaskIndex = PersistedFeatureMaskIndex,
    MaskIndexes = Vec<MaskIndex>,
    AtomDomain = PersistedFeatureDomain,
    BondDomain = PersistedFeatureDomain,
> {
    postings: Postings,
    mask_indexes: MaskIndexes,
    atom_domain: AtomDomain,
    bond_domain: BondDomain,
    mask_index: PhantomData<MaskIndex>,
}

/// Persisted flat representation of one target corpus index shard.
///
/// Retained [`super::TargetScreen`] values are intentionally not stored. They
/// are not needed for candidate generation, and omitting them keeps shard files
/// focused on the scalable target-side index.
#[derive(Debug, Clone, PartialEq, Eq, Epserde)]
pub struct PersistedTargetCorpusIndexShard<
    ScalarCounts = PersistedScalarCountIndexes,
    AtomPropertyCounts = PersistedAtomPropertyCountIndexes,
    Edge = PersistedLocalFeatureFamily,
    Path3 = PersistedLocalFeatureFamily,
    Path4 = PersistedLocalFeatureFamily,
    Star3 = PersistedLocalFeatureFamily,
> {
    format_version: u32,
    base_target_id: u64,
    target_count: u64,
    stats: PersistedTargetCorpusIndexStats,
    scalar_counts: ScalarCounts,
    atom_property_counts: AtomPropertyCounts,
    edge: Edge,
    path3: Path3,
    path4: Path4,
    star3: Star3,
}

impl PersistedTargetCorpusIndexShard {
    /// Builds a persisted shard payload from an owned runtime shard.
    ///
    /// # Panics
    ///
    /// Panics if the shard base id, target count, or an internal posting/mask
    /// offset exceeds the persisted format's `u64`/`u32` capacity.
    #[must_use]
    pub fn from_index_shard(shard: &TargetCorpusIndexShard) -> Self {
        let index = shard.index();
        Self {
            format_version: FORMAT_VERSION,
            base_target_id: u64::try_from(shard.base_target_id())
                .expect("target shard base id exceeds persisted index capacity"),
            target_count: u64::try_from(index.len())
                .expect("target shard length exceeds persisted index capacity"),
            stats: PersistedTargetCorpusIndexStats::from_stats(index.stats()),
            scalar_counts: PersistedScalarCountIndexes::from_target_corpus_index(index),
            atom_property_counts: PersistedAtomPropertyCountIndexes::from_target_corpus_index(
                index,
            ),
            edge: PersistedLocalFeatureFamily::from_edge_index(index),
            path3: PersistedLocalFeatureFamily::from_path3_index(index),
            path4: PersistedLocalFeatureFamily::from_path4_index(index),
            star3: PersistedLocalFeatureFamily::from_star3_index(index),
        }
    }

    /// Serializes this shard payload into an epserde writer.
    ///
    /// # Errors
    ///
    /// Returns an epserde serialization error if the writer fails.
    ///
    /// # Safety
    ///
    /// This wraps epserde serialization. The serialized bytes should be treated
    /// as trusted data for later epserde deserialization.
    pub unsafe fn serialize_into_unchecked(
        &self,
        writer: &mut impl ser::WriteNoStd,
    ) -> ser::Result<usize> {
        unsafe { ser::Serialize::serialize(self, writer) }
    }

    /// Stores this shard payload to a file.
    ///
    /// # Errors
    ///
    /// Returns an epserde serialization error if the file cannot be created or
    /// written.
    ///
    /// # Safety
    ///
    /// This wraps epserde serialization. The resulting file must be treated as
    /// trusted input when it is later deserialized.
    pub unsafe fn store_unchecked(&self, path: impl AsRef<std::path::Path>) -> ser::Result<()> {
        unsafe { ser::Serialize::store(self, path) }
    }

    /// Deserializes a full owned shard payload from an epserde reader.
    ///
    /// # Errors
    ///
    /// Returns an epserde deserialization error if the input is malformed.
    ///
    /// # Safety
    ///
    /// The input must be trusted bytes produced for this type by a compatible
    /// version of this crate and epserde.
    pub unsafe fn deserialize_full_unchecked(
        reader: &mut impl deser::ReadNoStd,
    ) -> deser::Result<Self> {
        unsafe { deser::Deserialize::deserialize_full(reader) }
    }

    /// Memory-maps and epsilon-deserializes a trusted persisted shard file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened, mapped, or deserialized.
    ///
    /// # Safety
    ///
    /// The file must be trusted bytes produced for this type by a compatible
    /// version of this crate and epserde. Memory-mapped files must not be
    /// mutated while the returned value is alive.
    pub unsafe fn mmap_unchecked(
        path: impl AsRef<std::path::Path>,
        flags: deser::Flags,
    ) -> Result<deser::MemCase<Self>, Box<dyn std::error::Error + Send + Sync + 'static>> {
        unsafe { <Self as deser::Deserialize>::mmap(path, flags) }
            .map_err(|error| -> Box<dyn std::error::Error + Send + Sync + 'static> { error.into() })
    }
}

impl<ScalarCounts, AtomPropertyCounts, Edge, Path3, Path4, Star3>
    PersistedTargetCorpusIndexShard<ScalarCounts, AtomPropertyCounts, Edge, Path3, Path4, Star3>
{
    /// Returns the persisted shard format version.
    #[inline]
    #[must_use]
    pub const fn format_version(&self) -> u32 {
        self.format_version
    }

    /// Returns the global id of local target `0` in this shard.
    #[inline]
    #[must_use]
    pub const fn base_target_id(&self) -> u64 {
        self.base_target_id
    }

    /// Returns the number of indexed targets in this persisted shard.
    #[inline]
    #[must_use]
    pub const fn target_count(&self) -> u64 {
        self.target_count
    }

    /// Returns persisted diagnostic stats for this shard.
    #[inline]
    #[must_use]
    pub const fn stats(&self) -> PersistedTargetCorpusIndexStats {
        self.stats
    }

    /// Returns persisted scalar count indexes.
    #[inline]
    #[must_use]
    pub const fn scalar_counts(&self) -> &ScalarCounts {
        &self.scalar_counts
    }

    /// Returns persisted atom-property count indexes.
    #[inline]
    #[must_use]
    pub const fn atom_property_counts(&self) -> &AtomPropertyCounts {
        &self.atom_property_counts
    }

    /// Returns persisted edge local-feature data.
    #[inline]
    #[must_use]
    pub const fn edge(&self) -> &Edge {
        &self.edge
    }

    /// Returns persisted path3 local-feature data.
    #[inline]
    #[must_use]
    pub const fn path3(&self) -> &Path3 {
        &self.path3
    }

    /// Returns persisted path4 local-feature data.
    #[inline]
    #[must_use]
    pub const fn path4(&self) -> &Path4 {
        &self.path4
    }

    /// Returns persisted star3 local-feature data.
    #[inline]
    #[must_use]
    pub const fn star3(&self) -> &Star3 {
        &self.star3
    }
}

impl PersistedTargetCorpusIndexStats {
    fn from_stats(stats: TargetCorpusIndexStats) -> Self {
        Self {
            target_count: persist_usize(stats.target_count),
            edge_feature_count: persist_usize(stats.edge_feature_count),
            edge_posting_count: persist_usize(stats.edge_posting_count),
            edge_atom_domain_count: persist_usize(stats.edge_atom_domain_count),
            edge_bond_domain_count: persist_usize(stats.edge_bond_domain_count),
            path3_feature_count: persist_usize(stats.path3_feature_count),
            path3_posting_count: persist_usize(stats.path3_posting_count),
            path3_atom_domain_count: persist_usize(stats.path3_atom_domain_count),
            path3_bond_domain_count: persist_usize(stats.path3_bond_domain_count),
            path4_feature_count: persist_usize(stats.path4_feature_count),
            path4_posting_count: persist_usize(stats.path4_posting_count),
            path4_atom_domain_count: persist_usize(stats.path4_atom_domain_count),
            path4_bond_domain_count: persist_usize(stats.path4_bond_domain_count),
            star3_feature_count: persist_usize(stats.star3_feature_count),
            star3_posting_count: persist_usize(stats.star3_posting_count),
            star3_atom_domain_count: persist_usize(stats.star3_atom_domain_count),
            star3_bond_domain_count: persist_usize(stats.star3_bond_domain_count),
        }
    }
}

impl PersistedKeyedCountIndexes {
    fn from_count_indexes<T, F>(entries: &IndexedFeatureCountIndex<T>, mut encode_key: F) -> Self
    where
        F: FnMut(&T) -> u16,
    {
        let mut keys = Vec::with_capacity(entries.len());
        let mut indexes = Vec::with_capacity(entries.len());
        for (feature, index) in entries {
            keys.push(encode_key(feature));
            indexes.push(PersistedCountBitsetIndex::from_count_bitset_index(index));
        }

        Self {
            keys,
            indexes,
            count_index: PhantomData,
        }
    }
}

impl<Keys, CountIndex, Indexes> PersistedKeyedCountIndexes<Keys, CountIndex, Indexes> {
    /// Returns persisted count-index keys.
    #[inline]
    #[must_use]
    pub fn keys(&self) -> &[u16]
    where
        Keys: AsRef<[u16]>,
    {
        self.keys.as_ref()
    }

    /// Returns persisted count indexes, one per key.
    #[inline]
    #[must_use]
    pub fn indexes(&self) -> &[CountIndex]
    where
        Indexes: AsRef<[CountIndex]>,
    {
        self.indexes.as_ref()
    }
}

impl PersistedScalarCountIndexes {
    /// Builds persisted scalar count indexes from an owned runtime index.
    #[must_use]
    pub fn from_target_corpus_index(index: &TargetCorpusIndex) -> Self {
        Self {
            atom: PersistedCountBitsetIndex::from_count_bitset_index(&index.atom_count_index),
            component: PersistedCountBitsetIndex::from_count_bitset_index(
                &index.component_count_index,
            ),
            aromatic_atom: PersistedCountBitsetIndex::from_count_bitset_index(
                &index.aromatic_atom_count_index,
            ),
            ring_atom: PersistedCountBitsetIndex::from_count_bitset_index(
                &index.ring_atom_count_index,
            ),
            single_bond: PersistedCountBitsetIndex::from_count_bitset_index(
                &index.single_bond_count_index,
            ),
            double_bond: PersistedCountBitsetIndex::from_count_bitset_index(
                &index.double_bond_count_index,
            ),
            triple_bond: PersistedCountBitsetIndex::from_count_bitset_index(
                &index.triple_bond_count_index,
            ),
            aromatic_bond: PersistedCountBitsetIndex::from_count_bitset_index(
                &index.aromatic_bond_count_index,
            ),
            ring_bond: PersistedCountBitsetIndex::from_count_bitset_index(
                &index.ring_bond_count_index,
            ),
        }
    }
}

impl<CountIndex> PersistedScalarCountIndexes<CountIndex> {
    /// Returns the persisted atom-count index.
    #[inline]
    #[must_use]
    pub const fn atom_counts(&self) -> &CountIndex {
        &self.atom
    }

    /// Returns the persisted component-count index.
    #[inline]
    #[must_use]
    pub const fn component_counts(&self) -> &CountIndex {
        &self.component
    }

    /// Returns the persisted aromatic-atom-count index.
    #[inline]
    #[must_use]
    pub const fn aromatic_atom_counts(&self) -> &CountIndex {
        &self.aromatic_atom
    }

    /// Returns the persisted ring-atom-count index.
    #[inline]
    #[must_use]
    pub const fn ring_atom_counts(&self) -> &CountIndex {
        &self.ring_atom
    }

    /// Returns the persisted single-bond-count index.
    #[inline]
    #[must_use]
    pub const fn single_bond_counts(&self) -> &CountIndex {
        &self.single_bond
    }

    /// Returns the persisted double-bond-count index.
    #[inline]
    #[must_use]
    pub const fn double_bond_counts(&self) -> &CountIndex {
        &self.double_bond
    }

    /// Returns the persisted triple-bond-count index.
    #[inline]
    #[must_use]
    pub const fn triple_bond_counts(&self) -> &CountIndex {
        &self.triple_bond
    }

    /// Returns the persisted aromatic-bond-count index.
    #[inline]
    #[must_use]
    pub const fn aromatic_bond_counts(&self) -> &CountIndex {
        &self.aromatic_bond
    }

    /// Returns the persisted ring-bond-count index.
    #[inline]
    #[must_use]
    pub const fn ring_bond_counts(&self) -> &CountIndex {
        &self.ring_bond
    }
}

impl PersistedAtomPropertyCountIndexes {
    fn from_target_corpus_index(index: &TargetCorpusIndex) -> Self {
        Self {
            element: PersistedKeyedCountIndexes::from_count_indexes(
                &index.indexed_element_count_index,
                |element| u16::from(u8::from(*element)),
            ),
            degree: PersistedKeyedCountIndexes::from_count_indexes(
                &index.indexed_degree_count_index,
                |degree| *degree,
            ),
            total_hydrogen: PersistedKeyedCountIndexes::from_count_indexes(
                &index.indexed_total_hydrogen_count_index,
                |total_hydrogen| *total_hydrogen,
            ),
        }
    }
}

impl<ElementCounts, DegreeCounts, TotalHydrogenCounts>
    PersistedAtomPropertyCountIndexes<ElementCounts, DegreeCounts, TotalHydrogenCounts>
{
    /// Returns persisted element count indexes keyed by atomic number.
    #[inline]
    #[must_use]
    pub const fn element_counts(&self) -> &ElementCounts {
        &self.element
    }

    /// Returns persisted exact-degree count indexes.
    #[inline]
    #[must_use]
    pub const fn degree_counts(&self) -> &DegreeCounts {
        &self.degree
    }

    /// Returns persisted total-hydrogen count indexes.
    #[inline]
    #[must_use]
    pub const fn total_hydrogen_counts(&self) -> &TotalHydrogenCounts {
        &self.total_hydrogen
    }
}

impl PersistedFeatureDomain {
    fn from_atom_features(features: &[AtomFeature]) -> Self {
        Self::from_features(features, ATOM_FEATURE_WIDTH, push_atom_feature)
    }

    fn from_bond_features(features: &[EdgeBondFeature]) -> Self {
        Self::from_features(features, BOND_FEATURE_WIDTH, push_bond_feature)
    }

    fn from_features<T, F>(features: &[T], feature_width: u16, mut push_feature: F) -> Self
    where
        T: Copy,
        F: FnMut(&mut Vec<u16>, T),
    {
        let mut encoded = Vec::with_capacity(features.len() * usize::from(feature_width));
        for &feature in features {
            push_feature(&mut encoded, feature);
        }

        Self {
            feature_width,
            features: encoded,
        }
    }
}

impl<Features> PersistedFeatureDomain<Features> {
    /// Returns the fixed feature width in `u16` code units.
    #[inline]
    #[must_use]
    pub const fn feature_width(&self) -> u16 {
        self.feature_width
    }

    /// Returns flattened fixed-width encoded features.
    #[inline]
    #[must_use]
    pub fn features(&self) -> &[u16]
    where
        Features: AsRef<[u16]>,
    {
        self.features.as_ref()
    }
}

impl PersistedFeatureMaskIndex {
    fn from_atom_feature_masks(entries: &IndexedFeatureIdMask<AtomFeature>) -> Self {
        Self::from_feature_masks(entries, ATOM_FEATURE_WIDTH, push_atom_feature)
    }

    fn from_bond_feature_masks(entries: &IndexedFeatureIdMask<EdgeBondFeature>) -> Self {
        Self::from_feature_masks(entries, BOND_FEATURE_WIDTH, push_bond_feature)
    }

    fn from_feature_masks<T, F>(
        entries: &IndexedFeatureIdMask<T>,
        feature_width: u16,
        mut push_feature: F,
    ) -> Self
    where
        T: Copy,
        F: FnMut(&mut Vec<u16>, T),
    {
        let mut features = Vec::with_capacity(entries.len() * usize::from(feature_width));
        let mut mask_offsets = Vec::with_capacity(entries.len() + 1);
        let mut mask_words = Vec::new();
        mask_offsets.push(0);
        for &(feature, ref mask) in entries {
            push_feature(&mut features, feature);
            mask_words.extend_from_slice(mask);
            mask_offsets.push(persist_usize(mask_words.len()));
        }

        Self {
            feature_width,
            features,
            mask_offsets,
            mask_words,
        }
    }
}

impl<Features, MaskOffsets, MaskWords> PersistedFeatureMaskIndex<Features, MaskOffsets, MaskWords> {
    /// Returns the fixed feature width in `u16` code units.
    #[inline]
    #[must_use]
    pub const fn feature_width(&self) -> u16 {
        self.feature_width
    }

    /// Returns flattened fixed-width encoded features.
    #[inline]
    #[must_use]
    pub fn features(&self) -> &[u16]
    where
        Features: AsRef<[u16]>,
    {
        self.features.as_ref()
    }

    /// Returns offsets into [`PersistedFeatureMaskIndex::mask_words`].
    #[inline]
    #[must_use]
    pub fn mask_offsets(&self) -> &[u64]
    where
        MaskOffsets: AsRef<[u64]>,
    {
        self.mask_offsets.as_ref()
    }

    /// Returns flattened mask words.
    #[inline]
    #[must_use]
    pub fn mask_words(&self) -> &[u64]
    where
        MaskWords: AsRef<[u64]>,
    {
        self.mask_words.as_ref()
    }
}

impl PersistedSparseFeatureIndex {
    fn from_sparse_features<T, F>(
        entries: &IndexedSparseFeatureCountIndex<T>,
        feature_width: u16,
        mut push_feature: F,
    ) -> Self
    where
        T: Copy,
        F: FnMut(&mut Vec<u16>, T),
    {
        let mut features = Vec::with_capacity(entries.len() * usize::from(feature_width));
        let mut singleton_offsets = Vec::with_capacity(entries.len() + 1);
        let mut singleton_target_ids = Vec::new();
        let mut counted_offsets = Vec::with_capacity(entries.len() + 1);
        let mut counted_target_ids = Vec::new();
        let mut counted_values = Vec::new();
        singleton_offsets.push(0);
        counted_offsets.push(0);

        for &(feature, ref counts) in entries {
            push_feature(&mut features, feature);
            singleton_target_ids.extend_from_slice(&counts.singleton_targets);
            singleton_offsets.push(persist_usize(singleton_target_ids.len()));
            for &(target_id, count) in &counts.counted_targets {
                counted_target_ids.push(target_id);
                counted_values.push(count);
            }
            counted_offsets.push(persist_usize(counted_target_ids.len()));
        }

        Self {
            feature_width,
            features,
            singleton_offsets,
            singleton_target_ids,
            counted_offsets,
            counted_target_ids,
            counted_values,
        }
    }
}

impl<
        Features,
        SingletonOffsets,
        SingletonTargetIds,
        CountedOffsets,
        CountedTargetIds,
        CountedValues,
    >
    PersistedSparseFeatureIndex<
        Features,
        SingletonOffsets,
        SingletonTargetIds,
        CountedOffsets,
        CountedTargetIds,
        CountedValues,
    >
{
    /// Returns the fixed feature width in `u16` code units.
    #[inline]
    #[must_use]
    pub const fn feature_width(&self) -> u16 {
        self.feature_width
    }

    /// Returns flattened fixed-width encoded features.
    #[inline]
    #[must_use]
    pub fn features(&self) -> &[u16]
    where
        Features: AsRef<[u16]>,
    {
        self.features.as_ref()
    }

    /// Returns offsets into [`PersistedSparseFeatureIndex::singleton_target_ids`].
    #[inline]
    #[must_use]
    pub fn singleton_offsets(&self) -> &[u64]
    where
        SingletonOffsets: AsRef<[u64]>,
    {
        self.singleton_offsets.as_ref()
    }

    /// Returns flattened singleton target ids.
    #[inline]
    #[must_use]
    pub fn singleton_target_ids(&self) -> &[u32]
    where
        SingletonTargetIds: AsRef<[u32]>,
    {
        self.singleton_target_ids.as_ref()
    }

    /// Returns offsets into [`PersistedSparseFeatureIndex::counted_target_ids`].
    #[inline]
    #[must_use]
    pub fn counted_offsets(&self) -> &[u64]
    where
        CountedOffsets: AsRef<[u64]>,
    {
        self.counted_offsets.as_ref()
    }

    /// Returns flattened target ids for repeated local-feature counts.
    #[inline]
    #[must_use]
    pub fn counted_target_ids(&self) -> &[u32]
    where
        CountedTargetIds: AsRef<[u32]>,
    {
        self.counted_target_ids.as_ref()
    }

    /// Returns flattened repeated local-feature count values.
    #[inline]
    #[must_use]
    pub fn counted_values(&self) -> &[u16]
    where
        CountedValues: AsRef<[u16]>,
    {
        self.counted_values.as_ref()
    }
}

impl PersistedLocalFeatureFamily {
    fn from_edge_index(index: &TargetCorpusIndex) -> Self {
        Self {
            postings: PersistedSparseFeatureIndex::from_sparse_features(
                &index.indexed_edge_feature_count_index,
                EDGE_FEATURE_WIDTH,
                push_edge_feature,
            ),
            mask_indexes: vec![
                PersistedFeatureMaskIndex::from_atom_feature_masks(
                    &index.edge_feature_mask_index.left_atoms,
                ),
                PersistedFeatureMaskIndex::from_bond_feature_masks(
                    &index.edge_feature_mask_index.bonds,
                ),
                PersistedFeatureMaskIndex::from_atom_feature_masks(
                    &index.edge_feature_mask_index.right_atoms,
                ),
            ],
            atom_domain: PersistedFeatureDomain::from_atom_features(
                &index.edge_atom_feature_domain,
            ),
            bond_domain: PersistedFeatureDomain::from_bond_features(
                &index.edge_bond_feature_domain,
            ),
            mask_index: PhantomData,
        }
    }

    fn from_path3_index(index: &TargetCorpusIndex) -> Self {
        Self {
            postings: PersistedSparseFeatureIndex::from_sparse_features(
                &index.indexed_path3_feature_count_index,
                PATH3_FEATURE_WIDTH,
                push_path3_feature,
            ),
            mask_indexes: vec![
                PersistedFeatureMaskIndex::from_atom_feature_masks(
                    &index.path3_feature_mask_index.left_atoms,
                ),
                PersistedFeatureMaskIndex::from_bond_feature_masks(
                    &index.path3_feature_mask_index.left_bonds,
                ),
                PersistedFeatureMaskIndex::from_atom_feature_masks(
                    &index.path3_feature_mask_index.center_atoms,
                ),
                PersistedFeatureMaskIndex::from_bond_feature_masks(
                    &index.path3_feature_mask_index.right_bonds,
                ),
                PersistedFeatureMaskIndex::from_atom_feature_masks(
                    &index.path3_feature_mask_index.right_atoms,
                ),
            ],
            atom_domain: PersistedFeatureDomain::from_atom_features(
                &index.path3_atom_feature_domain,
            ),
            bond_domain: PersistedFeatureDomain::from_bond_features(
                &index.path3_bond_feature_domain,
            ),
            mask_index: PhantomData,
        }
    }

    fn from_path4_index(index: &TargetCorpusIndex) -> Self {
        Self {
            postings: PersistedSparseFeatureIndex::from_sparse_features(
                &index.indexed_path4_feature_count_index,
                PATH4_FEATURE_WIDTH,
                push_path4_feature,
            ),
            mask_indexes: vec![
                PersistedFeatureMaskIndex::from_atom_feature_masks(
                    &index.path4_feature_mask_index.left_atoms,
                ),
                PersistedFeatureMaskIndex::from_bond_feature_masks(
                    &index.path4_feature_mask_index.left_bonds,
                ),
                PersistedFeatureMaskIndex::from_atom_feature_masks(
                    &index.path4_feature_mask_index.left_middle_atoms,
                ),
                PersistedFeatureMaskIndex::from_bond_feature_masks(
                    &index.path4_feature_mask_index.center_bonds,
                ),
                PersistedFeatureMaskIndex::from_atom_feature_masks(
                    &index.path4_feature_mask_index.right_middle_atoms,
                ),
                PersistedFeatureMaskIndex::from_bond_feature_masks(
                    &index.path4_feature_mask_index.right_bonds,
                ),
                PersistedFeatureMaskIndex::from_atom_feature_masks(
                    &index.path4_feature_mask_index.right_atoms,
                ),
            ],
            atom_domain: PersistedFeatureDomain::from_atom_features(
                &index.path4_atom_feature_domain,
            ),
            bond_domain: PersistedFeatureDomain::from_bond_features(
                &index.path4_bond_feature_domain,
            ),
            mask_index: PhantomData,
        }
    }

    fn from_star3_index(index: &TargetCorpusIndex) -> Self {
        Self {
            postings: PersistedSparseFeatureIndex::from_sparse_features(
                &index.indexed_star3_feature_count_index,
                STAR3_FEATURE_WIDTH,
                push_star3_feature,
            ),
            mask_indexes: vec![
                PersistedFeatureMaskIndex::from_atom_feature_masks(
                    &index.star3_feature_mask_index.center_atoms,
                ),
                PersistedFeatureMaskIndex::from_bond_feature_masks(
                    &index.star3_feature_mask_index.first_bonds,
                ),
                PersistedFeatureMaskIndex::from_atom_feature_masks(
                    &index.star3_feature_mask_index.first_atoms,
                ),
                PersistedFeatureMaskIndex::from_bond_feature_masks(
                    &index.star3_feature_mask_index.second_bonds,
                ),
                PersistedFeatureMaskIndex::from_atom_feature_masks(
                    &index.star3_feature_mask_index.second_atoms,
                ),
                PersistedFeatureMaskIndex::from_bond_feature_masks(
                    &index.star3_feature_mask_index.third_bonds,
                ),
                PersistedFeatureMaskIndex::from_atom_feature_masks(
                    &index.star3_feature_mask_index.third_atoms,
                ),
            ],
            atom_domain: PersistedFeatureDomain::from_atom_features(
                &index.star3_atom_feature_domain,
            ),
            bond_domain: PersistedFeatureDomain::from_bond_features(
                &index.star3_bond_feature_domain,
            ),
            mask_index: PhantomData,
        }
    }
}

impl<Postings, MaskIndex, MaskIndexes, AtomDomain, BondDomain>
    PersistedLocalFeatureFamily<Postings, MaskIndex, MaskIndexes, AtomDomain, BondDomain>
{
    /// Returns persisted sparse postings for this local-feature family.
    #[inline]
    #[must_use]
    pub const fn postings(&self) -> &Postings {
        &self.postings
    }

    /// Returns persisted feature-mask indexes in family-specific role order.
    #[inline]
    #[must_use]
    pub fn mask_indexes(&self) -> &[MaskIndex]
    where
        MaskIndexes: AsRef<[MaskIndex]>,
    {
        self.mask_indexes.as_ref()
    }

    /// Returns the atom-feature domain for expansion masks.
    #[inline]
    #[must_use]
    pub const fn atom_domain(&self) -> &AtomDomain {
        &self.atom_domain
    }

    /// Returns the bond-feature domain for expansion masks.
    #[inline]
    #[must_use]
    pub const fn bond_domain(&self) -> &BondDomain {
        &self.bond_domain
    }
}

fn persist_usize(value: usize) -> u64 {
    u64::try_from(value).expect("usize value exceeds persisted index capacity")
}

fn encode_option_u16(value: Option<u16>) -> u16 {
    value.map_or(0, |value| {
        value
            .checked_add(1)
            .expect("optional u16 feature value exceeds persisted index capacity")
    })
}

const fn encode_option_bool(value: Option<bool>) -> u16 {
    match value {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    }
}

const fn encode_bool(value: bool) -> u16 {
    if value {
        1
    } else {
        0
    }
}

const fn encode_bond_kind(kind: Option<RequiredBondKind>) -> u16 {
    match kind {
        None => 0,
        Some(RequiredBondKind::Single) => 1,
        Some(RequiredBondKind::Double) => 2,
        Some(RequiredBondKind::Triple) => 3,
        Some(RequiredBondKind::Aromatic) => 4,
    }
}

fn push_atom_feature(out: &mut Vec<u16>, feature: AtomFeature) {
    out.push(
        feature
            .element
            .map_or(0, |element| u16::from(u8::from(element))),
    );
    out.push(encode_option_bool(feature.aromatic));
    out.push(encode_bool(feature.requires_ring));
    out.push(encode_option_u16(feature.degree));
    out.push(encode_option_u16(feature.total_hydrogens));
}

fn push_bond_feature(out: &mut Vec<u16>, feature: EdgeBondFeature) {
    out.push(encode_bond_kind(feature.kind));
    out.push(encode_bool(feature.requires_ring));
}

fn push_edge_feature(out: &mut Vec<u16>, feature: EdgeFeature) {
    push_atom_feature(out, feature.left);
    push_bond_feature(out, feature.bond);
    push_atom_feature(out, feature.right);
}

fn push_path3_feature(out: &mut Vec<u16>, feature: Path3Feature) {
    push_atom_feature(out, feature.left);
    push_bond_feature(out, feature.left_bond);
    push_atom_feature(out, feature.center);
    push_bond_feature(out, feature.right_bond);
    push_atom_feature(out, feature.right);
}

fn push_path4_feature(out: &mut Vec<u16>, feature: Path4Feature) {
    push_atom_feature(out, feature.left);
    push_bond_feature(out, feature.left_bond);
    push_atom_feature(out, feature.left_middle);
    push_bond_feature(out, feature.center_bond);
    push_atom_feature(out, feature.right_middle);
    push_bond_feature(out, feature.right_bond);
    push_atom_feature(out, feature.right);
}

fn push_star3_feature(out: &mut Vec<u16>, feature: Star3Feature) {
    push_atom_feature(out, feature.center);
    for arm in feature.arms {
        push_bond_feature(out, arm.bond);
        push_atom_feature(out, arm.atom);
    }
}

fn encoded_atom_feature(feature: AtomFeature) -> [u16; ATOM_FEATURE_WIDTH_USIZE] {
    [
        feature
            .element
            .map_or(0, |element| u16::from(u8::from(element))),
        encode_option_bool(feature.aromatic),
        encode_bool(feature.requires_ring),
        encode_option_u16(feature.degree),
        encode_option_u16(feature.total_hydrogens),
    ]
}

const fn encoded_bond_feature(feature: EdgeBondFeature) -> [u16; BOND_FEATURE_WIDTH_USIZE] {
    [
        encode_bond_kind(feature.kind),
        encode_bool(feature.requires_ring),
    ]
}

fn decode_atom_feature(encoded: &[u16]) -> Option<AtomFeature> {
    let &[element, aromatic, requires_ring, degree, total_hydrogens] = encoded else {
        return None;
    };
    Some(AtomFeature {
        element: decode_optional_element(element).ok()?,
        aromatic: decode_option_bool(aromatic).ok()?,
        requires_ring: decode_bool(requires_ring)?,
        degree: decode_option_u16(degree),
        total_hydrogens: decode_option_u16(total_hydrogens),
    })
}

fn decode_bond_feature(encoded: &[u16]) -> Option<EdgeBondFeature> {
    let &[kind, requires_ring] = encoded else {
        return None;
    };
    Some(EdgeBondFeature {
        kind: decode_bond_kind(kind).ok()?,
        requires_ring: decode_bool(requires_ring)?,
    })
}

fn decode_optional_element(value: u16) -> Result<Option<elements_rs::Element>, ()> {
    if value == 0 {
        return Ok(None);
    }
    elements_rs::Element::try_from(u8::try_from(value).map_err(|_| ())?)
        .map(Some)
        .map_err(|_| ())
}

const fn decode_option_bool(value: u16) -> Result<Option<bool>, ()> {
    match value {
        0 => Ok(None),
        1 => Ok(Some(false)),
        2 => Ok(Some(true)),
        _ => Err(()),
    }
}

const fn decode_bool(value: u16) -> Option<bool> {
    match value {
        0 => Some(false),
        1 => Some(true),
        _ => None,
    }
}

const fn decode_option_u16(value: u16) -> Option<u16> {
    if value == 0 {
        None
    } else {
        Some(value - 1)
    }
}

const fn decode_bond_kind(value: u16) -> Result<Option<RequiredBondKind>, ()> {
    match value {
        0 => Ok(None),
        1 => Ok(Some(RequiredBondKind::Single)),
        2 => Ok(Some(RequiredBondKind::Double)),
        3 => Ok(Some(RequiredBondKind::Triple)),
        4 => Ok(Some(RequiredBondKind::Aromatic)),
        _ => Err(()),
    }
}

impl PersistedCountBitsetIndex {
    pub(super) fn from_count_bitset_index(index: &CountBitsetIndex) -> Self {
        let mut bitset_offsets = Vec::with_capacity(index.bitsets().len() + 1);
        let mut bitset_words = Vec::new();
        bitset_offsets.push(0);
        for bitset in index.bitsets() {
            bitset_words.extend_from_slice(bitset);
            bitset_offsets.push(
                u64::try_from(bitset_words.len())
                    .expect("count bitset word offset exceeds persisted index capacity"),
            );
        }

        Self {
            thresholds: index.thresholds().to_vec(),
            populations: index
                .populations()
                .iter()
                .map(|&population| {
                    u64::try_from(population)
                        .expect("count bitset population exceeds persisted index capacity")
                })
                .collect(),
            bitset_offsets,
            bitset_words,
        }
    }

    /// Serializes this flat index into an epserde writer.
    ///
    /// # Errors
    ///
    /// Returns an epserde serialization error if the writer fails.
    ///
    /// # Safety
    ///
    /// This wraps epserde serialization. The serialized bytes should be treated
    /// as trusted data for later epserde deserialization.
    pub unsafe fn serialize_into_unchecked(
        &self,
        writer: &mut impl ser::WriteNoStd,
    ) -> ser::Result<usize> {
        unsafe { ser::Serialize::serialize(self, writer) }
    }

    /// Deserializes a full owned flat index from an epserde reader.
    ///
    /// # Errors
    ///
    /// Returns an epserde deserialization error if the input is malformed.
    ///
    /// # Safety
    ///
    /// The input must be trusted bytes produced for this type by a compatible
    /// version of this crate and epserde.
    pub unsafe fn deserialize_full_unchecked(
        reader: &mut impl deser::ReadNoStd,
    ) -> deser::Result<Self> {
        unsafe { deser::Deserialize::deserialize_full(reader) }
    }

    /// Epsilon-deserializes a flat index from trusted bytes.
    ///
    /// # Errors
    ///
    /// Returns an epserde deserialization error if the input is malformed.
    ///
    /// # Safety
    ///
    /// The input must be trusted bytes produced for this type by a compatible
    /// version of this crate and epserde. This may borrow directly from
    /// `bytes`, so the returned value must not outlive that buffer.
    pub unsafe fn deserialize_eps_unchecked(
        bytes: &[u8],
    ) -> deser::Result<<Self as deser::DeserInner>::DeserType<'_>> {
        unsafe { <Self as deser::Deserialize>::deserialize_eps(bytes) }
    }
}

impl<Thresholds, Populations, BitsetOffsets, BitsetWords>
    PersistedCountBitsetIndex<Thresholds, Populations, BitsetOffsets, BitsetWords>
where
    Thresholds: AsRef<[u32]>,
    Populations: AsRef<[u64]>,
    BitsetOffsets: AsRef<[u64]>,
    BitsetWords: AsRef<[u64]>,
{
    /// Returns the count thresholds available in this index.
    #[inline]
    #[must_use]
    pub fn thresholds(&self) -> &[u32] {
        self.thresholds.as_ref()
    }

    /// Returns one population per count threshold.
    #[inline]
    #[must_use]
    pub fn populations(&self) -> &[u64] {
        self.populations.as_ref()
    }

    /// Returns offsets into [`PersistedCountBitsetIndex::bitset_words`].
    #[inline]
    #[must_use]
    pub fn bitset_offsets(&self) -> &[u64] {
        self.bitset_offsets.as_ref()
    }

    /// Returns the flattened bitset words for all count thresholds.
    #[inline]
    #[must_use]
    pub fn bitset_words(&self) -> &[u64] {
        self.bitset_words.as_ref()
    }

    /// Returns the number of count thresholds in this persisted index.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.thresholds().len()
    }

    /// Returns whether this persisted index has no count thresholds.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.thresholds().is_empty()
    }

    /// Returns the persisted filter for targets whose count is at least `required`.
    #[must_use]
    pub fn filter_for_at_least(&self, required: usize) -> Option<PersistedRequiredCountFilter<'_>> {
        let required = u32::try_from(required).ok()?;
        let index = self
            .thresholds()
            .partition_point(|&threshold| threshold < required);
        let population = usize::try_from(*self.populations().get(index)?).ok()?;
        let range = self.bitset_range(index)?;
        Some(PersistedRequiredCountFilter {
            population,
            source: self.bitset_words().get(range)?,
        })
    }

    fn bitset_range(&self, index: usize) -> Option<Range<usize>> {
        let offsets = self.bitset_offsets();
        let start = usize::try_from(*offsets.get(index)?).ok()?;
        let end = usize::try_from(*offsets.get(index + 1)?).ok()?;
        (start <= end && end <= self.bitset_words().len()).then_some(start..end)
    }
}

trait PersistedCountBitsetIndexAccess {
    fn persisted_filter_for_at_least(
        &self,
        required: usize,
    ) -> Option<PersistedRequiredCountFilter<'_>>;
}

impl<Thresholds, Populations, BitsetOffsets, BitsetWords> PersistedCountBitsetIndexAccess
    for PersistedCountBitsetIndex<Thresholds, Populations, BitsetOffsets, BitsetWords>
where
    Thresholds: AsRef<[u32]>,
    Populations: AsRef<[u64]>,
    BitsetOffsets: AsRef<[u64]>,
    BitsetWords: AsRef<[u64]>,
{
    fn persisted_filter_for_at_least(
        &self,
        required: usize,
    ) -> Option<PersistedRequiredCountFilter<'_>> {
        self.filter_for_at_least(required)
    }
}

trait PersistedFeatureDomainAccess {
    fn persisted_feature_width(&self) -> u16;
    fn persisted_features(&self) -> &[u16];
}

impl<Features> PersistedFeatureDomainAccess for PersistedFeatureDomain<Features>
where
    Features: AsRef<[u16]>,
{
    fn persisted_feature_width(&self) -> u16 {
        self.feature_width
    }

    fn persisted_features(&self) -> &[u16] {
        self.features.as_ref()
    }
}

trait PersistedFeatureMaskIndexAccess {
    fn persisted_mask_feature_width(&self) -> u16;
    fn persisted_mask_features(&self) -> &[u16];
    fn persisted_mask_offsets(&self) -> &[u64];
    fn persisted_mask_words(&self) -> &[u64];
}

impl<Features, MaskOffsets, MaskWords> PersistedFeatureMaskIndexAccess
    for PersistedFeatureMaskIndex<Features, MaskOffsets, MaskWords>
where
    Features: AsRef<[u16]>,
    MaskOffsets: AsRef<[u64]>,
    MaskWords: AsRef<[u64]>,
{
    fn persisted_mask_feature_width(&self) -> u16 {
        self.feature_width
    }

    fn persisted_mask_features(&self) -> &[u16] {
        self.features.as_ref()
    }

    fn persisted_mask_offsets(&self) -> &[u64] {
        self.mask_offsets.as_ref()
    }

    fn persisted_mask_words(&self) -> &[u64] {
        self.mask_words.as_ref()
    }
}

trait PersistedSparseFeatureIndexAccess {
    fn persisted_sparse_singleton_offsets(&self) -> &[u64];
    fn persisted_sparse_singleton_target_ids(&self) -> &[u32];
    fn persisted_sparse_counted_offsets(&self) -> &[u64];
    fn persisted_sparse_counted_target_ids(&self) -> &[u32];
    fn persisted_sparse_counted_values(&self) -> &[u16];
}

impl<
        Features,
        SingletonOffsets,
        SingletonTargetIds,
        CountedOffsets,
        CountedTargetIds,
        CountedValues,
    > PersistedSparseFeatureIndexAccess
    for PersistedSparseFeatureIndex<
        Features,
        SingletonOffsets,
        SingletonTargetIds,
        CountedOffsets,
        CountedTargetIds,
        CountedValues,
    >
where
    SingletonOffsets: AsRef<[u64]>,
    SingletonTargetIds: AsRef<[u32]>,
    CountedOffsets: AsRef<[u64]>,
    CountedTargetIds: AsRef<[u32]>,
    CountedValues: AsRef<[u16]>,
{
    fn persisted_sparse_singleton_offsets(&self) -> &[u64] {
        self.singleton_offsets.as_ref()
    }

    fn persisted_sparse_singleton_target_ids(&self) -> &[u32] {
        self.singleton_target_ids.as_ref()
    }

    fn persisted_sparse_counted_offsets(&self) -> &[u64] {
        self.counted_offsets.as_ref()
    }

    fn persisted_sparse_counted_target_ids(&self) -> &[u32] {
        self.counted_target_ids.as_ref()
    }

    fn persisted_sparse_counted_values(&self) -> &[u16] {
        self.counted_values.as_ref()
    }
}

trait PersistedKeyedCountIndexesAccess {
    type CountIndex: PersistedCountBitsetIndexAccess;

    fn push_filter_for_key<'idx>(
        &'idx self,
        filters: &mut Vec<RequiredCountFilter<'idx>>,
        key: u16,
        required: usize,
    ) -> bool;
}

trait PersistedCountIndexSlice {
    type Item: PersistedCountBitsetIndexAccess;

    fn as_count_index_slice(&self) -> &[Self::Item];
}

impl<T> PersistedCountIndexSlice for Vec<T>
where
    T: PersistedCountBitsetIndexAccess,
{
    type Item = T;

    fn as_count_index_slice(&self) -> &[Self::Item] {
        self.as_slice()
    }
}

impl<T> PersistedCountIndexSlice for &[T]
where
    T: PersistedCountBitsetIndexAccess,
{
    type Item = T;

    fn as_count_index_slice(&self) -> &[Self::Item] {
        self
    }
}

impl<Keys, CountIndexMarker, Indexes> PersistedKeyedCountIndexesAccess
    for PersistedKeyedCountIndexes<Keys, CountIndexMarker, Indexes>
where
    Keys: AsRef<[u16]>,
    Indexes: PersistedCountIndexSlice,
{
    type CountIndex = Indexes::Item;

    fn push_filter_for_key<'idx>(
        &'idx self,
        filters: &mut Vec<RequiredCountFilter<'idx>>,
        key: u16,
        required: usize,
    ) -> bool {
        let keys = self.keys.as_ref();
        let Ok(index) = keys.binary_search(&key) else {
            return false;
        };
        let Some(count_index) = self.indexes.as_count_index_slice().get(index) else {
            return false;
        };
        push_persisted_required_filter(filters, count_index, required)
    }
}

trait PersistedScalarCountIndexesAccess {
    fn collect_basic_count_filters<'idx>(
        &'idx self,
        query: &QueryScreen,
        filters: &mut Vec<RequiredCountFilter<'idx>>,
    ) -> bool;
}

impl<CountIndex> PersistedScalarCountIndexesAccess for PersistedScalarCountIndexes<CountIndex>
where
    CountIndex: PersistedCountBitsetIndexAccess,
{
    fn collect_basic_count_filters<'idx>(
        &'idx self,
        query: &QueryScreen,
        filters: &mut Vec<RequiredCountFilter<'idx>>,
    ) -> bool {
        let required_counts = [
            (&self.atom, query.min_atom_count),
            (&self.component, query.min_target_component_count),
            (&self.aromatic_atom, query.min_aromatic_atom_count),
            (&self.ring_atom, query.min_ring_atom_count),
            (&self.single_bond, query.required_bond_counts.single),
            (&self.double_bond, query.required_bond_counts.double),
            (&self.triple_bond, query.required_bond_counts.triple),
            (&self.aromatic_bond, query.required_bond_counts.aromatic),
            (&self.ring_bond, query.required_bond_counts.ring),
        ];

        required_counts
            .into_iter()
            .all(|(index, required)| push_persisted_required_filter(filters, index, required))
    }
}

trait PersistedAtomPropertyCountIndexesAccess {
    fn collect_atom_property_count_filters<'idx>(
        &'idx self,
        query: &QueryScreen,
        filters: &mut Vec<RequiredCountFilter<'idx>>,
    ) -> bool;
}

impl<ElementCounts, DegreeCounts, TotalHydrogenCounts> PersistedAtomPropertyCountIndexesAccess
    for PersistedAtomPropertyCountIndexes<ElementCounts, DegreeCounts, TotalHydrogenCounts>
where
    ElementCounts: PersistedKeyedCountIndexesAccess,
    DegreeCounts: PersistedKeyedCountIndexesAccess,
    TotalHydrogenCounts: PersistedKeyedCountIndexesAccess,
{
    fn collect_atom_property_count_filters<'idx>(
        &'idx self,
        query: &QueryScreen,
        filters: &mut Vec<RequiredCountFilter<'idx>>,
    ) -> bool {
        for (element, required) in &query.required_element_counts {
            if !self
                .element
                .push_filter_for_key(filters, u16::from(u8::from(*element)), *required)
            {
                return false;
            }
        }
        collect_persisted_keyed_count_filters(filters, &self.degree, &query.required_degree_counts)
            && collect_persisted_keyed_count_filters(
                filters,
                &self.total_hydrogen,
                &query.required_total_hydrogen_counts,
            )
    }
}

trait PersistedLocalFeatureFamilyAccess {
    type Postings: PersistedSparseFeatureIndexAccess;
    type MaskIndex: PersistedFeatureMaskIndexAccess;
    type AtomDomain: PersistedFeatureDomainAccess;
    type BondDomain: PersistedFeatureDomainAccess;

    fn postings(&self) -> &Self::Postings;
    fn mask_indexes(&self) -> &[Self::MaskIndex];
    fn atom_domain(&self) -> &Self::AtomDomain;
    fn bond_domain(&self) -> &Self::BondDomain;
}

trait PersistedMaskIndexSlice {
    type Item: PersistedFeatureMaskIndexAccess;

    fn as_mask_index_slice(&self) -> &[Self::Item];
}

impl<T> PersistedMaskIndexSlice for Vec<T>
where
    T: PersistedFeatureMaskIndexAccess,
{
    type Item = T;

    fn as_mask_index_slice(&self) -> &[Self::Item] {
        self.as_slice()
    }
}

impl<T> PersistedMaskIndexSlice for &[T]
where
    T: PersistedFeatureMaskIndexAccess,
{
    type Item = T;

    fn as_mask_index_slice(&self) -> &[Self::Item] {
        self
    }
}

impl<Postings, MaskIndexMarker, MaskIndexes, AtomDomain, BondDomain>
    PersistedLocalFeatureFamilyAccess
    for PersistedLocalFeatureFamily<Postings, MaskIndexMarker, MaskIndexes, AtomDomain, BondDomain>
where
    Postings: PersistedSparseFeatureIndexAccess,
    MaskIndexes: PersistedMaskIndexSlice,
    AtomDomain: PersistedFeatureDomainAccess,
    BondDomain: PersistedFeatureDomainAccess,
{
    type Postings = Postings;
    type MaskIndex = MaskIndexes::Item;
    type AtomDomain = AtomDomain;
    type BondDomain = BondDomain;

    fn postings(&self) -> &Self::Postings {
        &self.postings
    }

    fn mask_indexes(&self) -> &[Self::MaskIndex] {
        self.mask_indexes.as_mask_index_slice()
    }

    fn atom_domain(&self) -> &Self::AtomDomain {
        &self.atom_domain
    }

    fn bond_domain(&self) -> &Self::BondDomain {
        &self.bond_domain
    }
}

#[allow(private_bounds)]
impl<ScalarCounts, AtomPropertyCounts, Edge, Path3, Path4, Star3>
    PersistedTargetCorpusIndexShard<ScalarCounts, AtomPropertyCounts, Edge, Path3, Path4, Star3>
where
    ScalarCounts: PersistedScalarCountIndexesAccess,
    AtomPropertyCounts: PersistedAtomPropertyCountIndexesAccess,
    Edge: PersistedLocalFeatureFamilyAccess,
    Path3: PersistedLocalFeatureFamilyAccess,
    Path4: PersistedLocalFeatureFamilyAccess,
    Star3: PersistedLocalFeatureFamilyAccess,
{
    /// Collects global candidate target ids that pass this persisted shard.
    pub fn candidate_ids_into(&self, query: &QueryScreen, out: &mut Vec<usize>) {
        let mut scratch = TargetCorpusScratch::new();
        self.candidate_ids_with_scratch_into(query, &mut scratch, out);
    }

    /// Collects global candidate target ids using reusable scratch buffers.
    pub fn candidate_ids_with_scratch_into<'idx>(
        &'idx self,
        query: &QueryScreen,
        scratch: &mut TargetCorpusScratch<'idx>,
        out: &mut Vec<usize>,
    ) {
        out.clear();
        let Some(target_count) = self.target_count_usize() else {
            return;
        };
        let Some(base_target_id) = self.base_target_id_usize() else {
            return;
        };
        let Some(state) = self.populate_candidate_mask_with_scratch(query, scratch) else {
            return;
        };
        out.reserve(state.population);
        if !state.has_active_source {
            for local_target_id in 0..target_count {
                if let Some(target_id) = base_target_id.checked_add(local_target_id) {
                    out.push(target_id);
                }
            }
            return;
        }
        for_each_set_bit(&scratch.candidate_mask, target_count, |local_target_id| {
            if let Some(target_id) = base_target_id.checked_add(local_target_id) {
                out.push(target_id);
            }
        });
    }

    /// Counts candidate targets that pass this persisted shard.
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

    /// Returns global candidate target ids that pass this persisted shard.
    #[must_use]
    pub fn candidate_ids(&self, query: &QueryScreen) -> Vec<usize> {
        let mut out = Vec::new();
        self.candidate_ids_into(query, &mut out);
        out
    }

    /// Builds a reusable candidate set for one query.
    #[must_use]
    pub fn candidate_set(&self, query: &QueryScreen) -> TargetCandidateSet {
        let mut scratch = TargetCorpusScratch::new();
        self.candidate_set_with_scratch(query, &mut scratch)
    }

    /// Builds a reusable candidate set using reusable scratch buffers.
    pub fn candidate_set_with_scratch<'idx>(
        &'idx self,
        query: &QueryScreen,
        scratch: &mut TargetCorpusScratch<'idx>,
    ) -> TargetCandidateSet {
        let mut target_ids = Vec::new();
        self.candidate_ids_with_scratch_into(query, scratch, &mut target_ids);
        TargetCandidateSet::new(target_ids)
    }

    /// Streams global candidate target ids using reusable scratch buffers.
    pub fn for_each_candidate_id_with_scratch<'idx, F>(
        &'idx self,
        query: &QueryScreen,
        scratch: &mut TargetCorpusScratch<'idx>,
        mut f: F,
    ) where
        F: FnMut(usize),
    {
        let Some(target_count) = self.target_count_usize() else {
            return;
        };
        let Some(base_target_id) = self.base_target_id_usize() else {
            return;
        };
        let Some(state) = self.populate_candidate_mask_with_scratch(query, scratch) else {
            return;
        };
        if !state.has_active_source {
            for local_target_id in 0..target_count {
                if let Some(target_id) = base_target_id.checked_add(local_target_id) {
                    f(target_id);
                }
            }
            return;
        }
        for_each_set_bit(&scratch.candidate_mask, target_count, |local_target_id| {
            if let Some(target_id) = base_target_id.checked_add(local_target_id) {
                f(target_id);
            }
        });
    }

    fn target_count_usize(&self) -> Option<usize> {
        usize::try_from(self.target_count).ok()
    }

    fn base_target_id_usize(&self) -> Option<usize> {
        usize::try_from(self.base_target_id).ok()
    }

    fn populate_candidate_mask_with_scratch<'idx>(
        &'idx self,
        query: &QueryScreen,
        scratch: &mut TargetCorpusScratch<'idx>,
    ) -> Option<CandidateMaskState> {
        let target_count = self.target_count_usize()?;
        if target_count == 0 {
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

        scratch.ensure_word_count(bitset_word_count(target_count));
        let mut has_active_source = false;
        scratch
            .filters
            .sort_unstable_by_key(|filter| filter.population);
        let mut candidate_population = target_count;
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

    fn collect_required_count_filters<'idx>(
        &'idx self,
        query: &QueryScreen,
        filters: &mut Vec<RequiredCountFilter<'idx>>,
    ) -> bool {
        self.scalar_counts
            .collect_basic_count_filters(query, filters)
            && self
                .atom_property_counts
                .collect_atom_property_count_filters(query, filters)
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
            let population = self.populate_feature_candidate_mask(
                filter,
                active_candidate_mask.as_deref(),
                scratch,
            );
            if let Some(mask) = active_candidate_mask.take() {
                scratch.active_candidate_mask = mask;
            }
            if population == 0 {
                return false;
            }

            if used_active_candidate_mask || !*has_active_source {
                move_persisted_candidate_mask_to_active(filter, scratch);
                *has_active_source = true;
                *candidate_population = population;
            } else {
                let population = intersect_persisted_active_candidate_mask(filter, scratch);
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
        let target_count = self.target_count_usize()?;
        if !should_filter_sparse_counts(has_active_source, candidate_population, target_count) {
            return None;
        }

        let mut mask = core::mem::take(&mut scratch.active_candidate_mask);
        mask.clear();
        mask.extend_from_slice(&scratch.candidate_mask);
        Some(mask)
    }

    fn populate_feature_candidate_mask(
        &self,
        filter: QueryFeatureFilter,
        active_candidate_mask: Option<&[u64]>,
        scratch: &mut TargetCorpusScratch<'_>,
    ) -> usize {
        let Some(target_count) = self.target_count_usize() else {
            return 0;
        };
        match filter {
            QueryFeatureFilter::Edge {
                feature,
                required_count,
            } => populate_persisted_edge_candidate_mask(
                &self.edge,
                target_count,
                feature,
                required_count,
                active_candidate_mask,
                scratch,
            ),
            QueryFeatureFilter::Path3 {
                feature,
                required_count,
            } => populate_persisted_path3_candidate_mask(
                &self.path3,
                target_count,
                feature,
                required_count,
                active_candidate_mask,
                scratch,
            ),
            QueryFeatureFilter::Path4 {
                feature,
                required_count,
            } => populate_persisted_path4_candidate_mask(
                &self.path4,
                target_count,
                feature,
                required_count,
                active_candidate_mask,
                scratch,
            ),
            QueryFeatureFilter::Star3 {
                feature,
                required_count,
            } => populate_persisted_star3_candidate_mask(
                &self.star3,
                target_count,
                feature,
                required_count,
                active_candidate_mask,
                scratch,
            ),
        }
    }
}

fn push_persisted_required_filter<'idx>(
    filters: &mut Vec<RequiredCountFilter<'idx>>,
    index: &'idx impl PersistedCountBitsetIndexAccess,
    required: usize,
) -> bool {
    if required == 0 {
        return true;
    }
    let Some(filter) = index.persisted_filter_for_at_least(required) else {
        return false;
    };
    filters.push(RequiredCountFilter {
        population: filter.population,
        source: filter.source,
    });
    true
}

fn collect_persisted_keyed_count_filters<'idx, T>(
    filters: &mut Vec<RequiredCountFilter<'idx>>,
    index: &'idx impl PersistedKeyedCountIndexesAccess,
    required_counts: &alloc::collections::BTreeMap<T, usize>,
) -> bool
where
    T: Copy + Into<u16>,
{
    required_counts
        .iter()
        .all(|(&key, &required)| index.push_filter_for_key(filters, key.into(), required))
}

const fn move_persisted_candidate_mask_to_active(
    filter: QueryFeatureFilter,
    scratch: &mut TargetCorpusScratch<'_>,
) {
    match filter {
        QueryFeatureFilter::Edge { .. } => {
            core::mem::swap(
                &mut scratch.candidate_mask,
                &mut scratch.edge_candidate_mask,
            );
        }
        QueryFeatureFilter::Path3 { .. } => {
            core::mem::swap(
                &mut scratch.candidate_mask,
                &mut scratch.path3_candidate_mask,
            );
        }
        QueryFeatureFilter::Path4 { .. } => {
            core::mem::swap(
                &mut scratch.candidate_mask,
                &mut scratch.path4_candidate_mask,
            );
        }
        QueryFeatureFilter::Star3 { .. } => {
            core::mem::swap(
                &mut scratch.candidate_mask,
                &mut scratch.star3_candidate_mask,
            );
        }
    }
}

fn intersect_persisted_active_candidate_mask(
    filter: QueryFeatureFilter,
    scratch: &mut TargetCorpusScratch<'_>,
) -> usize {
    let source = match filter {
        QueryFeatureFilter::Edge { .. } => &scratch.edge_candidate_mask,
        QueryFeatureFilter::Path3 { .. } => &scratch.path3_candidate_mask,
        QueryFeatureFilter::Path4 { .. } => &scratch.path4_candidate_mask,
        QueryFeatureFilter::Star3 { .. } => &scratch.star3_candidate_mask,
    };

    let mut population = 0usize;
    for (candidate_word, &source_word) in scratch.candidate_mask.iter_mut().zip(source) {
        *candidate_word &= source_word;
        population += candidate_word.count_ones() as usize;
    }
    population
}

fn populate_persisted_edge_candidate_mask<Family>(
    family: &Family,
    target_count: usize,
    query_feature: EdgeFeature,
    required_count: u16,
    active_candidate_mask: Option<&[u64]>,
    scratch: &mut TargetCorpusScratch<'_>,
) -> usize
where
    Family: PersistedLocalFeatureFamilyAccess,
{
    if required_count == 0 {
        return target_count;
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
        target_count,
        &mut scratch.edge_count_by_target,
        &mut scratch.edge_touched_targets,
    );

    collect_persisted_edge_domain_matches(family, query_feature, scratch);
    accumulate_persisted_edge_matches(family, query_feature, scratch, active_candidate_mask);

    finalize_cached_sparse_candidate_mask(
        target_count,
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

fn collect_persisted_edge_domain_matches<Family>(
    family: &Family,
    query_feature: EdgeFeature,
    scratch: &mut TargetCorpusScratch<'_>,
) where
    Family: PersistedLocalFeatureFamilyAccess,
{
    collect_matching_persisted_atom_features(
        family.atom_domain(),
        query_feature.left,
        &mut scratch.edge_left_atoms,
    );
    collect_matching_persisted_atom_features(
        family.atom_domain(),
        query_feature.right,
        &mut scratch.edge_right_atoms,
    );
    collect_matching_persisted_bond_features(
        family.bond_domain(),
        query_feature.bond,
        &mut scratch.edge_bonds,
    );
}

fn accumulate_persisted_edge_matches<Family>(
    family: &Family,
    query_feature: EdgeFeature,
    scratch: &mut TargetCorpusScratch<'_>,
    active_candidate_mask: Option<&[u64]>,
) where
    Family: PersistedLocalFeatureFamilyAccess,
{
    let feature_count = persisted_feature_count(family.postings());
    let masks = family.mask_indexes();
    let Some(left_atoms) = masks.first() else {
        return;
    };
    let Some(bonds) = masks.get(1) else {
        return;
    };
    let Some(right_atoms) = masks.get(2) else {
        return;
    };

    let forward_has_candidates = build_persisted_edge_orientation_feature_mask(
        [left_atoms, bonds, right_atoms],
        &PersistedEdgeOrientationFeatureMask {
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
    let reverse_has_candidates = build_persisted_edge_orientation_feature_mask(
        [left_atoms, bonds, right_atoms],
        &PersistedEdgeOrientationFeatureMask {
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

    accumulate_persisted_feature_id_mask_counts(
        family.postings(),
        &scratch.edge_feature_candidate_mask,
        &mut scratch.edge_count_by_target,
        &mut scratch.edge_touched_targets,
        active_candidate_mask,
    );
}

fn populate_persisted_path3_candidate_mask<Family>(
    family: &Family,
    target_count: usize,
    query_feature: Path3Feature,
    required_count: u16,
    active_candidate_mask: Option<&[u64]>,
    scratch: &mut TargetCorpusScratch<'_>,
) -> usize
where
    Family: PersistedLocalFeatureFamilyAccess,
{
    if required_count == 0 {
        return target_count;
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
        target_count,
        &mut scratch.path3_count_by_target,
        &mut scratch.path3_touched_targets,
    );

    collect_persisted_path3_domain_matches(family, query_feature, scratch);
    accumulate_persisted_path3_matches(family, query_feature, scratch, active_candidate_mask);

    finalize_cached_sparse_candidate_mask(
        target_count,
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

fn collect_persisted_path3_domain_matches<Family>(
    family: &Family,
    query_feature: Path3Feature,
    scratch: &mut TargetCorpusScratch<'_>,
) where
    Family: PersistedLocalFeatureFamilyAccess,
{
    collect_matching_persisted_atom_features(
        family.atom_domain(),
        query_feature.left,
        &mut scratch.path3_left_atoms,
    );
    collect_matching_persisted_atom_features(
        family.atom_domain(),
        query_feature.center,
        &mut scratch.path3_center_atoms,
    );
    collect_matching_persisted_atom_features(
        family.atom_domain(),
        query_feature.right,
        &mut scratch.path3_right_atoms,
    );
    collect_matching_persisted_bond_features(
        family.bond_domain(),
        query_feature.left_bond,
        &mut scratch.path3_left_bonds,
    );
    collect_matching_persisted_bond_features(
        family.bond_domain(),
        query_feature.right_bond,
        &mut scratch.path3_right_bonds,
    );
}

fn accumulate_persisted_path3_matches<Family>(
    family: &Family,
    query_feature: Path3Feature,
    scratch: &mut TargetCorpusScratch<'_>,
    active_candidate_mask: Option<&[u64]>,
) where
    Family: PersistedLocalFeatureFamilyAccess,
{
    let feature_count = persisted_feature_count(family.postings());
    let masks = family.mask_indexes();
    let Some(left_atoms) = masks.first() else {
        return;
    };
    let Some(left_bonds) = masks.get(1) else {
        return;
    };
    let Some(center_atoms) = masks.get(2) else {
        return;
    };
    let Some(right_bonds) = masks.get(3) else {
        return;
    };
    let Some(right_atoms) = masks.get(4) else {
        return;
    };

    let forward_has_candidates = build_persisted_path3_orientation_feature_mask(
        [
            left_atoms,
            left_bonds,
            center_atoms,
            right_bonds,
            right_atoms,
        ],
        &PersistedPath3OrientationFeatureMask {
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
    let reverse_has_candidates = build_persisted_path3_orientation_feature_mask(
        [
            left_atoms,
            left_bonds,
            center_atoms,
            right_bonds,
            right_atoms,
        ],
        &PersistedPath3OrientationFeatureMask {
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

    accumulate_persisted_feature_id_mask_counts(
        family.postings(),
        &scratch.path3_feature_candidate_mask,
        &mut scratch.path3_count_by_target,
        &mut scratch.path3_touched_targets,
        active_candidate_mask,
    );
}

fn populate_persisted_path4_candidate_mask<Family>(
    family: &Family,
    target_count: usize,
    query_feature: Path4Feature,
    required_count: u16,
    active_candidate_mask: Option<&[u64]>,
    scratch: &mut TargetCorpusScratch<'_>,
) -> usize
where
    Family: PersistedLocalFeatureFamilyAccess,
{
    if required_count == 0 {
        return target_count;
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
        target_count,
        &mut scratch.path4_count_by_target,
        &mut scratch.path4_touched_targets,
    );

    collect_persisted_path4_domain_matches(family, query_feature, scratch);
    accumulate_persisted_path4_matches(family, query_feature, scratch, active_candidate_mask);

    finalize_cached_sparse_candidate_mask(
        target_count,
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

fn collect_persisted_path4_domain_matches<Family>(
    family: &Family,
    query_feature: Path4Feature,
    scratch: &mut TargetCorpusScratch<'_>,
) where
    Family: PersistedLocalFeatureFamilyAccess,
{
    collect_matching_persisted_atom_features(
        family.atom_domain(),
        query_feature.left,
        &mut scratch.path4_left_atoms,
    );
    collect_matching_persisted_atom_features(
        family.atom_domain(),
        query_feature.left_middle,
        &mut scratch.path4_left_middle_atoms,
    );
    collect_matching_persisted_atom_features(
        family.atom_domain(),
        query_feature.right_middle,
        &mut scratch.path4_right_middle_atoms,
    );
    collect_matching_persisted_atom_features(
        family.atom_domain(),
        query_feature.right,
        &mut scratch.path4_right_atoms,
    );
    collect_matching_persisted_bond_features(
        family.bond_domain(),
        query_feature.left_bond,
        &mut scratch.path4_left_bonds,
    );
    collect_matching_persisted_bond_features(
        family.bond_domain(),
        query_feature.center_bond,
        &mut scratch.path4_center_bonds,
    );
    collect_matching_persisted_bond_features(
        family.bond_domain(),
        query_feature.right_bond,
        &mut scratch.path4_right_bonds,
    );
}

fn accumulate_persisted_path4_matches<Family>(
    family: &Family,
    query_feature: Path4Feature,
    scratch: &mut TargetCorpusScratch<'_>,
    active_candidate_mask: Option<&[u64]>,
) where
    Family: PersistedLocalFeatureFamilyAccess,
{
    let feature_count = persisted_feature_count(family.postings());
    let masks = family.mask_indexes();
    let Some(left_atoms) = masks.first() else {
        return;
    };
    let Some(left_bonds) = masks.get(1) else {
        return;
    };
    let Some(left_middle_atoms) = masks.get(2) else {
        return;
    };
    let Some(center_bonds) = masks.get(3) else {
        return;
    };
    let Some(right_middle_atoms) = masks.get(4) else {
        return;
    };
    let Some(right_bonds) = masks.get(5) else {
        return;
    };
    let Some(right_atoms) = masks.get(6) else {
        return;
    };

    let forward_has_candidates = build_persisted_path4_orientation_feature_mask(
        [
            left_atoms,
            left_bonds,
            left_middle_atoms,
            center_bonds,
            right_middle_atoms,
            right_bonds,
            right_atoms,
        ],
        &PersistedPath4OrientationFeatureMask {
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
    let reverse_has_candidates = build_persisted_path4_orientation_feature_mask(
        [
            left_atoms,
            left_bonds,
            left_middle_atoms,
            center_bonds,
            right_middle_atoms,
            right_bonds,
            right_atoms,
        ],
        &PersistedPath4OrientationFeatureMask {
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

    accumulate_persisted_feature_id_mask_counts(
        family.postings(),
        &scratch.path4_feature_candidate_mask,
        &mut scratch.path4_count_by_target,
        &mut scratch.path4_touched_targets,
        active_candidate_mask,
    );
}

fn populate_persisted_star3_candidate_mask<Family>(
    family: &Family,
    target_count: usize,
    query_feature: Star3Feature,
    required_count: u16,
    active_candidate_mask: Option<&[u64]>,
    scratch: &mut TargetCorpusScratch<'_>,
) -> usize
where
    Family: PersistedLocalFeatureFamilyAccess,
{
    if required_count == 0 {
        return target_count;
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
        target_count,
        &mut scratch.star3_count_by_target,
        &mut scratch.star3_touched_targets,
    );

    collect_persisted_star3_domain_matches(family, query_feature, scratch);
    accumulate_persisted_star3_matches(family, query_feature, scratch, active_candidate_mask);

    finalize_cached_sparse_candidate_mask(
        target_count,
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

fn collect_persisted_star3_domain_matches<Family>(
    family: &Family,
    query_feature: Star3Feature,
    scratch: &mut TargetCorpusScratch<'_>,
) where
    Family: PersistedLocalFeatureFamilyAccess,
{
    collect_matching_persisted_atom_features(
        family.atom_domain(),
        query_feature.center,
        &mut scratch.star3_center_atoms,
    );
    collect_matching_persisted_atom_features(
        family.atom_domain(),
        query_feature.arms[0].atom,
        &mut scratch.star3_first_atoms,
    );
    collect_matching_persisted_atom_features(
        family.atom_domain(),
        query_feature.arms[1].atom,
        &mut scratch.star3_second_atoms,
    );
    collect_matching_persisted_atom_features(
        family.atom_domain(),
        query_feature.arms[2].atom,
        &mut scratch.star3_third_atoms,
    );
    collect_matching_persisted_bond_features(
        family.bond_domain(),
        query_feature.arms[0].bond,
        &mut scratch.star3_first_bonds,
    );
    collect_matching_persisted_bond_features(
        family.bond_domain(),
        query_feature.arms[1].bond,
        &mut scratch.star3_second_bonds,
    );
    collect_matching_persisted_bond_features(
        family.bond_domain(),
        query_feature.arms[2].bond,
        &mut scratch.star3_third_bonds,
    );
}

fn accumulate_persisted_star3_matches<Family>(
    family: &Family,
    query_feature: Star3Feature,
    scratch: &mut TargetCorpusScratch<'_>,
    active_candidate_mask: Option<&[u64]>,
) where
    Family: PersistedLocalFeatureFamilyAccess,
{
    const ARM_ORDERS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];

    let feature_count = persisted_feature_count(family.postings());
    let masks = family.mask_indexes();
    let Some(center_atoms) = masks.first() else {
        return;
    };
    let Some(first_bonds) = masks.get(1) else {
        return;
    };
    let Some(first_atoms) = masks.get(2) else {
        return;
    };
    let Some(second_bonds) = masks.get(3) else {
        return;
    };
    let Some(second_atoms) = masks.get(4) else {
        return;
    };
    let Some(third_bonds) = masks.get(5) else {
        return;
    };
    let Some(third_atoms) = masks.get(6) else {
        return;
    };

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
    let target_atom_indexes = [first_atoms, second_atoms, third_atoms];
    let target_bond_indexes = [first_bonds, second_bonds, third_bonds];

    for order in ARM_ORDERS {
        if build_persisted_star3_orientation_feature_mask(
            &PersistedStar3OrientationFeatureMask {
                query_center: query_feature.center,
                center_atoms: &scratch.star3_center_atoms,
                center_atom_index: center_atoms,
                arms: [
                    PersistedStar3OrientationArmFeatureMask {
                        query: query_feature.arms[0],
                        atoms: target_atom_matches[0],
                        atom_index: target_atom_indexes[order[0]],
                        bonds: target_bond_matches[0],
                        bond_index: target_bond_indexes[order[0]],
                    },
                    PersistedStar3OrientationArmFeatureMask {
                        query: query_feature.arms[1],
                        atoms: target_atom_matches[1],
                        atom_index: target_atom_indexes[order[1]],
                        bonds: target_bond_matches[1],
                        bond_index: target_bond_indexes[order[1]],
                    },
                    PersistedStar3OrientationArmFeatureMask {
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

    accumulate_persisted_feature_id_mask_counts(
        family.postings(),
        &scratch.star3_feature_candidate_mask,
        &mut scratch.star3_count_by_target,
        &mut scratch.star3_touched_targets,
        active_candidate_mask,
    );
}

fn persisted_feature_count(postings: &impl PersistedSparseFeatureIndexAccess) -> usize {
    postings
        .persisted_sparse_singleton_offsets()
        .len()
        .saturating_sub(1)
}

fn collect_matching_persisted_atom_features(
    domain: &impl PersistedFeatureDomainAccess,
    query: AtomFeature,
    out: &mut Vec<AtomFeature>,
) {
    out.clear();
    if atom_feature_is_unconstrained(query)
        || domain.persisted_feature_width() != ATOM_FEATURE_WIDTH
    {
        return;
    }
    out.extend(
        domain
            .persisted_features()
            .chunks_exact(ATOM_FEATURE_WIDTH_USIZE)
            .filter_map(decode_atom_feature)
            .filter(|&feature| atom_feature_satisfies_query(feature, query)),
    );
}

fn collect_matching_persisted_bond_features(
    domain: &impl PersistedFeatureDomainAccess,
    query: EdgeBondFeature,
    out: &mut Vec<EdgeBondFeature>,
) {
    out.clear();
    if bond_feature_is_unconstrained(query)
        || domain.persisted_feature_width() != BOND_FEATURE_WIDTH
    {
        return;
    }
    out.extend(
        domain
            .persisted_features()
            .chunks_exact(BOND_FEATURE_WIDTH_USIZE)
            .filter_map(decode_bond_feature)
            .filter(|&feature| bond_feature_satisfies_query(feature, query)),
    );
}

fn find_persisted_feature_id_mask<'a>(
    index: &'a impl PersistedFeatureMaskIndexAccess,
    encoded_feature: &[u16],
) -> Option<&'a [u64]> {
    let width = usize::from(index.persisted_mask_feature_width());
    if width != encoded_feature.len() {
        return None;
    }
    let features = index.persisted_mask_features();
    let feature_count = features.len() / width;
    let mut low = 0usize;
    let mut high = feature_count;
    while low < high {
        let mid = low + (high - low) / 2;
        let start = mid * width;
        match features[start..start + width].cmp(encoded_feature) {
            core::cmp::Ordering::Less => low = mid + 1,
            core::cmp::Ordering::Equal => {
                let offsets = index.persisted_mask_offsets();
                let start = usize::try_from(*offsets.get(mid)?).ok()?;
                let end = usize::try_from(*offsets.get(mid + 1)?).ok()?;
                return (start <= end && end <= index.persisted_mask_words().len())
                    .then_some(&index.persisted_mask_words()[start..end]);
            }
            core::cmp::Ordering::Greater => high = mid,
        }
    }
    None
}

fn union_persisted_atom_feature_id_masks(
    index: &impl PersistedFeatureMaskIndexAccess,
    matching_values: &[AtomFeature],
    out: &mut Vec<u64>,
    word_count: usize,
) -> bool {
    ensure_zeroed_words(out, word_count);
    for &value in matching_values {
        let encoded = encoded_atom_feature(value);
        let Some(source) = find_persisted_feature_id_mask(index, &encoded) else {
            continue;
        };
        for (dst, src) in out.iter_mut().zip(source) {
            *dst |= *src;
        }
    }
    out.iter().any(|&word| word != 0)
}

fn union_persisted_bond_feature_id_masks(
    index: &impl PersistedFeatureMaskIndexAccess,
    matching_values: &[EdgeBondFeature],
    out: &mut Vec<u64>,
    word_count: usize,
) -> bool {
    ensure_zeroed_words(out, word_count);
    for &value in matching_values {
        let encoded = encoded_bond_feature(value);
        let Some(source) = find_persisted_feature_id_mask(index, &encoded) else {
            continue;
        };
        for (dst, src) in out.iter_mut().zip(source) {
            *dst |= *src;
        }
    }
    out.iter().any(|&word| word != 0)
}

struct PersistedFeatureMaskBuilder<'a> {
    candidate_mask: &'a mut Vec<u64>,
    component_mask: &'a mut Vec<u64>,
    word_count: usize,
    has_active_candidate: bool,
}

impl<'a> PersistedFeatureMaskBuilder<'a> {
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

    fn intersect_atom(
        &mut self,
        index: &impl PersistedFeatureMaskIndexAccess,
        matching_values: &[AtomFeature],
    ) -> bool {
        if !union_persisted_atom_feature_id_masks(
            index,
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

    fn intersect_bond(
        &mut self,
        index: &impl PersistedFeatureMaskIndexAccess,
        matching_values: &[EdgeBondFeature],
    ) -> bool {
        if !union_persisted_bond_feature_id_masks(
            index,
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

enum PersistedFeatureMaskConstraint<'a, MaskIndex> {
    Atom {
        query: AtomFeature,
        index: &'a MaskIndex,
        matching_values: &'a [AtomFeature],
    },
    Bond {
        query: EdgeBondFeature,
        index: &'a MaskIndex,
        matching_values: &'a [EdgeBondFeature],
    },
}

impl<'a, MaskIndex> PersistedFeatureMaskConstraint<'a, MaskIndex>
where
    MaskIndex: PersistedFeatureMaskIndexAccess + 'a,
{
    const fn atom(
        query: AtomFeature,
        index: &'a MaskIndex,
        matching_values: &'a [AtomFeature],
    ) -> Self {
        Self::Atom {
            query,
            index,
            matching_values,
        }
    }

    const fn bond(
        query: EdgeBondFeature,
        index: &'a MaskIndex,
        matching_values: &'a [EdgeBondFeature],
    ) -> Self {
        Self::Bond {
            query,
            index,
            matching_values,
        }
    }

    fn intersects(self, builder: &mut PersistedFeatureMaskBuilder<'_>) -> bool {
        match self {
            Self::Atom {
                query,
                index,
                matching_values,
            } => {
                atom_feature_is_unconstrained(query)
                    || builder.intersect_atom(index, matching_values)
            }
            Self::Bond {
                query,
                index,
                matching_values,
            } => {
                bond_feature_is_unconstrained(query)
                    || builder.intersect_bond(index, matching_values)
            }
        }
    }
}

fn build_persisted_feature_mask<'a, MaskIndex>(
    feature_count: usize,
    candidate_mask: &mut Vec<u64>,
    component_mask: &mut Vec<u64>,
    constraints: impl IntoIterator<Item = PersistedFeatureMaskConstraint<'a, MaskIndex>>,
) -> bool
where
    MaskIndex: PersistedFeatureMaskIndexAccess + 'a,
{
    let mut builder =
        PersistedFeatureMaskBuilder::new(feature_count, candidate_mask, component_mask);
    for constraint in constraints {
        if !constraint.intersects(&mut builder) {
            return false;
        }
    }
    builder.has_candidates()
}

struct PersistedEdgeOrientationFeatureMask<'a> {
    query_left: AtomFeature,
    left_atoms: &'a [AtomFeature],
    query_bond: EdgeBondFeature,
    bonds: &'a [EdgeBondFeature],
    query_right: AtomFeature,
    right_atoms: &'a [AtomFeature],
}

fn build_persisted_edge_orientation_feature_mask<MaskIndex>(
    masks: [&MaskIndex; 3],
    orientation: &PersistedEdgeOrientationFeatureMask<'_>,
    feature_count: usize,
    candidate_mask: &mut Vec<u64>,
    component_mask: &mut Vec<u64>,
) -> bool
where
    MaskIndex: PersistedFeatureMaskIndexAccess,
{
    let [left_atoms, bonds, right_atoms] = masks;
    build_persisted_feature_mask(
        feature_count,
        candidate_mask,
        component_mask,
        [
            PersistedFeatureMaskConstraint::atom(
                orientation.query_left,
                left_atoms,
                orientation.left_atoms,
            ),
            PersistedFeatureMaskConstraint::bond(orientation.query_bond, bonds, orientation.bonds),
            PersistedFeatureMaskConstraint::atom(
                orientation.query_right,
                right_atoms,
                orientation.right_atoms,
            ),
        ],
    )
}

struct PersistedPath3OrientationFeatureMask<'a> {
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

fn build_persisted_path3_orientation_feature_mask<MaskIndex>(
    masks: [&MaskIndex; 5],
    orientation: &PersistedPath3OrientationFeatureMask<'_>,
    feature_count: usize,
    candidate_mask: &mut Vec<u64>,
    component_mask: &mut Vec<u64>,
) -> bool
where
    MaskIndex: PersistedFeatureMaskIndexAccess,
{
    let [left_atoms, left_bonds, center_atoms, right_bonds, right_atoms] = masks;
    build_persisted_feature_mask(
        feature_count,
        candidate_mask,
        component_mask,
        [
            PersistedFeatureMaskConstraint::atom(
                orientation.query_left,
                left_atoms,
                orientation.left_atoms,
            ),
            PersistedFeatureMaskConstraint::bond(
                orientation.query_left_bond,
                left_bonds,
                orientation.left_bonds,
            ),
            PersistedFeatureMaskConstraint::atom(
                orientation.query_center,
                center_atoms,
                orientation.center_atoms,
            ),
            PersistedFeatureMaskConstraint::bond(
                orientation.query_right_bond,
                right_bonds,
                orientation.right_bonds,
            ),
            PersistedFeatureMaskConstraint::atom(
                orientation.query_right,
                right_atoms,
                orientation.right_atoms,
            ),
        ],
    )
}

struct PersistedPath4OrientationFeatureMask<'a> {
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

fn build_persisted_path4_orientation_feature_mask<MaskIndex>(
    masks: [&MaskIndex; 7],
    orientation: &PersistedPath4OrientationFeatureMask<'_>,
    feature_count: usize,
    candidate_mask: &mut Vec<u64>,
    component_mask: &mut Vec<u64>,
) -> bool
where
    MaskIndex: PersistedFeatureMaskIndexAccess,
{
    let [left_atoms, left_bonds, left_middle_atoms, center_bonds, right_middle_atoms, right_bonds, right_atoms] =
        masks;
    build_persisted_feature_mask(
        feature_count,
        candidate_mask,
        component_mask,
        [
            PersistedFeatureMaskConstraint::atom(
                orientation.query_left,
                left_atoms,
                orientation.left_atoms,
            ),
            PersistedFeatureMaskConstraint::bond(
                orientation.query_left_bond,
                left_bonds,
                orientation.left_bonds,
            ),
            PersistedFeatureMaskConstraint::atom(
                orientation.query_left_middle,
                left_middle_atoms,
                orientation.left_middle_atoms,
            ),
            PersistedFeatureMaskConstraint::bond(
                orientation.query_center_bond,
                center_bonds,
                orientation.center_bonds,
            ),
            PersistedFeatureMaskConstraint::atom(
                orientation.query_right_middle,
                right_middle_atoms,
                orientation.right_middle_atoms,
            ),
            PersistedFeatureMaskConstraint::bond(
                orientation.query_right_bond,
                right_bonds,
                orientation.right_bonds,
            ),
            PersistedFeatureMaskConstraint::atom(
                orientation.query_right,
                right_atoms,
                orientation.right_atoms,
            ),
        ],
    )
}

struct PersistedStar3OrientationArmFeatureMask<'a, MaskIndex> {
    query: Star3Arm,
    atoms: &'a [AtomFeature],
    atom_index: &'a MaskIndex,
    bonds: &'a [EdgeBondFeature],
    bond_index: &'a MaskIndex,
}

struct PersistedStar3OrientationFeatureMask<'a, MaskIndex> {
    query_center: AtomFeature,
    center_atoms: &'a [AtomFeature],
    center_atom_index: &'a MaskIndex,
    arms: [PersistedStar3OrientationArmFeatureMask<'a, MaskIndex>; 3],
}

fn build_persisted_star3_orientation_feature_mask<MaskIndex>(
    orientation: &PersistedStar3OrientationFeatureMask<'_, MaskIndex>,
    feature_count: usize,
    candidate_mask: &mut Vec<u64>,
    component_mask: &mut Vec<u64>,
) -> bool
where
    MaskIndex: PersistedFeatureMaskIndexAccess,
{
    let [first, second, third] = &orientation.arms;
    build_persisted_feature_mask(
        feature_count,
        candidate_mask,
        component_mask,
        [
            PersistedFeatureMaskConstraint::atom(
                orientation.query_center,
                orientation.center_atom_index,
                orientation.center_atoms,
            ),
            PersistedFeatureMaskConstraint::bond(first.query.bond, first.bond_index, first.bonds),
            PersistedFeatureMaskConstraint::atom(first.query.atom, first.atom_index, first.atoms),
            PersistedFeatureMaskConstraint::bond(
                second.query.bond,
                second.bond_index,
                second.bonds,
            ),
            PersistedFeatureMaskConstraint::atom(
                second.query.atom,
                second.atom_index,
                second.atoms,
            ),
            PersistedFeatureMaskConstraint::bond(third.query.bond, third.bond_index, third.bonds),
            PersistedFeatureMaskConstraint::atom(third.query.atom, third.atom_index, third.atoms),
        ],
    )
}

fn accumulate_persisted_feature_id_mask_counts(
    postings: &impl PersistedSparseFeatureIndexAccess,
    feature_mask: &[u64],
    count_by_target: &mut [u16],
    touched_targets: &mut Vec<usize>,
    active_candidate_mask: Option<&[u64]>,
) {
    for (word_index, mut word) in feature_mask.iter().copied().enumerate() {
        while word != 0 {
            let bit = word.trailing_zeros() as usize;
            let feature_id = word_index * u64::BITS as usize + bit;
            accumulate_persisted_sparse_feature_counts(
                postings,
                feature_id,
                count_by_target,
                touched_targets,
                active_candidate_mask,
            );
            word &= word - 1;
        }
    }
}

fn accumulate_persisted_sparse_feature_counts(
    postings: &impl PersistedSparseFeatureIndexAccess,
    feature_id: usize,
    count_by_target: &mut [u16],
    touched_targets: &mut Vec<usize>,
    active_candidate_mask: Option<&[u64]>,
) {
    let Some(singleton_range) = persisted_sparse_range(
        postings.persisted_sparse_singleton_offsets(),
        feature_id,
        postings.persisted_sparse_singleton_target_ids().len(),
    ) else {
        return;
    };
    let Some(singleton_targets) = postings
        .persisted_sparse_singleton_target_ids()
        .get(singleton_range)
    else {
        return;
    };
    for &target_id in singleton_targets {
        if !persisted_target_is_active(target_id, active_candidate_mask) {
            continue;
        }
        accumulate_persisted_sparse_target_count(count_by_target, touched_targets, target_id, 1);
    }

    let Some(counted_range) = persisted_sparse_range(
        postings.persisted_sparse_counted_offsets(),
        feature_id,
        postings.persisted_sparse_counted_target_ids().len(),
    ) else {
        return;
    };
    let Some(counted_targets) = postings
        .persisted_sparse_counted_target_ids()
        .get(counted_range.clone())
    else {
        return;
    };
    let Some(counted_values) = postings
        .persisted_sparse_counted_values()
        .get(counted_range)
    else {
        return;
    };
    for (&target_id, &count) in counted_targets.iter().zip(counted_values) {
        if !persisted_target_is_active(target_id, active_candidate_mask) {
            continue;
        }
        accumulate_persisted_sparse_target_count(
            count_by_target,
            touched_targets,
            target_id,
            count,
        );
    }
}

fn persisted_sparse_range(
    offsets: &[u64],
    feature_id: usize,
    value_len: usize,
) -> Option<Range<usize>> {
    let start = usize::try_from(*offsets.get(feature_id)?).ok()?;
    let end = usize::try_from(*offsets.get(feature_id + 1)?).ok()?;
    (start <= end && end <= value_len).then_some(start..end)
}

fn persisted_target_is_active(target_id: u32, active_candidate_mask: Option<&[u64]>) -> bool {
    let Some(mask) = active_candidate_mask else {
        return true;
    };
    let target_id = target_id as usize;
    let word = target_id / u64::BITS as usize;
    let bit = target_id % u64::BITS as usize;
    mask.get(word)
        .is_some_and(|word| (word & (1u64 << bit)) != 0)
}

fn accumulate_persisted_sparse_target_count(
    count_by_target: &mut [u16],
    touched_targets: &mut Vec<usize>,
    target_id: u32,
    count: u16,
) {
    let target_id = target_id as usize;
    let Some(target_count) = count_by_target.get_mut(target_id) else {
        return;
    };
    if *target_count == 0 {
        touched_targets.push(target_id);
    }
    *target_count = target_count.saturating_add(count);
}

#[cfg(test)]
mod tests {
    use core::str::FromStr;

    use alloc::format;
    use epserde::deser::Flags;
    use epserde::prelude::*;
    use smiles_parser::Smiles;

    use super::*;
    use crate::{prepared::PreparedTarget, screening::TargetCorpusIndexShard};

    #[test]
    fn persisted_count_bitset_index_round_trips_with_eps_deserialization() {
        let index = CountBitsetIndex::from_compact_counts(5, [0, 2, 1, 2, 3]);
        let persisted = PersistedCountBitsetIndex::from_count_bitset_index(&index);

        let mut cursor = <AlignedCursor<Aligned16>>::new();
        unsafe {
            persisted.serialize_into_unchecked(&mut cursor).unwrap();
        }

        let loaded = unsafe {
            PersistedCountBitsetIndex::deserialize_eps_unchecked(cursor.as_bytes()).unwrap()
        };
        assert_eq!(loaded.thresholds(), persisted.thresholds());
        assert_eq!(loaded.populations(), persisted.populations());
        assert_eq!(loaded.bitset_offsets(), persisted.bitset_offsets());
        assert_eq!(loaded.bitset_words(), persisted.bitset_words());

        let expected = persisted.filter_for_at_least(2).unwrap();
        let actual = loaded.filter_for_at_least(2).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn persisted_scalar_count_indexes_keep_queryable_count_filters() {
        let targets = ["C", "CC", "CCC"]
            .into_iter()
            .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
            .collect::<Vec<_>>();
        let index = TargetCorpusIndex::new(&targets);

        let persisted = PersistedScalarCountIndexes::from_target_corpus_index(&index);
        let at_least_two_atoms = persisted.atom_counts().filter_for_at_least(2).unwrap();

        assert_eq!(at_least_two_atoms.population, 2);
    }

    #[test]
    fn persisted_target_corpus_index_shard_round_trips_owned_payload() {
        let targets = persisted_candidate_targets();
        let index = TargetCorpusIndex::new(&targets);
        let shard = TargetCorpusIndexShard::new(17, index);
        let persisted = PersistedTargetCorpusIndexShard::from_index_shard(&shard);

        let mut cursor = <AlignedCursor<Aligned16>>::new();
        unsafe {
            persisted.serialize_into_unchecked(&mut cursor).unwrap();
        }
        cursor.set_position(0);
        let loaded = unsafe {
            PersistedTargetCorpusIndexShard::deserialize_full_unchecked(&mut cursor).unwrap()
        };

        assert_eq!(loaded, persisted);
        assert_eq!(loaded.format_version(), FORMAT_VERSION);
        assert_eq!(loaded.base_target_id(), 17);
        assert_eq!(loaded.target_count(), 5);
        assert_eq!(loaded.edge().postings().feature_width(), EDGE_FEATURE_WIDTH);
        assert_persisted_candidates_match_runtime_index(&persisted, &shard);
        assert_persisted_candidates_match_runtime_index(&loaded, &shard);
    }

    #[test]
    fn persisted_target_corpus_index_shard_mmap_keeps_candidate_filtering() {
        let targets = persisted_candidate_targets();
        let index = TargetCorpusIndex::new(&targets);
        let shard = TargetCorpusIndexShard::new(17, index);
        let persisted = PersistedTargetCorpusIndexShard::from_index_shard(&shard);
        let path = persisted_shard_temp_path();

        unsafe {
            persisted.store_unchecked(&path).unwrap();
        }
        let mapped = unsafe {
            PersistedTargetCorpusIndexShard::mmap_unchecked(&path, Flags::empty()).unwrap()
        };
        let loaded = mapped.uncase();

        assert_eq!(loaded.format_version(), FORMAT_VERSION);
        assert_eq!(loaded.base_target_id(), 17);
        assert_eq!(loaded.target_count(), 5);
        assert_persisted_candidates_match_runtime_index(loaded, &shard);

        std::fs::remove_file(path).unwrap();
    }

    fn persisted_candidate_targets() -> Vec<PreparedTarget> {
        ["CCO", "COC", "CC(C)C", "C1CCCCC1", "CC(O)(N)Cl"]
            .into_iter()
            .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
            .collect()
    }

    fn persisted_candidate_queries() -> Vec<QueryScreen> {
        [
            "CO",
            "[#6;D1;H3]-[#6;D3]",
            "[#6;R;D2]-[#6;R;D2]-[#6;R;D2]",
            "COC",
            "COCO",
            "C(O)(N)Cl",
        ]
        .into_iter()
        .map(|smarts| QueryScreen::new(&crate::QueryMol::from_str(smarts).unwrap()))
        .collect()
    }

    fn assert_persisted_candidates_match_runtime_index<
        ScalarCounts,
        AtomPropertyCounts,
        Edge,
        Path3,
        Path4,
        Star3,
    >(
        persisted: &PersistedTargetCorpusIndexShard<
            ScalarCounts,
            AtomPropertyCounts,
            Edge,
            Path3,
            Path4,
            Star3,
        >,
        shard: &TargetCorpusIndexShard,
    ) where
        ScalarCounts: PersistedScalarCountIndexesAccess,
        AtomPropertyCounts: PersistedAtomPropertyCountIndexesAccess,
        Edge: PersistedLocalFeatureFamilyAccess,
        Path3: PersistedLocalFeatureFamilyAccess,
        Path4: PersistedLocalFeatureFamilyAccess,
        Star3: PersistedLocalFeatureFamilyAccess,
    {
        for query in persisted_candidate_queries() {
            let expected = shard
                .index()
                .candidate_ids(&query)
                .into_iter()
                .map(|target_id| shard.base_target_id() + target_id)
                .collect::<Vec<_>>();
            assert_eq!(persisted.candidate_ids(&query), expected);
            assert_eq!(persisted.candidate_count(&query), expected.len());
        }
    }

    fn persisted_shard_temp_path() -> std::path::PathBuf {
        let suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "smarts-rs-persisted-shard-{}-{suffix}.eps",
            std::process::id()
        ))
    }
}
