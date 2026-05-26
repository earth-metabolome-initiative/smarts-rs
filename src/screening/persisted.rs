//! Epserde-backed persisted screening-index components.
//!
//! This module is feature-gated because epserde deserialization is unsafe and
//! because persisted indexes require `std` for file-backed use. The persisted
//! layouts here are explicit flat formats, not direct dumps of private runtime
//! structs.

use alloc::{boxed::Box, format, string::String, vec, vec::Vec};
use core::{
    fmt,
    marker::PhantomData,
    ops::Range,
    str::{self, FromStr},
};
#[cfg(all(feature = "zstd", feature = "terminal-progress"))]
use std::io::IsTerminal;
use std::io::{Error as IoError, ErrorKind};
use std::path::{Path, PathBuf};
#[cfg(feature = "zstd")]
use std::{
    fs::File,
    io::{BufWriter, Write},
    time::{Duration, Instant},
};

use epserde::{deser, prelude::Epserde, ser};
#[cfg(all(feature = "zstd", feature = "terminal-progress"))]
use indicatif::{ProgressBar, ProgressStyle};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

use super::{
    atom_feature_is_unconstrained, atom_feature_satisfies_query,
    bitset::{
        bitset_word_count, ensure_zeroed_words, for_each_set_bit, intersect_source,
        intersect_source_with_population, CountBitsetIndex, RequiredCountFilter,
    },
    bitset_population, bond_feature_is_unconstrained, bond_feature_satisfies_query,
    features::{
        AtomFeature, EdgeBondFeature, EdgeFeature, Path3Feature, Path4Feature, RequiredBondKind,
        Star3Arm, Star3Feature,
    },
    fill_all_target_bits, finalize_cached_sparse_candidate_mask, load_cached_candidate_mask,
    merge_reversible_feature_mask, prepare_sparse_candidate_counts, should_filter_sparse_counts,
    BondKindPair, CandidateMaskState, IndexedFeatureCountIndex, IndexedFeatureIdMask,
    IndexedSparseFeatureCountIndex, QueryFeatureFilter, QueryScreen, SparseCandidateMaskBuffers,
    TargetCandidateSet, TargetCorpusIndex, TargetCorpusIndexShard, TargetCorpusIndexStats,
    TargetCorpusScratch,
};
use crate::{
    matching::{CompiledQuery, MatchScratch},
    prepared::PreparedTarget,
};
use smiles_parser::{Smiles, SmilesError};

#[cfg(feature = "zstd")]
use PersistedTargetCorpusIndexShardBuildEvent::{Finished, Started};
#[cfg(feature = "zstd")]
use PersistedTargetCorpusIndexShardBuildStep::{
    BuildPersistedLayout, BuildRuntimeIndex, StorePayload,
};

const FORMAT_VERSION: u32 = 2;
const MANIFEST_VERSION: u32 = 1;
const MANIFEST_SUFFIX: &str = ".manifest.tsv";
const TARGET_SMILES_SHARD_SUFFIX: &str = ".target-smiles.eps";
const EXTERNAL_IDS_SHARD_SUFFIX: &str = ".external-ids.eps";
const ATOM_FEATURE_WIDTH: u16 = 5;
const BOND_FEATURE_WIDTH: u16 = 2;
const ATOM_FEATURE_WIDTH_USIZE: usize = ATOM_FEATURE_WIDTH as usize;
const BOND_FEATURE_WIDTH_USIZE: usize = BOND_FEATURE_WIDTH as usize;
const EDGE_FEATURE_WIDTH: u16 = ATOM_FEATURE_WIDTH * 2 + BOND_FEATURE_WIDTH;
const PATH3_FEATURE_WIDTH: u16 = ATOM_FEATURE_WIDTH * 3 + BOND_FEATURE_WIDTH * 2;
const PATH4_FEATURE_WIDTH: u16 = ATOM_FEATURE_WIDTH * 4 + BOND_FEATURE_WIDTH * 3;
const STAR3_FEATURE_WIDTH: u16 = ATOM_FEATURE_WIDTH * 4 + BOND_FEATURE_WIDTH * 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PersistedShardPayloadKind {
    TargetIndex,
    TargetSmiles,
    ExternalIds,
}

impl PersistedShardPayloadKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::TargetIndex => "target-index",
            Self::TargetSmiles => "target-smiles",
            Self::ExternalIds => "external-ids",
        }
    }

    fn from_str(value: &str) -> std::io::Result<Self> {
        match value {
            "target-index" => Ok(Self::TargetIndex),
            "target-smiles" => Ok(Self::TargetSmiles),
            "external-ids" => Ok(Self::ExternalIds),
            _ => Err(invalid_manifest_data(format!(
                "unknown persisted shard payload kind {value}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PersistedShardStoredCompression {
    None,
    Zstd,
}

impl PersistedShardStoredCompression {
    const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Zstd => "zstd",
        }
    }

    fn from_str(value: &str) -> std::io::Result<Self> {
        match value {
            "none" => Ok(Self::None),
            "zstd" => Ok(Self::Zstd),
            _ => Err(invalid_manifest_data(format!(
                "unknown persisted shard compression {value}"
            ))),
        }
    }
}

#[cfg(feature = "zstd")]
impl From<PersistedShardCompression> for PersistedShardStoredCompression {
    fn from(compression: PersistedShardCompression) -> Self {
        match compression {
            PersistedShardCompression::None => Self::None,
            PersistedShardCompression::Zstd { .. } => Self::Zstd,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PersistedShardManifest {
    manifest_version: u32,
    payload_kind: PersistedShardPayloadKind,
    format_version: u32,
    base_target_id: u64,
    target_count: u64,
    compression: PersistedShardStoredCompression,
    payload_bytes: u64,
}

/// A count-bitset filter loaded from a persisted flat count index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PersistedRequiredCountFilter<'a> {
    /// Number of targets selected by `source`.
    pub population: usize,
    /// Bitset words for targets whose count is at least the requested value.
    pub source: &'a [u64],
}

/// Compression mode used when storing persisted target-index shards.
#[cfg(feature = "zstd")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PersistedShardCompression {
    /// Store raw epserde bytes.
    #[default]
    None,
    /// Store zstd-compressed epserde bytes.
    ///
    /// `worker_threads` values `0` and `1` use zstd's single-threaded encoder.
    /// Values greater than `1` enable zstd's multithreaded encoder.
    Zstd {
        /// zstd compression level.
        level: i32,
        /// Number of compression worker threads.
        worker_threads: u32,
    },
}

#[cfg(feature = "zstd")]
impl PersistedShardCompression {
    /// Returns true when this mode stores mmap-loadable raw epserde bytes.
    #[inline]
    #[must_use]
    pub const fn is_raw(self) -> bool {
        matches!(self, Self::None)
    }

    /// Returns the conventional file extension for this shard payload.
    #[inline]
    #[must_use]
    pub const fn file_extension(self) -> &'static str {
        match self {
            Self::None => "eps",
            Self::Zstd { .. } => "eps.zst",
        }
    }
}

#[cfg(feature = "zstd")]
impl fmt::Display for PersistedShardCompression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => f.write_str("none"),
            Self::Zstd {
                level,
                worker_threads,
            } => write!(f, "zstd(level={level}, threads={worker_threads})"),
        }
    }
}

/// Size information returned after storing a persisted target-index shard.
#[cfg(feature = "zstd")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PersistedShardStoreStats {
    /// Number of raw epserde bytes serialized before any compression.
    pub serialized_bytes: usize,
    /// Number of bytes written to disk.
    pub disk_bytes: u64,
}

/// Build and storage measurements for one persisted target-index shard.
#[cfg(feature = "zstd")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PersistedTargetCorpusIndexShardBuildStats {
    /// Number of targets included in the shard.
    pub target_count: usize,
    /// Runtime index feature/posting counts for this shard.
    pub index_stats: TargetCorpusIndexStats,
    /// Serialized and on-disk byte counts for the shard payload.
    pub store_stats: PersistedShardStoreStats,
}

/// Build and storage measurements for one queryable persisted target-index shard.
#[cfg(feature = "zstd")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PersistedQueryableTargetCorpusIndexShardBuildStats {
    /// Main persisted target-index shard measurements.
    pub index: PersistedTargetCorpusIndexShardBuildStats,
    /// Serialized and on-disk byte counts for the target SMILES sidecar.
    pub target_smiles_store_stats: PersistedShardStoreStats,
    /// Serialized and on-disk byte counts for the optional external-id sidecar.
    pub external_ids_store_stats: Option<PersistedShardStoreStats>,
}

/// Major step in persisted target-index shard construction.
#[cfg(feature = "zstd")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PersistedTargetCorpusIndexShardBuildStep {
    /// Build the runtime [`TargetCorpusIndex`] from prepared targets.
    BuildRuntimeIndex,
    /// Convert the runtime index into its persisted epserde layout.
    BuildPersistedLayout,
    /// Serialize and optionally compress the persisted shard payload.
    StorePayload,
}

#[cfg(feature = "zstd")]
impl PersistedTargetCorpusIndexShardBuildStep {
    /// Returns a short human-readable label for the step.
    #[inline]
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::BuildRuntimeIndex => "build runtime target index",
            Self::BuildPersistedLayout => "build persisted shard layout",
            Self::StorePayload => "store shard payload",
        }
    }
}

/// Progress event emitted while building one persisted target-index shard.
#[cfg(feature = "zstd")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PersistedTargetCorpusIndexShardBuildEvent {
    /// A shard-build step started.
    Started {
        /// Step that started.
        step: PersistedTargetCorpusIndexShardBuildStep,
        /// Number of targets in the shard being built.
        target_count: usize,
    },
    /// A shard-build step finished.
    Finished {
        /// Step that finished.
        step: PersistedTargetCorpusIndexShardBuildStep,
        /// Number of targets in the shard being built.
        target_count: usize,
        /// Wall-clock time spent in this step.
        elapsed: Duration,
    },
}

#[cfg(feature = "zstd")]
impl PersistedTargetCorpusIndexShardBuildEvent {
    /// Returns the step associated with this event.
    #[inline]
    #[must_use]
    pub const fn step(self) -> PersistedTargetCorpusIndexShardBuildStep {
        match self {
            Self::Started { step, .. } | Self::Finished { step, .. } => step,
        }
    }

    /// Returns the number of targets in the shard being built.
    #[inline]
    #[must_use]
    pub const fn target_count(self) -> usize {
        match self {
            Self::Started { target_count, .. } | Self::Finished { target_count, .. } => {
                target_count
            }
        }
    }

    /// Returns elapsed time for finished steps.
    #[inline]
    #[must_use]
    pub const fn elapsed(self) -> Option<Duration> {
        match self {
            Self::Started { .. } => None,
            Self::Finished { elapsed, .. } => Some(elapsed),
        }
    }
}

#[cfg(all(feature = "zstd", feature = "terminal-progress"))]
const SHARD_BUILD_PROGRESS_TEMPLATE: &str =
    "{spinner:.green} shard {prefix:.bold} [{bar:32.cyan/blue}] {pos}/{len} {msg} {elapsed_precise}";

/// Terminal progress renderer for persisted target-index shard construction.
#[cfg(all(feature = "zstd", feature = "terminal-progress"))]
#[derive(Debug, Clone)]
pub struct PersistedTargetCorpusIndexShardTerminalProgress {
    progress_bar: Option<ProgressBar>,
}

#[cfg(all(feature = "zstd", feature = "terminal-progress"))]
impl PersistedTargetCorpusIndexShardTerminalProgress {
    /// Creates an enabled progress bar for one shard.
    ///
    /// The bar is hidden automatically when stderr is not attached to a
    /// terminal, matching the backend's usual non-interactive behavior.
    #[must_use]
    pub fn new(shard_label: impl Into<String>, target_count: usize) -> Self {
        Self::with_enabled(true, shard_label, target_count)
    }

    /// Creates a progress bar only when `enabled` is true and stderr is a TTY.
    #[must_use]
    pub fn with_enabled(
        enabled: bool,
        shard_label: impl Into<String>,
        target_count: usize,
    ) -> Self {
        if !enabled || !std::io::stderr().is_terminal() {
            return Self { progress_bar: None };
        }

        let progress_bar = ProgressBar::new(3);
        progress_bar.set_style(
            ProgressStyle::with_template(SHARD_BUILD_PROGRESS_TEMPLATE)
                .unwrap_or_else(|_| unreachable!("progress template is static and valid")),
        );
        progress_bar.set_prefix(shard_label.into());
        progress_bar.set_message(format!("queued {target_count} targets"));
        progress_bar.enable_steady_tick(Duration::from_millis(100));
        Self {
            progress_bar: Some(progress_bar),
        }
    }

    /// Updates the progress bar with a shard-build event.
    pub fn on_event(&self, event: PersistedTargetCorpusIndexShardBuildEvent) {
        let Some(progress_bar) = &self.progress_bar else {
            return;
        };

        match event.elapsed() {
            None => {
                progress_bar.set_message(format!(
                    "{} for {} targets",
                    event.step().label(),
                    event.target_count()
                ));
            }
            Some(elapsed) => {
                progress_bar.inc(1);
                progress_bar
                    .set_message(format!("{} finished in {elapsed:?}", event.step().label()));
            }
        }
    }

    /// Finishes the progress bar with a target-count summary.
    pub fn finish(self, target_count: usize) {
        if let Some(progress_bar) = self.progress_bar {
            progress_bar.finish_with_message(format!("stored {target_count} targets"));
        }
    }
}

/// Result of extracting one zstd-compressed shard to raw epserde bytes.
#[cfg(feature = "zstd")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedShardExtraction {
    /// Compressed source path.
    pub source: PathBuf,
    /// Raw destination path.
    pub destination: PathBuf,
    /// Whether this call performed decompression.
    pub extracted: bool,
    /// Current raw destination size in bytes.
    pub disk_bytes: u64,
}

/// Error returned while storing a persisted target-index shard.
#[cfg(feature = "zstd")]
#[derive(Debug)]
pub enum PersistedShardStoreError {
    /// A filesystem or compression writer operation failed.
    Io(std::io::Error),
    /// Epserde serialization failed.
    Serialization(ser::Error),
}

#[cfg(feature = "zstd")]
impl fmt::Display for PersistedShardStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::Serialization(error) => write!(f, "{error}"),
        }
    }
}

#[cfg(feature = "zstd")]
impl std::error::Error for PersistedShardStoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Serialization(error) => Some(error),
        }
    }
}

#[cfg(feature = "zstd")]
impl From<std::io::Error> for PersistedShardStoreError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

#[cfg(feature = "zstd")]
impl From<ser::Error> for PersistedShardStoreError {
    fn from(error: ser::Error) -> Self {
        Self::Serialization(error)
    }
}

#[cfg(feature = "zstd")]
fn invalid_store_input(message: impl Into<String>) -> PersistedShardStoreError {
    IoError::new(ErrorKind::InvalidInput, message.into()).into()
}

fn invalid_manifest_data(message: impl Into<String>) -> IoError {
    IoError::new(ErrorKind::InvalidData, message.into())
}

#[cfg(feature = "zstd")]
fn write_payload_manifest(
    path: impl AsRef<Path>,
    payload_kind: PersistedShardPayloadKind,
    format_version: u32,
    base_target_id: u64,
    target_count: u64,
    compression: PersistedShardStoredCompression,
    payload_bytes: u64,
) -> std::io::Result<()> {
    let mut writer = BufWriter::new(File::create(persisted_manifest_path_for_payload_path(
        path,
    ))?);
    writeln!(writer, "manifest_version\t{MANIFEST_VERSION}")?;
    writeln!(writer, "payload_kind\t{}", payload_kind.as_str())?;
    writeln!(writer, "format_version\t{format_version}")?;
    writeln!(writer, "base_target_id\t{base_target_id}")?;
    writeln!(writer, "target_count\t{target_count}")?;
    writeln!(writer, "compression\t{}", compression.as_str())?;
    writeln!(writer, "payload_bytes\t{payload_bytes}")?;
    writeln!(writer, "crate_version\t{}", env!("CARGO_PKG_VERSION"))?;
    writer.flush()
}

fn validate_payload_manifest(
    path: impl AsRef<Path>,
    expected_kind: Option<PersistedShardPayloadKind>,
    expected_compression: PersistedShardStoredCompression,
) -> std::io::Result<PersistedShardManifest> {
    let path = path.as_ref();
    let manifest = read_payload_manifest(path)?;
    if manifest.manifest_version != MANIFEST_VERSION {
        return Err(invalid_manifest_data(format!(
            "persisted shard manifest {} has version {}, expected {MANIFEST_VERSION}",
            persisted_manifest_path_for_payload_path(path).display(),
            manifest.manifest_version
        )));
    }
    if let Some(expected_kind) = expected_kind {
        if manifest.payload_kind != expected_kind {
            return Err(invalid_manifest_data(format!(
                "persisted shard manifest {} has payload kind {}, expected {}",
                persisted_manifest_path_for_payload_path(path).display(),
                manifest.payload_kind.as_str(),
                expected_kind.as_str()
            )));
        }
    }
    if manifest.format_version != FORMAT_VERSION {
        return Err(invalid_manifest_data(format!(
            "persisted shard manifest {} has format version {}, expected {FORMAT_VERSION}",
            persisted_manifest_path_for_payload_path(path).display(),
            manifest.format_version
        )));
    }
    if manifest.compression != expected_compression {
        return Err(invalid_manifest_data(format!(
            "persisted shard manifest {} has compression {}, expected {}",
            persisted_manifest_path_for_payload_path(path).display(),
            manifest.compression.as_str(),
            expected_compression.as_str()
        )));
    }
    let payload_bytes = std::fs::metadata(path)?.len();
    if manifest.payload_bytes != payload_bytes {
        return Err(invalid_manifest_data(format!(
            "persisted shard manifest {} records {} payload bytes, but {} has {payload_bytes}",
            persisted_manifest_path_for_payload_path(path).display(),
            manifest.payload_bytes,
            path.display()
        )));
    }
    Ok(manifest)
}

fn validate_next_shard_base(
    path: &Path,
    manifest: &PersistedShardManifest,
    expected_base_target_id: u64,
) -> std::io::Result<()> {
    if manifest.base_target_id == expected_base_target_id {
        return Ok(());
    }
    if manifest.base_target_id < expected_base_target_id {
        return Err(invalid_manifest_data(format!(
            "persisted shard {} overlaps previous shard range: base target id {}, expected {}",
            path.display(),
            manifest.base_target_id,
            expected_base_target_id
        )));
    }
    Err(invalid_manifest_data(format!(
        "persisted shard {} leaves a target-id gap: base target id {}, expected {}",
        path.display(),
        manifest.base_target_id,
        expected_base_target_id
    )))
}

fn validate_sidecar_manifest_header(
    sidecar_name: &str,
    sidecar_path: &Path,
    sidecar_manifest: &PersistedShardManifest,
    index_path: &Path,
    index_manifest: &PersistedShardManifest,
) -> std::io::Result<()> {
    if sidecar_manifest.format_version == index_manifest.format_version
        && sidecar_manifest.base_target_id == index_manifest.base_target_id
        && sidecar_manifest.target_count == index_manifest.target_count
    {
        return Ok(());
    }
    Err(invalid_manifest_data(format!(
        "{sidecar_name} sidecar manifest {} does not match index shard manifest {}: sidecar format={}, base={}, len={}; index format={}, base={}, len={}",
        persisted_manifest_path_for_payload_path(sidecar_path).display(),
        persisted_manifest_path_for_payload_path(index_path).display(),
        sidecar_manifest.format_version,
        sidecar_manifest.base_target_id,
        sidecar_manifest.target_count,
        index_manifest.format_version,
        index_manifest.base_target_id,
        index_manifest.target_count
    )))
}

fn read_payload_manifest(path: &Path) -> std::io::Result<PersistedShardManifest> {
    let manifest_path = persisted_manifest_path_for_payload_path(path);
    let raw = std::fs::read_to_string(&manifest_path).map_err(|error| {
        IoError::new(
            error.kind(),
            format!(
                "failed to read persisted shard manifest {}: {error}",
                manifest_path.display()
            ),
        )
    })?;
    parse_payload_manifest(&manifest_path, &raw)
}

fn parse_payload_manifest(
    manifest_path: &Path,
    raw: &str,
) -> std::io::Result<PersistedShardManifest> {
    let mut manifest_version = None;
    let mut payload_kind = None;
    let mut format_version = None;
    let mut base_target_id = None;
    let mut target_count = None;
    let mut compression = None;
    let mut payload_bytes = None;

    for (line_index, line) in raw.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let Some((key, value)) = line.split_once('\t') else {
            return Err(invalid_manifest_data(format!(
                "invalid persisted shard manifest {} line {}",
                manifest_path.display(),
                line_index + 1
            )));
        };
        match key {
            "manifest_version" => {
                manifest_version = Some(parse_manifest_value(manifest_path, key, value)?);
            }
            "payload_kind" => {
                payload_kind = Some(PersistedShardPayloadKind::from_str(value)?);
            }
            "format_version" => {
                format_version = Some(parse_manifest_value(manifest_path, key, value)?);
            }
            "base_target_id" => {
                base_target_id = Some(parse_manifest_value(manifest_path, key, value)?);
            }
            "target_count" => {
                target_count = Some(parse_manifest_value(manifest_path, key, value)?);
            }
            "compression" => {
                compression = Some(PersistedShardStoredCompression::from_str(value)?);
            }
            "payload_bytes" => {
                payload_bytes = Some(parse_manifest_value(manifest_path, key, value)?);
            }
            "crate_version" => {}
            _ => {
                return Err(invalid_manifest_data(format!(
                    "unknown persisted shard manifest key {key} in {}",
                    manifest_path.display()
                )));
            }
        }
    }

    Ok(PersistedShardManifest {
        manifest_version: require_manifest_field(
            manifest_path,
            "manifest_version",
            manifest_version,
        )?,
        payload_kind: require_manifest_field(manifest_path, "payload_kind", payload_kind)?,
        format_version: require_manifest_field(manifest_path, "format_version", format_version)?,
        base_target_id: require_manifest_field(manifest_path, "base_target_id", base_target_id)?,
        target_count: require_manifest_field(manifest_path, "target_count", target_count)?,
        compression: require_manifest_field(manifest_path, "compression", compression)?,
        payload_bytes: require_manifest_field(manifest_path, "payload_bytes", payload_bytes)?,
    })
}

fn parse_manifest_value<T>(manifest_path: &Path, key: &str, value: &str) -> std::io::Result<T>
where
    T: FromStr,
    T::Err: fmt::Display,
{
    value.parse().map_err(|error| {
        invalid_manifest_data(format!(
            "invalid persisted shard manifest value for {key} in {}: {error}",
            manifest_path.display()
        ))
    })
}

fn require_manifest_field<T>(
    manifest_path: &Path,
    key: &str,
    value: Option<T>,
) -> std::io::Result<T> {
    value.ok_or_else(|| {
        invalid_manifest_data(format!(
            "missing persisted shard manifest key {key} in {}",
            manifest_path.display()
        ))
    })
}

#[cfg(feature = "zstd")]
unsafe fn store_epserde_value_with_compression_unchecked<T>(
    value: &T,
    path: impl AsRef<Path>,
    compression: PersistedShardCompression,
) -> Result<PersistedShardStoreStats, PersistedShardStoreError>
where
    T: ser::Serialize,
{
    match compression {
        PersistedShardCompression::None => {
            unsafe { ser::Serialize::store(value, path.as_ref())? };
            let disk_bytes = std::fs::metadata(path.as_ref())?.len();
            let serialized_bytes = usize::try_from(disk_bytes).unwrap_or(usize::MAX);
            Ok(PersistedShardStoreStats {
                serialized_bytes,
                disk_bytes,
            })
        }
        PersistedShardCompression::Zstd {
            level,
            worker_threads,
        } => {
            let file = File::create(path.as_ref())?;
            let writer = BufWriter::new(file);
            let mut encoder = zstd::stream::write::Encoder::new(writer, level)?;
            if worker_threads > 1 {
                encoder.multithread(worker_threads)?;
            }
            let serialized_bytes = unsafe { ser::Serialize::serialize(value, &mut encoder)? };
            let mut writer = encoder.finish()?;
            writer.flush()?;
            let disk_bytes = std::fs::metadata(path.as_ref())?.len();
            Ok(PersistedShardStoreStats {
                serialized_bytes,
                disk_bytes,
            })
        }
    }
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

/// Persisted flat form of keyed atom-property count indexes.
#[derive(Debug, Clone, PartialEq, Eq, Epserde)]
pub struct PersistedAtomPropertyCountIndexes<
    ElementCounts = PersistedKeyedCountIndexes,
    RingElementCounts = PersistedKeyedCountIndexes,
    AromaticElementCounts = PersistedKeyedCountIndexes,
    DegreeCounts = PersistedKeyedCountIndexes,
    TotalHydrogenCounts = PersistedKeyedCountIndexes,
    RingMembershipCounts = PersistedKeyedCountIndexes,
    RingSizeCounts = PersistedKeyedCountIndexes,
    RingConnectivityCounts = PersistedKeyedCountIndexes,
> {
    element: ElementCounts,
    ring_element: RingElementCounts,
    aromatic_element: AromaticElementCounts,
    degree: DegreeCounts,
    total_hydrogen: TotalHydrogenCounts,
    ring_membership: RingMembershipCounts,
    ring_size: RingSizeCounts,
    ring_connectivity: RingConnectivityCounts,
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

/// Persisted target SMILES payload aligned to one target-index shard.
#[derive(Debug, Clone, PartialEq, Eq, Epserde)]
pub struct PersistedTargetSmilesShard {
    format_version: u32,
    base_target_id: u64,
    target_count: u64,
    offsets: Box<[u64]>,
    bytes: Box<[u8]>,
}

/// Persisted external target ids aligned to one target-index shard.
#[derive(Debug, Clone, PartialEq, Eq, Epserde)]
pub struct PersistedExternalIdShard {
    format_version: u32,
    base_target_id: u64,
    target_count: u64,
    external_ids: Box<[u64]>,
}

/// Exact SMARTS hit returned by persisted queryable shards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PersistedTargetHit {
    /// Global target id within the indexed target corpus.
    pub target_id: usize,
    /// Optional caller-provided external target id.
    pub external_id: Option<u64>,
}

/// Exact SMARTS hit with the matched target SMILES borrowed from a persisted sidecar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PersistedTargetRecordHit<'a> {
    /// Global target id within the indexed target corpus.
    pub target_id: usize,
    /// Matched target SMILES from the target-SMILES sidecar.
    pub target_smiles: &'a str,
    /// Optional caller-provided external target id.
    pub external_id: Option<u64>,
}

/// Policy for malformed target SMILES encountered while querying persisted shards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PersistedTargetSmilesParsePolicy {
    /// Return an error when any candidate target SMILES cannot be parsed.
    Error,
    /// Skip candidate target SMILES that contain wildcard atoms.
    ///
    /// Other parse errors still return an error. This is useful for `PubChem`
    /// shards that retain a small number of upstream `*` records.
    SkipWildcard,
}

/// Builder for storing one persisted target-index shard.
#[cfg(feature = "zstd")]
#[derive(Debug, Clone, Copy)]
pub struct PersistedTargetCorpusIndexShardBuilder<'a> {
    base_target_id: usize,
    targets: &'a [PreparedTarget],
    target_smiles: Option<&'a [String]>,
    external_ids: Option<&'a [u64]>,
    compression: PersistedShardCompression,
}

#[cfg(feature = "zstd")]
impl<'a> PersistedTargetCorpusIndexShardBuilder<'a> {
    /// Starts building a persisted shard from prepared targets.
    #[inline]
    #[must_use]
    pub const fn new(targets: &'a [PreparedTarget]) -> Self {
        Self {
            base_target_id: 0,
            targets,
            target_smiles: None,
            external_ids: None,
            compression: PersistedShardCompression::None,
        }
    }

    /// Sets the global id of local target `0` in this shard.
    #[inline]
    #[must_use]
    pub const fn base_target_id(mut self, base_target_id: usize) -> Self {
        self.base_target_id = base_target_id;
        self
    }

    /// Sets the storage compression mode.
    #[inline]
    #[must_use]
    pub const fn compression(mut self, compression: PersistedShardCompression) -> Self {
        self.compression = compression;
        self
    }

    /// Sets target SMILES strings for exact persisted querying.
    ///
    /// The slice order must match the prepared targets passed to [`Self::new`].
    #[inline]
    #[must_use]
    pub const fn target_smiles(mut self, target_smiles: &'a [String]) -> Self {
        self.target_smiles = Some(target_smiles);
        self
    }

    /// Sets optional external target ids for exact persisted querying.
    ///
    /// The slice order must match the prepared targets passed to [`Self::new`].
    #[inline]
    #[must_use]
    pub const fn external_ids(mut self, external_ids: &'a [u64]) -> Self {
        self.external_ids = Some(external_ids);
        self
    }

    /// Builds and stores the shard at `path`.
    ///
    /// # Errors
    ///
    /// Returns an error if the shard cannot be serialized or written.
    ///
    /// # Panics
    ///
    /// Panics if the shard base id, target count, or an internal posting/mask
    /// offset exceeds the persisted format's `u64`/`u32` capacity.
    ///
    /// # Safety
    ///
    /// This writes epserde bytes. The resulting file must be treated as trusted
    /// input when it is later deserialized.
    pub unsafe fn store_unchecked(
        self,
        path: impl AsRef<Path>,
    ) -> Result<PersistedTargetCorpusIndexShardBuildStats, PersistedShardStoreError> {
        unsafe { self.store_unchecked_with_progress(path, |_| {}) }
    }

    /// Builds and stores the shard at `path`, emitting coarse progress events.
    ///
    /// The callback is intentionally generic: terminal tools can render
    /// progress bars, while library users can collect timings without pulling
    /// in a UI dependency.
    ///
    /// # Errors
    ///
    /// Returns an error if the shard cannot be serialized or written.
    ///
    /// # Panics
    ///
    /// Panics if the shard base id, target count, or an internal posting/mask
    /// offset exceeds the persisted format's `u64`/`u32` capacity.
    ///
    /// # Safety
    ///
    /// This writes epserde bytes. The resulting file must be treated as trusted
    /// input when it is later deserialized.
    pub unsafe fn store_unchecked_with_progress(
        self,
        path: impl AsRef<Path>,
        mut progress: impl FnMut(PersistedTargetCorpusIndexShardBuildEvent),
    ) -> Result<PersistedTargetCorpusIndexShardBuildStats, PersistedShardStoreError> {
        let target_count = self.targets.len();

        progress(Started {
            step: BuildRuntimeIndex,
            target_count,
        });
        let started = Instant::now();
        let index = TargetCorpusIndex::new(self.targets);
        progress(Finished {
            step: BuildRuntimeIndex,
            target_count,
            elapsed: started.elapsed(),
        });

        let index_stats = index.stats();
        let target_count = index.len();

        progress(Started {
            step: BuildPersistedLayout,
            target_count,
        });
        let started = Instant::now();
        let shard = TargetCorpusIndexShard::new(self.base_target_id, index);
        let persisted = PersistedTargetCorpusIndexShard::from_index_shard(&shard);
        progress(Finished {
            step: BuildPersistedLayout,
            target_count,
            elapsed: started.elapsed(),
        });

        progress(Started {
            step: StorePayload,
            target_count,
        });
        let started = Instant::now();
        let store_stats =
            unsafe { persisted.store_with_compression_unchecked(path, self.compression)? };
        progress(Finished {
            step: StorePayload,
            target_count,
            elapsed: started.elapsed(),
        });

        Ok(PersistedTargetCorpusIndexShardBuildStats {
            target_count,
            index_stats,
            store_stats,
        })
    }

    /// Builds and stores a queryable shard at `path`.
    ///
    /// A queryable shard stores the normal persisted screening index plus a
    /// target-SMILES sidecar. If external ids were provided, it also stores an
    /// external-id sidecar aligned to the same target order.
    ///
    /// # Errors
    ///
    /// Returns an error if target SMILES were not provided, if sidecar lengths
    /// do not match `targets`, or if any payload cannot be serialized or
    /// written.
    ///
    /// # Panics
    ///
    /// Panics if the shard base id, target count, target SMILES byte offsets,
    /// or an internal posting/mask offset exceeds the persisted format's
    /// `u64`/`u32` capacity.
    ///
    /// # Safety
    ///
    /// This writes epserde bytes. The resulting files must be treated as
    /// trusted input when they are later deserialized.
    pub unsafe fn store_queryable_unchecked(
        self,
        path: impl AsRef<Path>,
    ) -> Result<PersistedQueryableTargetCorpusIndexShardBuildStats, PersistedShardStoreError> {
        unsafe { self.store_queryable_unchecked_with_progress(path, |_| {}) }
    }

    /// Builds and stores a queryable shard at `path`, emitting progress events
    /// for the main target-index shard construction.
    ///
    /// # Errors
    ///
    /// Returns an error if target SMILES were not provided, if sidecar lengths
    /// do not match `targets`, or if any payload cannot be serialized or
    /// written.
    ///
    /// # Panics
    ///
    /// Panics if the shard base id, target count, target SMILES byte offsets,
    /// or an internal posting/mask offset exceeds the persisted format's
    /// `u64`/`u32` capacity.
    ///
    /// # Safety
    ///
    /// This writes epserde bytes. The resulting files must be treated as
    /// trusted input when they are later deserialized.
    pub unsafe fn store_queryable_unchecked_with_progress(
        self,
        path: impl AsRef<Path>,
        progress: impl FnMut(PersistedTargetCorpusIndexShardBuildEvent),
    ) -> Result<PersistedQueryableTargetCorpusIndexShardBuildStats, PersistedShardStoreError> {
        let path = path.as_ref();
        let target_smiles = self.target_smiles.ok_or_else(|| {
            invalid_store_input("queryable persisted shards require target SMILES")
        })?;
        if target_smiles.len() != self.targets.len() {
            return Err(invalid_store_input(format!(
                "target SMILES count {} does not match prepared target count {}",
                target_smiles.len(),
                self.targets.len()
            )));
        }
        if let Some(external_ids) = self.external_ids {
            if external_ids.len() != self.targets.len() {
                return Err(invalid_store_input(format!(
                    "external id count {} does not match prepared target count {}",
                    external_ids.len(),
                    self.targets.len()
                )));
            }
        }

        let index = unsafe { self.store_unchecked_with_progress(path, progress)? };
        let target_smiles_shard =
            PersistedTargetSmilesShard::from_target_smiles(self.base_target_id, target_smiles)?;
        let target_smiles_store_stats = unsafe {
            target_smiles_shard.store_with_compression_unchecked(
                persisted_target_smiles_path_for_index_shard_path(path),
                self.compression,
            )?
        };
        let external_ids_store_stats = self
            .external_ids
            .map(|external_ids| {
                let external_ids_shard =
                    PersistedExternalIdShard::from_external_ids(self.base_target_id, external_ids);
                unsafe {
                    external_ids_shard.store_with_compression_unchecked(
                        persisted_external_ids_path_for_index_shard_path(path),
                        self.compression,
                    )
                }
            })
            .transpose()?;

        Ok(PersistedQueryableTargetCorpusIndexShardBuildStats {
            index,
            target_smiles_store_stats,
            external_ids_store_stats,
        })
    }

    /// Builds and stores the shard at `path` with a terminal progress bar.
    ///
    /// The bar is hidden automatically when stderr is not attached to a
    /// terminal. Use `shard_label` for a compact label such as a zero-padded
    /// shard number.
    ///
    /// # Errors
    ///
    /// Returns an error if the shard cannot be serialized or written.
    ///
    /// # Panics
    ///
    /// Panics if the shard base id, target count, or an internal posting/mask
    /// offset exceeds the persisted format's `u64`/`u32` capacity.
    ///
    /// # Safety
    ///
    /// This writes epserde bytes. The resulting file must be treated as trusted
    /// input when it is later deserialized.
    #[cfg(feature = "terminal-progress")]
    pub unsafe fn store_unchecked_with_terminal_progress(
        self,
        path: impl AsRef<Path>,
        shard_label: impl Into<String>,
    ) -> Result<PersistedTargetCorpusIndexShardBuildStats, PersistedShardStoreError> {
        let progress =
            PersistedTargetCorpusIndexShardTerminalProgress::new(shard_label, self.targets.len());
        let stats =
            unsafe { self.store_unchecked_with_progress(path, |event| progress.on_event(event))? };
        progress.finish(stats.target_count);
        Ok(stats)
    }

    /// Builds and stores a queryable shard at `path` with a terminal progress bar.
    ///
    /// # Errors
    ///
    /// Returns an error if target SMILES were not provided, if sidecar lengths
    /// do not match `targets`, or if any payload cannot be serialized or
    /// written.
    ///
    /// # Panics
    ///
    /// Panics if the shard base id, target count, target SMILES byte offsets,
    /// or an internal posting/mask offset exceeds the persisted format's
    /// `u64`/`u32` capacity.
    ///
    /// # Safety
    ///
    /// This writes epserde bytes. The resulting files must be treated as
    /// trusted input when they are later deserialized.
    #[cfg(feature = "terminal-progress")]
    pub unsafe fn store_queryable_unchecked_with_terminal_progress(
        self,
        path: impl AsRef<Path>,
        shard_label: impl Into<String>,
    ) -> Result<PersistedQueryableTargetCorpusIndexShardBuildStats, PersistedShardStoreError> {
        let progress =
            PersistedTargetCorpusIndexShardTerminalProgress::new(shard_label, self.targets.len());
        let stats = unsafe {
            self.store_queryable_unchecked_with_progress(path, |event| progress.on_event(event))?
        };
        progress.finish(stats.index.target_count);
        Ok(stats)
    }
}

impl PersistedTargetSmilesShard {
    /// Builds a target-SMILES sidecar from shard-aligned SMILES strings.
    ///
    /// # Errors
    ///
    /// Returns an error if a SMILES string contains a newline or if the byte
    /// payload exceeds the persisted format's `u64` offset capacity.
    ///
    /// # Panics
    ///
    /// Panics if `base_target_id` or `target_smiles.len()` exceed `u64`.
    #[cfg(feature = "zstd")]
    pub fn from_target_smiles(
        base_target_id: usize,
        target_smiles: &[String],
    ) -> Result<Self, PersistedShardStoreError> {
        let mut offsets = Vec::with_capacity(target_smiles.len() + 1);
        let mut bytes = Vec::new();
        offsets.push(0);
        for smiles in target_smiles {
            if smiles.contains(['\r', '\n']) {
                return Err(invalid_store_input(
                    "target SMILES must not contain newlines",
                ));
            }
            bytes.extend_from_slice(smiles.as_bytes());
            offsets.push(
                u64::try_from(bytes.len())
                    .map_err(|_| invalid_store_input("target SMILES payload is too large"))?,
            );
        }
        Ok(Self {
            format_version: FORMAT_VERSION,
            base_target_id: u64::try_from(base_target_id)
                .expect("target shard base id exceeds persisted target SMILES capacity"),
            target_count: u64::try_from(target_smiles.len())
                .expect("target shard length exceeds persisted target SMILES capacity"),
            offsets: offsets.into_boxed_slice(),
            bytes: bytes.into_boxed_slice(),
        })
    }

    /// Returns the persisted target-SMILES format version.
    #[inline]
    #[must_use]
    pub const fn format_version(&self) -> u32 {
        self.format_version
    }

    /// Returns the global id of local target `0` in this sidecar.
    #[inline]
    #[must_use]
    pub const fn base_target_id(&self) -> u64 {
        self.base_target_id
    }

    /// Returns the number of target SMILES records in this sidecar.
    #[inline]
    #[must_use]
    pub const fn target_count(&self) -> u64 {
        self.target_count
    }

    /// Returns the target SMILES for a local target id.
    #[must_use]
    pub fn target_smiles(&self, local_target_id: usize) -> Option<&str> {
        let start = usize::try_from(*self.offsets.get(local_target_id)?).ok()?;
        let end = usize::try_from(*self.offsets.get(local_target_id + 1)?).ok()?;
        (start <= end && end <= self.bytes.len())
            .then(|| str::from_utf8(&self.bytes[start..end]).ok())
            .flatten()
    }

    /// Stores this sidecar with the requested compression mode.
    ///
    /// # Errors
    ///
    /// Returns a store error if the file cannot be created, compressed, or
    /// serialized.
    ///
    /// # Safety
    ///
    /// This wraps epserde serialization. The resulting raw bytes, whether
    /// compressed or not, must be treated as trusted input when they are later
    /// deserialized.
    #[cfg(feature = "zstd")]
    pub unsafe fn store_with_compression_unchecked(
        &self,
        path: impl AsRef<Path>,
        compression: PersistedShardCompression,
    ) -> Result<PersistedShardStoreStats, PersistedShardStoreError> {
        let path = path.as_ref();
        let stats =
            unsafe { store_epserde_value_with_compression_unchecked(self, path, compression)? };
        write_payload_manifest(
            path,
            PersistedShardPayloadKind::TargetSmiles,
            self.format_version,
            self.base_target_id,
            self.target_count,
            compression.into(),
            stats.disk_bytes,
        )?;
        Ok(stats)
    }

    /// Memory-maps and epsilon-deserializes a trusted target-SMILES sidecar.
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
        path: impl AsRef<Path>,
        flags: deser::Flags,
    ) -> Result<deser::MemCase<Self>, Box<dyn std::error::Error + Send + Sync + 'static>> {
        unsafe { <Self as deser::Deserialize>::mmap(path, flags) }
            .map_err(|error| -> Box<dyn std::error::Error + Send + Sync + 'static> { error.into() })
    }

    /// Validates the manifest and memory-maps a raw target-SMILES sidecar.
    ///
    /// # Errors
    ///
    /// Returns an error if the manifest is missing or incompatible, if the
    /// payload byte length differs from the manifest, or if mmap-backed
    /// deserialization fails.
    pub fn mmap(
        path: impl AsRef<Path>,
        flags: deser::Flags,
    ) -> Result<deser::MemCase<Self>, Box<dyn std::error::Error + Send + Sync + 'static>> {
        let path = path.as_ref();
        let manifest = validate_payload_manifest(
            path,
            Some(PersistedShardPayloadKind::TargetSmiles),
            PersistedShardStoredCompression::None,
        )?;
        let mapped = unsafe { Self::mmap_unchecked(path, flags)? };
        validate_loaded_payload_header(
            "target SMILES",
            path,
            &manifest,
            mapped.uncase().format_version(),
            mapped.uncase().base_target_id(),
            mapped.uncase().target_count(),
        )?;
        Ok(mapped)
    }
}

impl PersistedExternalIdShard {
    /// Builds an external-id sidecar from shard-aligned ids.
    ///
    /// # Panics
    ///
    /// Panics if `base_target_id` or `external_ids.len()` exceed `u64`.
    #[cfg(feature = "zstd")]
    #[must_use]
    pub fn from_external_ids(base_target_id: usize, external_ids: &[u64]) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            base_target_id: u64::try_from(base_target_id)
                .expect("target shard base id exceeds persisted external-id capacity"),
            target_count: u64::try_from(external_ids.len())
                .expect("target shard length exceeds persisted external-id capacity"),
            external_ids: external_ids.into(),
        }
    }

    /// Returns the persisted external-id format version.
    #[inline]
    #[must_use]
    pub const fn format_version(&self) -> u32 {
        self.format_version
    }

    /// Returns the global id of local target `0` in this sidecar.
    #[inline]
    #[must_use]
    pub const fn base_target_id(&self) -> u64 {
        self.base_target_id
    }

    /// Returns the number of external ids in this sidecar.
    #[inline]
    #[must_use]
    pub const fn target_count(&self) -> u64 {
        self.target_count
    }

    /// Returns the external id for a local target id.
    #[must_use]
    pub fn external_id(&self, local_target_id: usize) -> Option<u64> {
        self.external_ids.get(local_target_id).copied()
    }

    /// Stores this sidecar with the requested compression mode.
    ///
    /// # Errors
    ///
    /// Returns a store error if the file cannot be created, compressed, or
    /// serialized.
    ///
    /// # Safety
    ///
    /// This wraps epserde serialization. The resulting raw bytes, whether
    /// compressed or not, must be treated as trusted input when they are later
    /// deserialized.
    #[cfg(feature = "zstd")]
    pub unsafe fn store_with_compression_unchecked(
        &self,
        path: impl AsRef<Path>,
        compression: PersistedShardCompression,
    ) -> Result<PersistedShardStoreStats, PersistedShardStoreError> {
        let path = path.as_ref();
        let stats =
            unsafe { store_epserde_value_with_compression_unchecked(self, path, compression)? };
        write_payload_manifest(
            path,
            PersistedShardPayloadKind::ExternalIds,
            self.format_version,
            self.base_target_id,
            self.target_count,
            compression.into(),
            stats.disk_bytes,
        )?;
        Ok(stats)
    }

    /// Memory-maps and epsilon-deserializes a trusted external-id sidecar.
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
        path: impl AsRef<Path>,
        flags: deser::Flags,
    ) -> Result<deser::MemCase<Self>, Box<dyn std::error::Error + Send + Sync + 'static>> {
        unsafe { <Self as deser::Deserialize>::mmap(path, flags) }
            .map_err(|error| -> Box<dyn std::error::Error + Send + Sync + 'static> { error.into() })
    }

    /// Validates the manifest and memory-maps a raw external-id sidecar.
    ///
    /// # Errors
    ///
    /// Returns an error if the manifest is missing or incompatible, if the
    /// payload byte length differs from the manifest, or if mmap-backed
    /// deserialization fails.
    pub fn mmap(
        path: impl AsRef<Path>,
        flags: deser::Flags,
    ) -> Result<deser::MemCase<Self>, Box<dyn std::error::Error + Send + Sync + 'static>> {
        let path = path.as_ref();
        let manifest = validate_payload_manifest(
            path,
            Some(PersistedShardPayloadKind::ExternalIds),
            PersistedShardStoredCompression::None,
        )?;
        let mapped = unsafe { Self::mmap_unchecked(path, flags)? };
        validate_loaded_payload_header(
            "external id",
            path,
            &manifest,
            mapped.uncase().format_version(),
            mapped.uncase().base_target_id(),
            mapped.uncase().target_count(),
        )?;
        Ok(mapped)
    }
}

/// Ordered raw persisted target-index shard paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedTargetCorpusIndexShardPaths {
    paths: Box<[PathBuf]>,
}

impl PersistedTargetCorpusIndexShardPaths {
    /// Builds a shard-path collection from explicit paths.
    #[must_use]
    pub fn from_paths<I, P>(paths: I) -> Self
    where
        I: IntoIterator<Item = P>,
        P: Into<PathBuf>,
    {
        let mut paths = paths.into_iter().map(Into::into).collect::<Vec<_>>();
        paths.sort();
        Self {
            paths: paths.into_boxed_slice(),
        }
    }

    /// Finds raw `.eps` shard files in a directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be read.
    pub fn raw_in_dir(dir: impl AsRef<Path>) -> std::io::Result<Self> {
        Ok(Self::from_paths(shard_paths_with_suffix(dir, ".eps")?))
    }

    /// Finds zstd-compressed `.eps.zst` shard files in a directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be read.
    #[cfg(feature = "zstd")]
    pub fn zstd_in_dir(dir: impl AsRef<Path>) -> std::io::Result<Self> {
        Ok(Self::from_paths(shard_paths_with_suffix(dir, ".eps.zst")?))
    }

    /// Returns the ordered shard paths.
    #[inline]
    #[must_use]
    pub const fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    /// Returns the number of shard paths.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.paths.len()
    }

    /// Returns whether there are no shard paths.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    /// Validates raw shard manifests as a contiguous target-id range in path order.
    ///
    /// # Errors
    ///
    /// Returns an error if any shard manifest is missing or incompatible, if a
    /// shard payload length differs from its manifest, or if consecutive
    /// manifests overlap or leave a gap in target ids.
    pub fn validate_manifests(&self) -> std::io::Result<()> {
        self.validated_target_index_manifests().map(|_| ())
    }

    /// Counts candidates across all raw shard paths after manifest validation.
    ///
    /// # Errors
    ///
    /// Returns an error if any manifest is missing or incompatible, or if any
    /// raw shard cannot be memory-mapped.
    pub fn candidate_count(
        &self,
        query: &QueryScreen,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync + 'static>> {
        self.validate_manifests()?;
        let mut count = 0usize;
        for path in &self.paths {
            count = count.saturating_add(candidate_count_for_raw_shard(path, query)?);
        }
        Ok(count)
    }

    /// Counts candidates across all raw shard paths in parallel after manifest validation.
    ///
    /// # Errors
    ///
    /// Returns an error if any manifest is missing or incompatible, or if any
    /// raw shard cannot be memory-mapped.
    #[cfg(feature = "rayon")]
    pub fn par_candidate_count(
        &self,
        query: &QueryScreen,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync + 'static>> {
        self.validate_manifests()?;
        self.paths
            .par_iter()
            .map(|path| candidate_count_for_raw_shard(path, query))
            .try_reduce(|| 0usize, |left, right| Ok(left.saturating_add(right)))
    }

    /// Collects candidates across all raw shard paths after manifest validation.
    ///
    /// # Errors
    ///
    /// Returns an error if any manifest is missing or incompatible, or if any
    /// raw shard cannot be memory-mapped.
    pub fn candidate_ids(
        &self,
        query: &QueryScreen,
    ) -> Result<Vec<usize>, Box<dyn std::error::Error + Send + Sync + 'static>> {
        self.validate_manifests()?;
        let mut out = Vec::new();
        for path in &self.paths {
            candidate_ids_for_raw_shard_into(path, query, &mut out)?;
        }
        Ok(out)
    }

    /// Collects candidates across all raw shard paths in parallel after manifest validation.
    ///
    /// # Errors
    ///
    /// Returns an error if any manifest is missing or incompatible, or if any
    /// raw shard cannot be memory-mapped.
    #[cfg(feature = "rayon")]
    pub fn par_candidate_ids(
        &self,
        query: &QueryScreen,
    ) -> Result<Vec<usize>, Box<dyn std::error::Error + Send + Sync + 'static>> {
        self.validate_manifests()?;
        let chunks = self
            .paths
            .par_iter()
            .map(|path| candidate_ids_for_raw_shard(path, query))
            .collect::<Result<Vec<_>, _>>()?;
        let target_count = chunks.iter().map(Vec::len).sum();
        let mut out = Vec::with_capacity(target_count);
        for chunk in chunks {
            out.extend(chunk);
        }
        Ok(out)
    }

    /// Collects exact SMARTS hits across all raw queryable shard paths after manifest validation.
    ///
    /// This screens each persisted index shard first, then runs exact SMARTS
    /// matching against the aligned persisted target-SMILES sidecar.
    ///
    /// # Errors
    ///
    /// Returns an error if any manifest is missing or incompatible, if any
    /// shard or target sidecar cannot be memory-mapped, if sidecar metadata
    /// does not match the index shard, or if a candidate target SMILES cannot
    /// be parsed.
    pub fn matching_hits(
        &self,
        query: &CompiledQuery,
    ) -> Result<Vec<PersistedTargetHit>, Box<dyn std::error::Error + Send + Sync + 'static>> {
        self.validate_queryable_manifests()?;
        let mut out = Vec::new();
        for path in &self.paths {
            out.extend(matching_hits_for_raw_shard(path, query)?);
        }
        Ok(out)
    }

    /// Visits exact SMARTS hits across all raw queryable shard paths after manifest validation.
    ///
    /// This screens each persisted index shard first, then runs exact SMARTS
    /// matching against the aligned persisted target-SMILES sidecar. Hits are
    /// passed to `visit` as soon as they are found, so broad queries do not
    /// need to retain every hit in memory before producing output.
    ///
    /// Returns the number of exact hits visited.
    ///
    /// # Errors
    ///
    /// Returns an error if any manifest is missing or incompatible, if any
    /// shard or target sidecar cannot be memory-mapped, if sidecar metadata
    /// does not match the index shard, if a candidate target SMILES cannot be
    /// parsed, or if `visit` returns an error.
    pub fn try_for_each_matching_hit(
        &self,
        query: &CompiledQuery,
        mut visit: impl FnMut(
            PersistedTargetHit,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>>,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync + 'static>> {
        self.validate_queryable_manifests()?;
        let mut hit_count = 0usize;
        for path in &self.paths {
            hit_count = hit_count
                .saturating_add(matching_hits_for_raw_shard_with(path, query, &mut visit)?);
        }
        Ok(hit_count)
    }

    /// Visits exact SMARTS hits with their target SMILES across all raw queryable shard paths.
    ///
    /// This is the streaming form to use when callers need the matched SMILES
    /// payload without collecting all hits or re-reading sidecars externally.
    /// The `target_smiles` field passed to `visit` is borrowed from the mapped
    /// shard sidecar and is valid only for that callback invocation.
    ///
    /// Returns the number of exact hits visited.
    ///
    /// # Errors
    ///
    /// Returns an error if any manifest is missing or incompatible, if any
    /// shard or target sidecar cannot be memory-mapped, if sidecar metadata
    /// does not match the index shard, if a candidate target SMILES cannot be
    /// parsed, or if `visit` returns an error.
    pub fn try_for_each_matching_record_hit(
        &self,
        query: &CompiledQuery,
        mut visit: impl for<'a> FnMut(
            PersistedTargetRecordHit<'a>,
        )
            -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>>,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync + 'static>> {
        self.try_for_each_matching_record_hit_with_target_smiles_parse_policy(
            query,
            PersistedTargetSmilesParsePolicy::Error,
            &mut visit,
        )
    }

    /// Visits exact SMARTS hits with their target SMILES and a target-parse policy.
    ///
    /// This is equivalent to [`Self::try_for_each_matching_record_hit`], but
    /// lets callers opt into skipping wildcard target SMILES when querying
    /// trusted shards built from upstream corpora that contain those records.
    ///
    /// Returns the number of exact hits visited.
    ///
    /// # Errors
    ///
    /// Returns an error if any manifest is missing or incompatible, if any
    /// shard or target sidecar cannot be memory-mapped, if sidecar metadata
    /// does not match the index shard, if a candidate target SMILES cannot be
    /// parsed under `target_smiles_parse_policy`, or if `visit` returns an
    /// error.
    pub fn try_for_each_matching_record_hit_with_target_smiles_parse_policy(
        &self,
        query: &CompiledQuery,
        target_smiles_parse_policy: PersistedTargetSmilesParsePolicy,
        mut visit: impl for<'a> FnMut(
            PersistedTargetRecordHit<'a>,
        )
            -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>>,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync + 'static>> {
        self.validate_queryable_manifests()?;
        let mut hit_count = 0usize;
        for path in &self.paths {
            hit_count = hit_count.saturating_add(matching_record_hits_for_raw_shard_with(
                path,
                query,
                target_smiles_parse_policy,
                &mut visit,
            )?);
        }
        Ok(hit_count)
    }

    /// Collects exact SMARTS hits across all raw queryable shard paths in parallel after manifest validation.
    ///
    /// This screens each persisted index shard first, then runs exact SMARTS
    /// matching against the aligned persisted target-SMILES sidecar.
    ///
    /// # Errors
    ///
    /// Returns an error if any manifest is missing or incompatible, if any
    /// shard or target sidecar cannot be memory-mapped, if sidecar metadata
    /// does not match the index shard, or if a candidate target SMILES cannot
    /// be parsed.
    #[cfg(feature = "rayon")]
    pub fn par_matching_hits(
        &self,
        query: &CompiledQuery,
    ) -> Result<Vec<PersistedTargetHit>, Box<dyn std::error::Error + Send + Sync + 'static>> {
        self.validate_queryable_manifests()?;
        let chunks = self
            .paths
            .par_iter()
            .map(|path| matching_hits_for_raw_shard(path, query))
            .collect::<Result<Vec<_>, _>>()?;
        let target_count = chunks.iter().map(Vec::len).sum();
        let mut out = Vec::with_capacity(target_count);
        for chunk in chunks {
            out.extend(chunk);
        }
        Ok(out)
    }

    /// Counts candidates across all raw shard paths.
    ///
    /// # Errors
    ///
    /// Returns an error if any shard cannot be memory-mapped.
    ///
    /// # Safety
    ///
    /// Raw shard files must be trusted epserde payloads produced by a
    /// compatible version of this crate and epserde.
    pub unsafe fn candidate_count_unchecked(
        &self,
        query: &QueryScreen,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync + 'static>> {
        let mut count = 0usize;
        for path in &self.paths {
            count = count
                .saturating_add(unsafe { candidate_count_for_raw_shard_unchecked(path, query)? });
        }
        Ok(count)
    }

    /// Counts candidates across all raw shard paths in parallel.
    ///
    /// # Errors
    ///
    /// Returns an error if any shard cannot be memory-mapped.
    ///
    /// # Safety
    ///
    /// Raw shard files must be trusted epserde payloads produced by a
    /// compatible version of this crate and epserde.
    #[cfg(feature = "rayon")]
    pub unsafe fn par_candidate_count_unchecked(
        &self,
        query: &QueryScreen,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync + 'static>> {
        self.paths
            .par_iter()
            .map(|path| unsafe { candidate_count_for_raw_shard_unchecked(path, query) })
            .try_reduce(|| 0usize, |left, right| Ok(left.saturating_add(right)))
    }

    /// Collects candidates across all raw shard paths.
    ///
    /// # Errors
    ///
    /// Returns an error if any shard cannot be memory-mapped.
    ///
    /// # Safety
    ///
    /// Raw shard files must be trusted epserde payloads produced by a
    /// compatible version of this crate and epserde.
    pub unsafe fn candidate_ids_unchecked(
        &self,
        query: &QueryScreen,
    ) -> Result<Vec<usize>, Box<dyn std::error::Error + Send + Sync + 'static>> {
        let mut out = Vec::new();
        for path in &self.paths {
            unsafe { candidate_ids_for_raw_shard_into_unchecked(path, query, &mut out)? };
        }
        Ok(out)
    }

    /// Collects exact SMARTS hits across all raw queryable shard paths.
    ///
    /// This screens each persisted index shard first, then runs exact SMARTS
    /// matching against the aligned persisted target-SMILES sidecar.
    ///
    /// # Errors
    ///
    /// Returns an error if any shard or target sidecar cannot be memory-mapped,
    /// if sidecar metadata does not match the index shard, or if a candidate
    /// target SMILES cannot be parsed.
    ///
    /// # Safety
    ///
    /// Raw shard and sidecar files must be trusted epserde payloads produced by
    /// a compatible version of this crate and epserde.
    pub unsafe fn matching_hits_unchecked(
        &self,
        query: &CompiledQuery,
    ) -> Result<Vec<PersistedTargetHit>, Box<dyn std::error::Error + Send + Sync + 'static>> {
        let mut out = Vec::new();
        for path in &self.paths {
            out.extend(unsafe { matching_hits_for_raw_shard_unchecked(path, query)? });
        }
        Ok(out)
    }

    /// Visits exact SMARTS hits across all raw queryable shard paths.
    ///
    /// This screens each persisted index shard first, then runs exact SMARTS
    /// matching against the aligned persisted target-SMILES sidecar. Hits are
    /// passed to `visit` as soon as they are found, so broad queries do not
    /// need to retain every hit in memory before producing output.
    ///
    /// Returns the number of exact hits visited.
    ///
    /// # Errors
    ///
    /// Returns an error if any shard or target sidecar cannot be memory-mapped,
    /// if sidecar metadata does not match the index shard, if a candidate
    /// target SMILES cannot be parsed, or if `visit` returns an error.
    ///
    /// # Safety
    ///
    /// Raw shard and sidecar files must be trusted epserde payloads produced by
    /// a compatible version of this crate and epserde.
    pub unsafe fn try_for_each_matching_hit_unchecked(
        &self,
        query: &CompiledQuery,
        mut visit: impl FnMut(
            PersistedTargetHit,
        ) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>>,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync + 'static>> {
        let mut hit_count = 0usize;
        for path in &self.paths {
            hit_count = hit_count.saturating_add(unsafe {
                matching_hits_for_raw_shard_unchecked_with(path, query, &mut visit)?
            });
        }
        Ok(hit_count)
    }

    /// Visits exact SMARTS hits with their target SMILES across all raw queryable shard paths.
    ///
    /// This is the streaming form to use when callers need the matched SMILES
    /// payload without collecting all hits or re-reading sidecars externally.
    /// The `target_smiles` field passed to `visit` is borrowed from the mapped
    /// shard sidecar and is valid only for that callback invocation.
    ///
    /// Returns the number of exact hits visited.
    ///
    /// # Errors
    ///
    /// Returns an error if any shard or target sidecar cannot be memory-mapped,
    /// if sidecar metadata does not match the index shard, if a candidate
    /// target SMILES cannot be parsed, or if `visit` returns an error.
    ///
    /// # Safety
    ///
    /// Raw shard and sidecar files must be trusted epserde payloads produced by
    /// a compatible version of this crate and epserde.
    pub unsafe fn try_for_each_matching_record_hit_unchecked(
        &self,
        query: &CompiledQuery,
        mut visit: impl for<'a> FnMut(
            PersistedTargetRecordHit<'a>,
        )
            -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>>,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync + 'static>> {
        unsafe {
            self.try_for_each_matching_record_hit_unchecked_with_target_smiles_parse_policy(
                query,
                PersistedTargetSmilesParsePolicy::Error,
                &mut visit,
            )
        }
    }

    /// Visits exact SMARTS hits with their target SMILES and a target-parse policy.
    ///
    /// This is equivalent to [`Self::try_for_each_matching_record_hit_unchecked`],
    /// but lets callers opt into skipping wildcard target SMILES when querying
    /// trusted shards built from upstream corpora that contain those records.
    ///
    /// Returns the number of exact hits visited.
    ///
    /// # Errors
    ///
    /// Returns an error if any shard or target sidecar cannot be memory-mapped,
    /// if sidecar metadata does not match the index shard, if a candidate
    /// target SMILES cannot be parsed under `target_smiles_parse_policy`, or if
    /// `visit` returns an error.
    ///
    /// # Safety
    ///
    /// Raw shard and sidecar files must be trusted epserde payloads produced by
    /// a compatible version of this crate and epserde.
    pub unsafe fn try_for_each_matching_record_hit_unchecked_with_target_smiles_parse_policy(
        &self,
        query: &CompiledQuery,
        target_smiles_parse_policy: PersistedTargetSmilesParsePolicy,
        mut visit: impl for<'a> FnMut(
            PersistedTargetRecordHit<'a>,
        )
            -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>>,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync + 'static>> {
        let mut hit_count = 0usize;
        for path in &self.paths {
            hit_count = hit_count.saturating_add(unsafe {
                matching_record_hits_for_raw_shard_unchecked_with(
                    path,
                    query,
                    target_smiles_parse_policy,
                    &mut visit,
                )?
            });
        }
        Ok(hit_count)
    }

    /// Collects candidates across all raw shard paths in parallel.
    ///
    /// # Errors
    ///
    /// Returns an error if any shard cannot be memory-mapped.
    ///
    /// # Safety
    ///
    /// Raw shard files must be trusted epserde payloads produced by a
    /// compatible version of this crate and epserde.
    #[cfg(feature = "rayon")]
    pub unsafe fn par_candidate_ids_unchecked(
        &self,
        query: &QueryScreen,
    ) -> Result<Vec<usize>, Box<dyn std::error::Error + Send + Sync + 'static>> {
        let chunks = self
            .paths
            .par_iter()
            .map(|path| unsafe { candidate_ids_for_raw_shard_unchecked(path, query) })
            .collect::<Result<Vec<_>, _>>()?;
        let target_count = chunks.iter().map(Vec::len).sum();
        let mut out = Vec::with_capacity(target_count);
        for chunk in chunks {
            out.extend(chunk);
        }
        Ok(out)
    }

    /// Collects exact SMARTS hits across all raw queryable shard paths in parallel.
    ///
    /// This screens each persisted index shard first, then runs exact SMARTS
    /// matching against the aligned persisted target-SMILES sidecar.
    ///
    /// # Errors
    ///
    /// Returns an error if any shard or target sidecar cannot be memory-mapped,
    /// if sidecar metadata does not match the index shard, or if a candidate
    /// target SMILES cannot be parsed.
    ///
    /// # Safety
    ///
    /// Raw shard and sidecar files must be trusted epserde payloads produced by
    /// a compatible version of this crate and epserde.
    #[cfg(feature = "rayon")]
    pub unsafe fn par_matching_hits_unchecked(
        &self,
        query: &CompiledQuery,
    ) -> Result<Vec<PersistedTargetHit>, Box<dyn std::error::Error + Send + Sync + 'static>> {
        let chunks = self
            .paths
            .par_iter()
            .map(|path| unsafe { matching_hits_for_raw_shard_unchecked(path, query) })
            .collect::<Result<Vec<_>, _>>()?;
        let target_count = chunks.iter().map(Vec::len).sum();
        let mut out = Vec::with_capacity(target_count);
        for chunk in chunks {
            out.extend(chunk);
        }
        Ok(out)
    }

    /// Extracts `.eps.zst` shards to adjacent raw `.eps` files when needed.
    ///
    /// # Errors
    ///
    /// Returns an error if any shard cannot be decompressed or written.
    #[cfg(all(feature = "zstd", feature = "rayon"))]
    pub fn extract_zstd_if_missing_in_parallel(
        &self,
    ) -> Result<Vec<PersistedShardExtraction>, PersistedShardStoreError> {
        self.paths
            .par_iter()
            .map(extract_zstd_shard_if_missing)
            .collect()
    }

    /// Extracts compressed queryable shard payloads to adjacent raw files when needed.
    ///
    /// This extracts each index shard, its required target-SMILES sidecar, and
    /// its optional external-id sidecar when the compressed external-id sidecar
    /// exists.
    ///
    /// # Errors
    ///
    /// Returns an error if any required shard or sidecar cannot be decompressed
    /// or written.
    #[cfg(all(feature = "zstd", feature = "rayon"))]
    pub fn extract_queryable_zstd_if_missing_in_parallel(
        &self,
    ) -> Result<Vec<PersistedShardExtraction>, PersistedShardStoreError> {
        let mut paths = Vec::with_capacity(self.paths.len().saturating_mul(3));
        for path in &self.paths {
            paths.push(path.clone());
            paths.push(persisted_target_smiles_path_for_index_shard_path(path));
            let external_ids_path = persisted_external_ids_path_for_index_shard_path(path);
            if external_ids_path.is_file() {
                paths.push(external_ids_path);
            }
        }
        paths
            .par_iter()
            .map(extract_zstd_shard_if_missing)
            .collect()
    }

    /// Returns the raw `.eps` paths corresponding to these `.eps.zst` paths.
    #[cfg(feature = "zstd")]
    #[must_use]
    pub fn extracted_raw_paths(&self) -> Self {
        Self::from_paths(self.paths.iter().map(raw_path_for_zstd_shard))
    }

    /// Validates and memory-maps all raw shard paths for repeated queries.
    ///
    /// # Errors
    ///
    /// Returns an error if any manifest is missing or incompatible, or if any
    /// shard cannot be memory-mapped.
    pub fn mmap(
        &self,
    ) -> Result<
        MappedPersistedTargetCorpusIndexShards,
        Box<dyn std::error::Error + Send + Sync + 'static>,
    > {
        self.validate_manifests()?;
        MappedPersistedTargetCorpusIndexShards::mmap(self)
    }

    /// Memory-maps all raw shard paths for repeated queries.
    ///
    /// # Errors
    ///
    /// Returns an error if any shard cannot be memory-mapped.
    ///
    /// # Safety
    ///
    /// Raw shard files must be trusted epserde payloads produced by a
    /// compatible version of this crate and epserde.
    pub unsafe fn mmap_unchecked(
        &self,
    ) -> Result<
        MappedPersistedTargetCorpusIndexShards,
        Box<dyn std::error::Error + Send + Sync + 'static>,
    > {
        unsafe { MappedPersistedTargetCorpusIndexShards::mmap_unchecked(self) }
    }

    fn validate_queryable_manifests(&self) -> std::io::Result<()> {
        let index_manifests = self.validated_target_index_manifests()?;
        for (path, index_manifest) in self.paths.iter().zip(index_manifests) {
            let smiles_path = persisted_target_smiles_path_for_index_shard_path(path);
            let smiles_manifest = validate_payload_manifest(
                &smiles_path,
                Some(PersistedShardPayloadKind::TargetSmiles),
                PersistedShardStoredCompression::None,
            )?;
            validate_sidecar_manifest_header(
                "target SMILES",
                &smiles_path,
                &smiles_manifest,
                path,
                &index_manifest,
            )?;

            let external_ids_path = persisted_external_ids_path_for_index_shard_path(path);
            if external_ids_path.is_file() {
                let external_ids_manifest = validate_payload_manifest(
                    &external_ids_path,
                    Some(PersistedShardPayloadKind::ExternalIds),
                    PersistedShardStoredCompression::None,
                )?;
                validate_sidecar_manifest_header(
                    "external id",
                    &external_ids_path,
                    &external_ids_manifest,
                    path,
                    &index_manifest,
                )?;
            }
        }
        Ok(())
    }

    fn validated_target_index_manifests(&self) -> std::io::Result<Vec<PersistedShardManifest>> {
        let mut manifests = Vec::with_capacity(self.paths.len());
        let mut expected_base_target_id = None;
        for path in &self.paths {
            let manifest = validate_payload_manifest(
                path,
                Some(PersistedShardPayloadKind::TargetIndex),
                PersistedShardStoredCompression::None,
            )?;
            if let Some(expected_base_target_id) = expected_base_target_id {
                validate_next_shard_base(path, &manifest, expected_base_target_id)?;
            }
            expected_base_target_id = Some(
                manifest
                    .base_target_id
                    .checked_add(manifest.target_count)
                    .ok_or_else(|| {
                        invalid_manifest_data(format!(
                            "persisted shard target range overflows u64 at {}",
                            path.display()
                        ))
                    })?,
            );
            manifests.push(manifest);
        }
        Ok(manifests)
    }
}

/// Reusable memory-mapped persisted target-index shards.
///
/// Use this for workloads that run many SMARTS queries against the same
/// persisted corpus. It keeps the epserde mappings open and avoids remapping
/// shard files for every query.
#[derive(Debug)]
pub struct MappedPersistedTargetCorpusIndexShards {
    shards: Box<[deser::MemCase<PersistedTargetCorpusIndexShard>]>,
}

impl MappedPersistedTargetCorpusIndexShards {
    /// Validates and memory-maps all raw shard paths.
    ///
    /// # Errors
    ///
    /// Returns an error if any manifest is missing or incompatible, or if any
    /// shard cannot be memory-mapped.
    pub fn mmap(
        paths: &PersistedTargetCorpusIndexShardPaths,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync + 'static>> {
        let shards = paths
            .paths()
            .iter()
            .map(|path| PersistedTargetCorpusIndexShard::mmap(path, deser::Flags::empty()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            shards: shards.into_boxed_slice(),
        })
    }

    /// Memory-maps all raw shard paths.
    ///
    /// # Errors
    ///
    /// Returns an error if any shard cannot be memory-mapped.
    ///
    /// # Safety
    ///
    /// Raw shard files must be trusted epserde payloads produced by a
    /// compatible version of this crate and epserde.
    pub unsafe fn mmap_unchecked(
        paths: &PersistedTargetCorpusIndexShardPaths,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync + 'static>> {
        let shards = paths
            .paths()
            .iter()
            .map(|path| unsafe {
                PersistedTargetCorpusIndexShard::mmap_unchecked(path, deser::Flags::empty())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            shards: shards.into_boxed_slice(),
        })
    }

    /// Returns the number of mapped shards.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.shards.len()
    }

    /// Returns whether there are no mapped shards.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.shards.is_empty()
    }

    /// Returns the mapped shards.
    #[inline]
    #[must_use]
    pub const fn shards(&self) -> &[deser::MemCase<PersistedTargetCorpusIndexShard>] {
        &self.shards
    }

    /// Counts candidates across the mapped shards.
    #[must_use]
    pub fn candidate_count(&self, query: &QueryScreen) -> usize {
        self.shards
            .iter()
            .map(|shard| shard.uncase().candidate_count(query))
            .sum()
    }

    /// Counts candidates across the mapped shards in parallel.
    #[cfg(feature = "rayon")]
    #[must_use]
    pub fn par_candidate_count(&self, query: &QueryScreen) -> usize {
        self.shards
            .par_iter()
            .map(|shard| shard.uncase().candidate_count(query))
            .sum()
    }

    /// Collects global candidate target ids across the mapped shards.
    #[must_use]
    pub fn candidate_ids(&self, query: &QueryScreen) -> Vec<usize> {
        let mut out = Vec::new();
        for shard in &self.shards {
            let mut shard_ids = Vec::new();
            shard.uncase().candidate_ids_into(query, &mut shard_ids);
            out.extend(shard_ids);
        }
        out
    }

    /// Collects global candidate target ids across the mapped shards in parallel.
    #[cfg(feature = "rayon")]
    #[must_use]
    pub fn par_candidate_ids(&self, query: &QueryScreen) -> Vec<usize> {
        let chunks = self
            .shards
            .par_iter()
            .map(|shard| shard.uncase().candidate_ids(query))
            .collect::<Vec<_>>();
        let target_count = chunks.iter().map(Vec::len).sum();
        let mut out = Vec::with_capacity(target_count);
        for chunk in chunks {
            out.extend(chunk);
        }
        out
    }
}

fn shard_paths_with_suffix(dir: impl AsRef<Path>, suffix: &str) -> std::io::Result<Vec<PathBuf>> {
    let mut paths = std::fs::read_dir(dir)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    paths.retain(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| {
                name.ends_with(suffix)
                    && !name.contains(".target-smiles.")
                    && !name.contains(".external-ids.")
            })
    });
    paths.sort();
    Ok(paths)
}

/// Returns the manifest sidecar path for a persisted payload path.
///
/// The manifest is intentionally outside the epserde payload so callers can
/// validate the payload kind, format version, compression mode, target range,
/// and byte length before using mmap-backed epserde deserialization.
#[must_use]
pub fn persisted_manifest_path_for_payload_path(path: impl AsRef<Path>) -> PathBuf {
    let path = path.as_ref();
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    path.with_file_name(format!("{file_name}{MANIFEST_SUFFIX}"))
}

/// Returns the target-SMILES sidecar path for an index shard path.
#[must_use]
pub fn persisted_target_smiles_path_for_index_shard_path(path: impl AsRef<Path>) -> PathBuf {
    persisted_sidecar_path_for_index_shard_path(path, TARGET_SMILES_SHARD_SUFFIX)
}

/// Returns the external-id sidecar path for an index shard path.
#[must_use]
pub fn persisted_external_ids_path_for_index_shard_path(path: impl AsRef<Path>) -> PathBuf {
    persisted_sidecar_path_for_index_shard_path(path, EXTERNAL_IDS_SHARD_SUFFIX)
}

fn persisted_sidecar_path_for_index_shard_path(path: impl AsRef<Path>, suffix: &str) -> PathBuf {
    let path = path.as_ref();
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    if let Some(stem) = file_name.strip_suffix(".eps.zst") {
        return path.with_file_name(format!("{stem}{suffix}.zst"));
    }
    if let Some(stem) = file_name.strip_suffix(".eps") {
        return path.with_file_name(format!("{stem}{suffix}"));
    }
    path.with_file_name(format!("{file_name}{suffix}"))
}

fn candidate_count_for_raw_shard(
    path: impl AsRef<Path>,
    query: &QueryScreen,
) -> Result<usize, Box<dyn std::error::Error + Send + Sync + 'static>> {
    let mapped = PersistedTargetCorpusIndexShard::mmap(path, deser::Flags::empty())?;
    let mut scratch = TargetCorpusScratch::new();
    Ok(mapped
        .uncase()
        .candidate_count_with_scratch(query, &mut scratch))
}

unsafe fn candidate_count_for_raw_shard_unchecked(
    path: impl AsRef<Path>,
    query: &QueryScreen,
) -> Result<usize, Box<dyn std::error::Error + Send + Sync + 'static>> {
    let mapped =
        unsafe { PersistedTargetCorpusIndexShard::mmap_unchecked(path, deser::Flags::empty())? };
    let mut scratch = TargetCorpusScratch::new();
    Ok(mapped
        .uncase()
        .candidate_count_with_scratch(query, &mut scratch))
}

fn candidate_ids_for_raw_shard(
    path: impl AsRef<Path>,
    query: &QueryScreen,
) -> Result<Vec<usize>, Box<dyn std::error::Error + Send + Sync + 'static>> {
    let mut out = Vec::new();
    candidate_ids_for_raw_shard_into(path, query, &mut out)?;
    Ok(out)
}

fn candidate_ids_for_raw_shard_into(
    path: impl AsRef<Path>,
    query: &QueryScreen,
    out: &mut Vec<usize>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
    let mapped = PersistedTargetCorpusIndexShard::mmap(path, deser::Flags::empty())?;
    let mut scratch = TargetCorpusScratch::new();
    let mut shard_ids = Vec::new();
    mapped
        .uncase()
        .candidate_ids_with_scratch_into(query, &mut scratch, &mut shard_ids);
    out.extend(shard_ids);
    Ok(())
}

unsafe fn candidate_ids_for_raw_shard_unchecked(
    path: impl AsRef<Path>,
    query: &QueryScreen,
) -> Result<Vec<usize>, Box<dyn std::error::Error + Send + Sync + 'static>> {
    let mut out = Vec::new();
    unsafe { candidate_ids_for_raw_shard_into_unchecked(path, query, &mut out)? };
    Ok(out)
}

unsafe fn candidate_ids_for_raw_shard_into_unchecked(
    path: impl AsRef<Path>,
    query: &QueryScreen,
    out: &mut Vec<usize>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
    let mapped =
        unsafe { PersistedTargetCorpusIndexShard::mmap_unchecked(path, deser::Flags::empty())? };
    let mut scratch = TargetCorpusScratch::new();
    let mut shard_ids = Vec::new();
    mapped
        .uncase()
        .candidate_ids_with_scratch_into(query, &mut scratch, &mut shard_ids);
    out.extend(shard_ids);
    Ok(())
}

fn matching_hits_for_raw_shard(
    path: impl AsRef<Path>,
    query: &CompiledQuery,
) -> Result<Vec<PersistedTargetHit>, Box<dyn std::error::Error + Send + Sync + 'static>> {
    let mut hits = Vec::new();
    matching_hits_for_raw_shard_with(path, query, &mut |hit| {
        hits.push(hit);
        Ok(())
    })?;
    Ok(hits)
}

fn matching_hits_for_raw_shard_with(
    path: impl AsRef<Path>,
    query: &CompiledQuery,
    visit: &mut impl FnMut(
        PersistedTargetHit,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>>,
) -> Result<usize, Box<dyn std::error::Error + Send + Sync + 'static>> {
    matching_record_hits_for_raw_shard_with(
        path,
        query,
        PersistedTargetSmilesParsePolicy::Error,
        &mut |hit| {
            visit(PersistedTargetHit {
                target_id: hit.target_id,
                external_id: hit.external_id,
            })
        },
    )
}

fn matching_record_hits_for_raw_shard_with(
    path: impl AsRef<Path>,
    query: &CompiledQuery,
    target_smiles_parse_policy: PersistedTargetSmilesParsePolicy,
    visit: &mut impl for<'a> FnMut(
        PersistedTargetRecordHit<'a>,
    )
        -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>>,
) -> Result<usize, Box<dyn std::error::Error + Send + Sync + 'static>> {
    let path = path.as_ref();
    let mapped_index = PersistedTargetCorpusIndexShard::mmap(path, deser::Flags::empty())?;
    let smiles_path = persisted_target_smiles_path_for_index_shard_path(path);
    let mapped_smiles = PersistedTargetSmilesShard::mmap(&smiles_path, deser::Flags::empty())?;
    let external_ids_path = persisted_external_ids_path_for_index_shard_path(path);
    let mapped_external_ids = if external_ids_path.is_file() {
        Some(PersistedExternalIdShard::mmap(
            &external_ids_path,
            deser::Flags::empty(),
        )?)
    } else {
        None
    };
    matching_record_hits_for_loaded_shard_with(
        path,
        mapped_index.uncase(),
        mapped_smiles.uncase(),
        mapped_external_ids.as_ref().map(deser::MemCase::uncase),
        query,
        target_smiles_parse_policy,
        visit,
    )
}

unsafe fn matching_hits_for_raw_shard_unchecked(
    path: impl AsRef<Path>,
    query: &CompiledQuery,
) -> Result<Vec<PersistedTargetHit>, Box<dyn std::error::Error + Send + Sync + 'static>> {
    let mut hits = Vec::new();
    unsafe {
        matching_hits_for_raw_shard_unchecked_with(path, query, &mut |hit| {
            hits.push(hit);
            Ok(())
        })?;
    }
    Ok(hits)
}

unsafe fn matching_hits_for_raw_shard_unchecked_with(
    path: impl AsRef<Path>,
    query: &CompiledQuery,
    visit: &mut impl FnMut(
        PersistedTargetHit,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>>,
) -> Result<usize, Box<dyn std::error::Error + Send + Sync + 'static>> {
    unsafe {
        matching_record_hits_for_raw_shard_unchecked_with(
            path,
            query,
            PersistedTargetSmilesParsePolicy::Error,
            &mut |hit| {
                visit(PersistedTargetHit {
                    target_id: hit.target_id,
                    external_id: hit.external_id,
                })
            },
        )
    }
}

unsafe fn matching_record_hits_for_raw_shard_unchecked_with(
    path: impl AsRef<Path>,
    query: &CompiledQuery,
    target_smiles_parse_policy: PersistedTargetSmilesParsePolicy,
    visit: &mut impl for<'a> FnMut(
        PersistedTargetRecordHit<'a>,
    )
        -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>>,
) -> Result<usize, Box<dyn std::error::Error + Send + Sync + 'static>> {
    let path = path.as_ref();
    let mapped_index =
        unsafe { PersistedTargetCorpusIndexShard::mmap_unchecked(path, deser::Flags::empty())? };
    let smiles_path = persisted_target_smiles_path_for_index_shard_path(path);
    let mapped_smiles =
        unsafe { PersistedTargetSmilesShard::mmap_unchecked(&smiles_path, deser::Flags::empty())? };
    let external_ids_path = persisted_external_ids_path_for_index_shard_path(path);
    let mapped_external_ids = if external_ids_path.is_file() {
        Some(unsafe {
            PersistedExternalIdShard::mmap_unchecked(&external_ids_path, deser::Flags::empty())?
        })
    } else {
        None
    };
    matching_record_hits_for_loaded_shard_with(
        path,
        mapped_index.uncase(),
        mapped_smiles.uncase(),
        mapped_external_ids.as_ref().map(deser::MemCase::uncase),
        query,
        target_smiles_parse_policy,
        visit,
    )
}

fn matching_record_hits_for_loaded_shard_with<
    ScalarCounts,
    AtomPropertyCounts,
    Edge,
    Path3,
    Path4,
    Star3,
>(
    path: &Path,
    index: &PersistedTargetCorpusIndexShard<
        ScalarCounts,
        AtomPropertyCounts,
        Edge,
        Path3,
        Path4,
        Star3,
    >,
    smiles: &PersistedTargetSmilesShard,
    external_ids: Option<&PersistedExternalIdShard>,
    query: &CompiledQuery,
    target_smiles_parse_policy: PersistedTargetSmilesParsePolicy,
    visit: &mut impl for<'a> FnMut(
        PersistedTargetRecordHit<'a>,
    )
        -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>>,
) -> Result<usize, Box<dyn std::error::Error + Send + Sync + 'static>>
where
    ScalarCounts: PersistedScalarCountIndexesAccess,
    AtomPropertyCounts: PersistedAtomPropertyCountIndexesAccess,
    Edge: PersistedLocalFeatureFamilyAccess,
    Path3: PersistedLocalFeatureFamilyAccess,
    Path4: PersistedLocalFeatureFamilyAccess,
    Star3: PersistedLocalFeatureFamilyAccess,
{
    let base_target_id = index
        .base_target_id_usize()
        .ok_or_else(|| invalid_data_error("persisted shard base target id exceeds usize"))?;
    let target_count = index
        .target_count_usize()
        .ok_or_else(|| invalid_data_error("persisted shard target count exceeds usize"))?;
    let mut scratch = TargetCorpusScratch::new();
    let mut candidate_ids = Vec::new();
    index.candidate_ids_with_scratch_into(
        &QueryScreen::new(query.query()),
        &mut scratch,
        &mut candidate_ids,
    );

    validate_query_sidecar_header(
        "target SMILES",
        path,
        smiles.base_target_id(),
        smiles.target_count(),
        index.base_target_id(),
        index.target_count(),
    )?;
    if let Some(external_ids) = external_ids {
        validate_query_sidecar_header(
            "external id",
            path,
            external_ids.base_target_id(),
            external_ids.target_count(),
            index.base_target_id(),
            index.target_count(),
        )?;
    }

    let mut match_scratch = MatchScratch::new();
    let mut hit_count = 0usize;
    for target_id in candidate_ids {
        let local_target_id = target_id.checked_sub(base_target_id).ok_or_else(|| {
            invalid_data_error(format!(
                "candidate target id {target_id} is below shard base target id {base_target_id}"
            ))
        })?;
        if local_target_id >= target_count {
            return Err(invalid_data_error(format!(
                "candidate target id {target_id} is outside shard local range 0..{target_count}"
            )));
        }
        let target_smiles = smiles.target_smiles(local_target_id).ok_or_else(|| {
            invalid_data_error(format!(
                "missing target SMILES for target id {target_id} in {}",
                persisted_target_smiles_path_for_index_shard_path(path).display()
            ))
        })?;
        let target = match Smiles::from_str(target_smiles) {
            Ok(smiles) => PreparedTarget::new(smiles),
            Err(error)
                if target_smiles_parse_policy == PersistedTargetSmilesParsePolicy::SkipWildcard
                    && error.smiles_error() == SmilesError::WildcardAtomNotAllowed =>
            {
                continue;
            }
            Err(error) => {
                return Err(invalid_data_error(format!(
                    "persisted target SMILES for target id {target_id} failed to parse: {error}"
                )));
            }
        };
        if query.matches_with_scratch(&target, &mut match_scratch) {
            visit(PersistedTargetRecordHit {
                target_id,
                target_smiles,
                external_id: external_ids.and_then(|ids| ids.external_id(local_target_id)),
            })?;
            hit_count = hit_count.saturating_add(1);
        }
    }
    Ok(hit_count)
}

fn validate_query_sidecar_header(
    sidecar_name: &str,
    index_path: &Path,
    sidecar_base_target_id: u64,
    sidecar_target_count: u64,
    index_base_target_id: u64,
    index_target_count: u64,
) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
    if sidecar_base_target_id == index_base_target_id && sidecar_target_count == index_target_count
    {
        return Ok(());
    }
    Err(invalid_data_error(format!(
        "{sidecar_name} sidecar metadata does not match index shard {}: sidecar base={}, len={}; index base={}, len={}",
        index_path.display(),
        sidecar_base_target_id,
        sidecar_target_count,
        index_base_target_id,
        index_target_count
    )))
}

fn validate_loaded_payload_header(
    payload_name: &str,
    path: &Path,
    manifest: &PersistedShardManifest,
    format_version: u32,
    base_target_id: u64,
    target_count: u64,
) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
    if format_version != manifest.format_version {
        return Err(invalid_data_error(format!(
            "{payload_name} payload {} has format version {}, manifest records {}",
            path.display(),
            format_version,
            manifest.format_version
        )));
    }
    if base_target_id != manifest.base_target_id || target_count != manifest.target_count {
        return Err(invalid_data_error(format!(
            "{payload_name} payload {} metadata does not match manifest: payload base={}, len={}; manifest base={}, len={}",
            path.display(),
            base_target_id,
            target_count,
            manifest.base_target_id,
            manifest.target_count
        )));
    }
    Ok(())
}

fn invalid_data_error(
    message: impl Into<String>,
) -> Box<dyn std::error::Error + Send + Sync + 'static> {
    Box::new(IoError::new(ErrorKind::InvalidData, message.into()))
}

#[cfg(feature = "zstd")]
fn extract_zstd_shard_if_missing(
    source: impl AsRef<Path>,
) -> Result<PersistedShardExtraction, PersistedShardStoreError> {
    let source = source.as_ref();
    let source_manifest =
        validate_payload_manifest(source, None, PersistedShardStoredCompression::Zstd)?;
    let destination = raw_path_for_zstd_shard(source);
    if destination.exists() {
        let disk_bytes = std::fs::metadata(&destination)?.len();
        write_payload_manifest(
            &destination,
            source_manifest.payload_kind,
            source_manifest.format_version,
            source_manifest.base_target_id,
            source_manifest.target_count,
            PersistedShardStoredCompression::None,
            disk_bytes,
        )?;
        return Ok(PersistedShardExtraction {
            source: source.to_path_buf(),
            destination,
            extracted: false,
            disk_bytes,
        });
    }

    let input = File::open(source)?;
    let mut decoder = zstd::stream::read::Decoder::new(input)?;
    let mut output = BufWriter::new(File::create(&destination)?);
    let disk_bytes = std::io::copy(&mut decoder, &mut output)?;
    output.flush()?;
    write_payload_manifest(
        &destination,
        source_manifest.payload_kind,
        source_manifest.format_version,
        source_manifest.base_target_id,
        source_manifest.target_count,
        PersistedShardStoredCompression::None,
        disk_bytes,
    )?;
    Ok(PersistedShardExtraction {
        source: source.to_path_buf(),
        destination,
        extracted: true,
        disk_bytes,
    })
}

#[cfg(feature = "zstd")]
fn raw_path_for_zstd_shard(path: impl AsRef<Path>) -> PathBuf {
    let path = path.as_ref();
    path.file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_suffix(".zst"))
        .map_or_else(
            || path.with_extension("eps"),
            |name| path.with_file_name(name),
        )
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
    pub unsafe fn store_unchecked(&self, path: impl AsRef<Path>) -> ser::Result<()> {
        unsafe { ser::Serialize::store(self, path) }
    }

    /// Stores this shard payload with the requested compression mode.
    ///
    /// Raw storage writes mmap-loadable epserde bytes. zstd storage writes the
    /// same epserde payload through a zstd encoder; compressed shards must be
    /// decompressed before mmap-backed querying.
    ///
    /// # Errors
    ///
    /// Returns a store error if the file cannot be created, compressed, or
    /// serialized.
    ///
    /// # Safety
    ///
    /// This wraps epserde serialization. The resulting raw bytes, whether
    /// compressed or not, must be treated as trusted input when they are later
    /// deserialized.
    #[cfg(feature = "zstd")]
    pub unsafe fn store_with_compression_unchecked(
        &self,
        path: impl AsRef<Path>,
        compression: PersistedShardCompression,
    ) -> Result<PersistedShardStoreStats, PersistedShardStoreError> {
        match compression {
            PersistedShardCompression::None => {
                let path = path.as_ref();
                unsafe { self.store_unchecked(path)? };
                let disk_bytes = std::fs::metadata(path)?.len();
                let serialized_bytes = usize::try_from(disk_bytes).unwrap_or(usize::MAX);
                let stats = PersistedShardStoreStats {
                    serialized_bytes,
                    disk_bytes,
                };
                write_payload_manifest(
                    path,
                    PersistedShardPayloadKind::TargetIndex,
                    self.format_version,
                    self.base_target_id,
                    self.target_count,
                    PersistedShardStoredCompression::None,
                    stats.disk_bytes,
                )?;
                Ok(stats)
            }
            PersistedShardCompression::Zstd {
                level,
                worker_threads,
            } => unsafe { self.store_zstd_unchecked(path, level, worker_threads) },
        }
    }

    /// Stores this shard payload as zstd-compressed epserde bytes.
    ///
    /// # Errors
    ///
    /// Returns a store error if the file cannot be created, compressed, or
    /// serialized.
    ///
    /// # Safety
    ///
    /// This wraps epserde serialization. The compressed file must be
    /// decompressed back to trusted epserde bytes before deserialization.
    #[cfg(feature = "zstd")]
    pub unsafe fn store_zstd_unchecked(
        &self,
        path: impl AsRef<Path>,
        level: i32,
        worker_threads: u32,
    ) -> Result<PersistedShardStoreStats, PersistedShardStoreError> {
        let file = File::create(path.as_ref())?;
        let writer = BufWriter::new(file);
        let mut encoder = zstd::stream::write::Encoder::new(writer, level)?;
        if worker_threads > 1 {
            encoder.multithread(worker_threads)?;
        }
        let serialized_bytes = unsafe { self.serialize_into_unchecked(&mut encoder)? };
        let mut writer = encoder.finish()?;
        writer.flush()?;
        let disk_bytes = std::fs::metadata(path.as_ref())?.len();
        let stats = PersistedShardStoreStats {
            serialized_bytes,
            disk_bytes,
        };
        write_payload_manifest(
            path,
            PersistedShardPayloadKind::TargetIndex,
            self.format_version,
            self.base_target_id,
            self.target_count,
            PersistedShardStoredCompression::Zstd,
            stats.disk_bytes,
        )?;
        Ok(stats)
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
        path: impl AsRef<Path>,
        flags: deser::Flags,
    ) -> Result<deser::MemCase<Self>, Box<dyn std::error::Error + Send + Sync + 'static>> {
        unsafe { <Self as deser::Deserialize>::mmap(path, flags) }
            .map_err(|error| -> Box<dyn std::error::Error + Send + Sync + 'static> { error.into() })
    }

    /// Validates the manifest and memory-maps a raw persisted index shard.
    ///
    /// # Errors
    ///
    /// Returns an error if the manifest is missing or incompatible, if the
    /// payload byte length differs from the manifest, or if mmap-backed
    /// deserialization fails.
    pub fn mmap(
        path: impl AsRef<Path>,
        flags: deser::Flags,
    ) -> Result<deser::MemCase<Self>, Box<dyn std::error::Error + Send + Sync + 'static>> {
        let path = path.as_ref();
        let manifest = validate_payload_manifest(
            path,
            Some(PersistedShardPayloadKind::TargetIndex),
            PersistedShardStoredCompression::None,
        )?;
        let mapped = unsafe { Self::mmap_unchecked(path, flags)? };
        validate_loaded_payload_header(
            "target index",
            path,
            &manifest,
            mapped.uncase().format_version(),
            mapped.uncase().base_target_id(),
            mapped.uncase().target_count(),
        )?;
        Ok(mapped)
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
            ring_element: PersistedKeyedCountIndexes::from_count_indexes(
                &index.indexed_ring_element_count_index,
                |element| u16::from(u8::from(*element)),
            ),
            aromatic_element: PersistedKeyedCountIndexes::from_count_indexes(
                &index.indexed_aromatic_element_count_index,
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
            ring_membership: PersistedKeyedCountIndexes::from_count_indexes(
                &index.indexed_ring_membership_count_index,
                |ring_membership| *ring_membership,
            ),
            ring_size: PersistedKeyedCountIndexes::from_count_indexes(
                &index.indexed_ring_size_count_index,
                |ring_size| *ring_size,
            ),
            ring_connectivity: PersistedKeyedCountIndexes::from_count_indexes(
                &index.indexed_ring_connectivity_count_index,
                |ring_connectivity| *ring_connectivity,
            ),
        }
    }
}

impl<
        ElementCounts,
        RingElementCounts,
        AromaticElementCounts,
        DegreeCounts,
        TotalHydrogenCounts,
        RingMembershipCounts,
        RingSizeCounts,
        RingConnectivityCounts,
    >
    PersistedAtomPropertyCountIndexes<
        ElementCounts,
        RingElementCounts,
        AromaticElementCounts,
        DegreeCounts,
        TotalHydrogenCounts,
        RingMembershipCounts,
        RingSizeCounts,
        RingConnectivityCounts,
    >
{
    /// Returns persisted element count indexes keyed by atomic number.
    #[inline]
    #[must_use]
    pub const fn element_counts(&self) -> &ElementCounts {
        &self.element
    }

    /// Returns persisted ring-element count indexes keyed by atomic number.
    #[inline]
    #[must_use]
    pub const fn ring_element_counts(&self) -> &RingElementCounts {
        &self.ring_element
    }

    /// Returns persisted aromatic-element count indexes keyed by atomic number.
    #[inline]
    #[must_use]
    pub const fn aromatic_element_counts(&self) -> &AromaticElementCounts {
        &self.aromatic_element
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

    /// Returns persisted exact ring-membership count indexes.
    #[inline]
    #[must_use]
    pub const fn ring_membership_counts(&self) -> &RingMembershipCounts {
        &self.ring_membership
    }

    /// Returns persisted exact smallest-ring-size count indexes.
    #[inline]
    #[must_use]
    pub const fn ring_size_counts(&self) -> &RingSizeCounts {
        &self.ring_size
    }

    /// Returns persisted exact ring-connectivity count indexes.
    #[inline]
    #[must_use]
    pub const fn ring_connectivity_counts(&self) -> &RingConnectivityCounts {
        &self.ring_connectivity
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

    fn bond_kind_count_filter(
        &self,
        kind: RequiredBondKind,
        required: usize,
    ) -> Option<RequiredCountFilter<'_>>;
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

    fn bond_kind_count_filter(
        &self,
        kind: RequiredBondKind,
        required: usize,
    ) -> Option<RequiredCountFilter<'_>> {
        let index = match kind {
            RequiredBondKind::Single => &self.single_bond,
            RequiredBondKind::Double => &self.double_bond,
            RequiredBondKind::Triple => &self.triple_bond,
            RequiredBondKind::Aromatic => &self.aromatic_bond,
        };
        let filter = index.persisted_filter_for_at_least(required)?;
        Some(RequiredCountFilter {
            population: filter.population,
            source: filter.source,
        })
    }
}

trait PersistedAtomPropertyCountIndexesAccess {
    fn collect_atom_property_count_filters<'idx>(
        &'idx self,
        query: &QueryScreen,
        filters: &mut Vec<RequiredCountFilter<'idx>>,
    ) -> bool;
}

impl<
        ElementCounts,
        RingElementCounts,
        AromaticElementCounts,
        DegreeCounts,
        TotalHydrogenCounts,
        RingMembershipCounts,
        RingSizeCounts,
        RingConnectivityCounts,
    > PersistedAtomPropertyCountIndexesAccess
    for PersistedAtomPropertyCountIndexes<
        ElementCounts,
        RingElementCounts,
        AromaticElementCounts,
        DegreeCounts,
        TotalHydrogenCounts,
        RingMembershipCounts,
        RingSizeCounts,
        RingConnectivityCounts,
    >
where
    ElementCounts: PersistedKeyedCountIndexesAccess,
    RingElementCounts: PersistedKeyedCountIndexesAccess,
    AromaticElementCounts: PersistedKeyedCountIndexesAccess,
    DegreeCounts: PersistedKeyedCountIndexesAccess,
    TotalHydrogenCounts: PersistedKeyedCountIndexesAccess,
    RingMembershipCounts: PersistedKeyedCountIndexesAccess,
    RingSizeCounts: PersistedKeyedCountIndexesAccess,
    RingConnectivityCounts: PersistedKeyedCountIndexesAccess,
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
        for (element, required) in &query.required_ring_element_counts {
            if !self.ring_element.push_filter_for_key(
                filters,
                u16::from(u8::from(*element)),
                *required,
            ) {
                return false;
            }
        }
        for (element, required) in &query.required_aromatic_element_counts {
            if !self.aromatic_element.push_filter_for_key(
                filters,
                u16::from(u8::from(*element)),
                *required,
            ) {
                return false;
            }
        }
        collect_persisted_keyed_count_filters(filters, &self.degree, &query.required_degree_counts)
            && collect_persisted_keyed_count_filters(
                filters,
                &self.total_hydrogen,
                &query.required_total_hydrogen_counts,
            )
            && collect_persisted_keyed_count_filters(
                filters,
                &self.ring_membership,
                &query.required_ring_membership_counts,
            )
            && collect_persisted_keyed_count_filters(
                filters,
                &self.ring_size,
                &query.required_ring_size_counts,
            )
            && collect_persisted_keyed_count_filters(
                filters,
                &self.ring_connectivity,
                &query.required_ring_connectivity_counts,
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
        self.populate_candidate_mask_with_initial_source(query, scratch, None)
    }

    fn populate_candidate_mask_with_initial_source<'idx>(
        &'idx self,
        query: &QueryScreen,
        scratch: &mut TargetCorpusScratch<'idx>,
        initial_source: Option<(&[u64], usize)>,
    ) -> Option<CandidateMaskState> {
        let target_count = self.target_count_usize()?;
        if target_count == 0 {
            return None;
        }

        scratch.ensure_cache_owner(core::ptr::from_ref(self) as usize);
        scratch.filters.clear();
        scratch.filters.reserve(
            12 + query.required_element_counts.len()
                + query.required_ring_element_counts.len()
                + query.required_aromatic_element_counts.len()
                + query.required_degree_counts.len()
                + query.required_total_hydrogen_counts.len()
                + query.required_ring_membership_counts.len()
                + query.required_ring_size_counts.len()
                + query.required_ring_connectivity_counts.len()
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
        let mut candidate_population = if let Some((source, source_population)) = initial_source {
            intersect_source_with_population(
                &mut scratch.candidate_mask,
                &mut has_active_source,
                source,
                source_population,
            )?
        } else {
            target_count
        };
        scratch
            .filters
            .sort_unstable_by_key(|filter| filter.population);
        for &filter in &scratch.filters {
            let population = intersect_source_with_population(
                &mut scratch.candidate_mask,
                &mut has_active_source,
                filter.source,
                filter.population,
            )?;
            candidate_population = population;
        }
        if !self.apply_bond_pair_count_filters(
            query,
            scratch,
            &mut has_active_source,
            &mut candidate_population,
        ) {
            return None;
        }
        if !self.apply_feature_count_filters(
            query,
            scratch,
            &mut has_active_source,
            &mut candidate_population,
        ) {
            return None;
        }
        if !self.apply_alternative_screen_groups(
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

    fn apply_bond_pair_count_filters(
        &self,
        query: &QueryScreen,
        scratch: &mut TargetCorpusScratch<'_>,
        has_active_source: &mut bool,
        candidate_population: &mut usize,
    ) -> bool {
        for (&pair, &required) in &query.required_bond_pair_counts {
            let Some(pair_population) =
                self.populate_bond_pair_count_candidate_mask(pair, required, scratch)
            else {
                return false;
            };
            let Some(population) = intersect_source_with_population(
                &mut scratch.candidate_mask,
                has_active_source,
                &scratch.bond_pair_candidate_mask,
                pair_population,
            ) else {
                return false;
            };
            *candidate_population = population;
        }

        true
    }

    fn populate_bond_pair_count_candidate_mask(
        &self,
        pair: BondKindPair,
        required: usize,
        scratch: &mut TargetCorpusScratch<'_>,
    ) -> Option<usize> {
        let target_count = self.target_count_usize()?;
        if required == 0 {
            return Some(target_count);
        }

        let word_count = bitset_word_count(target_count);
        ensure_zeroed_words(&mut scratch.bond_pair_candidate_mask, word_count);
        for first_required in 0..=required {
            let second_required = required - first_required;
            if self.populate_bond_pair_count_term_mask(
                pair,
                first_required,
                second_required,
                scratch,
            ) {
                for (candidate_word, &term_word) in scratch
                    .bond_pair_candidate_mask
                    .iter_mut()
                    .zip(&scratch.bond_pair_term_mask)
                {
                    *candidate_word |= term_word;
                }
            }
        }

        let population = bitset_population(&scratch.bond_pair_candidate_mask, target_count);
        (population != 0).then_some(population)
    }

    fn populate_bond_pair_count_term_mask(
        &self,
        pair: BondKindPair,
        first_required: usize,
        second_required: usize,
        scratch: &mut TargetCorpusScratch<'_>,
    ) -> bool {
        let Some(target_count) = self.target_count_usize() else {
            return false;
        };
        let word_count = bitset_word_count(target_count);
        ensure_zeroed_words(&mut scratch.bond_pair_term_mask, word_count);
        let mut has_term_source = false;
        if first_required > 0 {
            let Some(filter) = self
                .scalar_counts
                .bond_kind_count_filter(pair.first, first_required)
            else {
                return false;
            };
            if intersect_source_with_population(
                &mut scratch.bond_pair_term_mask,
                &mut has_term_source,
                filter.source,
                filter.population,
            )
            .is_none()
            {
                return false;
            }
        }
        if second_required > 0 {
            let Some(filter) = self
                .scalar_counts
                .bond_kind_count_filter(pair.second, second_required)
            else {
                return false;
            };
            if intersect_source_with_population(
                &mut scratch.bond_pair_term_mask,
                &mut has_term_source,
                filter.source,
                filter.population,
            )
            .is_none()
            {
                return false;
            }
        }
        has_term_source
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

    fn apply_alternative_screen_groups<'idx>(
        &'idx self,
        query: &QueryScreen,
        scratch: &mut TargetCorpusScratch<'idx>,
        has_active_source: &mut bool,
        candidate_population: &mut usize,
    ) -> bool {
        for group in &query.alternative_screen_groups {
            let group_result = {
                let active_source = if *has_active_source {
                    Some((scratch.candidate_mask.as_slice(), *candidate_population))
                } else {
                    None
                };
                self.alternative_screen_group_candidate_mask(group, active_source)
            };
            let Some((group_mask, group_population)) = group_result else {
                return false;
            };
            let Some(population) = intersect_source_with_population(
                &mut scratch.candidate_mask,
                has_active_source,
                &group_mask,
                group_population,
            ) else {
                return false;
            };
            *candidate_population = population;
        }
        true
    }

    fn alternative_screen_group_candidate_mask(
        &self,
        group: &[QueryScreen],
        active_source: Option<(&[u64], usize)>,
    ) -> Option<(Vec<u64>, usize)> {
        let target_count = self.target_count_usize()?;
        let word_count = bitset_word_count(target_count);
        let mut group_mask = vec![0; word_count];
        let mut alternative_scratch = TargetCorpusScratch::new();
        for alternative in group {
            let Some(state) = self.populate_candidate_mask_with_initial_source(
                alternative,
                &mut alternative_scratch,
                active_source,
            ) else {
                continue;
            };
            if !state.has_active_source {
                fill_all_target_bits(&mut group_mask, target_count);
                return Some((group_mask, target_count));
            }
            for (group_word, &alternative_word) in group_mask
                .iter_mut()
                .zip(&alternative_scratch.candidate_mask)
            {
                *group_word |= alternative_word;
            }
        }
        let population = bitset_population(&group_mask, target_count);
        (population != 0).then_some((group_mask, population))
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

    use alloc::{format, string::ToString};
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

    #[cfg(feature = "zstd")]
    #[test]
    fn persisted_target_corpus_index_shard_checked_mmap_requires_manifest() {
        let targets = persisted_candidate_targets();
        let index = TargetCorpusIndex::new(&targets);
        let shard = TargetCorpusIndexShard::new(17, index);
        let persisted = PersistedTargetCorpusIndexShard::from_index_shard(&shard);
        let legacy_path = persisted_shard_temp_path();

        unsafe {
            persisted.store_unchecked(&legacy_path).unwrap();
        }
        let error =
            PersistedTargetCorpusIndexShard::mmap(&legacy_path, Flags::empty()).unwrap_err();
        assert!(error.to_string().contains("manifest"));
        std::fs::remove_file(legacy_path).unwrap();

        let path = persisted_shard_temp_path();
        unsafe {
            PersistedTargetCorpusIndexShardBuilder::new(&targets)
                .base_target_id(17)
                .store_unchecked(&path)
                .unwrap();
        }
        let mapped = PersistedTargetCorpusIndexShard::mmap(&path, Flags::empty()).unwrap();
        let loaded = mapped.uncase();

        assert_eq!(loaded.format_version(), FORMAT_VERSION);
        assert_eq!(loaded.base_target_id(), 17);
        assert_eq!(loaded.target_count(), 5);
        assert_persisted_candidates_match_runtime_index(loaded, &shard);

        replace_manifest_value(&path, "format_version", "999");
        let error = PersistedTargetCorpusIndexShard::mmap(&path, Flags::empty()).unwrap_err();
        assert!(error.to_string().contains("format version"));

        remove_payload_and_manifest(path);
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn persisted_shard_paths_validate_contiguous_manifests() {
        let targets = persisted_candidate_targets();
        let first_path = persisted_shard_temp_path();
        let second_path = persisted_shard_temp_path();

        unsafe {
            PersistedTargetCorpusIndexShardBuilder::new(&targets)
                .base_target_id(17)
                .store_unchecked(&first_path)
                .unwrap();
            PersistedTargetCorpusIndexShardBuilder::new(&targets)
                .base_target_id(22)
                .store_unchecked(&second_path)
                .unwrap();
        }

        PersistedTargetCorpusIndexShardPaths::from_paths([first_path.clone(), second_path.clone()])
            .validate_manifests()
            .unwrap();

        remove_payload_and_manifest(first_path);
        remove_payload_and_manifest(second_path);
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn persisted_shard_paths_reject_manifest_gaps_and_overlaps() {
        for (second_base_target_id, expected_error) in [(21, "overlaps"), (23, "gap")] {
            let targets = persisted_candidate_targets();
            let first_path = persisted_shard_temp_path();
            let second_path = persisted_shard_temp_path();

            unsafe {
                PersistedTargetCorpusIndexShardBuilder::new(&targets)
                    .base_target_id(17)
                    .store_unchecked(&first_path)
                    .unwrap();
                PersistedTargetCorpusIndexShardBuilder::new(&targets)
                    .base_target_id(second_base_target_id)
                    .store_unchecked(&second_path)
                    .unwrap();
            }

            let error = PersistedTargetCorpusIndexShardPaths::from_paths([
                first_path.clone(),
                second_path.clone(),
            ])
            .validate_manifests()
            .unwrap_err();
            assert!(
                error.to_string().contains(expected_error),
                "unexpected error: {error}"
            );

            remove_payload_and_manifest(first_path);
            remove_payload_and_manifest(second_path);
        }
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn persisted_shard_paths_reject_manifest_payload_byte_mismatch() {
        let targets = persisted_candidate_targets();
        let path = persisted_shard_temp_path();

        unsafe {
            PersistedTargetCorpusIndexShardBuilder::new(&targets)
                .base_target_id(17)
                .store_unchecked(&path)
                .unwrap();
        }
        replace_manifest_value(&path, "payload_bytes", "0");

        let error = PersistedTargetCorpusIndexShardPaths::from_paths([path.clone()])
            .validate_manifests()
            .unwrap_err();
        assert!(
            error.to_string().contains("payload bytes"),
            "unexpected error: {error}"
        );

        remove_payload_and_manifest(path);
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn persisted_queryable_shard_paths_reject_sidecar_manifest_mismatch() {
        let target_smiles = ["CCO", "CC=O", "COC"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let targets = target_smiles
            .iter()
            .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
            .collect::<Vec<_>>();
        let path = persisted_shard_temp_path();

        unsafe {
            PersistedTargetCorpusIndexShardBuilder::new(&targets)
                .base_target_id(17)
                .target_smiles(&target_smiles)
                .store_queryable_unchecked(&path)
                .unwrap();
        }
        let smiles_path = persisted_target_smiles_path_for_index_shard_path(&path);
        replace_manifest_value(&smiles_path, "target_count", "99");

        let query = CompiledQuery::new(crate::QueryMol::from_str("[#6]=[#8]").unwrap()).unwrap();
        let error = PersistedTargetCorpusIndexShardPaths::from_paths([path.clone()])
            .matching_hits(&query)
            .unwrap_err();
        assert!(
            error.to_string().contains("target SMILES sidecar manifest"),
            "unexpected error: {error}"
        );

        remove_payload_and_manifest(smiles_path);
        remove_payload_and_manifest(path);
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn persisted_target_corpus_index_shard_stores_zstd_payload() {
        let targets = persisted_candidate_targets();
        let index = TargetCorpusIndex::new(&targets);
        let shard = TargetCorpusIndexShard::new(17, index);
        let persisted = PersistedTargetCorpusIndexShard::from_index_shard(&shard);
        let path = persisted_shard_temp_path().with_extension("eps.zst");

        let stats = unsafe {
            persisted
                .store_with_compression_unchecked(
                    &path,
                    PersistedShardCompression::Zstd {
                        level: 1,
                        worker_threads: 0,
                    },
                )
                .unwrap()
        };
        assert!(stats.serialized_bytes > 0);
        assert!(stats.disk_bytes > 0);

        let file = File::open(&path).unwrap();
        let mut decoder = zstd::stream::read::Decoder::new(file).unwrap();
        let loaded = unsafe {
            PersistedTargetCorpusIndexShard::deserialize_full_unchecked(&mut decoder).unwrap()
        };

        assert_eq!(loaded, persisted);
        assert_persisted_candidates_match_runtime_index(&loaded, &shard);

        remove_payload_and_manifest(path);
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn persisted_target_corpus_index_shard_builder_reports_progress() {
        let targets = persisted_candidate_targets();
        let path = persisted_shard_temp_path();
        let mut events = Vec::new();

        let stats = unsafe {
            PersistedTargetCorpusIndexShardBuilder::new(&targets)
                .base_target_id(17)
                .store_unchecked_with_progress(&path, |event| {
                    events.push((event.step(), event.elapsed().is_some()));
                })
                .unwrap()
        };

        assert_eq!(stats.target_count, targets.len());
        assert_eq!(
            events,
            [
                (BuildRuntimeIndex, false),
                (BuildRuntimeIndex, true),
                (BuildPersistedLayout, false),
                (BuildPersistedLayout, true),
                (StorePayload, false),
                (StorePayload, true),
            ]
        );

        remove_payload_and_manifest(path);
    }

    #[cfg(all(feature = "zstd", feature = "terminal-progress"))]
    #[test]
    fn persisted_target_corpus_index_shard_builder_can_render_terminal_progress() {
        let targets = persisted_candidate_targets();
        let path = persisted_shard_temp_path();

        let stats = unsafe {
            PersistedTargetCorpusIndexShardBuilder::new(&targets)
                .base_target_id(17)
                .store_unchecked_with_terminal_progress(&path, "test-shard")
                .unwrap()
        };

        assert_eq!(stats.target_count, targets.len());

        remove_payload_and_manifest(path);
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn persisted_queryable_shard_returns_exact_hits_and_external_ids() {
        let target_smiles = ["CCO", "CC=O", "COC"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let targets = target_smiles
            .iter()
            .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
            .collect::<Vec<_>>();
        let external_ids = [101_u64, 102, 103];
        let path = persisted_shard_temp_path();

        let stats = unsafe {
            PersistedTargetCorpusIndexShardBuilder::new(&targets)
                .base_target_id(17)
                .target_smiles(&target_smiles)
                .external_ids(&external_ids)
                .store_queryable_unchecked(&path)
                .unwrap()
        };
        assert_eq!(stats.index.target_count, targets.len());
        assert!(stats.target_smiles_store_stats.disk_bytes > 0);
        assert!(stats.external_ids_store_stats.is_some());

        let query = CompiledQuery::new(crate::QueryMol::from_str("[#6]=[#8]").unwrap()).unwrap();
        let hits = PersistedTargetCorpusIndexShardPaths::from_paths([path.clone()])
            .matching_hits(&query)
            .unwrap();
        assert_eq!(
            hits,
            [PersistedTargetHit {
                target_id: 18,
                external_id: Some(102)
            }]
        );
        let mut streamed_hits = Vec::new();
        let streamed_count = PersistedTargetCorpusIndexShardPaths::from_paths([path.clone()])
            .try_for_each_matching_hit(&query, |hit| {
                streamed_hits.push(hit);
                Ok(())
            })
            .unwrap();
        assert_eq!(streamed_count, hits.len());
        assert_eq!(streamed_hits, hits);

        let mut streamed_record_hits = Vec::new();
        let streamed_record_count =
            PersistedTargetCorpusIndexShardPaths::from_paths([path.clone()])
                .try_for_each_matching_record_hit(&query, |hit| {
                    streamed_record_hits.push((
                        hit.target_id,
                        String::from(hit.target_smiles),
                        hit.external_id,
                    ));
                    Ok(())
                })
                .unwrap();
        assert_eq!(streamed_record_count, hits.len());
        assert_eq!(
            streamed_record_hits,
            [(18_usize, String::from("CC=O"), Some(102_u64))]
        );

        remove_payload_and_manifest(persisted_target_smiles_path_for_index_shard_path(&path));
        remove_payload_and_manifest(persisted_external_ids_path_for_index_shard_path(&path));
        remove_payload_and_manifest(path);
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn persisted_queryable_shard_can_skip_wildcard_target_smiles() {
        let indexed_smiles = ["CCO", "CC=O", "COC"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let target_smiles = ["CCO", "*", "COC"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let targets = indexed_smiles
            .iter()
            .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
            .collect::<Vec<_>>();
        let external_ids = [101_u64, 102, 103];
        let path = persisted_shard_temp_path();

        unsafe {
            PersistedTargetCorpusIndexShardBuilder::new(&targets)
                .base_target_id(17)
                .target_smiles(&target_smiles)
                .external_ids(&external_ids)
                .store_queryable_unchecked(&path)
                .unwrap();
        }

        let query = CompiledQuery::new(crate::QueryMol::from_str("[#6]=[#8]").unwrap()).unwrap();
        let error = PersistedTargetCorpusIndexShardPaths::from_paths([path.clone()])
            .try_for_each_matching_record_hit(&query, |_hit| Ok(()))
            .unwrap_err();
        assert!(
            error.to_string().contains("Wildcard atom not allowed"),
            "unexpected error: {error}"
        );

        let mut streamed_record_hits = Vec::new();
        let skipped_count = PersistedTargetCorpusIndexShardPaths::from_paths([path.clone()])
            .try_for_each_matching_record_hit_with_target_smiles_parse_policy(
                &query,
                PersistedTargetSmilesParsePolicy::SkipWildcard,
                |hit| {
                    streamed_record_hits.push(hit.target_id);
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(skipped_count, 0);
        assert!(streamed_record_hits.is_empty());

        remove_payload_and_manifest(persisted_target_smiles_path_for_index_shard_path(&path));
        remove_payload_and_manifest(persisted_external_ids_path_for_index_shard_path(&path));
        remove_payload_and_manifest(path);
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn persisted_shard_matches_runtime_ring_property_filters() {
        let targets = [
            "CCN",
            "c1ccccc1",
            "n1ccccc1",
            "C1CCCCC1",
            "c1ccc2ccccc2c1",
            "C1CC1",
        ]
        .into_iter()
        .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
        .collect::<Vec<_>>();
        let runtime_index = TargetCorpusIndex::new(&targets);
        let runtime_shard = TargetCorpusIndexShard::new(17, runtime_index);
        let path = persisted_shard_temp_path();

        unsafe {
            PersistedTargetCorpusIndexShardBuilder::new(&targets)
                .base_target_id(17)
                .store_unchecked(&path)
                .unwrap();
        }

        let persisted_paths = PersistedTargetCorpusIndexShardPaths::from_paths([path.clone()]);
        for smarts in [
            "[#7&R]", "[n]", "[#6;R2]", "[#6;r6]", "[#6;x3]", "[#6;r3]", "[#6;R0]",
        ] {
            let query = QueryScreen::new(&crate::QueryMol::from_str(smarts).unwrap());
            let expected = runtime_shard
                .index()
                .candidate_ids(&query)
                .into_iter()
                .map(|target_id| runtime_shard.base_target_id() + target_id)
                .collect::<Vec<_>>();
            assert_eq!(persisted_paths.candidate_ids(&query).unwrap(), expected);
        }

        remove_payload_and_manifest(path);
    }

    #[cfg(all(feature = "zstd", feature = "rayon"))]
    #[test]
    fn persisted_shard_paths_extract_and_query_in_parallel() {
        let targets = persisted_candidate_targets();
        let compressed_path = persisted_shard_temp_path().with_extension("eps.zst");
        let raw_path = raw_path_for_zstd_shard(&compressed_path);
        let build_stats = unsafe {
            PersistedTargetCorpusIndexShardBuilder::new(&targets)
                .base_target_id(17)
                .compression(PersistedShardCompression::Zstd {
                    level: 1,
                    worker_threads: 0,
                })
                .store_unchecked(&compressed_path)
                .unwrap()
        };
        assert_eq!(build_stats.target_count, targets.len());
        assert!(build_stats.store_stats.disk_bytes > 0);
        assert!(!raw_path.exists());

        let compressed_shards =
            PersistedTargetCorpusIndexShardPaths::from_paths([compressed_path.clone()]);
        let extracted = compressed_shards
            .extract_zstd_if_missing_in_parallel()
            .unwrap();
        assert_eq!(extracted.len(), 1);
        assert_eq!(extracted[0].destination, raw_path);
        assert!(extracted[0].extracted);

        let extracted_again = compressed_shards
            .extract_zstd_if_missing_in_parallel()
            .unwrap();
        assert!(!extracted_again[0].extracted);

        let query = QueryScreen::new(&crate::QueryMol::from_str("CO").unwrap());
        let raw_shards = compressed_shards.extracted_raw_paths();
        let index = TargetCorpusIndex::new(&targets);
        let shard = TargetCorpusIndexShard::new(17, index);
        let expected = shard
            .index()
            .candidate_ids(&query)
            .into_iter()
            .map(|target_id| shard.base_target_id() + target_id)
            .collect::<Vec<_>>();
        let candidate_count = raw_shards.par_candidate_count(&query).unwrap();
        let candidate_ids = raw_shards.par_candidate_ids(&query).unwrap();
        let mapped_shards = raw_shards.mmap().unwrap();

        assert_eq!(candidate_count, expected.len());
        assert_eq!(candidate_ids, expected);
        assert_eq!(mapped_shards.len(), 1);
        assert!(!mapped_shards.is_empty());
        assert_eq!(mapped_shards.shards().len(), 1);
        assert_eq!(mapped_shards.candidate_count(&query), expected.len());
        assert_eq!(mapped_shards.par_candidate_count(&query), expected.len());
        assert_eq!(mapped_shards.candidate_ids(&query), expected);
        assert_eq!(mapped_shards.par_candidate_ids(&query), expected);

        remove_payload_and_manifest(raw_path);
        remove_payload_and_manifest(compressed_path);
    }

    #[cfg(all(feature = "zstd", feature = "rayon"))]
    #[test]
    fn persisted_queryable_shard_paths_extract_and_match_in_parallel() {
        let target_smiles = ["CCO", "CC=O", "COC"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let targets = target_smiles
            .iter()
            .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
            .collect::<Vec<_>>();
        let external_ids = [101_u64, 102, 103];
        let compressed_path = persisted_shard_temp_path().with_extension("eps.zst");
        let raw_path = raw_path_for_zstd_shard(&compressed_path);
        let compressed_smiles_path =
            persisted_target_smiles_path_for_index_shard_path(&compressed_path);
        let raw_smiles_path = persisted_target_smiles_path_for_index_shard_path(&raw_path);
        let compressed_external_ids_path =
            persisted_external_ids_path_for_index_shard_path(&compressed_path);
        let raw_external_ids_path = persisted_external_ids_path_for_index_shard_path(&raw_path);

        unsafe {
            PersistedTargetCorpusIndexShardBuilder::new(&targets)
                .base_target_id(17)
                .target_smiles(&target_smiles)
                .external_ids(&external_ids)
                .compression(PersistedShardCompression::Zstd {
                    level: 1,
                    worker_threads: 0,
                })
                .store_queryable_unchecked(&compressed_path)
                .unwrap()
        };
        assert!(compressed_smiles_path.is_file());
        assert!(compressed_external_ids_path.is_file());
        assert!(!raw_path.exists());
        assert!(!raw_smiles_path.exists());
        assert!(!raw_external_ids_path.exists());

        let compressed_shards =
            PersistedTargetCorpusIndexShardPaths::from_paths([compressed_path.clone()]);
        let extracted = compressed_shards
            .extract_queryable_zstd_if_missing_in_parallel()
            .unwrap();
        assert_eq!(extracted.len(), 3);
        assert!(raw_path.is_file());
        assert!(raw_smiles_path.is_file());
        assert!(raw_external_ids_path.is_file());

        let query = CompiledQuery::new(crate::QueryMol::from_str("[#6]=[#8]").unwrap()).unwrap();
        let hits = compressed_shards
            .extracted_raw_paths()
            .par_matching_hits(&query)
            .unwrap();
        assert_eq!(
            hits,
            [PersistedTargetHit {
                target_id: 18,
                external_id: Some(102)
            }]
        );

        for path in [
            raw_path,
            raw_smiles_path,
            raw_external_ids_path,
            compressed_path,
            compressed_smiles_path,
            compressed_external_ids_path,
        ] {
            remove_payload_and_manifest(path);
        }
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn persisted_shard_matches_runtime_path_and_star_feature_filters() {
        let targets = [
            "CCCCCC",    // long chain: many path3 and path4 features
            "CC(C)CC",   // branched aliphatic carbon
            "CCN(CC)CC", // star3 environment around nitrogen
            "CCOCC",     // heteroatom-interrupted chain
            "c1ccccc1",  // aromatic ring
            "C1CCCCC1",  // aliphatic ring
            "CC(=O)OC",  // ester with a branch and a double bond
        ]
        .into_iter()
        .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
        .collect::<Vec<_>>();
        let runtime_index = TargetCorpusIndex::new(&targets);
        let runtime_shard = TargetCorpusIndexShard::new(17, runtime_index);
        let path = persisted_shard_temp_path();

        unsafe {
            PersistedTargetCorpusIndexShardBuilder::new(&targets)
                .base_target_id(17)
                .store_unchecked(&path)
                .unwrap();
        }

        let persisted_paths = PersistedTargetCorpusIndexShardPaths::from_paths([path.clone()]);
        for smarts in [
            "CCC",       // three-atom path
            "CCCC",      // four-atom path (forward and reverse orientations)
            "CC(C)C",    // star around a central carbon
            "CCOC",      // heteroatom-bearing path
            "CC(=O)O",   // branch with a double bond
            "C-C-C-C-C", // five-atom chain
            "CN(C)C",    // star around a central nitrogen
        ] {
            let query = QueryScreen::new(&crate::QueryMol::from_str(smarts).unwrap());
            let expected = runtime_shard
                .index()
                .candidate_ids(&query)
                .into_iter()
                .map(|target_id| runtime_shard.base_target_id() + target_id)
                .collect::<Vec<_>>();
            assert_eq!(
                persisted_paths.candidate_ids(&query).unwrap(),
                expected,
                "persisted candidates diverged from runtime for {smarts}"
            );
        }

        remove_payload_and_manifest(path);
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn persisted_queryable_shard_unchecked_variants_match_checked_results() {
        let target_smiles = ["CCO", "CC=O", "COC"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let targets = target_smiles
            .iter()
            .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
            .collect::<Vec<_>>();
        let external_ids = [101_u64, 102, 103];
        let path = persisted_shard_temp_path();

        unsafe {
            PersistedTargetCorpusIndexShardBuilder::new(&targets)
                .base_target_id(17)
                .target_smiles(&target_smiles)
                .external_ids(&external_ids)
                .store_queryable_unchecked(&path)
                .unwrap();
        }

        let paths = PersistedTargetCorpusIndexShardPaths::from_paths([path.clone()]);
        let screen = QueryScreen::new(&crate::QueryMol::from_str("CO").unwrap());
        let compiled = CompiledQuery::new(crate::QueryMol::from_str("[#6]=[#8]").unwrap()).unwrap();

        // The checked accessors validate manifests; the unchecked variants skip
        // that step but must return identical results for trusted shards.
        let checked_count = paths.candidate_count(&screen).unwrap();
        let checked_ids = paths.candidate_ids(&screen).unwrap();
        let checked_hits = paths.matching_hits(&compiled).unwrap();
        assert_eq!(
            checked_hits,
            [PersistedTargetHit {
                target_id: 18,
                external_id: Some(102)
            }]
        );

        unsafe {
            assert_eq!(
                paths.candidate_count_unchecked(&screen).unwrap(),
                checked_count
            );
            assert_eq!(paths.candidate_ids_unchecked(&screen).unwrap(), checked_ids);
            assert_eq!(
                paths.matching_hits_unchecked(&compiled).unwrap(),
                checked_hits
            );

            #[cfg(feature = "rayon")]
            {
                assert_eq!(
                    paths.par_candidate_count_unchecked(&screen).unwrap(),
                    checked_count
                );
                assert_eq!(
                    paths.par_candidate_ids_unchecked(&screen).unwrap(),
                    checked_ids
                );
                assert_eq!(
                    paths.par_matching_hits_unchecked(&compiled).unwrap(),
                    checked_hits
                );
            }

            let mut streamed = Vec::new();
            let streamed_count = paths
                .try_for_each_matching_hit_unchecked(&compiled, |hit| {
                    streamed.push(hit);
                    Ok(())
                })
                .unwrap();
            assert_eq!(streamed_count, checked_hits.len());
            assert_eq!(streamed, checked_hits);

            let mut records = Vec::new();
            let record_count = paths
                .try_for_each_matching_record_hit_unchecked(&compiled, |hit| {
                    records.push((
                        hit.target_id,
                        String::from(hit.target_smiles),
                        hit.external_id,
                    ));
                    Ok(())
                })
                .unwrap();
            assert_eq!(record_count, checked_hits.len());
            assert_eq!(records, [(18_usize, String::from("CC=O"), Some(102_u64))]);

            let mut policy_records = Vec::new();
            let policy_count = paths
                .try_for_each_matching_record_hit_unchecked_with_target_smiles_parse_policy(
                    &compiled,
                    PersistedTargetSmilesParsePolicy::SkipWildcard,
                    |hit| {
                        policy_records.push(hit.target_id);
                        Ok(())
                    },
                )
                .unwrap();
            assert_eq!(policy_count, checked_hits.len());
            assert_eq!(policy_records, [18_usize]);
        }

        remove_payload_and_manifest(persisted_target_smiles_path_for_index_shard_path(&path));
        remove_payload_and_manifest(persisted_external_ids_path_for_index_shard_path(&path));
        remove_payload_and_manifest(path);
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
            "[!#1;$([#6]-[#8]),$([#6;R]-[#6;R]-[#6;R])]",
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

    fn persisted_shard_temp_path() -> PathBuf {
        let suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "smarts-rs-persisted-shard-{}-{suffix}.eps",
            std::process::id()
        ))
    }

    fn remove_payload_and_manifest(path: impl AsRef<Path>) {
        let path = path.as_ref();
        std::fs::remove_file(persisted_manifest_path_for_payload_path(path)).unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[cfg(feature = "zstd")]
    fn replace_manifest_value(path: impl AsRef<Path>, key: &str, replacement: &str) {
        let manifest_path = persisted_manifest_path_for_payload_path(path);
        let raw = std::fs::read_to_string(&manifest_path).unwrap();
        let mut updated = String::new();
        let mut replaced = false;
        for line in raw.lines() {
            if line
                .split_once('\t')
                .is_some_and(|(line_key, _)| line_key == key)
            {
                updated.push_str(key);
                updated.push('\t');
                updated.push_str(replacement);
                updated.push('\n');
                replaced = true;
            } else {
                updated.push_str(line);
                updated.push('\n');
            }
        }
        assert!(replaced, "manifest key {key} was not present");
        std::fs::write(manifest_path, updated).unwrap();
    }
}
