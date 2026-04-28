use std::{env, error::Error, path::PathBuf, str::FromStr, time::Instant};

use mem_dbg::{MemSize, SizeFlags};
use smarts_rs::{PreparedTarget, TargetCorpusIndex, TargetCorpusIndexStats};
use smiles_parser::{DatasetFetchOptions, GzipMode, Smiles, SmilesDatasetSource, PUBCHEM_SMILES};

const CACHE_DIR_ENV: &str = "SMARTS_RS_PUBCHEM_CACHE_DIR";
const DEFAULT_RECORD_LIMIT: usize = 1_000_000;
const INITIAL_TARGET_CAPACITY: usize = 1_000_000;
const LIMIT_ENV: &str = "SMARTS_RS_PUBCHEM_INDEX_LIMIT";

#[test]
#[ignore = "downloads/streams PubChem CID-SMILES and builds a large target index"]
fn pubchem_smiles_can_prepare_targets_and_build_index() -> Result<(), Box<dyn Error>> {
    let limit = pubchem_record_limit();

    eprintln!(
        "PubChem target limit: {}",
        limit.map_or_else(|| "full".to_owned(), |limit| limit.to_string())
    );

    let load_started = Instant::now();
    let targets = load_pubchem_targets(limit)?;
    eprintln!(
        "loaded and prepared {} PubChem targets in {:?}",
        targets.len(),
        load_started.elapsed()
    );
    assert!(!targets.is_empty(), "PubChem dataset yielded no targets");

    let build_started = Instant::now();
    let index = TargetCorpusIndex::new(&targets);
    let build_elapsed = build_started.elapsed();
    let stats = index.stats();

    assert_eq!(index.len(), targets.len());
    assert_eq!(stats.target_count, targets.len());

    eprintln!(
        "built TargetCorpusIndex for {} targets in {:?}",
        targets.len(),
        build_elapsed
    );
    print_index_stats(stats);

    let mem_size = index.mem_size(SizeFlags::default());
    let mem_capacity = index.mem_size(SizeFlags::CAPACITY);
    eprintln!(
        "target_index_mem_size: {} bytes ({:.2} MiB)",
        mem_size,
        bytes_to_mib(mem_size)
    );
    eprintln!(
        "target_index_mem_capacity: {} bytes ({:.2} MiB)",
        mem_capacity,
        bytes_to_mib(mem_capacity)
    );

    Ok(())
}

fn load_pubchem_targets(limit: Option<usize>) -> Result<Vec<PreparedTarget>, Box<dyn Error>> {
    let options = DatasetFetchOptions {
        cache_dir: pubchem_cache_dir(),
        gzip_mode: GzipMode::KeepCompressed,
        ..DatasetFetchOptions::default()
    };
    let smiles_iter = PUBCHEM_SMILES.iter_smiles_with_options(&options)?;
    let initial_capacity = limit
        .unwrap_or(DEFAULT_RECORD_LIMIT)
        .min(INITIAL_TARGET_CAPACITY);
    let mut targets = Vec::with_capacity(initial_capacity);

    for (record_idx, smiles) in smiles_iter.enumerate() {
        if limit.is_some_and(|limit| record_idx >= limit) {
            break;
        }

        let smiles = smiles?;
        let molecule = Smiles::from_str(&smiles).unwrap_or_else(|error| {
            panic!(
                "PubChem SMILES record {} failed to parse: {error}",
                record_idx + 1
            )
        });
        targets.push(PreparedTarget::new(molecule));
    }

    Ok(targets)
}

fn pubchem_cache_dir() -> Option<PathBuf> {
    env::var_os(CACHE_DIR_ENV).map(PathBuf::from)
}

fn pubchem_record_limit() -> Option<usize> {
    let Some(raw_limit) = env::var_os(LIMIT_ENV) else {
        return Some(DEFAULT_RECORD_LIMIT);
    };
    let raw_limit = raw_limit.to_string_lossy();
    if raw_limit == "0" || raw_limit.eq_ignore_ascii_case("full") {
        return None;
    }

    Some(
        raw_limit
            .parse::<usize>()
            .unwrap_or_else(|_| panic!("{LIMIT_ENV} must be a positive integer, 0, or full")),
    )
}

fn print_index_stats(stats: TargetCorpusIndexStats) {
    eprintln!(
        "target_index_features: edge {} ({} postings, atom domain {}, bond domain {}), path3 {} ({} postings, atom domain {}, bond domain {}), path4 {} ({} postings, atom domain {}, bond domain {}), star3 {} ({} postings, atom domain {}, bond domain {})",
        stats.edge_feature_count,
        stats.edge_posting_count,
        stats.edge_atom_domain_count,
        stats.edge_bond_domain_count,
        stats.path3_feature_count,
        stats.path3_posting_count,
        stats.path3_atom_domain_count,
        stats.path3_bond_domain_count,
        stats.path4_feature_count,
        stats.path4_posting_count,
        stats.path4_atom_domain_count,
        stats.path4_bond_domain_count,
        stats.star3_feature_count,
        stats.star3_posting_count,
        stats.star3_atom_domain_count,
        stats.star3_bond_domain_count
    );
}

fn bytes_to_mib(bytes: usize) -> f64 {
    bytes as f64 / 1_048_576.0
}
