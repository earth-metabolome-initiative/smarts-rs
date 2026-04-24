//! Reports screening/index selectivity on the fixed evolution benchmark corpora.

use core::str::FromStr;
use std::{fs, path::PathBuf, time::Instant};

use smarts_rs::QueryMol;
use smarts_rs::{
    QueryScreen, TargetCorpusIndex, TargetCorpusIndexStats, TargetCorpusScratch, TargetScreen,
};
use smiles_parser::Smiles;

const TARGET_FIXTURE: &str = "corpus/benchmark/smarts-evolution-example-smiles-v0.tsv";
const QUERY_FIXTURES: [(&str, &str); 2] = [
    (
        "smarts_evolution_complex_queries_x_examples",
        "corpus/benchmark/smarts-evolution-complex-queries-v0.smarts",
    ),
    (
        "smarts_evolution_large_complex_queries_x_examples",
        "corpus/benchmark/smarts-evolution-complex-queries-large-v0.smarts",
    ),
];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn load_lines(relative_path: &str) -> Vec<String> {
    let path = repo_root().join(relative_path);
    fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("failed to read {}", path.display()))
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn load_target_smiles(relative_path: &str) -> Vec<String> {
    let path = repo_root().join(relative_path);
    fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("failed to read {}", path.display()))
        .lines()
        .enumerate()
        .filter_map(|(line_idx, line)| {
            if line_idx == 0 {
                return None;
            }
            let mut fields = line.splitn(3, '\t');
            let _dataset = fields.next()?;
            let _label = fields.next()?;
            fields.next().map(ToOwned::to_owned)
        })
        .collect()
}

fn to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("benchmark counts must fit in u32"))
}

fn percentage(part: usize, total: usize) -> f64 {
    (to_f64(part) * 100.0) / to_f64(total)
}

fn mean_per_query(total: usize, query_count: usize) -> f64 {
    to_f64(total) / to_f64(query_count)
}

fn percentile(sorted_counts: &[usize], percentile: usize) -> usize {
    assert!(
        !sorted_counts.is_empty(),
        "percentiles need at least one value"
    );
    let numerator = percentile * (sorted_counts.len() - 1);
    sorted_counts[numerator.div_ceil(100)]
}

fn print_summary(
    query_smarts: &[String],
    query_screens: &[QueryScreen],
    index_stats: TargetCorpusIndexStats,
    target_count: usize,
    total_pairs: usize,
    coarse_pairs: usize,
    indexed_counts: &[usize],
) {
    let indexed_pairs = indexed_counts.iter().sum::<usize>();
    let mut sorted_indexed_counts = indexed_counts.to_vec();
    sorted_indexed_counts.sort_unstable();
    let mut ranked_queries = indexed_counts
        .iter()
        .copied()
        .enumerate()
        .collect::<Vec<_>>();
    ranked_queries.sort_unstable_by_key(|&(_, count)| core::cmp::Reverse(count));

    let feature_stats = query_screens
        .iter()
        .map(QueryScreen::feature_stats)
        .collect::<Vec<_>>();
    let edge_features = feature_stats
        .iter()
        .map(|stats| stats.edge_features)
        .sum::<usize>();
    let path3_features = feature_stats
        .iter()
        .map(|stats| stats.path3_features)
        .sum::<usize>();
    let path4_features = feature_stats
        .iter()
        .map(|stats| stats.path4_features)
        .sum::<usize>();
    let star3_features = feature_stats
        .iter()
        .map(|stats| stats.star3_features)
        .sum::<usize>();
    let edge_masks = feature_stats
        .iter()
        .map(|stats| stats.edge_feature_masks)
        .sum::<usize>();
    let path3_masks = feature_stats
        .iter()
        .map(|stats| stats.path3_feature_masks)
        .sum::<usize>();
    let path4_masks = feature_stats
        .iter()
        .map(|stats| stats.path4_feature_masks)
        .sum::<usize>();
    let star3_masks = feature_stats
        .iter()
        .map(|stats| stats.star3_feature_masks)
        .sum::<usize>();
    let query_count = query_screens.len();

    println!("queries: {query_count}");
    println!("targets: {target_count}");
    println!("total_pairs: {total_pairs}");
    println!(
        "coarse_screen_pairs: {} ({:.2}%)",
        coarse_pairs,
        percentage(coarse_pairs, total_pairs)
    );
    println!(
        "indexed_pairs: {} ({:.2}%)",
        indexed_pairs,
        percentage(indexed_pairs, total_pairs)
    );
    println!(
        "indexed_candidates_per_query: min {}, p50 {}, p90 {}, p99 {}, max {}",
        sorted_indexed_counts.first().copied().unwrap_or_default(),
        percentile(&sorted_indexed_counts, 50),
        percentile(&sorted_indexed_counts, 90),
        percentile(&sorted_indexed_counts, 99),
        sorted_indexed_counts.last().copied().unwrap_or_default()
    );
    println!("top_indexed_candidate_queries:");
    for (query_id, count) in ranked_queries.iter().take(10) {
        println!("  {query_id}: {count} {}", query_smarts[*query_id]);
    }
    println!(
        "avg_required_features_per_query: edge {:.2}, path3 {:.2}, path4 {:.2}, star3 {:.2}",
        mean_per_query(edge_features, query_count),
        mean_per_query(path3_features, query_count),
        mean_per_query(path4_features, query_count),
        mean_per_query(star3_features, query_count)
    );
    println!(
        "required_feature_masks: edge {edge_masks}, path3 {path3_masks}, path4 {path4_masks}, star3 {star3_masks}"
    );
    println!(
        "target_index_features: edge {} ({} postings, atom domain {}, bond domain {}), path3 {} ({} postings, atom domain {}, bond domain {}), path4 {} ({} postings, atom domain {}, bond domain {}), star3 {} ({} postings, atom domain {}, bond domain {})",
        index_stats.edge_feature_count,
        index_stats.edge_posting_count,
        index_stats.edge_atom_domain_count,
        index_stats.edge_bond_domain_count,
        index_stats.path3_feature_count,
        index_stats.path3_posting_count,
        index_stats.path3_atom_domain_count,
        index_stats.path3_bond_domain_count,
        index_stats.path4_feature_count,
        index_stats.path4_posting_count,
        index_stats.path4_atom_domain_count,
        index_stats.path4_bond_domain_count,
        index_stats.star3_feature_count,
        index_stats.star3_posting_count,
        index_stats.star3_atom_domain_count,
        index_stats.star3_bond_domain_count
    );
}

fn main() {
    let started = Instant::now();
    let query_sets = QUERY_FIXTURES
        .iter()
        .map(|&(name, path)| (name, load_lines(path)))
        .collect::<Vec<_>>();
    let target_smiles = load_target_smiles(TARGET_FIXTURE);
    eprintln!("loaded fixtures in {:?}", started.elapsed());

    let phase = Instant::now();
    let targets = target_smiles
        .iter()
        .map(|smiles| {
            smarts_rs::PreparedTarget::new(
                Smiles::from_str(smiles).expect("benchmark SMILES must parse"),
            )
        })
        .collect::<Vec<_>>();
    let target_screens = targets.iter().map(TargetScreen::new).collect::<Vec<_>>();
    eprintln!("prepared targets/screens in {:?}", phase.elapsed());

    let phase = Instant::now();
    let index = TargetCorpusIndex::new(&targets);
    let index_stats = index.stats();
    eprintln!("built target index in {:?}", phase.elapsed());

    let target_count = target_screens.len();

    for (name, query_smarts) in query_sets {
        let phase = Instant::now();
        let queries = query_smarts
            .iter()
            .map(|smarts| QueryMol::from_str(smarts).expect("benchmark SMARTS must parse"))
            .collect::<Vec<_>>();
        let query_screens = queries.iter().map(QueryScreen::new).collect::<Vec<_>>();
        eprintln!("prepared {name} queries in {:?}", phase.elapsed());

        let phase = Instant::now();
        let total_pairs = query_screens.len() * target_screens.len();
        let coarse_pairs = query_screens
            .iter()
            .map(|query| {
                target_screens
                    .iter()
                    .filter(|target| query.may_match(target))
                    .count()
            })
            .sum::<usize>();
        eprintln!("counted {name} coarse pairs in {:?}", phase.elapsed());

        let phase = Instant::now();
        let mut scratch = TargetCorpusScratch::new();
        let candidate_sets = index.candidate_sets_with_scratch(&query_screens, &mut scratch);
        let indexed_counts = candidate_sets
            .iter()
            .map(smarts_rs::TargetCandidateSet::len)
            .collect::<Vec<_>>();
        eprintln!("counted {name} indexed pairs in {:?}", phase.elapsed());

        println!("fixture: {name}");
        print_summary(
            &query_smarts,
            &query_screens,
            index_stats,
            target_count,
            total_pairs,
            coarse_pairs,
            &indexed_counts,
        );
    }
}
