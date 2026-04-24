//! Large heterogeneous n-SMARTS x m-SMILES benchmark based on downstream
//! smarts-evolution seeds and example corpora.

use core::str::FromStr;
use std::{collections::BTreeMap, fs, hint::black_box, path::PathBuf, time::Duration};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rayon::prelude::*;
use smarts_rs::{
    CompiledQuery, MatchScratch, PreparedTarget, QueryMol, QueryScreen, TargetCandidateSet,
    TargetCorpusIndex, TargetCorpusScratch,
};
use smiles_parser::Smiles;

const COMPLEX_QUERY_FIXTURE: &str = "corpus/benchmark/smarts-evolution-complex-queries-v0.smarts";
const LARGE_COMPLEX_QUERY_FIXTURE: &str =
    "corpus/benchmark/smarts-evolution-complex-queries-large-v0.smarts";
const TARGET_FIXTURE: &str = "corpus/benchmark/smarts-evolution-example-smiles-v0.tsv";

struct MatrixDataset {
    id: &'static str,
    description: String,
    queries: Vec<CompiledQuery>,
    query_screens: Vec<QueryScreen>,
    candidate_sets: Vec<TargetCandidateSet>,
    total_pairs: u64,
    screened_pairs: u64,
    expected_total_matches: usize,
    benchmark_scalar: bool,
}

struct RawMatrixInput {
    id: &'static str,
    query_smarts: Vec<String>,
    target_smiles: Vec<String>,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn load_query_smarts(relative_path: &str) -> Vec<String> {
    let path = repo_root().join(relative_path);
    let raw =
        fs::read_to_string(&path).unwrap_or_else(|_| panic!("failed to read {}", path.display()));
    raw.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn load_target_smiles() -> (Vec<String>, BTreeMap<String, usize>) {
    let path = repo_root().join(TARGET_FIXTURE);
    let raw =
        fs::read_to_string(&path).unwrap_or_else(|_| panic!("failed to read {}", path.display()));

    let mut targets = Vec::new();
    let mut per_dataset = BTreeMap::new();
    for (line_idx, line) in raw.lines().enumerate() {
        if line_idx == 0 {
            assert_eq!(line, "dataset\tlabel\tsmiles", "fixture header drifted");
            continue;
        }

        let mut fields = line.splitn(3, '\t');
        let dataset = fields
            .next()
            .unwrap_or_else(|| panic!("missing dataset field at line {}", line_idx + 1));
        let _label = fields
            .next()
            .unwrap_or_else(|| panic!("missing label field at line {}", line_idx + 1));
        let smiles = fields
            .next()
            .unwrap_or_else(|| panic!("missing SMILES field at line {}", line_idx + 1));
        assert!(
            !dataset.is_empty() && !smiles.is_empty(),
            "fixture row {} must stay populated",
            line_idx + 1
        );
        *per_dataset.entry(dataset.to_string()).or_default() += 1;
        targets.push(smiles.to_string());
    }

    (targets, per_dataset)
}

fn load_raw_input(id: &'static str, query_fixture: &str) -> RawMatrixInput {
    let query_smarts = load_query_smarts(query_fixture);
    let (target_smiles, _) = load_target_smiles();
    RawMatrixInput {
        id,
        query_smarts,
        target_smiles,
    }
}

fn compile_queries(query_smarts: &[String]) -> Vec<CompiledQuery> {
    query_smarts
        .iter()
        .map(|smarts| {
            CompiledQuery::new(
                QueryMol::from_str(smarts)
                    .unwrap_or_else(|_| panic!("benchmark SMARTS must parse: {smarts}")),
            )
            .unwrap_or_else(|_| panic!("benchmark SMARTS must stay supported: {smarts}"))
        })
        .collect()
}

fn prepare_targets(target_smiles: &[String]) -> Vec<PreparedTarget> {
    target_smiles
        .iter()
        .map(|smiles| {
            PreparedTarget::new(
                Smiles::from_str(smiles)
                    .unwrap_or_else(|_| panic!("benchmark SMILES must parse: {smiles}")),
            )
        })
        .collect()
}

fn build_dataset(
    id: &'static str,
    query_fixture: &str,
    query_description: &str,
    targets: &[PreparedTarget],
    target_index: &TargetCorpusIndex,
    per_dataset: &BTreeMap<String, usize>,
    benchmark_scalar: bool,
) -> MatrixDataset {
    let query_smarts = load_query_smarts(query_fixture);
    let queries = compile_queries(&query_smarts);
    let query_screens = queries
        .iter()
        .map(|query| QueryScreen::new(query.query()))
        .collect::<Vec<_>>();
    let candidate_sets = build_candidate_sets(&query_screens, target_index);

    let total_pairs = (queries.len() * targets.len()) as u64;
    let expected_total_matches = if benchmark_scalar {
        matrix_match_count_scalar(&queries, targets)
    } else {
        matrix_match_count_indexed_scalar(&queries, &candidate_sets, targets)
    };
    let screened_pairs = candidate_sets
        .iter()
        .map(TargetCandidateSet::len)
        .sum::<usize>() as u64;
    let screened_percent_x100 = screened_pairs.saturating_mul(10_000) / total_pairs;
    let description = format!(
        "{query_description}: {} queries x {} targets across {} downstream example families ({}), indexed candidates {} ({}.{:02}% of full matrix)",
        queries.len(),
        targets.len(),
        per_dataset.len(),
        per_dataset
            .iter()
            .map(|(dataset, count)| format!("{dataset}:{count}"))
            .collect::<Vec<_>>()
            .join(", "),
        screened_pairs,
        screened_percent_x100 / 100,
        screened_percent_x100 % 100,
    );

    MatrixDataset {
        id,
        description,
        queries,
        query_screens,
        candidate_sets,
        total_pairs,
        screened_pairs,
        expected_total_matches,
        benchmark_scalar,
    }
}

fn build_candidate_sets(
    query_screens: &[QueryScreen],
    target_index: &TargetCorpusIndex,
) -> Vec<TargetCandidateSet> {
    let mut scratch = TargetCorpusScratch::new();
    target_index.candidate_sets_with_scratch(query_screens, &mut scratch)
}

fn matrix_match_count_scalar(queries: &[CompiledQuery], targets: &[PreparedTarget]) -> usize {
    let mut total = 0usize;
    let mut scratch = MatchScratch::new();
    for query in queries {
        for target in targets {
            total += usize::from(query.matches_with_scratch(target, &mut scratch));
        }
    }
    total
}

fn matrix_match_count_indexed_scalar(
    queries: &[CompiledQuery],
    candidate_sets: &[TargetCandidateSet],
    targets: &[PreparedTarget],
) -> usize {
    let mut total = 0usize;
    let mut scratch = MatchScratch::new();
    for (query, candidates) in queries.iter().zip(candidate_sets) {
        for &target_id in candidates.target_ids() {
            total += usize::from(query.matches_with_scratch(&targets[target_id], &mut scratch));
        }
    }
    total
}

fn matrix_match_count_indexed_rayon_queries(
    queries: &[CompiledQuery],
    candidate_sets: &[TargetCandidateSet],
    targets: &[PreparedTarget],
) -> usize {
    queries
        .par_iter()
        .zip(candidate_sets.par_iter())
        .map_init(MatchScratch::new, |match_scratch, (query, candidates)| {
            candidates
                .target_ids()
                .iter()
                .map(|&target_id| {
                    usize::from(query.matches_with_scratch(&targets[target_id], match_scratch))
                })
                .sum::<usize>()
        })
        .sum()
}

fn matrix_match_count_indexed_batched_scalar(
    queries: &[CompiledQuery],
    query_screens: &[QueryScreen],
    target_index: &TargetCorpusIndex,
    targets: &[PreparedTarget],
) -> usize {
    let mut corpus_scratch = TargetCorpusScratch::new();
    let candidate_sets =
        target_index.candidate_sets_with_scratch(query_screens, &mut corpus_scratch);
    matrix_match_count_indexed_scalar(queries, &candidate_sets, targets)
}

fn matrix_match_count_indexed_batched_rayon_queries(
    queries: &[CompiledQuery],
    query_screens: &[QueryScreen],
    target_index: &TargetCorpusIndex,
    targets: &[PreparedTarget],
) -> usize {
    let mut corpus_scratch = TargetCorpusScratch::new();
    let candidate_sets =
        target_index.candidate_sets_with_scratch(query_screens, &mut corpus_scratch);
    matrix_match_count_indexed_rayon_queries(queries, &candidate_sets, targets)
}

fn matrix_match_count_indexed_streaming_scalar(
    queries: &[CompiledQuery],
    query_screens: &[QueryScreen],
    target_index: &TargetCorpusIndex,
    targets: &[PreparedTarget],
) -> usize {
    let mut total = 0usize;
    let mut corpus_scratch = TargetCorpusScratch::new();
    let mut match_scratch = MatchScratch::new();
    for (query, screen) in queries.iter().zip(query_screens) {
        target_index.for_each_candidate_id_with_scratch(screen, &mut corpus_scratch, |target_id| {
            total +=
                usize::from(query.matches_with_scratch(&targets[target_id], &mut match_scratch));
        });
    }
    total
}

fn matrix_match_count_indexed_streaming_rayon_queries(
    queries: &[CompiledQuery],
    query_screens: &[QueryScreen],
    target_index: &TargetCorpusIndex,
    targets: &[PreparedTarget],
) -> usize {
    queries
        .par_iter()
        .zip(query_screens.par_iter())
        .map_init(
            || (TargetCorpusScratch::new(), MatchScratch::new()),
            |(corpus_scratch, match_scratch), (query, screen)| {
                let mut total = 0usize;
                target_index.for_each_candidate_id_with_scratch(
                    screen,
                    corpus_scratch,
                    |target_id| {
                        total += usize::from(
                            query.matches_with_scratch(&targets[target_id], match_scratch),
                        );
                    },
                );
                total
            },
        )
        .sum()
}

fn configure_group(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));
}

#[allow(clippy::too_many_lines)]
fn bench_evolution_matrix(c: &mut Criterion) {
    let (target_smiles, per_dataset) = load_target_smiles();
    let targets = prepare_targets(&target_smiles);
    let target_index = TargetCorpusIndex::new(&targets);

    let mut group = c.benchmark_group("matcher_evolution_matrix_boolean");
    configure_group(&mut group);

    let datasets = [
        build_dataset(
            "smarts_evolution_complex_queries_x_examples",
            COMPLEX_QUERY_FIXTURE,
            "mined downstream complex SMARTS",
            &targets,
            &target_index,
            &per_dataset,
            true,
        ),
        build_dataset(
            "smarts_evolution_large_complex_queries_x_examples",
            LARGE_COMPLEX_QUERY_FIXTURE,
            "larger mined downstream complex SMARTS generation",
            &targets,
            &target_index,
            &per_dataset,
            false,
        ),
    ];

    for dataset in &datasets {
        group.throughput(Throughput::Elements(dataset.total_pairs));

        if dataset.benchmark_scalar {
            group.bench_function(BenchmarkId::new("scalar", dataset.id), |b| {
                b.iter(|| {
                    let total =
                        matrix_match_count_scalar(black_box(&dataset.queries), black_box(&targets));
                    assert_eq!(
                        total, dataset.expected_total_matches,
                        "benchmark fixture drifted"
                    );
                    black_box(total);
                });
            });
        }

        group.bench_function(BenchmarkId::new("indexed_scalar", dataset.id), |b| {
            b.iter(|| {
                let total = matrix_match_count_indexed_scalar(
                    black_box(&dataset.queries),
                    black_box(&dataset.candidate_sets),
                    black_box(&targets),
                );
                assert_eq!(
                    total, dataset.expected_total_matches,
                    "indexed benchmark changed match count"
                );
                black_box(total);
            });
        });

        group.bench_function(BenchmarkId::new("indexed_rayon_queries", dataset.id), |b| {
            b.iter(|| {
                let total = matrix_match_count_indexed_rayon_queries(
                    black_box(&dataset.queries),
                    black_box(&dataset.candidate_sets),
                    black_box(&targets),
                );
                assert_eq!(
                    total, dataset.expected_total_matches,
                    "indexed rayon benchmark changed match count"
                );
                black_box(total);
            });
        });

        group.bench_function(
            BenchmarkId::new("indexed_batched_scalar", dataset.id),
            |b| {
                b.iter(|| {
                    let total = matrix_match_count_indexed_batched_scalar(
                        black_box(&dataset.queries),
                        black_box(&dataset.query_screens),
                        black_box(&target_index),
                        black_box(&targets),
                    );
                    assert_eq!(
                        total, dataset.expected_total_matches,
                        "batched indexed benchmark changed match count"
                    );
                    black_box(total);
                });
            },
        );

        group.bench_function(
            BenchmarkId::new("indexed_batched_rayon_queries", dataset.id),
            |b| {
                b.iter(|| {
                    let total = matrix_match_count_indexed_batched_rayon_queries(
                        black_box(&dataset.queries),
                        black_box(&dataset.query_screens),
                        black_box(&target_index),
                        black_box(&targets),
                    );
                    assert_eq!(
                        total, dataset.expected_total_matches,
                        "batched indexed rayon benchmark changed match count"
                    );
                    black_box(total);
                });
            },
        );

        group.bench_function(
            BenchmarkId::new("indexed_streaming_scalar", dataset.id),
            |b| {
                b.iter(|| {
                    let total = matrix_match_count_indexed_streaming_scalar(
                        black_box(&dataset.queries),
                        black_box(&dataset.query_screens),
                        black_box(&target_index),
                        black_box(&targets),
                    );
                    assert_eq!(
                        total, dataset.expected_total_matches,
                        "streaming indexed benchmark changed match count"
                    );
                    black_box(total);
                });
            },
        );

        group.bench_function(
            BenchmarkId::new("indexed_streaming_rayon_queries", dataset.id),
            |b| {
                b.iter(|| {
                    let total = matrix_match_count_indexed_streaming_rayon_queries(
                        black_box(&dataset.queries),
                        black_box(&dataset.query_screens),
                        black_box(&target_index),
                        black_box(&targets),
                    );
                    assert_eq!(
                        total, dataset.expected_total_matches,
                        "streaming indexed rayon benchmark changed match count"
                    );
                    black_box(total);
                });
            },
        );

        group.throughput(Throughput::Elements(0));
        black_box(&dataset.description);
        black_box(dataset.screened_pairs);
    }

    group.finish();
}

fn bench_evolution_setup(c: &mut Criterion) {
    let inputs = [
        load_raw_input(
            "smarts_evolution_complex_queries_x_examples",
            COMPLEX_QUERY_FIXTURE,
        ),
        load_raw_input(
            "smarts_evolution_large_complex_queries_x_examples",
            LARGE_COMPLEX_QUERY_FIXTURE,
        ),
    ];
    let targets = prepare_targets(&inputs[0].target_smiles);
    let target_index = TargetCorpusIndex::new(&targets);

    let mut group = c.benchmark_group("matcher_evolution_matrix_setup");
    configure_group(&mut group);

    for input in &inputs {
        let queries = compile_queries(&input.query_smarts);
        let query_screens = queries
            .iter()
            .map(|query| QueryScreen::new(query.query()))
            .collect::<Vec<_>>();

        group.throughput(Throughput::Elements(input.query_smarts.len() as u64));
        group.bench_function(BenchmarkId::new("compile_queries", input.id), |b| {
            b.iter(|| {
                let queries = compile_queries(black_box(&input.query_smarts));
                black_box(queries);
            });
        });

        group.bench_function(BenchmarkId::new("build_candidate_sets", input.id), |b| {
            b.iter(|| {
                let candidates =
                    build_candidate_sets(black_box(&query_screens), black_box(&target_index));
                black_box(candidates);
            });
        });
    }

    group.throughput(Throughput::Elements(inputs[0].target_smiles.len() as u64));
    group.bench_function(BenchmarkId::new("prepare_targets", inputs[0].id), |b| {
        b.iter(|| {
            let targets = prepare_targets(black_box(&inputs[0].target_smiles));
            black_box(targets);
        });
    });

    group.bench_function(BenchmarkId::new("build_target_index", inputs[0].id), |b| {
        b.iter(|| {
            let index = TargetCorpusIndex::new(black_box(&targets));
            black_box(index);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_evolution_matrix, bench_evolution_setup);
criterion_main!(benches);
