//! Criterion benchmark for compiled SMARTS matching on shared matching workloads.

use core::str::FromStr;
use std::{fs, hint::black_box, path::PathBuf, time::Duration};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use serde::Deserialize;
use smarts_rs::{CompiledQuery, PreparedTarget, QueryMol};
use smiles_parser::Smiles;

const TARGET_BATCH_SIZE: usize = 30_000;

#[derive(Debug, Deserialize)]
struct BenchmarkCase {
    name: String,
    smarts: String,
    smiles: String,
    expected_match: bool,
    #[serde(default)]
    expected_count: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct Workload {
    id: String,
    description: String,
    case_files: Vec<String>,
}

struct RuntimeCase {
    query: CompiledQuery,
    target: PreparedTarget,
    expected_match: bool,
    expected_count: Option<usize>,
}

struct RuntimeDataset {
    cases: Vec<RuntimeCase>,
    repeat_count: usize,
}

#[derive(Debug, Clone, Copy)]
enum MatchBenchmark {
    Boolean,
    Count,
    Materialized,
}

impl MatchBenchmark {
    const fn group_name(self) -> &'static str {
        match self {
            Self::Boolean => "matcher_boolean_workloads",
            Self::Count => "matcher_count_workloads",
            Self::Materialized => "matcher_materialized_workloads",
        }
    }

    fn run_case(self, case: &RuntimeCase) {
        match self {
            Self::Boolean => {
                let matched = black_box(&case.query).matches(black_box(&case.target));
                assert_eq!(matched, case.expected_match, "benchmark fixture drifted");
                black_box(matched);
            }
            Self::Count => {
                let count = black_box(&case.query).match_count(black_box(&case.target));
                assert_eq!(count > 0, case.expected_match, "benchmark fixture drifted");
                if let Some(expected_count) = case.expected_count {
                    assert_eq!(count, expected_count, "benchmark fixture drifted");
                }
                black_box(count);
            }
            Self::Materialized => {
                let matches = black_box(&case.query).substructure_matches(black_box(&case.target));
                assert_eq!(
                    !matches.is_empty(),
                    case.expected_match,
                    "benchmark fixture drifted"
                );
                if let Some(expected_count) = case.expected_count {
                    assert_eq!(matches.len(), expected_count, "benchmark fixture drifted");
                }
                black_box(matches);
            }
        }
    }
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn load_workloads() -> Vec<Workload> {
    let path = repo_root().join("corpus/benchmark/matching-workloads.json");
    let raw = fs::read_to_string(path).expect("failed to read matching-workloads.json");
    serde_json::from_str(&raw).expect("failed to parse matching workloads JSON")
}

fn load_cases(case_files: &[String]) -> Vec<BenchmarkCase> {
    let mut cases = Vec::new();
    let matching_corpus_root = repo_root().join("corpus/matching");
    for case_file in case_files {
        let path = matching_corpus_root.join(case_file);
        let raw = fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("failed to read matching fixture {}", path.display()));
        let mut file_cases: Vec<BenchmarkCase> = serde_json::from_str(&raw)
            .unwrap_or_else(|_| panic!("failed to parse matching fixture {}", path.display()));
        cases.append(&mut file_cases);
    }
    cases
}

fn build_dataset(cases: &[BenchmarkCase]) -> RuntimeDataset {
    let repeat_count = TARGET_BATCH_SIZE.div_ceil(cases.len());
    let runtime_cases = cases
        .iter()
        .map(|case| RuntimeCase {
            query: CompiledQuery::new(
                QueryMol::from_str(&case.smarts)
                    .unwrap_or_else(|_| panic!("benchmark SMARTS must parse: {}", case.name)),
            )
            .unwrap_or_else(|_| panic!("benchmark SMARTS must stay supported: {}", case.name)),
            target: PreparedTarget::new(
                Smiles::from_str(&case.smiles)
                    .unwrap_or_else(|_| panic!("benchmark SMILES must parse: {}", case.name)),
            ),
            expected_match: case.expected_match,
            expected_count: case.expected_count,
        })
        .collect();
    RuntimeDataset {
        cases: runtime_cases,
        repeat_count,
    }
}

fn bench_match_corpus(c: &mut Criterion) {
    let workloads = load_workloads();
    for benchmark in [
        MatchBenchmark::Boolean,
        MatchBenchmark::Count,
        MatchBenchmark::Materialized,
    ] {
        bench_workloads(c, &workloads, benchmark);
    }
}

fn configure_group(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));
}

fn bench_workloads(c: &mut Criterion, workloads: &[Workload], benchmark: MatchBenchmark) {
    let mut group = c.benchmark_group(benchmark.group_name());
    configure_group(&mut group);

    for workload in workloads {
        let cases = load_cases(&workload.case_files);
        let dataset = build_dataset(&cases);
        let k = (dataset.cases.len() * dataset.repeat_count) as u64;
        group.throughput(Throughput::Elements(k));
        group.bench_function(
            BenchmarkId::new("rust_smarts_rs_matcher", &workload.id),
            |b| {
                b.iter(|| {
                    for _ in 0..dataset.repeat_count {
                        for case in &dataset.cases {
                            benchmark.run_case(case);
                        }
                    }
                });
            },
        );
        group.throughput(Throughput::Elements(0));
        black_box(&workload.description);
    }

    group.finish();
}

criterion_group!(benches, bench_match_corpus);
criterion_main!(benches);
