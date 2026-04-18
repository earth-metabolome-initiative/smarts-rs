//! Criterion benchmark for compiled SMARTS matching on shared validator workloads.

use core::str::FromStr;
use std::{fs, hint::black_box, path::PathBuf, time::Duration};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use serde::Deserialize;
use smarts_parser::QueryMol;
use smarts_validator::{matches_compiled, CompiledQuery, PreparedTarget};
use smiles_parser::Smiles;

const TARGET_BATCH_SIZE: usize = 30_000;

#[derive(Debug, Deserialize)]
struct BenchmarkCase {
    name: String,
    smarts: String,
    smiles: String,
    expected_match: bool,
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
}

struct RuntimeDataset {
    cases: Vec<RuntimeCase>,
    repeat_count: usize,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn load_workloads() -> Vec<Workload> {
    let path = repo_root().join("corpus/benchmark/validator-workloads.json");
    let raw = fs::read_to_string(path).expect("failed to read validator-workloads.json");
    serde_json::from_str(&raw).expect("failed to parse validator workloads JSON")
}

fn load_cases(case_files: &[String]) -> Vec<BenchmarkCase> {
    let mut cases = Vec::new();
    let corpus_root = repo_root().join("corpus/validator");
    for case_file in case_files {
        let path = corpus_root.join(case_file);
        let raw = fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("failed to read validator fixture {}", path.display()));
        let mut file_cases: Vec<BenchmarkCase> = serde_json::from_str(&raw)
            .unwrap_or_else(|_| panic!("failed to parse validator fixture {}", path.display()));
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
        })
        .collect();
    RuntimeDataset {
        cases: runtime_cases,
        repeat_count,
    }
}

fn bench_match_corpus(c: &mut Criterion) {
    let workloads = load_workloads();
    let mut group = c.benchmark_group("validator_workloads");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));

    for workload in workloads {
        let cases = load_cases(&workload.case_files);
        let dataset = build_dataset(&cases);
        let k = (dataset.cases.len() * dataset.repeat_count) as u64;
        group.throughput(Throughput::Elements(k));
        group.bench_function(
            BenchmarkId::new("rust_smarts_validator", &workload.id),
            |b| {
                b.iter(|| {
                    for _ in 0..dataset.repeat_count {
                        for case in &dataset.cases {
                            let matched =
                                matches_compiled(black_box(&case.query), black_box(&case.target));
                            assert_eq!(matched, case.expected_match, "benchmark fixture drifted");
                            black_box(matched);
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
