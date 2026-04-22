//! Criterion benchmark for SMARTS canonicalization over shared workload manifests.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use serde::Deserialize;
use smarts_parser::QueryMol;
use std::collections::HashMap;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Duration;

const TARGET_BATCH_SIZE: usize = 60_000;

#[derive(Debug, Deserialize)]
struct BenchmarkCase {
    id: String,
    smarts: String,
}

#[derive(Debug, Deserialize)]
struct Workload {
    id: String,
    description: String,
    case_ids: Vec<String>,
}

fn load_cases() -> Vec<BenchmarkCase> {
    let parser_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../corpus/parser/parse-valid-v0.json");
    let extra_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../corpus/benchmark/parse-extra-cases.json");

    let mut cases: Vec<BenchmarkCase> = serde_json::from_str(
        &fs::read_to_string(parser_path).expect("failed to read parse-valid-v0.json"),
    )
    .expect("failed to parse parser corpus JSON");

    let extra_cases: Vec<BenchmarkCase> = serde_json::from_str(
        &fs::read_to_string(extra_path).expect("failed to read parse-extra-cases.json"),
    )
    .expect("failed to parse benchmark extra cases JSON");
    cases.extend(extra_cases);
    cases
}

fn load_workloads() -> Vec<Workload> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../corpus/benchmark/parse-workloads.json");
    let raw = fs::read_to_string(path).expect("failed to read parse-workloads.json");
    serde_json::from_str(&raw).expect("failed to parse workloads JSON")
}

fn build_dataset(workload: &Workload, queries_by_id: &HashMap<&str, &QueryMol>) -> Vec<QueryMol> {
    let unique: Vec<QueryMol> = workload
        .case_ids
        .iter()
        .map(|case_id| {
            (*queries_by_id
                .get(case_id.as_str())
                .unwrap_or_else(|| panic!("missing SMARTS fixture for workload case id {case_id}")))
            .clone()
        })
        .collect();
    let repeat_count = TARGET_BATCH_SIZE.div_ceil(unique.len());
    let mut dataset = Vec::with_capacity(unique.len() * repeat_count);
    for _ in 0..repeat_count {
        dataset.extend(unique.iter().cloned());
    }
    dataset
}

fn bench_canonicalization_corpus(c: &mut Criterion) {
    let cases = load_cases();
    let workloads = load_workloads();
    let parsed_queries: Vec<(String, QueryMol)> = cases
        .into_iter()
        .map(|case| {
            let query = case
                .smarts
                .parse()
                .unwrap_or_else(|error| panic!("benchmark SMARTS {} must parse: {error}", case.id));
            (case.id, query)
        })
        .collect();
    let queries_by_id: HashMap<&str, &QueryMol> = parsed_queries
        .iter()
        .map(|(id, query)| (id.as_str(), query))
        .collect();

    let mut group = c.benchmark_group("canonicalization_workloads");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));

    for workload in workloads {
        let dataset = build_dataset(&workload, &queries_by_id);
        let k = dataset.len() as u64;
        group.throughput(Throughput::Elements(k));
        group.bench_function(BenchmarkId::new("canonicalize_query", &workload.id), |b| {
            b.iter(|| {
                for query in &dataset {
                    let canonical = query.canonicalize();
                    black_box(canonical);
                }
            });
        });
        group.bench_function(
            BenchmarkId::new("canonical_smarts_string", &workload.id),
            |b| {
                b.iter(|| {
                    for query in &dataset {
                        let canonical = query.to_canonical_smarts();
                        black_box(canonical);
                    }
                });
            },
        );
        group.throughput(Throughput::Elements(0));
        black_box(&workload.description);
    }

    group.finish();
}

criterion_group!(benches, bench_canonicalization_corpus);
criterion_main!(benches);
