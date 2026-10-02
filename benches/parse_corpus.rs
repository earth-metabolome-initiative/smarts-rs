//! Criterion benchmark for parsing shared SMARTS workload manifests.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use serde::Deserialize;
use smarts_rs::parse_smarts;
use std::collections::HashMap;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Duration;

const TARGET_BATCH_SIZE: usize = 120_000;

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
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus/parser/parse-valid-v0.json");
    let extra_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus/benchmark/parse-extra-cases.json");

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
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("corpus/benchmark/parse-workloads.json");
    let raw = fs::read_to_string(path).expect("failed to read parse-workloads.json");
    serde_json::from_str(&raw).expect("failed to parse workloads JSON")
}

fn build_dataset(workload: &Workload, smarts_by_id: &HashMap<&str, &str>) -> Vec<String> {
    let unique: Vec<String> = workload
        .case_ids
        .iter()
        .map(|case_id| {
            smarts_by_id
                .get(case_id.as_str())
                .unwrap_or_else(|| panic!("missing SMARTS fixture for workload case id {case_id}"))
                .to_string()
        })
        .collect();
    let repeat_count = TARGET_BATCH_SIZE.div_ceil(unique.len());
    let mut dataset = Vec::with_capacity(unique.len() * repeat_count);
    for _ in 0..repeat_count {
        dataset.extend(unique.iter().cloned());
    }
    dataset
}

fn bench_parse_corpus(c: &mut Criterion) {
    let cases = load_cases();
    let workloads = load_workloads();
    let smarts_by_id: HashMap<&str, &str> = cases
        .iter()
        .map(|case| (case.id.as_str(), case.smarts.as_str()))
        .collect();

    let mut group = c.benchmark_group("parse_workloads");
    group.sample_size(12);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));

    for workload in workloads {
        let dataset = build_dataset(&workload, &smarts_by_id);
        let k = dataset.len() as u64;
        group.throughput(Throughput::Elements(k));
        group.bench_function(
            BenchmarkId::new("rust_smarts_rs_parser", &workload.id),
            |b| {
                b.iter(|| {
                    for smarts in &dataset {
                        let query = parse_smarts(black_box(smarts))
                            .expect("valid corpus SMARTS must parse");
                        black_box(query);
                    }
                });
            },
        );
        group.throughput(Throughput::Elements(0));
        black_box(&workload.description);
    }

    group.finish();
}

criterion_group!(benches, bench_parse_corpus);
criterion_main!(benches);
