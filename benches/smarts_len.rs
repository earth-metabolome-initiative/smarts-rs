//! Criterion benchmark comparing `QueryMol::smarts_len` against the
//! allocation-heavy `to_string().len()` it replaces, bucketed by rendered
//! query size to show the gap widening for larger queries.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use serde::Deserialize;
use smarts_rs::QueryMol;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;

/// Number of queries replayed per bucket, repeating the bucket members so each
/// measurement covers a stable, comparable amount of work.
const TARGET_BATCH_SIZE: usize = 20_000;

/// Inclusive upper rendered-byte bounds defining each size bucket. The final
/// bucket captures everything larger than the previous bound.
const BUCKET_BOUNDS: [(&str, usize); 3] = [("small", 8), ("medium", 32), ("large", usize::MAX)];

#[derive(Debug, Deserialize)]
struct Case {
    #[allow(dead_code)]
    id: String,
    smarts: String,
}

fn corpus_path(file_name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("corpus")
        .join(file_name)
}

fn load_cases(file_name: &str) -> Vec<Case> {
    let path = corpus_path(file_name);
    let raw = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
    serde_json::from_str(&raw)
        .unwrap_or_else(|err| panic!("failed to parse {}: {err}", path.display()))
}

fn load_smarts_lines(file_name: &str) -> Vec<String> {
    let path = corpus_path(file_name);
    let raw = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
    raw.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn bucket_label(rendered_len: usize) -> &'static str {
    BUCKET_BOUNDS
        .iter()
        .find_map(|&(label, upper)| (rendered_len <= upper).then_some(label))
        .unwrap_or("large")
}

fn build_dataset(members: &[QueryMol]) -> Vec<QueryMol> {
    let repeat_count = TARGET_BATCH_SIZE.div_ceil(members.len());
    let mut dataset = Vec::with_capacity(members.len() * repeat_count);
    for _ in 0..repeat_count {
        dataset.extend(members.iter().cloned());
    }
    dataset
}

fn bench_smarts_len(c: &mut Criterion) {
    // The JSON corpora supply small and medium queries; the smarts-evolution
    // complex-query fixtures (the downstream hot path this method serves)
    // supply the large bucket.
    let mut smarts = load_cases("parser/parse-valid-v0.json")
        .into_iter()
        .chain(load_cases("benchmark/parse-extra-cases.json"))
        .map(|case| case.smarts)
        .collect::<Vec<_>>();
    smarts.extend(load_smarts_lines(
        "benchmark/smarts-evolution-complex-queries-v0.smarts",
    ));
    smarts.extend(load_smarts_lines(
        "benchmark/smarts-evolution-complex-queries-large-v0.smarts",
    ));

    let mut buckets: Vec<(&'static str, Vec<QueryMol>)> = BUCKET_BOUNDS
        .iter()
        .map(|&(label, _)| (label, Vec::new()))
        .collect();
    for smarts in smarts {
        let Ok(query) = QueryMol::from_str(&smarts) else {
            continue;
        };
        let label = bucket_label(query.smarts_len());
        if let Some((_, members)) = buckets.iter_mut().find(|(bucket, _)| *bucket == label) {
            members.push(query);
        }
    }

    let mut group = c.benchmark_group("smarts_len");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));

    for (label, members) in &buckets {
        if members.is_empty() {
            continue;
        }
        let dataset = build_dataset(members);
        group.throughput(Throughput::Elements(dataset.len() as u64));

        group.bench_function(BenchmarkId::new("smarts_len", label), |b| {
            b.iter(|| {
                let mut total = 0usize;
                for query in &dataset {
                    total += black_box(query.smarts_len());
                }
                black_box(total)
            });
        });
        group.bench_function(BenchmarkId::new("to_string_len", label), |b| {
            b.iter(|| {
                let mut total = 0usize;
                for query in &dataset {
                    total += black_box(query.to_string().len());
                }
                black_box(total)
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_smarts_len);
criterion_main!(benches);
