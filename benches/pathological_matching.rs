//! Criterion benchmarks for exact matching cases that can trigger large
//! backtracking search spaces.

use core::str::FromStr;
use std::{fs, hint::black_box, path::PathBuf, time::Duration};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use smarts_rs::{
    CompiledQuery, MatchScratch, PreparedTarget, QueryMol, QueryScreen, TargetCandidateSet,
    TargetCorpusIndex, TargetCorpusScratch,
};
use smiles_parser::Smiles;

const TARGET_FIXTURE: &str = "corpus/benchmark/smarts-evolution-example-smiles-v0.tsv";
const LOGGED_SLOW_TARGET_LIMIT: usize = 4096;

struct PathologicalCase {
    id: &'static str,
    description: &'static str,
    smarts: &'static str,
    smiles: String,
    expected_match: bool,
    repeat_count: usize,
}

struct LoggedSlowCase {
    id: &'static str,
    description: &'static str,
    smarts: &'static str,
    reported_generation: usize,
    reported_elapsed_ms: usize,
    reported_targets: usize,
    reported_complexity: usize,
}

struct RuntimeCase {
    query: CompiledQuery,
    target: PreparedTarget,
    expected_match: bool,
    repeat_count: usize,
}

struct LoggedSlowRuntimeCase {
    id: &'static str,
    description: &'static str,
    query: CompiledQuery,
    query_screen: QueryScreen,
    candidate_set: TargetCandidateSet,
    expected_matches: usize,
    reported_generation: usize,
    reported_elapsed_ms: usize,
    reported_targets: usize,
    reported_complexity: usize,
}

fn repeated_components(component: &str, count: usize) -> String {
    vec![component; count].join(".")
}

fn alkane_chain(atom_count: usize) -> String {
    "C".repeat(atom_count)
}

fn build_cases() -> Vec<PathologicalCase> {
    vec![
        PathologicalCase {
            id: "branched_missing_hetero_leaf",
            description: "many branch-compatible carbon centers miss one required hetero leaf",
            smarts: "C(O)(N)(S)F",
            smiles: repeated_components("C(O)(O)(S)F", 96),
            expected_match: false,
            repeat_count: 200,
        },
        PathologicalCase {
            id: "recursive_branched_missing_hetero_leaf",
            description: "same branch failure hidden behind a recursive SMARTS predicate",
            smarts: "[$(C(O)(N)(S)F)]",
            smiles: repeated_components("C(O)(O)(S)F", 96),
            expected_match: false,
            repeat_count: 200,
        },
        PathologicalCase {
            id: "six_cycle_query_against_long_chain",
            description: "underconstrained six-cycle query must fail late on many acyclic paths",
            smarts: "*1~*~*~*~*~*1",
            smiles: alkane_chain(96),
            expected_match: false,
            repeat_count: 100,
        },
        PathologicalCase {
            id: "branched_positive_control",
            description: "branch-heavy positive case guards against tuning only negative failures",
            smarts: "C(O)(N)(S)F",
            smiles: repeated_components("C(O)(N)(S)F", 96),
            expected_match: true,
            repeat_count: 500,
        },
    ]
}

fn build_logged_slow_cases() -> Vec<LoggedSlowCase> {
    let mut cases = Vec::new();
    cases.extend(build_alkaloids_logged_slow_cases());
    cases.extend(build_carotenoids_logged_slow_cases());
    cases
}

const fn build_alkaloids_logged_slow_cases() -> [LoggedSlowCase; 6] {
    [
        LoggedSlowCase {
            id: "alkaloids_generation_19_len219_complexity3",
            description: "generation 19 Alkaloids slow evaluation: 34.839s over 232408 targets",
            smarts: "[!!a,+7,R,Z,Z{7-},!*,D16&!X{-12}&-3;@SP3&F&X;Cm&16O;A;A,v;Rh;Z;Z{5-};x9;z7;z{-7},!@TB10,-2;-7;14n;15n,^1;@OH17;D{-14};D{3-};r29;x6;z].[!14N;#95&-4&a;$([#6]~[#7&$([#6])])&X14;x{14-},#41,@,A,X6;a;r,x16&!Bk&X{-13}].[#7;H2]",
            reported_generation: 19,
            reported_elapsed_ms: 34_839,
            reported_targets: 232_408,
            reported_complexity: 3,
        },
        LoggedSlowCase {
            id: "alkaloids_generation_70_len117_complexity5",
            description: "generation 70 Alkaloids slow evaluation: 61.364s over 232408 targets",
            smarts: "*[!12C,D,D11,D{-16},h,v14].([!!37585*,!v{10-};#79;#85;v9;x11,o,z11;53903*;A;A;r30;r7,$([#7:43954]),r15].[#7;R:30837])",
            reported_generation: 70,
            reported_elapsed_ms: 61_364,
            reported_targets: 232_408,
            reported_complexity: 5,
        },
        LoggedSlowCase {
            id: "alkaloids_generation_70_len144_complexity9",
            description: "generation 70 Alkaloids slow evaluation: 94.046s over 232408 targets",
            smarts: "*[!!12C,D,D11,D{-16},h,v14].([!37585*,!v{10-};#79;#85;v9;x11,o,z11;53903*;A;A;r30;r7,$([#7:43954]),r15:62709](!:&~[!r6:36126])@[@].[#7;R:30837])",
            reported_generation: 70,
            reported_elapsed_ms: 94_046,
            reported_targets: 232_408,
            reported_complexity: 9,
        },
        LoggedSlowCase {
            id: "alkaloids_generation_73_len106_complexity7",
            description: "generation 73 Alkaloids slow evaluation: 48.408s over 232408 targets",
            smarts: "**.[!!37585*,!v{10-};#79;#85;v9;x11,o,z11;53903*;A;A;r30;r7,$([#7:43954]),r15]!:&~[!r6:36126].[#7;R:30837]",
            reported_generation: 73,
            reported_elapsed_ms: 48_408,
            reported_targets: 232_408,
            reported_complexity: 7,
        },
        LoggedSlowCase {
            id: "alkaloids_generation_73_len64_complexity3",
            description: "generation 73 Alkaloids slow evaluation: 164.781s over 232408 targets",
            smarts: "[!!12C,16O,D,D11,h,v14].([!r6&$([#6,#7:44963]):57773].[R:30837])",
            reported_generation: 73,
            reported_elapsed_ms: 164_781,
            reported_targets: 232_408,
            reported_complexity: 3,
        },
        LoggedSlowCase {
            id: "alkaloids_generation_74_len80_complexity5",
            description: "generation 74 Alkaloids slow evaluation: 33.486s over 232408 targets",
            smarts: "*.[!!37585*,!v{10-};#79;#85;v9;x11,o,z11;53903*;A;A;r;r30,r15].[!#6][$([!#1])&c]",
            reported_generation: 74,
            reported_elapsed_ms: 33_486,
            reported_targets: 232_408,
            reported_complexity: 5,
        },
    ]
}

const fn build_carotenoids_logged_slow_cases() -> [LoggedSlowCase; 9] {
    [
        LoggedSlowCase {
            id: "carotenoids_generation_5_len236_complexity8",
            description: "generation 5 Carotenoids slow evaluation: 42.425s over 229455 targets",
            smarts: "*[!!H,H10;z{8-},as,x14,#50;#67,#84;a,$([!67*;#71&$([#6]~[#8])&$([#8])&z{2-};$([#6]~[#8]);-3;9875*;R&14n&16o&Z11;A;D{-13};r{-31}]@[#7]),$([#6]),+0,21451*,@TB13,A,A,D{2-};H;Ir&x0;X;X{-12},z4,z{13-15};z,R{-3}].*([!$([!#1]);R:32752])-,/[#6]",
            reported_generation: 5,
            reported_elapsed_ms: 42_425,
            reported_targets: 229_455,
            reported_complexity: 8,
        },
        LoggedSlowCase {
            id: "carotenoids_generation_12_len135_complexity7",
            description: "generation 12 Carotenoids slow evaluation: 1008.763s over 229455 targets",
            smarts: "([!!-2,8610*,A,X16&!58236*&@TH1&R15,Z{-16},h16,#50].[!#6;X3][!a,!r{8-}&40476*;X;v&A;R1;Z{7-};$([#6]~[#8]);a&R{14-}&v2][#6].[!$([!#1])])",
            reported_generation: 12,
            reported_elapsed_ms: 1_008_763,
            reported_targets: 229_455,
            reported_complexity: 7,
        },
        LoggedSlowCase {
            id: "carotenoids_generation_14_len119_complexity10",
            description: "generation 14 Carotenoids slow evaluation: 39.875s over 229455 targets",
            smarts: "*:[!!41626*&-4&18O&30998*&@,r{27-}&x16,v].([!!#75,-2,A,X16&!58236*&@TH1&R15,Z{-16},h16,#50].[!$([!#1])].[#6]#[#6]-[#6])",
            reported_generation: 14,
            reported_elapsed_ms: 39_875,
            reported_targets: 229_455,
            reported_complexity: 10,
        },
        LoggedSlowCase {
            id: "carotenoids_generation_14_len29_complexity5",
            description: "generation 14 Carotenoids slow evaluation: 212.402s over 229455 targets",
            smarts: "*!:;~[!^{-16}].[r].[v2:14569]",
            reported_generation: 14,
            reported_elapsed_ms: 212_402,
            reported_targets: 229_455,
            reported_complexity: 5,
        },
        LoggedSlowCase {
            id: "carotenoids_generation_15_len72_complexity5",
            description: "generation 15 Carotenoids slow evaluation: 43.168s over 229455 targets",
            smarts: "([!#6]-[#6].[!-2,8610*,A,X16&!58236*&@TH1&R15,Z{-16},h16,#50].[$([#1])])",
            reported_generation: 15,
            reported_elapsed_ms: 43_168,
            reported_targets: 229_455,
            reported_complexity: 5,
        },
        LoggedSlowCase {
            id: "carotenoids_generation_16_len64_complexity6",
            description: "generation 16 Carotenoids slow evaluation: 36.591s over 229455 targets",
            smarts: "([!#6]-[!$([#7]~[48054*]):23558]#&/[!$([!#1:31672])].[$([!#1])])",
            reported_generation: 16,
            reported_elapsed_ms: 36_591,
            reported_targets: 229_455,
            reported_complexity: 6,
        },
        LoggedSlowCase {
            id: "carotenoids_generation_17_len99_complexity5",
            description: "generation 17 Carotenoids slow evaluation: 49.502s over 229455 targets",
            smarts: "[!-2,8610*,A,X16&!58236*&@TH1&R15,Z{-16},h16,#50:41291].(*-[#6].[$([!#1&$([#6]-[#8].[#7])]):26257])",
            reported_generation: 17,
            reported_elapsed_ms: 49_502,
            reported_targets: 229_455,
            reported_complexity: 5,
        },
        LoggedSlowCase {
            id: "carotenoids_generation_18_len171_complexity3",
            description: "generation 18 Carotenoids slow evaluation: 101.203s over 229455 targets",
            smarts: "[!-2,8610*,A,X16&!58236*&@TH1&R15,Z{-16},h16,#50:41291].[#6].[$([!#6,#7]!-[$([!R&!z10&$([#8]);A;H8;z{-13};$([#6,#7]),$([#6,#7]),H{-12},^12&A&H{-0};Sc&Pu&r32;Z{6-};z15])])]",
            reported_generation: 18,
            reported_elapsed_ms: 101_203,
            reported_targets: 229_455,
            reported_complexity: 3,
        },
        LoggedSlowCase {
            id: "carotenoids_generation_18_len257_complexity13",
            description: "generation 18 Carotenoids slow evaluation: 685.804s over 229455 targets",
            smarts: "*(~[!$([#6]-[#8].[#7])&$([#6]=,~[#7]),R5;13C&^{13-}&v{16-}])#,@[!$([#7]~[48054*]):23558][*:28867].[!$([#8]),^{-14},c&Cm&x&x16,!@@;X{-4};!r20&5753*&H{-12}&r{16-},!R0,!Z3,a;h&#118;#83&$([!#1])&@AL2&x{16-};A;A&r9;Cs;X&z]/&:[!z12:29594].[#6;r5:51116]-[$([!#1])]",
            reported_generation: 18,
            reported_elapsed_ms: 685_804,
            reported_targets: 229_455,
            reported_complexity: 13,
        },
    ]
}

fn build_runtime_case(case: &PathologicalCase) -> RuntimeCase {
    let query = CompiledQuery::new(
        QueryMol::from_str(case.smarts)
            .unwrap_or_else(|_| panic!("pathological benchmark SMARTS must parse: {}", case.id)),
    )
    .unwrap_or_else(|_| panic!("pathological benchmark SMARTS must compile: {}", case.id));
    let target = PreparedTarget::new(
        Smiles::from_str(&case.smiles)
            .unwrap_or_else(|_| panic!("pathological benchmark SMILES must parse: {}", case.id)),
    );

    RuntimeCase {
        query,
        target,
        expected_match: case.expected_match,
        repeat_count: case.repeat_count,
    }
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn load_target_smiles() -> Vec<String> {
    let path = repo_root().join(TARGET_FIXTURE);
    let raw =
        fs::read_to_string(&path).unwrap_or_else(|_| panic!("failed to read {}", path.display()));

    raw.lines()
        .enumerate()
        .filter_map(|(line_idx, line)| {
            if line_idx == 0 {
                assert_eq!(line, "dataset\tlabel\tsmiles", "fixture header drifted");
                return None;
            }
            let mut fields = line.splitn(3, '\t');
            let _dataset = fields
                .next()
                .unwrap_or_else(|| panic!("missing dataset field at line {}", line_idx + 1));
            let _label = fields
                .next()
                .unwrap_or_else(|| panic!("missing label field at line {}", line_idx + 1));
            let smiles = fields
                .next()
                .unwrap_or_else(|| panic!("missing SMILES field at line {}", line_idx + 1));
            Some(smiles.to_string())
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

fn logged_slow_match_count(query: &CompiledQuery, targets: &[PreparedTarget]) -> usize {
    let mut scratch = MatchScratch::new();
    targets
        .iter()
        .map(|target| usize::from(query.matches_with_scratch(target, &mut scratch)))
        .sum()
}

fn logged_slow_candidate_match_count(
    query: &CompiledQuery,
    candidate_set: &TargetCandidateSet,
    targets: &[PreparedTarget],
) -> usize {
    let mut scratch = MatchScratch::new();
    candidate_set
        .target_ids()
        .iter()
        .map(|&target_id| {
            usize::from(query.matches_with_scratch(&targets[target_id], &mut scratch))
        })
        .sum()
}

fn logged_slow_indexed_match_count(
    query: &CompiledQuery,
    query_screen: &QueryScreen,
    target_index: &TargetCorpusIndex,
    targets: &[PreparedTarget],
) -> usize {
    let mut corpus_scratch = TargetCorpusScratch::new();
    let candidate_set = target_index.candidate_set_with_scratch(query_screen, &mut corpus_scratch);
    logged_slow_candidate_match_count(query, &candidate_set, targets)
}

fn build_logged_slow_runtime_cases(
    targets: &[PreparedTarget],
    target_index: &TargetCorpusIndex,
) -> Vec<LoggedSlowRuntimeCase> {
    build_logged_slow_cases()
        .into_iter()
        .map(|case| {
            let query = CompiledQuery::new(
                QueryMol::from_str(case.smarts)
                    .unwrap_or_else(|_| panic!("logged slow SMARTS must parse: {}", case.id)),
            )
            .unwrap_or_else(|_| panic!("logged slow SMARTS must compile: {}", case.id));
            let query_screen = QueryScreen::new(query.query());
            let candidate_set = target_index.candidate_set(&query_screen);
            let expected_matches = logged_slow_match_count(&query, targets);
            assert_eq!(
                logged_slow_candidate_match_count(&query, &candidate_set, targets),
                expected_matches,
                "indexed benchmark changed match count for {}",
                case.id
            );

            LoggedSlowRuntimeCase {
                id: case.id,
                description: case.description,
                query,
                query_screen,
                candidate_set,
                expected_matches,
                reported_generation: case.reported_generation,
                reported_elapsed_ms: case.reported_elapsed_ms,
                reported_targets: case.reported_targets,
                reported_complexity: case.reported_complexity,
            }
        })
        .collect()
}

fn configure_group(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(8));
}

fn bench_pathological_boolean_matching(c: &mut Criterion) {
    let cases = build_cases();
    let mut group = c.benchmark_group("matcher_pathological_boolean");
    configure_group(&mut group);

    for case in &cases {
        let runtime = build_runtime_case(case);
        assert_eq!(
            runtime.query.matches(&runtime.target),
            runtime.expected_match,
            "pathological benchmark fixture drifted: {}",
            case.id
        );

        group.throughput(Throughput::Elements(runtime.repeat_count as u64));
        group.bench_function(
            BenchmarkId::new("matches_with_reused_scratch", case.id),
            |b| {
                b.iter(|| {
                    let mut scratch = MatchScratch::new();
                    for _ in 0..runtime.repeat_count {
                        let matched = black_box(&runtime.query)
                            .matches_with_scratch(black_box(&runtime.target), &mut scratch);
                        assert_eq!(
                            matched, runtime.expected_match,
                            "pathological benchmark fixture drifted: {}",
                            case.id
                        );
                        black_box(matched);
                    }
                });
            },
        );
        group.throughput(Throughput::Elements(0));
        black_box(case.description);
    }

    group.finish();
}

fn bench_logged_slow_smarts_over_example_targets(c: &mut Criterion) {
    let mut target_smiles = load_target_smiles();
    target_smiles.truncate(LOGGED_SLOW_TARGET_LIMIT);
    let targets = prepare_targets(&target_smiles);
    let target_index = TargetCorpusIndex::new(&targets);
    let runtime_cases = build_logged_slow_runtime_cases(&targets, &target_index);

    let mut group = c.benchmark_group("matcher_logged_slow_smarts_boolean");
    configure_group(&mut group);

    for case in &runtime_cases {
        group.throughput(Throughput::Elements(targets.len() as u64));
        group.bench_function(
            BenchmarkId::new("scalar_over_example_targets", case.id),
            |b| {
                b.iter(|| {
                    let total =
                        logged_slow_match_count(black_box(&case.query), black_box(&targets));
                    assert_eq!(
                        total, case.expected_matches,
                        "logged slow SMARTS benchmark fixture drifted: {}",
                        case.id
                    );
                    black_box(total);
                });
            },
        );
        group.throughput(Throughput::Elements(case.candidate_set.len() as u64));
        group.bench_function(
            BenchmarkId::new("indexed_reused_candidates", case.id),
            |b| {
                b.iter(|| {
                    let total = logged_slow_candidate_match_count(
                        black_box(&case.query),
                        black_box(&case.candidate_set),
                        black_box(&targets),
                    );
                    assert_eq!(
                        total, case.expected_matches,
                        "logged slow SMARTS indexed benchmark fixture drifted: {}",
                        case.id
                    );
                    black_box(total);
                });
            },
        );
        group.throughput(Throughput::Elements(targets.len() as u64));
        group.bench_function(
            BenchmarkId::new("indexed_build_candidates_and_match", case.id),
            |b| {
                b.iter(|| {
                    let total = logged_slow_indexed_match_count(
                        black_box(&case.query),
                        black_box(&case.query_screen),
                        black_box(&target_index),
                        black_box(&targets),
                    );
                    assert_eq!(
                        total, case.expected_matches,
                        "logged slow SMARTS indexed benchmark fixture drifted: {}",
                        case.id
                    );
                    black_box(total);
                });
            },
        );
        group.throughput(Throughput::Elements(0));
        black_box(case.description);
        black_box(case.candidate_set.len());
        black_box(case.reported_generation);
        black_box(case.reported_elapsed_ms);
        black_box(case.reported_targets);
        black_box(case.reported_complexity);
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_pathological_boolean_matching,
    bench_logged_slow_smarts_over_example_targets
);
criterion_main!(benches);
