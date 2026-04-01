# SMARTS Parser Benchmark: RDKit vs Rust

Date: 2026-04-01

## Dataset

- Source fixtures:
  - `corpus/parser/parse-valid-v0.json`
  - `corpus/benchmark/parse-extra-cases.json`
- Workload manifest: `corpus/benchmark/parse-workloads.json`
- Target batch size per workload: about `120,000` SMARTS

The comparison suite now uses a shared RDKit-compatible corpus instead of the older
29-case parser-fixture slice. The current shared benchmark corpus contains:

- `59` unique SMARTS
- `38` cases from the parser conformance corpus
- `21` benchmark-only synthetic shared cases
- median SMARTS length `6`
- average SMARTS length about `7.1`
- maximum SMARTS length `20`

Character inventory across the shared corpus:

```text
!#$%&()+,-./0123456789;=@BCDFHNOPRSX[\]aclnrtvx~
```

Zero-level grouped component SMARTS like `(C.C)` and `(C).(C).C` are intentionally
excluded from the comparison suite because RDKit does not parse them, even though
`smarts-parser` now does.

Each workload repeats its unique SMARTS enough times to reach roughly the same batch
size, so the comparison is not distorted by tiny category-specific batches.

## Commands

Rust benchmark:

```bash
cargo bench -p smarts-parser --bench parse_corpus
```

RDKit benchmark:

```bash
uv run --with rdkit python scripts/benchmark_rdkit_parse.py
```

## Results

The benchmark now reports multiple workloads instead of a single blended batch:

- `simple_atoms_bonds`
- `bond_expressions`
- `bracket_primitives`
- `topology_rings`
- `recursive_mixed`
- `full_shared_corpus`

Each Rust workload is timed with `criterion` 0.8.2. Each RDKit workload is timed
with 10 Python-driven repeats using `SmartsParserParams` with CX/name parsing disabled,
and the Python loop parses and drops immediately instead of retaining all parsed
`Mol` objects in a list.

Measured on 2026-04-01:

| Workload | Rust time | Rust ns / SMARTS | RDKit median ns / SMARTS | Speedup |
| --- | ---: | ---: | ---: | ---: |
| `simple_atoms_bonds` | `16.84 ms` for `120,000` SMARTS | `140.4` | `1985.6` | `14.1x` |
| `bond_expressions` | `30.58 ms` for `120,000` SMARTS | `254.8` | `2372.9` | `9.3x` |
| `bracket_primitives` | `29.30 ms` for `120,015` SMARTS | `244.1` | `1716.2` | `7.0x` |
| `topology_rings` | `37.06 ms` for `120,010` SMARTS | `308.8` | `5077.4` | `16.4x` |
| `recursive_mixed` | `57.79 ms` for `120,001` SMARTS | `481.6` | `3949.2` | `8.2x` |
| `full_shared_corpus` | `33.25 ms` for `120,006` SMARTS | `277.1` | `2940.4` | `10.6x` |

## Comparison

The widened corpus changes the headline a bit, but the workload split is more useful:

- The Rust parser stays below about `500 ns / SMARTS` even on the recursive-heavy slice.
- The bracket-heavy workload is now the smallest gap at about `7.0x`.
- Ring-heavy SMARTS are still the most expensive category for both parsers and still produce the largest gap.
- On the broader `59`-case shared corpus, the Rust parser is about `10.6x` faster.

Any future parser optimization should be read against these workload categories, not
only against the old tiny blended batch.

## Important RDKit Caveat

The RDKit numbers above still come from Python, so they still include:

- Python string conversion
- Python `Mol` wrapper creation
- Python call overhead

## Caveats

- This measures parser throughput only, not matching.
- RDKit is called through Python, so the result includes Python call overhead.
- The comparison suite is intentionally limited to the shared SMARTS subset that both parsers accept.
- The benchmark corpus is broader than the parser-fixture corpus, but it is still small compared with a real public production corpus.
