# SMARTS Validator Benchmark: RDKit vs Rust

Date: 2026-04-18

## Scope

This benchmark measures **compiled SMARTS matching** against **prepared target
molecules**.

It does **not** measure:

- SMARTS parsing
- SMILES parsing
- target preparation

Those costs are intentionally excluded because the realistic many-SMARTS ×
many-SMILES workflow compiles and prepares both sides once, then reuses them.

## Shared Dataset

The benchmark uses only the frozen RDKit-backed validator fixture suites:

- `corpus/validator/single-atom-v0.rdkit.json`
- `corpus/validator/connected-v0.rdkit.json`
- `corpus/validator/counts-v0.rdkit.json`
- `corpus/validator/ring-v0.rdkit.json`
- `corpus/validator/recursive-v0.rdkit.json`
- `corpus/validator/disconnected-v0.rdkit.json`
- `corpus/validator/stereo-v0.rdkit.json`
- `corpus/validator/stereo-gap-v1.rdkit.json`
- `corpus/validator/stereo-gap-v2.rdkit.json`
- `corpus/validator/stereo-gap-v3.rdkit.json`
- `corpus/validator/stereo-gap-v4.rdkit.json`
- `corpus/validator/stereo-gap-v5.rdkit.json`
- `corpus/validator/stereo-gap-v6.rdkit.json`
- `corpus/validator/stereo-gap-v7.rdkit.json`

The workload manifest is:

- `corpus/benchmark/validator-workloads.json`

Zero-level grouped local semantics are intentionally excluded from the
cross-tool benchmark because RDKit does not support them.

Each workload repeats its unique case set enough times to reach about
`30,000` query-target pairs.

## Reproduction

The Rust side is still reproducible from the repository with:

```bash
cargo bench -p smarts-validator --bench match_corpus
```

The RDKit side was measured with a now-removed local helper script, so the
numbers below should be treated as a historical comparison snapshot unless that
helper is reintroduced.

## Method

### Rust

- SMARTS is parsed once into `QueryMol`
- `QueryMol` is compiled once into `CompiledQuery`
- SMILES is parsed once into `Smiles`
- target molecules are prepared once into `PreparedTarget`
- the timed loop calls `matches_compiled(&CompiledQuery, &PreparedTarget)`
- the compiled query also keeps reusable stereo normalization cache state across
  repeated matches

### RDKit

- SMARTS is compiled once with `Chem.MolFromSmarts(...)`
- SMILES is compiled once with `Chem.MolFromSmiles(...)`
- the timed loop calls `mol.HasSubstructMatch(query, useChirality=...)`

Both sides assert that every benchmark case still matches the frozen expected
RDKit truth value, so the timing loop is also a correctness guard.

## Results

Measured on 2026-04-18:

| Workload | Rust ns / pair | RDKit median ns / pair | Relative result |
| --- | ---: | ---: | ---: |
| `single_atom` | `73.4` | `1079.6` | Rust `14.7x` faster |
| `connected` | `187.2` | `1142.3` | Rust `6.1x` faster |
| `counts` | `100.7` | `1111.3` | Rust `11.0x` faster |
| `ring` | `103.2` | `1162.4` | Rust `11.3x` faster |
| `recursive` | `485.6` | `1524.5` | Rust `3.1x` faster |
| `disconnected` | `167.0` | `1070.8` | Rust `6.4x` faster |
| `stereo` | `1076.2` | `1879.5` | Rust `1.7x` faster |
| `full_validator_corpus` | `518.5` | `1500.1` | Rust `2.9x` faster |

## Interpretation

The current validator core is now faster than RDKit on every measured workload
in the frozen corpus:

- single-atom predicates are already cheap
- count/range and ring workloads are comfortably faster
- recursive SMARTS stays faster than RDKit even after the validator grew more
  semantic coverage
- the full mixed workload is now much faster too, once the Rust benchmark uses
  a real compiled-query object and keeps reusable stereo normalization cache
  state on that compiled query

Stereo was the weak spot before the compiled-query and stereo-cache work. On the
current benchmark it is now ahead too:

- stereo matching is about `1.7x` faster than RDKit on the richer stereo corpus
- that removes the last major drag on the blended full-corpus result

So the benchmark gives a clear optimization priority:

1. keep reducing the cost of recursive SMARTS
2. then improve screening/indexing for many-query × many-target workloads

## Caveats

- RDKit is still called through Python, so the RDKit numbers include Python call
  overhead.
- The benchmark uses the current frozen validator corpus, not a large
  production library.
- Zero-level grouped local SMARTS are excluded because there is no RDKit
  reference path for them.
- RDKit emits warnings for some explicit-hydrogen benchmark molecules during
  initial compilation; those warnings do not affect the timed loop.
