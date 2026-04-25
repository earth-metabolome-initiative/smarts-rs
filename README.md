# smarts-rs

[![CI](https://github.com/earth-metabolome-initiative/smarts-rs/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/earth-metabolome-initiative/smarts-rs/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/earth-metabolome-initiative/smarts-rs/graph/badge.svg?branch=main)](https://codecov.io/gh/earth-metabolome-initiative/smarts-rs)

Rust support for parsing, canonicalizing, editing, screening, and matching
SMARTS queries against SMILES targets.

## Basic SMARTS Utilities

SMARTS strings parse through `FromStr` into `smarts_rs::QueryMol`. The
parsed query can be inspected directly, used for one-off matching, or compiled
once and reused against prepared targets.

```rust
use core::str::FromStr;

use smarts_rs::{CompiledQuery, PreparedTarget, QueryMol};
use smiles_parser::Smiles;

let query = QueryMol::from_str("[#6]-[#8]").unwrap();
assert_eq!(query.atom_count(), 2);
assert_eq!(query.bond_count(), 1);

assert!(query.matches("CCO").unwrap());
assert!(!query.matches("CCCC").unwrap());

let compiled = CompiledQuery::new(query).unwrap();
let ethanol = PreparedTarget::new(Smiles::from_str("CCO").unwrap());
assert!(compiled.matches(&ethanol));
```

## Matching Safety Fuse

Use `CompiledQuery::matches_with_scratch_and_interrupt` when query evaluation
must remain cancellable. The interrupt predicate is polled cooperatively from
the matcher, so this works on `wasm32-unknown-unknown` by closing over a
host-provided clock instead of relying on Rust's unsupported wasm `Instant`.

```rust
use core::str::FromStr;

use smarts_rs::{CompiledQuery, MatchLimitResult, MatchScratch, PreparedTarget, QueryMol};
use smiles_parser::Smiles;

let query = CompiledQuery::new(QueryMol::from_str("[#6]-[#8]").unwrap()).unwrap();
let target = PreparedTarget::new(Smiles::from_str("CCO").unwrap());
let mut scratch = MatchScratch::new();

let mut polls = 0;
let result = query.matches_with_scratch_and_interrupt(&target, &mut scratch, || {
    polls += 1;
    false
});

assert_eq!(result, MatchLimitResult::Complete(true));
assert!(polls > 0);
```

## Canonicalization

Canonicalization normalizes equivalent SMARTS spellings into one stable query
graph and rendering. This is useful for deduplication, corpus reduction, and
roundtrip stability checks.

```rust
use core::str::FromStr;

use smarts_rs::QueryMol;

let query = QueryMol::from_str("[$(OC)]").unwrap();
let canonical = query.canonicalize();

assert!(!query.is_canonical());
assert!(canonical.is_canonical());
assert_eq!(canonical.to_string(), "[$(CO)]");
assert_eq!(query.to_canonical_smarts(), canonical.to_string());
```

## Target Corpus Index

For many SMARTS queries against a fixed SMILES corpus, prepare the targets once,
build a `TargetCorpusIndex`, then use each new query to get conservative
candidate target ids. The index is only a prefilter: always run exact SMARTS
matching on the candidates before accepting hits.

```rust
use core::str::FromStr;

use smarts_rs::{
    CompiledQuery, MatchScratch, PreparedTarget, QueryMol, QueryScreen, TargetCorpusIndex,
    TargetCorpusScratch,
};
use smiles_parser::Smiles;

let targets = ["CCO", "CCCC", "CC=O", "CCN"]
    .into_iter()
    .map(|smiles| PreparedTarget::new(Smiles::from_str(smiles).unwrap()))
    .collect::<Vec<_>>();

let index = TargetCorpusIndex::new(&targets);
let query = CompiledQuery::new(QueryMol::from_str("[#6]=[#8]").unwrap()).unwrap();
let screen = QueryScreen::new(query.query());

let mut corpus_scratch = TargetCorpusScratch::new();
let candidates = index.candidate_set_with_scratch(&screen, &mut corpus_scratch);

let mut match_scratch = MatchScratch::new();
let hits = candidates
    .target_ids()
    .iter()
    .copied()
    .filter(|&target_id| {
        query.matches_with_scratch(&targets[target_id], &mut match_scratch)
    })
    .collect::<Vec<_>>();

assert_eq!(hits, vec![2]);
```
