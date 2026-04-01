# SMARTS Parser Performance Analysis

Date: 2026-03-31

## Scope

This note explains:

- what is likely making the current Rust parser slower than it could be
- what RDKit is doing during `MolFromSmarts()` beyond bare syntax parsing
- which optimizations are likely worth doing first

It is based on:

- the current `smarts-parser` implementation in this repo
- local workload benchmarks already added under `corpus/benchmark/parse-workloads.json`
- direct timing of RDKit through `uv run --with rdkit`
- inspection of RDKit parser sources

## Corrected RDKit Baseline

The earlier RDKit benchmark in `parser-benchmark-rdkit-vs-rust.md` timed this pattern:

```python
parsed = [Chem.MolFromSmarts(smarts) for smarts in dataset]
```

That benchmark includes:

- Python string conversion
- Python wrapper object creation for each `Mol`
- retaining the full list of parsed molecules until the batch completes

That is useful as an end-user number, but it overstates parser-core cost.

For the current `full_corpus` workload, I measured:

| RDKit Python pattern | Median time | ns / SMARTS |
| --- | ---: | ---: |
| no-op Python loop | `0.000220 s` | `7.3` |
| keep list, default `MolFromSmarts()` | `0.111333 s` | `3709.2` |
| parse and drop immediately | `0.067435 s` | `2246.7` |
| parse and drop with `allowCXSMILES=False`, `parseName=False`, `strictCXSMILES=False` | `0.063172 s` | `2104.7` |

So the more defensible current comparison is:

- Rust parser, `full_corpus`: about `204.1 ns / SMARTS`
- RDKit via Python, parse-and-drop, reduced parser options: about `2104.7 ns / SMARTS`

That is still about `10.3x` faster for the Rust parser, but it is not the earlier `16.7x` headline.

## Current Rust Parser: Likely Cost Centers

### 1. Full source retention in `QueryMol`

`QueryMol` stores the whole input SMARTS as an owned `String`.

Relevant file:

- `crates/smarts-parser/src/query.rs`

Today that is required only because `Display` is source-backed. This means every successful parse allocates and retains a full copy of the input even though the IR already contains spans and structure.

Expected impact:

- moderate allocation cost on every parse
- more retained memory than necessary
- especially wasteful once a real serializer exists

### 2. Bracket atoms duplicate both text and tree

`BracketExpr` stores:

- `text: String`
- `tree: BracketExprTree`

Relevant file:

- `crates/smarts-parser/src/query.rs`

That means bracket-heavy SMARTS pay for:

- one scan to isolate the bracket contents
- another parse to build the AST
- an owned copy of the bracket text even after the tree exists

This is almost certainly a major reason the `bracket_primitives` workload is notably slower than the `simple_atoms_bonds` workload.

### 3. Recursive SMARTS is parsed twice

Current flow:

1. `parse_recursive_smarts()` captures `$(...)` as `AtomPrimitive::Recursive(String)`
2. after the whole query is parsed, `lower_recursive_queries_in_atoms()` walks the atom tree
3. nested SMARTS is reparsed with `parse_smarts()`

Relevant files:

- `crates/smarts-parser/src/bracket.rs`
- `crates/smarts-parser/src/parse.rs`

This means recursive SMARTS pays for:

- payload string allocation
- a second parse pass for the nested query
- a full AST walk after the main parse completes

This is the clearest structural reason the `recursive_bond_ops` workload is the slowest Rust category.

### 4. Bracket atoms are scanned multiple times

For bracket atoms, the parser currently does all of this:

1. scan forward to find the matching `]`
2. call `find_atom_map(text)`, which rescans the bracket text
3. call `parse_bracket_text(text)`, which scans it again

Relevant file:

- `crates/smarts-parser/src/parse.rs`

That is avoidable repeated work on one of the most syntactically dense parts of SMARTS.

### 5. Symbol storage uses owned `String`s everywhere

Bare atoms, bracket symbols, and isotopes all store symbols as owned `String`.

Relevant files:

- `crates/smarts-parser/src/query.rs`
- `crates/smarts-parser/src/bracket.rs`
- `crates/smarts-parser/src/parse.rs`

Examples:

- `AtomExpr::Bare { symbol: String, aromatic: bool }`
- `AtomPrimitive::Symbol { symbol: String, aromatic: bool }`
- `AtomPrimitive::Isotope { mass, symbol: String, aromatic: bool }`

These are tiny ASCII tokens with a very small vocabulary. Heap-allocating them is unnecessary.

### 6. The parser is Unicode-safe even though the grammar is effectively ASCII

Hot-path helpers like `peek()` repeatedly do:

```rust
self.input[self.pos..].chars().next()
```

and parsing code uses `starts_with()` on slices and `char`-level stepping.

Relevant files:

- `crates/smarts-parser/src/parse.rs`
- `crates/smarts-parser/src/bracket.rs`

The current grammar is ASCII-only in practice. A byte-oriented parser would avoid repeated UTF-8 decoding and simplify many branches.

### 7. Ring closures use `BTreeMap`

Open ring closures are tracked with:

```rust
BTreeMap<u32, PendingRingClosure>
```

Relevant file:

- `crates/smarts-parser/src/parse.rs`

This is reasonable and simple, but likely not ideal for a hot parse path. Ring closures are typically sparse and small in count. A flat small map, `HashMap`, or custom tiny-vector strategy would probably be cheaper.

### 8. Numeric parsing goes through string parsing

Helpers like `parse_required_u16()` slice a substring and then do:

```rust
.parse::<u16>()
```

Relevant file:

- `crates/smarts-parser/src/bracket.rs`

For tiny unsigned integers in an ASCII grammar, manual digit accumulation is usually faster and allocates nothing.

## RDKit: What It Does Beyond Bare Parsing

The Rust parser currently builds a narrow immutable query IR. RDKit builds full query molecules and does extra post-processing.

### 1. It preprocesses the input string before parsing

Relevant source:

- `Code/GraphMol/SmilesParse/SmilesParse.cpp`

Before parsing SMARTS, RDKit:

- splits optional CXSMARTS / name suffixes
- applies replacement macros if configured
- relabels recursive SMARTS with `labelRecursivePatterns()`

`labelRecursivePatterns()` is a whole-string pass with its own nesting-state machine and string rewriting. That is extra work our parser does not currently do.

### 2. It initializes and tears down a flex scanner for each parse

Relevant source:

- `Code/GraphMol/SmilesParse/SmilesParse.cpp`
- `Code/GraphMol/SmilesParse/smarts.ll`

Each parse goes through:

- `yysmarts_lex_init()`
- `setup_smarts_string()`
- scanner buffer allocation and copy
- `yysmarts_parse(...)`
- `yysmarts_lex_destroy()`

That setup cost is real, especially for short SMARTS.

### 3. Grammar actions build full RDKit query objects while parsing

Relevant source:

- `Code/GraphMol/SmilesParse/smarts.yy`

RDKit’s parser actions do not just build a cheap syntax tree. They allocate and mutate:

- `RWMol`
- `QueryAtom`
- `QueryBond`
- bookmarks for ring closures and branches
- query predicates combined with `expandQuery()`
- bond/query property maps

This is substantially heavier than pushing small Rust enums into `Vec`s.

### 4. It performs post-parse molecule/query cleanup

Relevant source:

- `Code/GraphMol/SmilesParse/SmilesParse.cpp`
- `Code/GraphMol/SmilesParse/SmilesParseOps.cpp`

After parsing SMARTS, RDKit still does more:

- `MolOps::setBondStereoFromDirections(*res)`
- optional `MolOps::mergeQueryHs(*res)`
- `SmilesParseOps::CleanupAfterParsing(res.get())`

Even the generic `toMol()` path also does:

- `CloseMolRings()`
- `CheckChiralitySpecifications()`
- `SetUnspecifiedBondTypes()`
- `AdjustAtomChiralityFlags()`

Those are useful semantics, but they are not free.

### 5. Python `MolFromSmarts()` adds wrapper overhead

Relevant source:

- `Code/GraphMol/Wrap/rdmolfiles.cpp`

The Python entry point does at least this per call:

- convert `python::object` to `std::string`
- construct a replacement map from a Python dict
- call C++ parsing
- wrap the returned raw pointer as a Python-managed `Mol`

That wrapper layer is not the whole gap, but it is a meaningful part of the benchmark if you measure through Python.

## Prioritized Rust Optimizations

### Tier 1: high value, low conceptual risk

1. Implement a real serializer and remove source-backed `Display`.
   Then remove `QueryMol.source: String`.

2. Remove `BracketExpr.text` once parser diagnostics and serialization no longer depend on it.

3. Parse recursive SMARTS directly to nested `QueryMol` during bracket parsing.
   Avoid the `Recursive(String)` intermediate and the later lowering walk.

4. Replace symbol `String`s with compact atom-symbol types.
   Best options:
   - a small internal enum
   - `elements-rs::Element` plus an aromatic flag for aliphatic forms
   - a tiny borrowed/static symbol type

These changes should reduce allocation pressure immediately, especially on bracket-heavy and recursive workloads.

### Tier 2: likely worthwhile after Tier 1

5. Convert the parser to byte-oriented ASCII scanning.

6. Fold atom-map detection into bracket parsing instead of rescanning bracket text.

7. Replace `parse::<u16>()` and similar numeric parsing with manual digit accumulation.

8. Replace `BTreeMap` ring-closure tracking with a cheaper structure.

These are more mechanical but should shave steady-state parse cost.

### Tier 3: only if needed later

9. Pre-size `atoms` and `bonds` with a rough first pass or heuristic.

10. Split parser modes so the fastest path for common bare-atom SMARTS avoids some generic branching.

11. Consider a lower-level tokenizer if the byte parser still leaves too much overhead.

These are lower priority because the current parser is already very fast.

## Recommendation

If the goal is maximum return on engineering time, do the work in this order:

1. true serializer
2. remove `QueryMol.source`
3. remove `BracketExpr.text`
4. parse recursive SMARTS directly into nested IR
5. compact atom-symbol representation
6. byte-oriented parser

That sequence improves both performance and architecture. It removes bootstrap shortcuts that were useful early on but are now visible in the benchmark shape.

## Sources

Local code:

- `crates/smarts-parser/src/parse.rs`
- `crates/smarts-parser/src/bracket.rs`
- `crates/smarts-parser/src/query.rs`

RDKit sources:

- `https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/SmilesParse/SmilesParse.cpp`
- `https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/SmilesParse/SmilesParse.h`
- `https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/SmilesParse/SmilesParseOps.cpp`
- `https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/SmilesParse/smarts.yy`
- `https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/SmilesParse/smarts.ll`
- `https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/Wrap/rdmolfiles.cpp`
