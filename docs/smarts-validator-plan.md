## SMARTS Validator Plan

This document defines the implementation order for `smarts-validator`.

The goal is not "full SMARTS immediately". The goal is:

- correct matching for a scoped SMARTS subset
- differential validation against `RDKit`
- reusable preparation for many SMARTS x many SMILES workloads
- no hidden semantic guesses without a corpus proving the behavior

### Ground Rules

1. `RDKit` is the reference implementation.
   Every new supported SMARTS feature should be introduced with a frozen
   corpus generated from `RDKit`.

2. TDD is the default workflow.
   For each feature slice:
   - add or extend source cases
   - generate expected results from `RDKit`
   - add failing Rust tests against the frozen fixture
   - implement the minimal code to make those tests pass
   - widen the corpus only after green

3. Build matching in slices.
   Do not start with a full VF2 implementation and hope semantics converge
   later. Land query semantics and target preparation incrementally.

4. Keep parser and validator responsibilities separate.
   `smarts-parser` owns SMARTS syntax and query IR.
   `smarts-validator` owns:
   - target adaptation
   - prepared target properties
   - query evaluation
   - matching
   - conservative screening/indexing

### Target Architecture

There are three distinct layers on the target side:

1. Parsed molecule source
   First backend: `smiles_parser::Smiles`

2. Prepared target view
   Cached properties required by SMARTS evaluation, such as:
   - degree
   - connectivity (`X`)
   - implicit hydrogen count
   - total hydrogen count
   - total valence (`v`)
   - ring membership
   - aromatic atom/bond assignment

3. Matcher
   Uses the prepared target plus the compiled SMARTS query.

The public API can stay backend-friendly later, but the first real slice
should target `smiles-parser` directly instead of building more placeholder
abstractions.

### Validation Phases

#### Phase 0: Differential Test Foundation

Deliverables:

- `corpus/validator/*.json` source case files
- `scripts/generate_rdkit_validator_corpus.py`
- frozen `RDKit` result fixtures committed to the repo
- Rust fixture loader tests

The first fixture family should be intentionally small and exact:

- single-atom SMARTS
- single target SMILES
- boolean `HasSubstructMatch` expectation

This is the narrowest slice that still proves end-to-end correctness.

#### Phase 1: `smiles-parser` Adapter

Deliverables:

- real `matches()` implementation path through `smiles_parser::Smiles`
- a prepared target type for `Smiles`
- removal of the string placeholder target path

Prepared properties needed immediately:

- atom count
- atom labels:
  - element or wildcard
  - isotope mass number
  - formal charge
  - explicit hydrogen count
- degree
- connectivity
- implicit hydrogen count
- total hydrogen count
- total valence
- aromatic atom flags from `AromaticityPolicy::RdkitDefault`

Deferred:

- ring bond count
- smallest ring size
- stereo normalization

#### Phase 2: Single-Atom Query Evaluation

Deliverables:

- evaluation of one `AtomExpr` against one prepared target atom
- boolean tree evaluation for bracket atoms
- explicit unsupported-feature rejection for primitives not yet implemented

The first supported atom primitives should be:

- wildcard
- bare element/aromatic atom
- bracket symbol
- isotope
- atomic number
- `A`
- `a`
- `H`
- `h`
- charge
- `D`
- `X`
- `v`
- boolean logic: `!`, `&`, `,`, `;`

Explicitly unsupported in the first slice:

- `R`
- `r`
- `x`
- chirality matching

Those should produce an explicit validator error, not silently return `false`.

#### Phase 3: Single-Atom End-to-End Matching

Deliverables:

- `QueryMol` with one atom and zero bonds matches by scanning target atoms
- frozen `RDKit` corpus for the supported atom subset
- exact parity for that corpus

This phase gives us a real validator, even though it only handles the
single-atom subset.

#### Phase 4: Single-Component Topological Matching

Deliverables:

- connected query matching for one SMARTS component
- bond predicate evaluation
- backtracking matcher

Supported bond primitives in the first bond slice:

- elided bond semantics
- `-`
- `=`
- `#`
- `:`
- `~`
- bond boolean logic `!`, `&`, `,`, `;`

Still deferred:

- `/`
- `\`
- `@`

#### Phase 5: Ring-Aware Atom/Bond Semantics

Deliverables:

- prepared ring caches:
  - ring atom membership
  - ring bond membership
  - ring bond count per atom
  - smallest ring size per atom
- atom predicate support for:
  - `R`
  - `r`
  - `x`
- bond predicate support for:
  - `@`

This phase should add a dedicated ring-focused `RDKit` fixture family.

#### Phase 5.5: Connectivity And Valence Semantics

Deliverables:

- prepared target caches for:
  - SMARTS connectivity `X`
  - RDKit-style total valence `v`
- dedicated `RDKit` fixture family for:
  - neutral atoms
  - charged atoms
  - aromatic carbon and nitrogen
  - hypervalent phosphorus
  - small connected-query combinations using `X` and `v`

This phase is intentionally separate from ring work because the target-side
chemistry model matters more than graph topology here.

#### Phase 6: Multi-Component SMARTS

Deliverables:

- top-level disconnected components `.`
- zero-level grouping semantics
  - `(A.B)`
  - `(A).(B)`

This phase will need target connected-component information.

Status:

- implemented
- covered by frozen `RDKit` fixtures for plain disconnected SMARTS
- covered by local semantic fixtures for zero-level grouped SMARTS

#### Phase 7: Recursive SMARTS

Deliverables:

- nested query evaluation for `$(...)`
- memoization keyed by `(query_atom_id, target_atom_id)` or similar

Recursive SMARTS should not be enabled in generated/evolved query workflows
until this phase is green and benchmarked.

Status:

- implemented for molecule SMARTS matching
- covered by frozen `RDKit` fixtures

#### Phase 8: Stereo

Deliverables:

- atom chirality checks
- directional bond checks
- `RDKit` differential corpus for stereo-sensitive queries

Status:

- implemented for:
  - explicit tetrahedral SMARTS classes `@TH1` and `@TH2`
  - semantic alkene stereo from `/` and `\` around explicit double bonds
- covered by frozen `RDKit` fixtures

Still deferred in this phase:

- raw `@` / `@@` atom chirality in SMARTS queries
- `@AL*`, `@SP*`, `@TB*`, `@OH*`
- more exotic or boolean-composed directional bond expressions

#### Phase 9: Conservative Screening

Deliverables:

- query-side lower-bound summary
- target-side summary
- cheap conservative prefilter for many-SMARTS × many-SMILES workloads
- tests proving the screen never rejects known true matches from frozen
  fixtures

Status:

- implemented
- currently uses only:
  - minimum atom count
  - minimum grouped-component count
  - lower bounds for exact element occurrences
  - aromatic atom lower bounds
  - ring atom lower bounds

### Test Strategy

Each fixture family should have two files:

1. Source cases
   Human-edited cases with just:
   - name
   - SMARTS
   - SMILES

2. Frozen `RDKit` expectations
   Generated from the source cases with:
   - SMARTS
   - SMILES
   - expected boolean match

Rust tests should read only the frozen expectation file.

When widening the corpus:

- edit the source case file
- regenerate the expectation file with `RDKit`
- review the diff
- commit both

This keeps the reference traceable and deterministic.

### Current Status

Implemented so far:

- single-atom differential corpus
- connected-query differential corpus
- ring differential corpus
- connectivity/valence differential corpus
- stereo differential corpus
- `smiles-parser` backed prepared target with:
  - degree
  - connectivity
  - implicit hydrogen count
  - total hydrogen count
  - total valence
  - RDKit-default aromaticity
  - ring membership and ring-size caches
  - semantic tetrahedral chirality cache
  - semantic double-bond stereo cache
- matcher support for:
  - atom boolean logic
  - bond boolean logic
  - `D`
  - `H`
  - `h`
  - `R`
  - `r`
  - `x`
  - `X`
  - `v`
  - explicit `@TH1` / `@TH2`
  - bond `@`
  - semantic `/` and `\` around explicit double bonds

The next meaningful slices are:

1. widen atom stereo beyond explicit tetrahedral classes
2. widen bond stereo beyond the current explicit-double-bond slice
3. strengthen screening/indexing for many-SMARTS × many-SMILES workloads
4. benchmark the richer validator slice against larger frozen corpora

That gives us a real validator baseline with exact `RDKit` parity on a
meaningful subset, and it creates the harness needed for the larger matcher
phases.
