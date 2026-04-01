# Thread-Safe SMARTS Parser + Validator Plan

## Purpose

This document lays out a phased plan for building a **thread-safe SMARTS workspace in Rust** with two primary deliverables:

1. a **SMARTS parser crate**
2. a **SMARTS-vs-SMILES validator crate**

The immediate motivation is to avoid the shared mutable matcher/query state that shows up in existing implementations, especially around recursive SMARTS handling.

This is not a plan for "full RDKit compatibility on day one".
It is a plan for building a clean, thread-safe foundation that can grow.

## Executive summary

The right way to start is:

- treat **syntax** and **matching** as separate products
- keep **compiled SMARTS queries immutable**
- keep **all caches and mutable matcher state per-call**
- reuse EMI crates where they already own the right domain data
- target a **Daylight-like molecule SMARTS subset first**

The project should be organized as a workspace with two public crates and, only if necessary later, one shared internal crate.

Suggested public crates:

- `smarts-parser`
- `smarts-validator`

Possible internal crate later, only if it becomes necessary:

- `smarts-core`

## Scope

### In scope for the first serious build

- molecule SMARTS
- parsing with spans and good diagnostics
- immutable query graph / query predicate IR
- validation of SMARTS queries against `smiles_parser::Smiles`
- thread-safe matching
- recursive SMARTS
- ring and aromaticity preparation
- unique match handling

### Explicitly out of scope for v1

- reaction SMARTS
- atom-map semantics beyond parse-time retention
- CXSMARTS / CXSMILES layers
- component grouping across disconnected target components
- non-tetrahedral stereochemistry
- dialect extensions like ranges, dative bonds, hybridization shortcuts unless very cheap and isolated

## Why the split matters

There are two different systems here:

- a **language frontend**
- a **query engine**

The parser crate should answer:

- is this SMARTS syntactically valid?
- what query graph does it represent?
- what exact source span corresponds to each query node, bond, and expression?

The validator crate should answer:

- does this SMARTS match this SMILES-derived target graph?
- where are the matches?
- are the matches unique?

Keeping these separated prevents the parser crate from accreting chemistry-preparation and matcher state concerns.

## Relationship to the existing EMI crates

### `elements-rs`

This should be treated as the canonical source for:

- element identity
- isotope identity
- atomic metadata that is truly element-level

Where it fits:

- parser crate:
  - parse atom symbols and isotopes
  - normalize bracket atoms
- validator crate:
  - element identity checks
  - isotope checks
  - possible element-derived helpers

What it should **not** be forced to own:

- SMARTS query semantics
- aromaticity model behavior
- ring semantics
- SMARTS-specific wildcard logic

### `molecular-formulas`

This is **not** on the critical path for SMARTS parsing or general SMARTS matching.

It becomes relevant later for:

- optional screening / prefilters
- explicit-atom exact-query summaries
- derived target formula checks in tests or benchmarking

Important limitation:

- many SMARTS queries do **not** determine a molecular formula
- wildcard atoms, disjunctions, recursive SMARTS, ring predicates, and degree predicates are not formula-level constraints

So this crate should be integrated as an optional helper, not as a core dependency of the matcher semantics.

### `smiles-parser`

This is the most immediately relevant crate.

It already provides:

- a parsed molecular graph type: `smiles_parser::Smiles`
- atom and bond structures
- bracket atom metadata
- bond types and ring closure handling

The validator crate should be designed around adapting `smiles_parser::Smiles` instead of re-parsing or re-owning SMILES target graphs.

Recommended rule:

- `smarts-validator` depends on `smiles-parser`
- `smarts-parser` does **not** depend on `smiles-parser`

## Workspace shape

Recommended initial layout:

```text
smarts-rs/
  crates/
    smarts-parser/
    smarts-validator/
  docs/
```

If shared code starts to become awkward, introduce:

```text
  crates/
    smarts-core/
```

But do not create `smarts-core` immediately unless it is clearly pulling its weight.

Too-early shared crates often become a dumping ground.

## Public API direction

### `smarts-parser`

Suggested responsibilities:

- lexing/tokenization
- parsing
- syntax tree / query tree creation
- span-aware diagnostics
- lowering to immutable query IR
- unsupported-feature reporting

Possible API sketch:

```rust
pub struct SmartsQuery { /* immutable query graph */ }

pub fn parse_smarts(input: &str) -> Result<SmartsQuery, SmartsParseError>;
pub fn parse_cst(input: &str) -> Result<Cst, SmartsParseError>;
```

### `smarts-validator`

Suggested responsibilities:

- query compilation if needed
- target adaptation from `smiles_parser::Smiles`
- target preparation
- matching
- recursive SMARTS evaluation
- match enumeration / uniqueness

Possible API sketch:

```rust
pub fn matches(query: &SmartsQuery, target: &smiles_parser::Smiles) -> bool;
pub fn match_all(query: &SmartsQuery, target: &smiles_parser::Smiles) -> Vec<Match>;
```

## Thread-safety design contract

This is the most important architectural constraint.

The design contract should be:

- `SmartsQuery` is immutable after construction
- `SmartsQuery` should be `Send + Sync`
- target preparation state is either:
  - precomputed and stored in an immutable prepared wrapper, or
  - computed inside a per-call scratch context
- recursive SMARTS caches live in per-call state only
- no global mutable caches
- no query object mutation during matching

In other words:

- parser output is pure data
- matching uses a separate `MatchContext`

Suggested internal model:

```rust
pub struct MatchContext {
    recursion_cache: ...,
    candidate_sets: ...,
    visited_target_atoms: ...,
    prepared_target: ...,
}
```

This is the clean break from the older style where query objects accumulate runtime state.

## Internal representation strategy

The parser should lower into a query IR that is useful for matching, not just a pretty CST.

Suggested shapes:

```rust
pub struct QueryMol {
    pub atoms: Vec<QueryAtom>,
    pub bonds: Vec<QueryBond>,
    pub connected_components: Vec<ComponentId>,
}

pub struct QueryAtom {
    pub expr: AtomExpr,
    pub span: Span,
}

pub struct QueryBond {
    pub src: AtomId,
    pub dst: AtomId,
    pub expr: BondExpr,
    pub span: Span,
}

pub enum AtomExpr {
    True,
    Primitive(AtomPrimitive),
    Not(Box<AtomExpr>),
    And(Box<AtomExpr>, Box<AtomExpr>),
    Or(Box<AtomExpr>, Box<AtomExpr>),
    Recursive(Box<QueryMol>),
}

pub enum BondExpr {
    True,
    Primitive(BondPrimitive),
    Not(Box<BondExpr>),
    And(Box<BondExpr>, Box<BondExpr>),
    Or(Box<BondExpr>, Box<BondExpr>),
}
```

This aligns with what RDKit, Open Babel, and CDK all converge toward, even though they encode it differently.

## Parsing strategy

Recommended approach:

- hand-written recursive-descent for graph structure
- Pratt or precedence-climbing parser for atom and bond expressions

Why this is a good fit:

- SMARTS is not just an expression language
- graph assembly is stateful
- you already have a Rust `smiles-parser` codebase in the same ecosystem
- a hand-written parser will integrate better with span-rich diagnostics and unsupported-feature handling

Recommended parser split:

- tokenizer
- graph parser
- atom expression parser
- bond expression parser
- lowerer from parse nodes to `QueryMol`

## Validator strategy

The validator crate should not work directly on raw `smiles_parser::Smiles` internals everywhere.

Instead, define a target adapter boundary.

Suggested pattern:

```rust
pub trait TargetMol {
    fn atom_count(&self) -> usize;
    fn bond_count(&self) -> usize;
    fn atom(&self, id: usize) -> TargetAtomRef<'_>;
    fn bond(&self, id: usize) -> TargetBondRef<'_>;
    fn neighbors(&self, atom_id: usize) -> impl Iterator<Item = NeighborRef<'_>>;
}
```

Then implement or wrap that trait for `smiles_parser::Smiles`.

Benefits:

- test matcher logic without being tightly coupled to one graph type
- later support prepared wrappers
- keep validator modular

## Phase plan

### Phase 0: Requirements and dialect freeze

Goal:

- decide exactly what "SMARTS v1" means

Deliverables:

- dialect matrix
- supported/unsupported feature list
- parser and validator crate boundaries
- thread-safety contract

Decisions to freeze:

- Daylight-like molecule SMARTS subset
- no reaction SMARTS
- no CXSMARTS
- recursive SMARTS included
- tetrahedral stereo deferred unless target model is ready

Acceptance criteria:

- written support matrix exists
- all later phases are blocked from quietly expanding scope

### Phase 1: Type alignment with EMI crates

Goal:

- define where external crates are reused and where local abstractions begin

Deliverables:

- adapter notes for `elements-rs`
- target integration design for `smiles-parser`
- explicit statement that `molecular-formulas` is optional/later

Concrete work:

- element symbol parsing must use `elements_rs::Element`
- isotope parsing must use `elements_rs::Isotope` where appropriate
- identify which `smiles-parser` atom/bond fields are sufficient for SMARTS matching and which prepared properties must be layered on top

Acceptance criteria:

- no duplicate local element enum
- no local isotope registry
- no plan to fork `smiles-parser`

### Phase 2: Tokenizer and spans

Goal:

- build a robust SMARTS lexer with exact spans

Deliverables:

- token enum
- span type
- lexer errors

Tokens must include:

- atom bracket delimiters
- branch delimiters
- ring digits and `%` forms
- recursive `$(` start
- atom and bond operators
- `!`, `&`, `,`, `;`
- atom primitives and bond primitives

Important design choice:

- keep the lexer mostly dumb
- let the parser decide most semantics

Acceptance criteria:

- token corpus covers Daylight tutorial/examples
- lexer is reentrant and allocation-light
- all parse errors later can point back to exact spans

### Phase 3: Graph parser skeleton

Goal:

- parse SMARTS graph structure without full semantic depth yet

Deliverables:

- chains
- branches
- ring closures
- disconnected components with `.`
- bracket atom capture

At this stage, the parser can still represent atom and bond expressions as nested parse fragments.

Acceptance criteria:

- can parse:
  - `CC`
  - `C(O)N`
  - `c1ccccc1`
  - `[C,N]`
  - `[$(CC)]`

### Phase 4: Atom and bond expression parser

Goal:

- fully parse SMARTS boolean expression syntax

Deliverables:

- precedence-correct atom expression parser
- precedence-correct bond expression parser
- implicit high-precedence conjunction handling

Must handle:

- `!`
- `&`
- implicit conjunction
- `,`
- `;`
- recursive `$(...)`

Acceptance criteria:

- parser preserves distinctions like:
  - `[N,O&H1]`
  - `[N,O;H1]`
  - `[nH]`

### Phase 5: Query IR lowering and query validation

Goal:

- lower parsed syntax into immutable matching IR
- reject internally inconsistent queries early

Deliverables:

- `QueryMol`
- `AtomExpr`
- `BondExpr`
- query validation pass

Validation examples:

- invalid ring closure bookkeeping
- unsupported feature usage
- conflicting or nonsensical local constructs
- recursion depth limits if desired

Acceptance criteria:

- `SmartsQuery` is immutable
- `SmartsQuery` is detached from parser internals
- serialization/debug-printing is reasonable for tests

### Phase 6: Validator MVP against `smiles-parser`

Goal:

- achieve a thread-safe minimal matcher over `smiles_parser::Smiles`

Deliverables:

- target adapter
- basic candidate generation
- DFS/backtracking matcher

Supported semantics in MVP:

- element identity
- wildcard
- bond order
- branches
- ring closures as topological constraints
- simple boolean query logic

Not yet required in MVP:

- aromaticity
- recursive SMARTS
- stereo

Acceptance criteria:

- simple SMARTS work against parsed SMILES:
  - `C`
  - `CC`
  - `C=O`
  - `[O,N]`
  - `C(O)N`

### Phase 7: Target preparation layer

Goal:

- compute chemistry-derived properties needed by SMARTS predicates

Deliverables:

- prepared target wrapper
- ring membership
- ring bond count
- degree / connectivity summaries
- explicit/implicit hydrogen summaries
- aromaticity perception

Important note:

- this phase is where the validator stops being "graph only" and becomes chemistry-aware

Design rule:

- preparation results are immutable
- matching reads prepared data but never mutates query objects

Acceptance criteria:

- support predicates like:
  - aromatic/aliphatic atom
  - aromatic bond
  - ring membership
  - ring bond count
  - degree / valence / connectivity family

### Phase 8: Recursive SMARTS

Goal:

- support `$(...)` without sacrificing thread safety

Deliverables:

- recursive matcher entry points
- per-call recursion cache
- recursion guard policy

Thread-safety rule here is non-negotiable:

- recursive match state lives in `MatchContext`
- recursive results do not get written into shared query state

Acceptance criteria:

- nested recursive SMARTS work correctly
- concurrent matching of the same compiled query is safe

### Phase 9: Match enumeration, uniqueness, and diagnostics

Goal:

- make validator results actually useful

Deliverables:

- first-match API
- all-matches API
- unique-match mode
- match diagnostics for testing/debugging

Potential output types:

- atom mapping vector
- atom mapping set
- matched subgraph view

Acceptance criteria:

- unique/non-unique behavior is explicit and tested

### Phase 10: Stereo and directional bonds

Goal:

- add stereochemical semantics in a deliberate phase, not mixed into the core matcher

Deliverables:

- tetrahedral atom stereo support if target crate is ready
- `/` and `\` directional bond semantics

This phase should remain optional until `smiles-parser` exposes target stereo in a stable enough way.

Acceptance criteria:

- supported stereo behavior is documented
- unsupported stereo is rejected cleanly or ignored only by explicit policy

### Phase 11: Performance and search heuristics

Goal:

- improve practical runtime without destabilizing semantics

Deliverables:

- atom ordering heuristics
- candidate filtering
- cheap early rejects
- recursion cache tuning

Possible optimizations:

- choose most selective query atom first
- precompute candidate target atoms per query atom
- use adjacency-aware growth ordering
- avoid repeated predicate evaluation

Acceptance criteria:

- no semantic changes
- performance baselines exist

### Phase 12: Differential validation and fuzzing

Goal:

- prove the implementation against real cases

Deliverables:

- corpus from Daylight docs/tutorial
- corpus from RDKit/Open Babel/CDK-compatible features
- fuzz targets
- regression suite

Testing layers:

1. tokenizer tests
2. parser tests
3. lowering tests
4. query validation tests
5. simple matcher tests
6. recursive SMARTS tests
7. aromaticity/ring tests
8. stereo tests
9. differential tests against external toolkits on supported subsets

Acceptance criteria:

- every supported feature has:
  - syntax tests
  - semantic tests
  - matching tests

## Recommended initial milestone cut

If the goal is to get moving fast without overcommitting, the first major milestone should stop after:

- Phase 0
- Phase 1
- Phase 2
- Phase 3
- Phase 4
- Phase 5
- Phase 6

That gives:

- a real SMARTS parser
- immutable query IR
- a basic validator against `smiles-parser`
- no shared mutable state

This is enough to prove the architecture.

## Second milestone cut

The second major milestone should add:

- Phase 7
- Phase 8
- Phase 9

That turns the project from "proof of architecture" into "useful molecule SMARTS engine".

## Third milestone cut

The third milestone should add:

- Phase 10
- Phase 11
- Phase 12

That is where the project becomes robust enough for broad adoption.

## Project size assessment

### Parser crate alone

Medium project.

This is substantial but bounded.

### Validator crate without chemistry prep

Medium project.

### Validator crate with real SMARTS semantics

Large project.

### Full production-quality engine

Very large project.

## Major risks

### Risk 1: Scope creep into compatibility

Mitigation:

- freeze a supported subset in Phase 0

### Risk 2: Over-coupling to `smiles-parser`

Mitigation:

- define a target adapter boundary

### Risk 3: Aromaticity model mismatch

Mitigation:

- pick one model for v1
- document it
- test it separately

### Risk 4: Recursive SMARTS reintroducing shared mutable state

Mitigation:

- recursion cache only inside `MatchContext`

### Risk 5: Premature extensions

Mitigation:

- no ranges, CXSMARTS, reaction SMARTS, or dialect extras until base engine is stable

## Concrete next actions

1. create a workspace with `smarts-parser` and `smarts-validator`
2. write the support matrix for the v1 SMARTS subset
3. define the immutable query IR
4. define the `TargetMol` adapter boundary for `smiles_parser::Smiles`
5. build the tokenizer with spans
6. build the parser with precedence-correct atom/bond expressions
7. only then start the validator

## Sources

- Daylight SMARTS theory:
  - https://daylight.com/dayhtml/doc/theory/theory.smarts.html
- Daylight SMARTS tutorial:
  - https://www.daylight.com/dayhtml_tutorials/languages/smarts/index.html
- RDKit SMARTS grammar:
  - https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/SmilesParse/smarts.yy
- RDKit SMARTS lexer:
  - https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/SmilesParse/smarts.ll
- RDKit substructure matcher:
  - https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/Substruct/SubstructMatch.cpp
- RDKit VF2 implementation:
  - https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/Substruct/vf2.hpp
- Open Babel SMARTS parser and matcher:
  - https://github.com/openbabel/openbabel/blob/master/src/parsmart.cpp
- Open Babel SMARTS internal structures:
  - https://github.com/openbabel/openbabel/blob/master/include/openbabel/parsmart.h
- CDK SMARTS parser:
  - https://github.com/cdk/cdk/blob/main/tool/smarts/src/main/java/org/openscience/cdk/smarts/Smarts.java
- CDK SMARTS pattern wrapper and filters:
  - https://github.com/cdk/cdk/blob/main/base/isomorphism/src/main/java/org/openscience/cdk/isomorphism/Pattern.java
- CDK depth-first matcher:
  - https://github.com/cdk/cdk/blob/main/base/isomorphism/src/main/java/org/openscience/cdk/isomorphism/DfPattern.java
- CDK VF implementation:
  - https://github.com/cdk/cdk/blob/main/base/isomorphism/src/main/java/org/openscience/cdk/isomorphism/VentoFoggia.java
- `elements-rs`:
  - https://github.com/earth-metabolome-initiative/elements-rs
- `molecular-formulas`:
  - https://github.com/earth-metabolome-initiative/molecular-formulas
- `smiles-parser`:
  - https://github.com/earth-metabolome-initiative/smiles-parser
