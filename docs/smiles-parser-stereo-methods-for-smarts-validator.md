# `smiles-parser` Stereo Surface Needed By `smarts-validator`

Date: 2026-04-17

## Scope

This note defines the target-side stereo API that `smarts-validator` needs
from `smiles-parser` in order to implement SMARTS stereo matching without
duplicating chemistry logic downstream.

The goal is not to expose every internal stereo planning detail. The goal is
to expose just enough public, stable, semantics-facing API that:

- `smarts-validator` can match SMARTS stereo against `SMILES`
- `RDKit` remains the reference implementation for parity testing
- target-side stereo normalization stays owned by `smiles-parser`

## Why A New API Is Needed

`smarts-validator` already has:

- parsed SMARTS chirality tokens from `smarts-parser`
- parsed target bond labels and atom chirality tokens through
  `smiles-parser`

That is not enough.

For SMARTS stereo matching, the validator needs target-side answers to:

- what is the semantic tetrahedral chirality of this atom?
- what is the semantic double-bond stereo configuration of this bond?

Raw parsed tokens are not a sufficient contract because:

- tetrahedral chirality depends on neighbor ordering and implicit-hydrogen
  treatment
- directional `/` and `\` bond tokens are local surface syntax, while SMARTS
  matching needs the semantic stereochemical relationship around the double
  bond

So the validator should not read raw syntax and attempt to reconstruct
stereochemistry itself. That would duplicate target-side chemistry logic that
already belongs in `smiles-parser`.

## Minimal Public API

The exact names are flexible. The important part is the contract.

### 1. Tetrahedral / atom-centered stereo

```rust
pub fn smarts_tetrahedral_chirality(&self, atom_id: usize) -> Option<Chirality>;
```

Expected meaning:

- returns the target-side atom chirality in a form suitable for SMARTS
  matching
- returns a normalized value, not just the raw parsed token
- returns `None` when the atom has no usable tetrahedral/allene-like stereo
  for SMARTS matching

The existing public `Atom::chirality()` is not enough for this, because it
reports the parsed token, not the semantic answer that SMARTS matching wants.

### 2. Double-bond stereo

```rust
pub enum DoubleBondStereoConfig {
    E,
    Z,
}

pub fn double_bond_stereo_config(
    &self,
    node_a: usize,
    node_b: usize,
) -> Option<DoubleBondStereoConfig>;
```

Expected meaning:

- `node_a` and `node_b` identify the double bond
- returns `Some(E)` or `Some(Z)` when the bond has semantic stereo
- returns `None` when:
  - the edge is not a double bond
  - stereo is not specified
  - stereo is not semantically defined for that bond

This is the missing bridge between parsed `/` and `\` surface tokens and the
semantic bond stereo that SMARTS matching needs.

## Ownership Boundary

The split should be:

### `smiles-parser` owns

- target-side stereo normalization
- implicit-hydrogen effects on chirality
- neighbor-order normalization for semantic stereo
- conversion from parsed directional bond tokens to semantic double-bond
  stereo

### `smarts-validator` owns

- interpreting SMARTS query stereo primitives
- deciding which query stereo forms are supported in the first slice
- comparing query stereo against the target-side semantic answers
- differential testing against `RDKit`

This keeps the chemistry model authoritative upstream and keeps the validator
focused on query semantics.

## Recommended First Support Slice

Do not try to land every SMARTS stereo form at once.

The first useful validator slice should be:

### Atom stereo

- `@`
- `@@`
- `@TH1`
- `@TH2`

Potentially also:

- `@AL1`
- `@AL2`

if `smiles-parser` already has clean allene-like normalization and the public
API can expose it without ambiguity.

### Bond stereo

- `/`
- `\`

only in the context of double-bond semantics, aligned to `RDKit`.

### Still deferred

- `@SP`
- `@TB`
- `@OH`
- broader non-tetrahedral stereo matching

The parser already accepts those. The validator does not need to support them
in the first stereo slice.

## Implementation Guidance Upstream

The public helpers should be thin wrappers over existing internal stereo
machinery, not a second implementation.

From the current codebase, the useful internal building blocks already appear
to exist:

- tetrahedral normalization helpers in `src/smiles/stereo.rs`
- semantic double-bond stereo planning in `src/smiles/double_bond_stereo.rs`

So the upstream task should be:

1. choose the public semantic return types
2. wrap the existing internal normalization/classification logic
3. document the contract
4. add direct unit tests at the public method boundary

## Test Plan

Testing should happen at two levels.

### A. `smiles-parser` unit tests

These should prove the new public methods themselves.

#### Atom stereo method tests

Add direct tests for `smarts_tetrahedral_chirality()` using representative
inputs:

- tetrahedral carbon with explicit hydrogen
- tetrahedral carbon without explicit hydrogen
- reordered neighbors that should preserve semantic chirality
- atoms with no stereo that should return `None`
- atoms with parsed chirality tags that are not semantically valid and should
  return `None` or the chosen fallback

Candidate target SMILES:

- `F[C@H](Cl)Br`
- `F[C@@H](Cl)Br`
- `C[C@H](O)N`
- `C[C@@H](O)N`
- achiral controls such as `CC(O)N`

The exact expected answers should be asserted directly at the
`smiles-parser` method level.

#### Double-bond stereo method tests

Add direct tests for `double_bond_stereo_config()` using:

- clear stereo-specified alkenes
- unspecific double bonds
- non-double-bond controls
- ring cases where semantic double-bond stereo should be absent

Candidate target SMILES:

- `F/C=C/F`
- `F/C=C\\F`
- `Cl/C=C/Br`
- `Cl/C=C\\Br`
- `CC=CC`
- `C/C=C/C`
- `C/C=C\\C`
- ring or constrained cases where stereo is absent

The important point is not to trust memory for which spelling is `E` or `Z`.
The expected values should be checked against `RDKit` once, then asserted in
tests.

### B. `smarts-validator` differential tests

Once the upstream methods exist, the validator should add a new frozen
fixture family:

- `corpus/validator/stereo-v0.cases.json`
- `corpus/validator/stereo-v0.rdkit.json`
- `crates/smarts-validator/tests/stereo_rdkit.rs`

The workflow should stay the same as for every other validator slice:

1. add human-edited source cases
2. generate `RDKit` expectations
3. add failing Rust tests against the frozen fixture
4. implement the minimal validator code
5. widen only after green

#### Required validator corpus categories

Include all of these:

1. Positive atom-chirality matches
2. Negative atom-chirality mismatches
3. Positive directional-bond matches
4. Negative directional-bond mismatches
5. Queries with stereo against targets lacking stereo
6. Targets with stereo against queries without stereo
7. Mixed topological queries where stereo is only one part of the predicate

#### Candidate SMARTS/SMILES categories

Atom stereo:

- `[C@H](F)(Cl)Br`
- `[C@@H](F)(Cl)Br`
- `[C@](F)(Cl)(Br)I`
- `[C@@](F)(Cl)(Br)I`

Double-bond stereo:

- `F/C=C/F`
- `F/C=C\\F`
- `C/C=C/C`
- `C/C=C\\C`
- `C/C=C/Cl`
- `C/C=C\\Cl`

Negative controls:

- same topological scaffold without stereo in the target
- same topological scaffold without stereo in the query
- opposite stereo on the target

Again, the frozen expected booleans should come from `RDKit`, not from memory.

## Acceptance Criteria

The upstream work is complete when:

1. `smiles-parser` exposes public semantic stereo helpers
2. those helpers have direct unit tests on representative examples
3. `smarts-validator` can stop rejecting the first stereo subset
4. the new `stereo-v0` RDKit differential suite is green
5. no target-side stereo normalization logic is copied into
   `smarts-validator`

## What Not To Do

- do not expose only the raw parsed stereo tokens and call that sufficient
- do not reimplement tetrahedral normalization in `smarts-validator`
- do not reimplement semantic E/Z classification in `smarts-validator`
- do not attempt full non-tetrahedral SMARTS stereo support in the first
  slice

The shortest correct path is:

1. expose semantic target-side stereo upstream
2. add focused RDKit-backed validator tests
3. support the narrow useful stereo subset first
