# `smiles-parser` Methods Needed By `smarts-validator`

Date: 2026-04-17

## Scope

This note records the target-side helper methods that `smarts-validator`
needs from `smiles-parser`, and the exact semantics they must have in order
to match `RDKit` SMARTS behavior.

The immediate goal is not “general molecule convenience APIs”. The goal is:

- avoid duplicating chemistry logic in `smarts-validator`
- keep the validator aligned with `RDKit`
- make the target-side contract explicit enough that future upstream changes
  can be evaluated against known SMARTS behavior

## Status

Two helpers were requested first:

- `Smiles::connectivity_count(atom_id) -> u8`
- `Smiles::total_valence(atom_id) -> u8`

Current outcome:

- `connectivity_count()` is sufficient for SMARTS `X`
- raw `total_valence()` is not sufficient for SMARTS `v`
- `smarts_total_valence()` is the correct aromaticity-aware helper for
  SMARTS `v`

The reason is that SMARTS `v` follows `RDKit`’s aromaticity-aware behavior,
while the current `Smiles::total_valence()` is still the raw parsed-graph
bond-order sum plus explicit/implicit hydrogens.

So the useful surface is now:

- keep `connectivity_count()` as-is
- keep `total_valence()` as the raw parsed-graph helper
- use `smarts_total_valence()` for SMARTS / `RDKit` parity

## Method 1: `connectivity_count()`

### Desired signature

```rust
pub fn connectivity_count(&self, atom_id: usize) -> u8
```

### Required semantics

This should match SMARTS `X`:

```text
topological degree + explicit hydrogens + implicit hydrogens
```

Where:

- topological degree is the number of actual graph neighbors
- bracket-explicit hydrogens count as explicit hydrogens
- implicit hydrogens count as implicit hydrogens
- explicit hydrogen neighbor nodes must not be counted twice

### Examples

Expected `RDKit`-compatible values:

| SMILES | Atom | Expected `X` |
| --- | --- | ---: |
| `C` | carbon | `4` |
| `O` | oxygen | `2` |
| `[Na+]` | sodium | `0` |
| `[NH4+]` | nitrogen | `4` |
| `c1ccccc1` | aromatic carbon | `3` |
| `c1ncccc1` | aromatic pyridine nitrogen | `3` |
| `[nH]1cccc1` | pyrrolic nitrogen | `3` |

### Current status

The current upstream `connectivity_count()` matches the needed SMARTS
semantics and is used directly by `smarts-validator`.

## Method 2: Raw `total_valence()`

### Current signature

```rust
pub fn total_valence(&self, atom_id: usize) -> u8
```

### Current semantics

This currently behaves like:

```text
bond-order sum + explicit hydrogens + implicit hydrogens
```

using the parsed graph as written.

That is useful, but it is still a raw local graph property.

### Where it is sufficient

It works for many non-aromatic cases:

| SMILES | Atom | Expected raw total valence |
| --- | --- | ---: |
| `C` | carbon | `4` |
| `O` | oxygen | `2` |
| `[Na+]` | sodium | `0` |
| `[NH4+]` | nitrogen | `4` |
| `CC=O` | carbonyl oxygen | `2` |
| `P(=O)(O)(O)O` | phosphorus | `5` |

### Where it is not sufficient

It does not match SMARTS `v` once aromatic atoms are involved.

Examples from `RDKit` SMARTS behavior:

| SMARTS | SMILES | Expected match |
| --- | --- | --- |
| `[v4]` | `c1ccccc1` | `true` |
| `[v3]` | `c1ccccc1` | `false` |
| `[n&v3]` | `c1ncccc1` | `true` |
| `[nH&v3]` | `[nH]1cccc1` | `true` |

So for benzene carbon:

- raw parsed-graph total valence is effectively `3`
- SMARTS / `RDKit` `v` semantics require `4`

That is why `smarts-validator` needed an aromaticity-aware helper instead of
the raw parsed-graph one.

## Method 3: Aromaticity-aware SMARTS valence

### Required helper

What the validator actually wants is a helper with semantics closer to:

```rust
pub fn smarts_total_valence(
    &self,
    atom_id: usize,
    aromaticity: &AromaticityAssignment,
) -> u8
```

Possible alternative names:

- `rdkit_total_valence(...)`
- `smarts_valence(...)`
- `query_total_valence(...)`

The important part is not the name. The important part is the contract:

- raw non-aromatic atoms should behave like current `total_valence()`
- aromatic atoms should match `RDKit` SMARTS `v` semantics
- the method must take an aromaticity assignment explicitly, because this is
  not a pure parsed-graph property

### Required semantics

For SMARTS matching, this helper should satisfy:

| SMILES | Atom | Expected `v` |
| --- | --- | ---: |
| `C` | carbon | `4` |
| `O` | oxygen | `2` |
| `[Na+]` | sodium | `0` |
| `[NH4+]` | nitrogen | `4` |
| `c1ccccc1` | aromatic carbon | `4` |
| `c1ncccc1` | aromatic pyridine nitrogen | `3` |
| `[nH]1cccc1` | pyrrolic nitrogen | `3` |
| `P(=O)(O)(O)O` | phosphorus | `5` |

### Why this belongs upstream

The logic is target-side chemistry interpretation, not SMARTS matching policy.

`smarts-validator` should decide:

- which SMARTS primitive to evaluate
- how boolean query logic combines predicates

But the target-side answer to:

```text
what is this atom’s RDKit-style SMARTS valence under this aromaticity model?
```

fits naturally in `smiles-parser`.

That keeps:

- the target model authoritative
- one source of truth for valence semantics
- less duplicated chemistry logic downstream

## Minimal target-side contract for current validator work

For the current validator slices, the useful public `smiles-parser` helpers are:

```rust
pub fn node_by_id(&self, id: usize) -> Option<&Atom>;
pub fn edge_for_node_pair(&self, nodes: (usize, usize)) -> Option<BondEdge>;
pub fn edge_count_for_node(&self, id: usize) -> usize;
pub fn edges_for_node(&self, id: usize) -> impl Iterator<Item = BondEdge> + '_;
pub fn implicit_hydrogen_count(&self, id: usize) -> u8;
pub fn connectivity_count(&self, id: usize) -> u8;
pub fn total_valence(&self, id: usize) -> u8; // raw parsed-graph value
pub fn ring_membership(&self) -> RingMembership;
pub fn symm_sssr_result(&self) -> SymmSssrResult;
pub fn aromaticity_assignment_for(&self, policy: AromaticityPolicy) -> AromaticityAssignment;
```

And the aromaticity-aware helper that removes the last local workaround is:

```rust
pub fn smarts_total_valence(
    &self,
    atom_id: usize,
    aromaticity: &AromaticityAssignment,
) -> u8;
```

## Practical downstream impact

With `smarts_total_valence()` in `smiles-parser`, `smarts-validator` can:

- use upstream `connectivity_count()` for `X`
- use upstream SMARTS valence for `v`
- avoid local aromatic `v` adjustment logic entirely

That is the cleanest division of responsibility.

## Next likely need: stereo-facing helpers

The current validator work can continue without upstream stereo support, but
the next major SMARTS slice is blocked on a narrow public stereo API.

### Current public state

What is already available:

- `Atom::chirality() -> Option<Chirality>`
- parsed directional bond labels through `edge_for_node_pair()`

What is still missing for SMARTS / `RDKit` parity:

- a public tetrahedral-chirality view that is normalized enough for matching
- a public semantic double-bond stereo view (`E` / `Z`) derived from the
  parsed directional bonds

### Why parsed tokens are not enough

For SMARTS stereo matching, the validator needs target-side answers to:

- does this atom satisfy `@` / `@@` / `@TH1` style query chirality?
- does this bond environment satisfy `/` and `\` relative to the double bond
  it constrains?

Raw parsed tokens are not a sufficient contract because:

- atom chirality depends on neighbor ordering and implicit hydrogen handling
- directional bond tokens are local surface syntax, while SMARTS matching
  needs semantic double-bond stereo

### Likely useful upstream helpers

The exact names are flexible, but the validator will likely want something in
this shape:

```rust
pub fn smarts_tetrahedral_chirality(&self, atom_id: usize) -> Option<Chirality>;

pub fn double_bond_stereo_config(
    &self,
    node_a: usize,
    node_b: usize,
) -> Option<DoubleBondStereoConfig>;
```

Where:

- `smarts_tetrahedral_chirality()` returns the target-side chirality in a form
  suitable for SMARTS parity, not just the raw parsed token
- `double_bond_stereo_config()` returns the semantic `E` / `Z` classification
  for one double bond when defined

The validator should stay responsible for:

- interpreting SMARTS query chirality primitives
- deciding whether the query asks for equality, inversion, or an unsupported
  stereo class

But the target-side normalization and semantic double-bond classification fit
more naturally in `smiles-parser`.
