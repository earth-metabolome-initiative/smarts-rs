# `smiles-parser` Review For SMARTS Validation

## Purpose

This note reviews the current `earth-metabolome-initiative/smiles-parser` API from the point of view of `smarts-validator`.

The question is not "is `smiles-parser` a good SMILES parser?"

The question is:

- what can we already reuse as the target-molecule model for SMARTS matching?
- what should be added upstream because it is generally useful graph/chemistry access?
- what should stay out of `smiles-parser` and be implemented in `smarts-validator` as derived preparation?

## Reviewed revision

I reviewed the public repository at:

- `earth-metabolome-initiative/smiles-parser`
- commit `b890aa2f43055d9d90662cc86a5d34cccc4e414d`

Main files reviewed:

- `src/lib.rs`
- `src/smiles/mod.rs`
- `src/smiles/from_str.rs`
- `src/atom/mod.rs`
- `src/atom/atom_node.rs`
- `src/atom/atom_symbol.rs`
- `src/atom/unbracketed.rs`
- `src/atom/bracketed/mod.rs`
- `src/bond/mod.rs`
- `src/bond/bond_edge.rs`
- `src/bond/ring_num.rs`
- `src/token.rs`
- `src/errors.rs`
- `tests/test_aromatics.rs`
- `tests/test_square_brackets.rs`

## Executive summary

`smiles-parser` is already good enough to serve as the **owned parsed target graph** for SMARTS validation.

It is not yet shaped for efficient SMARTS matching.

The biggest gaps are not parser correctness. They are:

- missing **zero-allocation adjacency access**
- missing **direct target-property accessors** for exact isotope / explicit hydrogens / element-level identity
- missing **derived molecule properties** needed by SMARTS such as implicit hydrogen counts, degree, connectivity, ring membership, and ring size

The important design conclusion is:

- keep `smiles-parser` focused on **SMILES syntax + parsed graph**
- let it grow into a **usable molecule-graph API** where that remains generally useful
- keep SMARTS-specific query semantics and matching logic in `smarts-validator`

## What `smiles-parser` already gives us

### Parse entrypoint

The crate exposes:

- `Smiles: FromStr`
- parser errors as `SmilesErrorWithSpan`

That is a fine entrypoint for `smarts-validator`.

### Graph ownership model

`Smiles` stores:

- `Vec<AtomNode>`
- `Vec<BondEdge>`
- `HashMap<usize, usize>` for node-id lookup

Public methods on `Smiles` already include:

- `nodes() -> &[AtomNode]`
- `nodes_mut() -> &mut [AtomNode]`
- `node_by_id(id) -> Option<&AtomNode>`
- `edges() -> &[BondEdge]`
- `edges_mut() -> &mut [BondEdge]`
- `edge_for_node_pair((a, b)) -> Option<&BondEdge>`
- `edges_for_node(id) -> Vec<&BondEdge>`
- `render() -> Result<String, SmilesError>`

That means we already have:

- a concrete parsed graph type
- stable node ids
- edge objects with bond labels
- enough data to build an adapter in `smarts-validator`

### Atom surface

`AtomNode` provides:

- `id()`
- `atom()`
- `span()`

`Atom` provides:

- `aromatic()`
- `symbol()`
- `chirality()`
- `class()`
- `charge()`
- `charge_value()`
- `hydrogen_count()`
- `hydrogens()`
- `isotope() -> Result<Isotope, SmilesError>`

`BracketAtom` provides:

- `symbol()`
- `element()`
- `isotope_mass_number()`
- `aromatic()`
- `hydrogens()`
- `hydrogen_count()`
- `charge()`
- `charge_value()`
- `class()`
- `chirality()`

This is already a solid syntax-preserving atom model.

### Bond surface

`Bond` already distinguishes:

- `Single`
- `Double`
- `Triple`
- `Quadruple`
- `Aromatic`
- `Up`
- `Down`

`BondEdge` provides:

- `bond()`
- `vertices()`
- `node_a()`
- `node_b()`
- `contains(node_id)`
- `other(node_id)`
- `ring_num()`
- `ring_num_val()`

For v1 SMARTS matching, this is a good starting bond model.

### What is already represented syntactically

The current parser already preserves:

- aromatic atom spelling via lowercase atoms
- aromatic bonds via `Bond::Aromatic`
- directional bonds via `Bond::Up` and `Bond::Down`
- bracket isotope / charge / chirality / explicit hydrogen / class
- disconnected components via absence of edges

That is enough to start building a target adapter.

## What is missing for SMARTS use

The missing pieces fall into three groups:

1. small generic graph/property access that should probably exist upstream
2. derived molecule properties that are still reasonable upstream
3. SMARTS-specific preparation and matching state that should live in `smarts-validator`

## Upstream additions that make sense in `smiles-parser`

These are additions I would consider reasonable in the students' crate because they are still generic SMILES-graph or molecule-model functionality, not SMARTS logic.

### 1. Zero-allocation adjacency API

Current issue:

- `Smiles::edges_for_node(id)` allocates a fresh `Vec<&BondEdge>` every call
- `Smiles::edge_for_node_pair((a, b))` linearly scans all edges

That is acceptable for general use, but it is a bad fit for subgraph matching, where neighbor access is in the inner loop.

Recommended additions:

```rust
impl Smiles {
    pub fn neighbors(&self, id: usize) -> impl Iterator<Item = (usize, &BondEdge)> + '_;
    pub fn degree(&self, id: usize) -> usize;
}
```

If iterator-returning APIs are awkward, a cached adjacency table inside `Smiles` is also fine.

This is the single most valuable upstream change.

### 2. Direct atom-level identity accessors

`Atom` currently exposes `symbol()` and `isotope()`, but SMARTS matching needs a slightly more explicit surface.

Recommended additions:

```rust
impl Atom {
    pub fn element(&self) -> Option<elements_rs::Element>;
    pub fn is_wildcard(&self) -> bool;
    pub fn isotope_mass_number(&self) -> Option<u16>;
    pub fn explicit_hydrogen_count(&self) -> Option<u8>;
}
```

Why:

- `Atom::isotope()` is not ideal for SMARTS because unbracketed atoms currently map to the most abundant isotope
- SMARTS matching needs to distinguish:
  - isotope explicitly specified
  - isotope unspecified
- explicit H count should be readable without going through bracket-only internals

This is not SMARTS-specific. It is just better molecule inspection API.

### 3. Neighbor and edge convenience on `AtomNode`

This is optional, but would make downstream adapters cleaner:

```rust
impl Smiles {
    pub fn has_edge(&self, a: usize, b: usize) -> bool;
}
```

This is small, generic, and useful.

### 4. Optional edge span retention

This is not required for matching, but it would improve diagnostics and debugging.

Today:

- `AtomNode` has a source span
- `BondEdge` does not

If upstream ever wants better render/debug/error tooling, edge spans would be a good addition.

This is low priority.

### 5. Derived molecule properties that are still fair upstream

This is where I want to correct the first version of this note.

Properties like these are not merely SMARTS conveniences:

- implicit hydrogen count
- total hydrogen count
- degree
- connectivity
- valence helpers
- connected component id
- ring membership
- ring bond flag
- smallest ring size

These are all legitimate **molecule properties**.

So if `smiles-parser` is intended to remain only a syntax-to-graph converter, then these can stay downstream.
But if it is intended to become a reusable molecular graph crate, these are perfectly reasonable upstream additions.

My revised position is:

- these properties do **not** need to stay out of `smiles-parser`
- they are only inappropriate upstream if the project scope is intentionally limited to parsing

In practice, I would expect at least some of them to move upstream over time.

The most defensible early candidates are:

- `implicit_hydrogen_count(atom_id)`
- `degree(atom_id)`
- `connectivity(atom_id)`
- `component_id(atom_id)`
- `is_in_ring(atom_id)` / `is_bond_in_ring(edge)`

Whether `smallest_ring_size` and valence helpers should also move upstream depends on how far the crate wants to go toward a full molecule model.

## Things that should stay out of `smiles-parser`

These are needed for SMARTS, but I would still keep them out of the parser crate proper.

### 1. SMARTS query semantics and matching

Definitely keep these out of `smiles-parser`:

- query predicates
- recursive SMARTS
- subgraph isomorphism
- uniqueness filtering
- match enumeration
- SMARTS stereo semantics

Those are not molecule-model concerns. They are query-engine concerns.

### 2. SMARTS-specific per-call preparation state

Even if many derived molecule properties move upstream, the validator will still need its own per-query state, for example:

- candidate pruning tables
- recursive SMARTS memoization
- query-to-target compatibility caches
- match uniqueness bookkeeping
- temporary search frontiers

Those should remain in `smarts-validator`.

### 3. Aromaticity policy tied specifically to SMARTS behavior

The reviewed tests show that aromaticity is currently preserved syntactically:

- lowercase atoms stay aromatic
- aromatic bracket atoms can be parsed

That is useful, but it is not a full aromaticity model.

I am more cautious here than with implicit H or ring membership.

Aromaticity perception is a real molecule property, but it is also model- and policy-dependent.
So:

- a generic aromaticity module upstream could make sense later
- SMARTS-specific aromaticity preparation should not block validator development now
- `smarts-validator` should be prepared to own aromaticity prep until the upstream crate has a stable policy

## Concrete gaps for `smarts-validator`

Even if some of the molecule properties move upstream, `smarts-validator` will still need a prepared target wrapper.

That wrapper may compute or simply cache:

- dense atom indices
- adjacency lists
- atom degree
- heavy-atom degree if needed
- connectivity counts
- valence helpers
- explicit H / implicit H / total H
- ring membership flags
- ring bond flags
- smallest ring size
- connected component ids
- optional aromaticity preparation

In other words:

- `smiles-parser` should remain the source of parsed graph truth
- `smarts-validator` should own all match-oriented state
- some derived molecule properties may be reused from upstream instead of recomputed locally

## Important API caveats in the current design

### `Atom::isotope()` is not a SMARTS-ready accessor

This is the biggest semantic footgun in the current atom API for SMARTS work.

For unbracketed atoms, `Atom::isotope()` returns the most abundant isotope.

That is fine for chemistry-oriented convenience.
It is not fine as the primary API for SMARTS matching.

SMARTS needs:

- exact isotope specified, for example `[13C]`
- unspecified isotope, for example `C`

Those are different states.

So downstream matching should use explicit mass-number access, not inferred default isotopes.

### `hydrogen_count()` is explicit syntax, not total hydrogen semantics

Current atom APIs tell us what was written in bracket syntax.
They do not tell us:

- implicit hydrogen count
- total hydrogen count for matching

That distinction matters for SMARTS `H` and `h`.

### `Bond::Single` collapses explicit `-` and elided single

For target matching, this is usually acceptable.

SMARTS matching generally cares about target bond type, not whether the target SMILES wrote the single bond explicitly.

So I do not see this as a blocker.

I would not change this upstream unless there is another reason.

## Recommended upstream changes

I would separate this into two tiers.

### Tier 1: minimal changes that immediately help `smarts-validator`

If I were giving the students a narrow request list, I would start here:

1. add zero-allocation neighbor access on `Smiles`
2. add direct `Atom` accessors for:
   - `element()`
   - `is_wildcard()`
   - `isotope_mass_number()`
   - `explicit_hydrogen_count()`
3. optionally add `Smiles::degree(id)` and `Smiles::has_edge(a, b)`

That is enough to make `smarts-validator` much cleaner without changing the crate's scope too aggressively.

### Tier 2: broader molecule-model features that also fit upstream

If the intention is for `smiles-parser` to grow into a generally useful molecular graph crate, then these also fit upstream:

1. implicit hydrogen count
2. total hydrogen count
3. connectivity / valence helpers
4. connected components
5. ring membership and ring-bond flags
6. smallest ring size

These are all molecule properties, not SMARTS-only properties.

## Recommended `smarts-validator` plan against the current API

The practical path forward is:

1. depend on `smiles-parser`
2. write an adapter layer over `Smiles`
3. remap node ids to dense indices for matching
4. build a `PreparedTarget` that computes or caches the SMARTS-relevant derived properties
5. do not block on upstream perfection

So the answer is not "wait until `smiles-parser` is complete."

The answer is:

- ask for a few small upstream graph/accessor improvements
- let molecule-model features move upstream where that makes sense
- keep the real SMARTS engine in `smarts-validator`

## Bottom line

`smiles-parser` is already a usable target graph source for SMARTS work.

What it lacks is not "basic parsing."
What it lacks is the match-oriented access and derived chemistry state that a SMARTS engine needs.

The right split is:

- upstream:
  - better graph access
  - better direct atom accessors
  - optionally broader molecule properties like implicit H and ring membership
- downstream:
  - SMARTS-specific preparation policy
  - aromaticity policy if upstream does not settle it
  - matching

That keeps both crates conceptually clean.
