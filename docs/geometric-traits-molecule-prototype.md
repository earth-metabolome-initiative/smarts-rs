# `geometric-traits` Molecular Graph Prototype

This note records a concrete experiment in `smarts-rs` to answer a narrow
question:

Can we represent a molecule target for SMARTS on top of
`geometric-traits` + `elements_rs` without depending on `smiles-parser`
internals?

Short answer: yes, but the current API is still awkward enough that we should
not build the SMARTS matcher directly against the raw crate surface without a
small chemistry-oriented adapter layer.

That adapter layer now exists in
[`crates/smarts-validator/src/target.rs`](/home/luca/github/smarts-rs/crates/smarts-validator/src/target.rs),
[`crates/smarts-validator/src/geometric_target.rs`](/home/luca/github/smarts-rs/crates/smarts-validator/src/geometric_target.rs),
and
[`crates/smarts-validator/src/prepared.rs`](/home/luca/github/smarts-rs/crates/smarts-validator/src/prepared.rs).

## Prototype

The compile-checked probe lives in
`crates/smarts-validator/tests/geometric_traits_probe.rs`.

The matcher-facing wrapper now lives in
`crates/smarts-validator/src/geometric_target.rs`.

The working graph shape is:

```rust
type MoleculeBondMatrix = ValuedCSR2D<usize, usize, usize, BondLabel>;
type MoleculeGraph = GenericGraph<SortedVec<AtomLabel>, MoleculeBondMatrix>;
```

with:

- `AtomLabel` carrying:
  - `id`
  - `elements_rs::Element`
  - `aromatic`
  - `isotope_mass_number`
  - `formal_charge`
  - `explicit_hydrogens`
- `BondLabel` carrying bond kind

The prototype builds ethanol successfully, retrieves atom metadata, iterates
neighbor IDs, and reads a bond label from the sparse valued matrix.

## What Worked

- `elements_rs::Element` is a good atom-identity primitive.
- `ValuedCSR2D<usize, usize, usize, BondLabel>` works as a bond-labeled
  adjacency structure.
- `GenericGraph<Nodes, Edges>` is flexible enough to pair a node vocabulary
  with a labeled adjacency matrix.
- The overall approach is thread-safe and allocator-friendly.

## What Was Awkward

### 1. Node identity and node label are conflated

`geometric-traits` vocabularies are bijections between dense node IDs and
destination symbols. That means destination symbols must be unique.

This is a bad fit for chemistry at the raw trait level:

- molecules often contain many identical atoms
- SMARTS matching wants repeated labels naturally

The first probe worked around this by making the node symbol artificially
unique:

```rust
struct AtomLabel {
    id: usize,
    element: Element,
    ...
}
```

That works, but it means the graph crate is currently treating
"node symbol" as "node identity plus metadata", not as a reusable atom label.

There is one important nuance after the adapter experiment:

- `GenericGraph<Vec<AtomLabel>, ...>` does allow repeated atom labels
- but the `BidirectionalVocabulary` contract is still semantically awkward,
  because `invert(label)` becomes ambiguous and returns only the first match

So duplicate chemistry labels can be stored today, but the raw vocabulary model
is still not the right abstraction to expose to matcher code.

### 2. The obvious undirected labeled-edge route does not work cleanly

The first attempted shape was:

```rust
GenericUndirectedMonopartiteEdgesBuilder<
    _,
    UpperTriangularCSR2D<ValuedCSR2D<...>>,
    SymmetricCSR2D<ValuedCSR2D<...>>,
>
```

That failed at compile time because the square/upper-triangular wrappers expect
coordinate entries, not `(u, v, value)` entries.

So for the working prototype we had to fall back to a directed valued matrix and
insert both directions manually:

- `(0, 1, Single)`
- `(1, 0, Single)`
- `(1, 2, Single)`
- `(2, 1, Single)`

That is workable, but not ergonomic for molecule graphs.

### 3. Square/symmetric wrappers drop edge attributes at the `Edges` trait level

`SquareCSR2D<M>` and `SymmetricCSR2D<M>` expose:

```rust
type Edge = Coordinates;
```

even when the underlying matrix is valued.

That means graph-level edge typing is lost exactly where a chemistry application
needs it most.

The bond label is still recoverable from matrix methods like
`sparse_value_at(u, v)`, but it is no longer a first-class edge attribute in
the graph API.

### 4. Basic traversal requires trait-disambiguation boilerplate

Even a simple successor query hit method ambiguity:

```rust
Edges::successors(graph.edges(), 1)
```

instead of the more natural:

```rust
graph.successors(1)
```

or:

```rust
graph.neighbors(1)
```

This is not fatal, but it is a bad developer experience for a matcher that will
do this constantly.

### 5. The sparse matrix is useful, but SMARTS also needs dense sidecars

The sparse valued adjacency is appropriate for bond labels.

It is not enough by itself for SMARTS preparation, which also needs dense
per-node or per-edge derived properties such as:

- implicit hydrogen count
- degree
- heavy-atom degree
- valence
- connectivity
- ring membership
- ring bond count
- smallest ring size

Those are not naturally sparse adjacency values. They want dedicated dense
property maps keyed by node ID or edge ID.

### 6. Edge-context helpers need easier edge-ID lookup

`EdgeContexts` is promising for ring memberships or other per-edge prepared
contexts, but it is indexed by sparse edge index.

For SMARTS we will frequently start from `(u, v)` and need:

- the edge ID
- the bond label
- the prepared edge properties

The current API makes that possible, but not cheap to write repeatedly.

## What I Need For Ergonomic SMARTS Integration

These are the concrete improvements that would make the code pleasant enough to
target directly.

### A. Separate node identity from node label

Any of these would help:

- a graph type where node IDs are dense indices and labels live in a separate
  sidecar
- a vocabulary that allows repeated destination symbols
- a dedicated `NodeLabels<NodeId, Label>` abstraction

For chemistry, repeated atom labels are normal, so uniqueness should apply to
node IDs, not to labels.

### B. A first-class undirected valued graph type

Something like:

```rust
type UndirectedValuedGraph<Label> = ...;
type UndirectedValuedEdgesBuilder<Label> = ...;
```

where the edge entry is `(u, v, label)` and the resulting graph preserves the
edge label as part of the edge API.

### C. Preserve typed edge attributes through symmetric wrappers

If the underlying matrix is valued, then graph wrappers should expose either:

- an edge type that includes the value
- or direct graph-level accessors like `edge_value(u, v)`

without falling back to coordinate-only edges.

### D. Node and edge property maps

For SMARTS I want:

- `NodeProps<T>`
- `EdgeProps<T>`

indexed by dense IDs, not improvised from raw `Vec<T>`.

These should be cheap, immutable by default, and explicit about whether they
are dense or sparse.

### E. Graph-level convenience methods

I want a tiny surface that makes matcher code readable:

- `node_ref(id)`
- `neighbors(id)` for undirected graphs
- `bond_ref(u, v)` or `edge_value(u, v)`
- `has_bond(u, v)`
- `edge_id(u, v)`

The lower-level traits are fine, but the current call sites are too verbose for
matching code.

## Practical Recommendation

The architecture is viable:

- parse SMARTS into query IR in `smarts-parser`
- target a generic chemistry graph in `smarts-validator`
- use `elements_rs` for element identity
- use `geometric-traits` as the structural foundation

But I would not make `smarts-validator` depend directly on the raw
`geometric-traits` traits. I would put a thin molecule-target adapter layer in
between and only expose the chemistry-oriented operations the matcher actually
needs.

That keeps the SMARTS implementation clean while still letting us adopt
`geometric-traits` as the storage and algorithm substrate.

## What Landed

The first adapter pass now exists in `smarts-validator`:

- `MoleculeTarget` exposes:
  - `atom_count`
  - `atom`
  - `bond`
  - `neighbors`
  - `degree`
  - optional `edge_id`
- `MoleculeGraph` wraps
  `GenericGraph<Vec<AtomLabel>, ValuedCSR2D<usize, usize, usize, BondLabel>>`
- `PreparedMolecule<T>` demonstrates the preparation pattern with dense
  `NodeProps<T>` sidecars

This confirms the project does not need `smiles-parser` specifically. What it
needs is a chemistry-labeled graph plus a thin adapter that hides the raw graph
crate's vocabulary and matrix details from the SMARTS matcher.
