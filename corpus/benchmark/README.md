# Benchmark Corpus

This directory contains the fixed benchmark inputs for the `smarts-rs`
`evolution_matrix` Criterion benchmark.

- `smarts-evolution-example-smiles-v0.tsv` contains 20,000 downstream example
  SMILES grouped by source dataset.
- `smarts-evolution-complex-queries-v0.smarts` contains 200 mined SMARTS
  queries generated from those SMILES.

The benchmark models the expected genetic-algorithm workload: the target SMILES
corpus is fixed, while each generation evaluates a changing batch of SMARTS
queries. The corpus is intentionally heterogeneous so index work is not tuned
only for small or trivial SMARTS.

Regenerate the SMARTS fixture with:

```sh
cargo run --example generate_evolution_query_corpus
```

After regenerating, run the correctness and performance checks:

```sh
cargo test
cargo bench --bench evolution_matrix
```
