#!/bin/bash
set -eu

cd "$SRC/smarts-rs"
cargo fuzz build -O --debug-assertions --fuzz-dir fuzz

targets=$(cargo fuzz list --fuzz-dir fuzz)
if [[ -z "$targets" ]]; then
    echo "cargo fuzz list named no target" >&2
    exit 1
fi

target_dir=fuzz/target/x86_64-unknown-linux-gnu/release
for name in $targets; do
    cp "$target_dir/$name" "$OUT/"
    # a target-specific dictionary wins over the shared SMARTS one
    if [[ -f "fuzz/$name.dict" ]]; then
        cp "fuzz/$name.dict" "$OUT/$name.dict"
    else
        cp fuzz/dictionaries/smarts.dict "$OUT/$name.dict"
    fi
done

# the runner unpacks <target>_seed_corpus.zip as the starting corpus
for dir in fuzz/corpus/*/; do
    name=$(basename "$dir")
    zip -qj "$OUT/${name}_seed_corpus.zip" "$dir"*
done
