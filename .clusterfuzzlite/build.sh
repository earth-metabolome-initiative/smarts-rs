#!/bin/bash
set -eu

cd "$SRC/smarts-rs"

targets=$(cargo fuzz list --fuzz-dir fuzz)
if [[ -z "$targets" ]]; then
    echo "cargo fuzz list named no target" >&2
    exit 1
fi

# every target starts from a committed seed corpus
unseeded=0
for name in $targets; do
    if [[ -z "$(ls -A "fuzz/seeds/$name" 2>/dev/null)" ]]; then
        echo "fuzz target $name has no seed corpus in fuzz/seeds/$name" >&2
        unseeded=1
    fi
done
if ((unseeded)); then
    exit 1
fi

cargo fuzz build -O --debug-assertions --fuzz-dir fuzz

target_dir=fuzz/target/x86_64-unknown-linux-gnu/release
for name in $targets; do
    cp "$target_dir/$name" "$OUT/"
    # a target-specific dictionary wins over the shared SMARTS one
    if [[ -f "fuzz/$name.dict" ]]; then
        cp "fuzz/$name.dict" "$OUT/$name.dict"
    else
        cp fuzz/dictionaries/smarts.dict "$OUT/$name.dict"
    fi
    # the runner unpacks <target>_seed_corpus.zip as the starting corpus
    zip -qj "$OUT/${name}_seed_corpus.zip" "fuzz/seeds/$name"/*
done
