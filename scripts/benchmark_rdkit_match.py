#!/usr/bin/env python3
"""Benchmark RDKit substructure matching on the shared matching workloads."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem

TARGET_BATCH_SIZE = 30_000


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    smarts: str
    smiles: str
    expected_match: bool
    expected_count: int | None
    use_chirality: bool


@dataclass(frozen=True)
class Workload:
    id: str
    description: str
    case_files: tuple[str, ...]


@dataclass(frozen=True)
class RuntimeCase:
    name: str
    query: Chem.Mol
    target: Chem.Mol
    expected_match: bool
    expected_count: int | None
    use_chirality: bool


REPO_ROOT = Path(__file__).resolve().parent.parent
MATCHING_CORPUS_ROOT = REPO_ROOT / "corpus" / "matching"


def load_workloads() -> list[Workload]:
    raw = json.loads((REPO_ROOT / "corpus" / "benchmark" / "matching-workloads.json").read_text())
    return [
        Workload(
            id=item["id"],
            description=item["description"],
            case_files=tuple(item["case_files"]),
        )
        for item in raw
    ]


def load_cases(case_files: tuple[str, ...]) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for case_file in case_files:
        raw = json.loads((MATCHING_CORPUS_ROOT / case_file).read_text())
        for item in raw:
            cases.append(
                BenchmarkCase(
                    name=item["name"],
                    smarts=item["smarts"],
                    smiles=item["smiles"],
                    expected_match=item["expected_match"],
                    expected_count=item.get("expected_count"),
                    use_chirality=item.get("use_chirality", False),
                )
            )
    return cases


def build_runtime_cases(cases: list[BenchmarkCase]) -> tuple[list[RuntimeCase], int]:
    repeat_count = math.ceil(TARGET_BATCH_SIZE / len(cases))
    runtime_cases: list[RuntimeCase] = []
    for case in cases:
        query = Chem.MolFromSmarts(case.smarts)
        if query is None:
            raise RuntimeError(f"benchmark SMARTS must parse: {case.name}")
        target = Chem.MolFromSmiles(case.smiles)
        if target is None:
            raise RuntimeError(f"benchmark SMILES must parse: {case.name}")
        runtime_cases.append(
            RuntimeCase(
                name=case.name,
                query=query,
                target=target,
                expected_match=case.expected_match,
                expected_count=case.expected_count,
                use_chirality=case.use_chirality,
            )
        )
    return runtime_cases, repeat_count


def benchmark_boolean(runtime_cases: list[RuntimeCase], repeat_count: int) -> tuple[int, float]:
    started = time.perf_counter()
    iterations = 0
    for _ in range(repeat_count):
        for case in runtime_cases:
            matched = case.target.HasSubstructMatch(case.query, useChirality=case.use_chirality)
            if matched != case.expected_match:
                raise RuntimeError(f"boolean benchmark fixture drifted: {case.name}")
            iterations += 1
    elapsed = time.perf_counter() - started
    return iterations, elapsed


def benchmark_count(runtime_cases: list[RuntimeCase], repeat_count: int) -> tuple[int, float]:
    started = time.perf_counter()
    iterations = 0
    for _ in range(repeat_count):
        for case in runtime_cases:
            count = len(
                case.target.GetSubstructMatches(
                    case.query,
                    uniquify=True,
                    useChirality=case.use_chirality,
                )
            )
            if (count > 0) != case.expected_match:
                raise RuntimeError(f"count benchmark fixture drifted: {case.name}")
            if case.expected_count is not None and count != case.expected_count:
                raise RuntimeError(f"count benchmark fixture drifted: {case.name}")
            iterations += 1
    elapsed = time.perf_counter() - started
    return iterations, elapsed


def format_ns_per_pair(iterations: int, elapsed: float) -> float:
    return elapsed * 1_000_000_000.0 / iterations


def main() -> None:
    workloads = load_workloads()
    print("RDKit benchmark results")
    for workload in workloads:
        cases = load_cases(workload.case_files)
        runtime_cases, repeat_count = build_runtime_cases(cases)
        bool_iterations, bool_elapsed = benchmark_boolean(runtime_cases, repeat_count)
        count_iterations, count_elapsed = benchmark_count(runtime_cases, repeat_count)
        print(
            f"{workload.id}: boolean {format_ns_per_pair(bool_iterations, bool_elapsed):.1f} ns/pair, "
            f"count {format_ns_per_pair(count_iterations, count_elapsed):.1f} ns/pair"
        )


if __name__ == "__main__":
    main()
