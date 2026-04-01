import json
import math
import statistics
import time
from pathlib import Path

from rdkit import Chem

TARGET_BATCH_SIZE = 120_000


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_cases() -> dict[str, str]:
    parser_cases = json.loads((repo_root() / "corpus" / "parser" / "parse-valid-v0.json").read_text())
    extra_cases = json.loads((repo_root() / "corpus" / "benchmark" / "parse-extra-cases.json").read_text())
    return {case["id"]: case["smarts"] for case in [*parser_cases, *extra_cases]}


def load_workloads() -> list[dict]:
    path = repo_root() / "corpus" / "benchmark" / "parse-workloads.json"
    return json.loads(path.read_text())


def build_dataset(case_ids: list[str], smarts_by_id: dict[str, str]) -> tuple[list[str], int]:
    unique = [smarts_by_id[case_id] for case_id in case_ids]
    repeat_count = math.ceil(TARGET_BATCH_SIZE / len(unique))
    return unique * repeat_count, len(unique)


def main() -> None:
    smarts_by_id = load_cases()
    workloads = load_workloads()
    repeats = 10
    results = []
    params = Chem.SmartsParserParams()
    params.allowCXSMILES = False
    params.parseName = False
    params.strictCXSMILES = False

    for workload in workloads:
        dataset, unique_count = build_dataset(workload["case_ids"], smarts_by_id)
        assert dataset, f"dataset must not be empty for workload {workload['id']}"

        for smarts in dataset[:unique_count]:
            assert Chem.MolFromSmarts(smarts, params) is not None, (workload["id"], smarts)

        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            parsed_count = 0
            for smarts in dataset:
                mol = Chem.MolFromSmarts(smarts, params)
                assert mol is not None
                parsed_count += 1
            end = time.perf_counter()
            assert parsed_count == len(dataset)
            timings.append(end - start)

        median_s = statistics.median(timings)
        mean_s = statistics.fmean(timings)
        results.append(
            {
                "workload": workload["id"],
                "description": workload["description"],
                "k": len(dataset),
                "unique": unique_count,
                "repeats": repeats,
                "median_seconds": median_s,
                "mean_seconds": mean_s,
                "median_ns_per_smarts": median_s * 1e9 / len(dataset),
                "mean_ns_per_smarts": mean_s * 1e9 / len(dataset),
            }
        )

    print(json.dumps({"parser": "rdkit", "target_batch_size": TARGET_BATCH_SIZE, "results": results}, indent=2))


if __name__ == "__main__":
    main()
