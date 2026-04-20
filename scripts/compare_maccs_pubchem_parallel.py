#!/usr/bin/env python3
"""Run MACCS parity checks against a large PubChem-style CID/SMILES TSV in parallel."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import MACCSkeys


DEFAULT_INPUT_CANDIDATES = (
    Path("/home/luca/github/pubchem-topology/data/pubchem/CID-SMILES.tsv"),
    Path("/home/luca/github/npc-labeler/work/CID-SMILES.tsv"),
    Path("/home/luca/github/pubchem-rascal-mces/data/raw/CID-SMILES.gz"),
    Path("/home/luca/github/pubchem-fingerprints/data/CID-SMILES.gz"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare RDKit MACCS keys against the local smarts-rs implementation "
            "across a PubChem-style CID/SMILES TSV using multiple workers."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="CID/SMILES TSV or TSV.GZ. If omitted, use the first known local PubChem dump.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(os.cpu_count() or 1, 16)),
        help="Number of worker processes to launch for plain-text TSV input.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/maccs_pubchem_parallel"),
        help="Directory for per-worker mismatch logs and summaries.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100_000,
        help="Print progress every N checked rows per worker.",
    )
    parser.add_argument(
        "--cid-column",
        type=int,
        default=1,
        help="1-based CID column index in the input TSV.",
    )
    parser.add_argument(
        "--smiles-column",
        type=int,
        default=2,
        help="1-based SMILES column index in the input TSV.",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="Run the Rust MACCS evaluator in release mode.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed worker summaries already present in the output directory.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--start-offset", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--end-offset", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--catalog", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--mismatches-jsonl", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--summary-json", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def configure_rdkit_logging() -> None:
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")


def rdkit_maccs_catalog() -> list[dict[str, object]]:
    catalog: list[dict[str, object]] = []
    for key_id in range(1, 167):
        smarts, count_threshold = MACCSkeys.smartsPatts[key_id]
        if smarts == "?":
            special_case = {
                1: "undefined",
                125: "aromatic_ring_count_gt_1",
                166: "fragment_count_gt_1",
            }.get(key_id)
            catalog.append(
                {
                    "key_id": key_id,
                    "smarts": None,
                    "count_threshold": count_threshold,
                    "special_case": special_case,
                }
            )
        else:
            catalog.append(
                {
                    "key_id": key_id,
                    "smarts": smarts,
                    "count_threshold": count_threshold,
                    "special_case": None,
                }
            )
    return catalog


def resolve_input_path(path: Path | None) -> Path:
    if path is not None:
        if not path.exists():
            raise SystemExit(f"input file not found: {path}")
        return path

    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate

    raise SystemExit(
        "no local PubChem dump found; pass an explicit input path. "
        f"Tried: {', '.join(str(candidate) for candidate in DEFAULT_INPUT_CANDIDATES)}"
    )


def active_rdkit_bits(smiles: str) -> tuple[list[int] | None, str | None]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "rdkit_parse_failed"
    fp = MACCSkeys.GenMACCSKeys(mol)
    bits = [bit for bit in fp.GetOnBits() if bit != 0]
    return bits, None


def rust_command(catalog_path: Path, release: bool) -> list[str]:
    command = [
        "cargo",
        "run",
        "--quiet",
        "-p",
        "smarts-validator",
        "--bin",
        "maccs_eval",
    ]
    if release:
        command.append("--release")
    command.extend(["--", "--catalog", str(catalog_path)])
    return command


def shard_ranges(path: Path, workers: int) -> list[tuple[int, int | None]]:
    if path.suffix == ".gz":
        if workers != 1:
            raise SystemExit("parallel byte-range sharding requires an uncompressed TSV input")
        return [(0, None)]

    size = path.stat().st_size
    if size == 0:
        return [(0, None)]

    boundaries = [size * worker // workers for worker in range(workers + 1)]
    ranges: list[tuple[int, int | None]] = []
    for worker in range(workers):
        start = boundaries[worker]
        end = boundaries[worker + 1] if worker < workers - 1 else None
        ranges.append((start, end))
    return ranges


def extract_fields(line: str, cid_column: int, smiles_column: int) -> tuple[str, str] | None:
    fields = line.rstrip("\n\r").split("\t")
    needed = max(cid_column, smiles_column)
    if len(fields) < needed:
        return None
    cid = fields[cid_column - 1].strip()
    smiles = fields[smiles_column - 1].strip()
    if not cid or not smiles:
        return None
    return cid, smiles


def iter_shard_lines(path: Path, start_offset: int, end_offset: int | None) -> Iterable[tuple[int, str]]:
    if path.suffix == ".gz":
        import gzip

        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for row_index, line in enumerate(handle, start=1):
                yield row_index, line
        return

    with path.open("rb") as handle:
        if start_offset > 0:
            handle.seek(start_offset - 1)
            if handle.read(1) != b"\n":
                handle.readline()
        else:
            handle.seek(0)

        while True:
            line_start = handle.tell()
            if end_offset is not None and line_start >= end_offset:
                break
            raw_line = handle.readline()
            if not raw_line:
                break
            yield line_start, raw_line.decode("utf-8")


def write_catalog(path: Path) -> None:
    path.write_text(json.dumps(rdkit_maccs_catalog(), separators=(",", ":")), encoding="utf-8")


def worker_main(args: argparse.Namespace) -> int:
    configure_rdkit_logging()
    assert args.catalog is not None
    assert args.worker_index is not None
    assert args.mismatches_jsonl is not None
    assert args.summary_json is not None

    checked_rows = 0
    mismatches = 0
    rdkit_parse_failures = 0
    rust_errors = 0
    malformed_rows = 0
    signatures: Counter[str] = Counter()
    first_cid_for_signature: dict[str, str] = {}

    proc = subprocess.Popen(
        rust_command(args.catalog, args.release),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert proc.stdin is not None
    assert proc.stdout is not None

    try:
        with args.mismatches_jsonl.open("w", encoding="utf-8") as mismatch_handle:
            for line_start, raw_line in iter_shard_lines(
                args.input, args.start_offset or 0, args.end_offset
            ):
                extracted = extract_fields(raw_line, args.cid_column, args.smiles_column)
                if extracted is None:
                    malformed_rows += 1
                    continue

                cid, smiles = extracted
                rdkit_bits, rdkit_error = active_rdkit_bits(smiles)
                if rdkit_error is not None:
                    rdkit_parse_failures += 1
                    continue

                proc.stdin.write(smiles)
                proc.stdin.write("\n")
                proc.stdin.flush()
                rust_line = proc.stdout.readline()
                if not rust_line:
                    rust_errors += 1
                    break

                rust_output = json.loads(rust_line)
                if not rust_output["ok"]:
                    rust_errors += 1
                    continue

                rust_bits = rust_output["bits"]
                checked_rows += 1

                extra = sorted(set(rust_bits) - set(rdkit_bits))
                missing = sorted(set(rdkit_bits) - set(rust_bits))
                if extra or missing:
                    mismatches += 1
                    signature = f"extra={extra};missing={missing}"
                    signatures[signature] += 1
                    first_cid_for_signature.setdefault(signature, cid)
                    mismatch_handle.write(
                        json.dumps(
                            {
                                "worker": args.worker_index,
                                "cid": cid,
                                "smiles": smiles,
                                "byte_offset": line_start,
                                "extra": extra,
                                "missing": missing,
                                "rust_bits": rust_bits,
                                "rdkit_bits": rdkit_bits,
                            }
                        )
                    )
                    mismatch_handle.write("\n")
                    mismatch_handle.flush()

                if args.progress_every and checked_rows % args.progress_every == 0:
                    print(
                        f"worker={args.worker_index} checked_rows={checked_rows} "
                        f"mismatches={mismatches} rdkit_parse_failures={rdkit_parse_failures} "
                        f"rust_errors={rust_errors} malformed_rows={malformed_rows}",
                        file=sys.stderr,
                    )
    finally:
        proc.stdin.close()
        proc.wait()

    summary = {
        "completed": True,
        "worker": args.worker_index,
        "input": str(args.input),
        "start_offset": args.start_offset,
        "end_offset": args.end_offset,
        "checked_rows": checked_rows,
        "mismatches": mismatches,
        "rdkit_parse_failures": rdkit_parse_failures,
        "rust_errors": rust_errors,
        "malformed_rows": malformed_rows,
        "signatures": [
            {
                "signature": signature,
                "count": count,
                "first_cid": first_cid_for_signature[signature],
            }
            for signature, count in sorted(signatures.items())
        ],
    }
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if rust_errors == 0 else 1


def worker_command(
    script_path: Path,
    input_path: Path,
    output_dir: Path,
    catalog_path: Path,
    worker_index: int,
    start_offset: int,
    end_offset: int | None,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str(script_path),
        str(input_path),
        "--worker",
        "--worker-index",
        str(worker_index),
        "--cid-column",
        str(args.cid_column),
        "--smiles-column",
        str(args.smiles_column),
        "--progress-every",
        str(args.progress_every),
        "--catalog",
        str(catalog_path),
        "--mismatches-jsonl",
        str(output_dir / f"worker-{worker_index:03d}.jsonl"),
        "--summary-json",
        str(output_dir / f"worker-{worker_index:03d}.summary.json"),
        "--start-offset",
        str(start_offset),
    ]
    if end_offset is not None:
        command.extend(["--end-offset", str(end_offset)])
    if args.release:
        command.append("--release")
    return command


def load_worker_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_worker_outputs(output_dir: Path, worker_count: int) -> dict[str, object]:
    signatures: Counter[str] = Counter()
    first_cid_for_signature: dict[str, str] = {}
    checked_rows = 0
    mismatches = 0
    rdkit_parse_failures = 0
    rust_errors = 0
    malformed_rows = 0

    for worker_index in range(worker_count):
        summary_path = output_dir / f"worker-{worker_index:03d}.summary.json"
        if not summary_path.exists():
            raise SystemExit(f"missing worker summary: {summary_path}")
        summary = load_worker_summary(summary_path)
        checked_rows += int(summary["checked_rows"])
        mismatches += int(summary["mismatches"])
        rdkit_parse_failures += int(summary["rdkit_parse_failures"])
        rust_errors += int(summary["rust_errors"])
        malformed_rows += int(summary["malformed_rows"])
        for signature_entry in summary["signatures"]:
            signature = signature_entry["signature"]
            signatures[signature] += int(signature_entry["count"])
            first_cid_for_signature.setdefault(signature, signature_entry["first_cid"])

    merged_jsonl = output_dir / "mismatches.jsonl"
    with merged_jsonl.open("w", encoding="utf-8") as merged_handle:
        for worker_index in range(worker_count):
            worker_jsonl = output_dir / f"worker-{worker_index:03d}.jsonl"
            if not worker_jsonl.exists():
                continue
            merged_handle.write(worker_jsonl.read_text(encoding="utf-8"))

    return {
        "workers": worker_count,
        "checked_rows": checked_rows,
        "mismatches": mismatches,
        "distinct_mismatch_signatures": len(signatures),
        "rdkit_parse_failures": rdkit_parse_failures,
        "rust_errors": rust_errors,
        "malformed_rows": malformed_rows,
        "merged_mismatches_jsonl": str(merged_jsonl),
        "signatures": [
            {
                "signature": signature,
                "count": count,
                "first_cid": first_cid_for_signature[signature],
            }
            for signature, count in sorted(signatures.items())
        ],
    }


def coordinator_main(args: argparse.Namespace) -> int:
    configure_rdkit_logging()
    input_path = resolve_input_path(args.input)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = output_dir / "maccs_catalog.json"
    if not catalog_path.exists():
        write_catalog(catalog_path)

    ranges = shard_ranges(input_path, args.workers)
    script_path = Path(__file__).resolve()
    processes: list[subprocess.Popen[str]] = []

    for worker_index, (start_offset, end_offset) in enumerate(ranges):
        summary_path = output_dir / f"worker-{worker_index:03d}.summary.json"
        if args.resume and summary_path.exists():
            summary = load_worker_summary(summary_path)
            if summary.get("completed") is True:
                print(f"reusing completed worker {worker_index}", file=sys.stderr)
                continue

        command = worker_command(
            script_path,
            input_path,
            output_dir,
            catalog_path,
            worker_index,
            start_offset,
            end_offset,
            args,
        )
        processes.append(subprocess.Popen(command))

    exit_code = 0
    for process in processes:
        rc = process.wait()
        if rc != 0:
            exit_code = rc

    summary = merge_worker_outputs(output_dir, len(ranges))
    summary["input"] = str(input_path)
    summary["catalog"] = str(catalog_path)
    summary["output_dir"] = str(output_dir)
    summary["resume"] = args.resume
    summary["release"] = args.release
    summary["worker_count_requested"] = args.workers

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if exit_code != 0:
        return exit_code
    return 0 if summary["mismatches"] == 0 and summary["rust_errors"] == 0 else 1


def main() -> int:
    args = parse_args()
    if args.worker:
        return worker_main(args)
    return coordinator_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
