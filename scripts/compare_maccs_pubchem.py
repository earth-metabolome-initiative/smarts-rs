#!/usr/bin/env python3
"""Compare RDKit MACCS bits against the local smarts-rs matcher on a SMILES stream."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import MACCSkeys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream SMILES through RDKit and the local smarts-rs MACCS evaluator, "
            "then compare active bits."
        )
    )
    parser.add_argument("input", type=Path, help="SMILES file, plain text or .gz")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many non-empty rows",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=1,
        help="1-based row to start from",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        help="Print progress every N checked rows",
    )
    parser.add_argument(
        "--mismatches-jsonl",
        type=Path,
        default=Path("/tmp/maccs_pubchem_failures.jsonl"),
        help="Where to write mismatch rows",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("/tmp/maccs_pubchem_failure_summary.json"),
        help="Where to write the aggregated summary",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="Run the Rust MACCS evaluator in release mode",
    )
    return parser.parse_args()


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


def configure_rdkit_logging() -> None:
    RDLogger.DisableLog("rdApp.warning")


def active_rdkit_bits(smiles: str) -> tuple[list[int] | None, str | None]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "rdkit_parse_failed"
    fp = MACCSkeys.GenMACCSKeys(mol)
    bits = [bit for bit in fp.GetOnBits() if bit != 0]
    return bits, None


def open_text_lines(path: Path) -> Iterable[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            yield from handle
    else:
        with path.open("r", encoding="utf-8") as handle:
            yield from handle


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


def main() -> int:
    args = parse_args()
    configure_rdkit_logging()
    args.mismatches_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    catalog = rdkit_maccs_catalog()
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", suffix=".json", delete=False
    ) as handle:
        json.dump(catalog, handle, separators=(",", ":"))
        catalog_path = Path(handle.name)

    checked_rows = 0
    mismatches = 0
    rdkit_parse_failures = 0
    rust_errors = 0
    signatures: Counter[str] = Counter()
    first_row_for_signature: dict[str, int] = {}

    proc = subprocess.Popen(
        rust_command(catalog_path, args.release),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert proc.stdin is not None
    assert proc.stdout is not None

    try:
        with args.mismatches_jsonl.open("w", encoding="utf-8") as mismatch_handle:
            for row_index, raw_line in enumerate(open_text_lines(args.input), start=1):
                smiles = raw_line.strip()
                if not smiles:
                    continue
                if row_index < args.start_row:
                    continue
                if args.limit is not None and checked_rows >= args.limit:
                    break

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
                    first_row_for_signature.setdefault(signature, row_index)
                    mismatch_handle.write(
                        json.dumps(
                            {
                                "row": row_index,
                                "smiles": smiles,
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
                        f"checked_rows={checked_rows} mismatches={mismatches} "
                        f"rdkit_parse_failures={rdkit_parse_failures} rust_errors={rust_errors}",
                        file=sys.stderr,
                    )
    finally:
        proc.stdin.close()
        proc.wait()
        catalog_path.unlink(missing_ok=True)

    summary = {
        "input": str(args.input),
        "checked_rows": checked_rows,
        "mismatches": mismatches,
        "distinct_mismatch_signatures": len(signatures),
        "rdkit_parse_failures": rdkit_parse_failures,
        "rust_errors": rust_errors,
        "signatures": [
            {
                "signature": signature,
                "count": count,
                "first_row": first_row_for_signature[signature],
            }
            for signature, count in sorted(signatures.items())
        ],
    }
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0 if mismatches == 0 and rust_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
