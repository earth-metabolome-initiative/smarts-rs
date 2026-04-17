#!/usr/bin/env python3
"""Generate frozen RDKit expectations for SMARTS validator fixture cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rdkit import Chem


def load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"expected a list of cases in {path}")
    return data


def compile_smarts(smarts: str):
    query = Chem.MolFromSmarts(smarts)
    if query is None:
        raise ValueError(f"RDKit failed to parse SMARTS: {smarts!r}")
    return query


def compile_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles!r}")
    return mol


def build_expectations(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expectations: list[dict[str, Any]] = []
    for case in cases:
        name = case["name"]
        smarts = case["smarts"]
        smiles = case["smiles"]
        use_chirality = bool(case.get("use_chirality", False))

        query = compile_smarts(smarts)
        target = compile_smiles(smiles)
        expectations.append(
            {
                "name": name,
                "smarts": smarts,
                "smiles": smiles,
                "use_chirality": use_chirality,
                "expected_match": bool(
                    target.HasSubstructMatch(query, useChirality=use_chirality)
                ),
            }
        )
    return expectations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    expectations = build_expectations(load_cases(args.cases))
    args.output.write_text(
        json.dumps(expectations, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
