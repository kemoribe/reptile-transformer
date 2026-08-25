#!/usr/bin/env python3
"""Validate processed dataset layout and target-disjoint splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Optional, Tuple

SPLITS = ("train_set", "val_set", "test_set")
SEQUENCE_SUFFIXES = {".txt", ".fa", ".faa", ".fasta"}


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_activity_file(target_dir: Path) -> Optional[Path]:
    candidates = sorted(target_dir.glob("*_processed_activities.csv"))
    candidates.extend(sorted(target_dir.glob("*_activities.csv")))
    candidates.extend(sorted(target_dir.glob("activities.csv")))
    return candidates[0] if candidates else None


def read_sequence(target_dir: Path) -> Optional[str]:
    candidates = sorted(target_dir.glob("*protein_sequence*"))
    candidates.extend(sorted(target_dir.glob("sequence.*")))
    candidates.extend(
        path for path in sorted(target_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SEQUENCE_SUFFIXES
    )
    for path in candidates:
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
        sequence = "".join(
            line.strip() for line in text.splitlines() if not line.startswith(">")
        ).upper()
        sequence = "".join(char for char in sequence if char in "ACDEFGHIKLMNPQRSTVWY")
        if sequence:
            return sequence
    return None


def inspect_split(split_dir: Path) -> dict:
    target_dirs = sorted({path.parent for path in split_dir.rglob("*.csv")})
    rows = 0
    targets = set()
    sequences = set()
    missing_sequences = []

    for target_dir in target_dirs:
        activity_file = find_activity_file(target_dir)
        if activity_file is None:
            continue
        rows += count_csv_rows(activity_file)
        targets.add(target_dir.name)
        sequence = read_sequence(target_dir)
        if sequence:
            sequences.add(sequence)
        else:
            missing_sequences.append(str(target_dir.relative_to(split_dir)))

    return {
        "rows": rows,
        "targets": len(targets),
        "unique_sequences": len(sequences),
        "missing_sequence_targets": missing_sequences,
        "_target_ids": targets,
        "_sequences": sequences,
    }


def inspect_dataset(dataset_dir: Path) -> Tuple[dict, bool]:
    report = {"path": str(dataset_dir), "splits": {}, "overlap": {}}
    ok = True
    for split in SPLITS:
        split_dir = dataset_dir / split
        if not split_dir.is_dir():
            report["splits"][split] = {"error": "missing directory"}
            ok = False
            continue
        report["splits"][split] = inspect_split(split_dir)

    pairs = (("train_set", "val_set"), ("train_set", "test_set"), ("val_set", "test_set"))
    for left, right in pairs:
        if "error" in report["splits"][left] or "error" in report["splits"][right]:
            continue
        left_data = report["splits"][left]
        right_data = report["splits"][right]
        target_overlap = left_data["_target_ids"] & right_data["_target_ids"]
        sequence_overlap = left_data["_sequences"] & right_data["_sequences"]
        key = f"{left}_vs_{right}"
        report["overlap"][key] = {
            "target_ids": len(target_overlap),
            "protein_sequences": len(sequence_overlap),
        }
        ok = ok and not target_overlap and not sequence_overlap

    combined = dataset_dir / "combined_activities.csv"
    if combined.is_file():
        combined_rows = count_csv_rows(combined)
        split_rows = sum(
            split_data.get("rows", 0) for split_data in report["splits"].values()
        )
        report["combined_activities"] = {
            "rows": combined_rows,
            "bytes": combined.stat().st_size,
            "sha256": sha256(combined),
            "matches_split_rows": combined_rows == split_rows,
        }
        ok = ok and combined_rows == split_rows
    else:
        report["combined_activities"] = {"error": "missing file"}
        ok = False

    for split_data in report["splits"].values():
        split_data.pop("_target_ids", None)
        split_data.pop("_sequences", None)
    report["target_disjoint"] = ok
    return report, ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/data_manifest.json"))
    parser.add_argument("--datasets", nargs="+", default=["chembl", "davis", "kiba", "bindingdb"])
    args = parser.parse_args()

    reports = {}
    all_ok = True
    for name in args.datasets:
        dataset_dir = args.root / name
        portable_path = (Path("data") / "processed" / name).as_posix()
        if not dataset_dir.is_dir():
            reports[name] = {"path": portable_path, "error": "missing dataset directory"}
            all_ok = False
            continue
        reports[name], ok = inspect_dataset(dataset_dir)
        reports[name]["path"] = portable_path
        all_ok = all_ok and ok

    payload = {"datasets": reports, "all_target_disjoint": all_ok}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
