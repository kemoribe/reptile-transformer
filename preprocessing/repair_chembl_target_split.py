#!/usr/bin/env python3
"""Merge duplicate ChEMBL targets into one split (test > val > train)."""

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd

SPLITS = ("train_set", "val_set", "test_set")
PRIORITY = {"train_set": 0, "val_set": 1, "test_set": 2}


def activity_file(target_dir: Path):
    files = sorted(target_dir.glob("*_processed_activities.csv"))
    return files[0] if files else None


def sequence_file(target_dir: Path):
    files = sorted(target_dir.glob("*_processed_protein_sequence.txt"))
    return files[0] if files else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.dataset_dir.resolve()

    locations = defaultdict(list)
    for split in SPLITS:
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(split_dir)
        for csv_path in split_dir.rglob("*_processed_activities.csv"):
            locations[csv_path.parent.name].append((split, csv_path.parent))

    assignments = {}
    duplicate_count = 0
    for target, entries in locations.items():
        selected_split, selected_dir = max(entries, key=lambda item: PRIORITY[item[0]])
        assignments[target] = selected_split
        if len(entries) == 1:
            continue
        duplicate_count += 1
        print(f"{target}: {sorted(split for split, _ in entries)} -> {selected_split}")
        if args.dry_run:
            continue

        frames = [pd.read_csv(activity_file(path)) for _, path in entries]
        merged = pd.concat(frames, ignore_index=True).drop_duplicates()
        selected_dir.mkdir(parents=True, exist_ok=True)
        merged.to_csv(selected_dir / f"{target}_processed_activities.csv", index=False)

        if sequence_file(selected_dir) is None:
            source_sequence = next((sequence_file(path) for _, path in entries if sequence_file(path)), None)
            if source_sequence:
                shutil.copy2(source_sequence, selected_dir / f"{target}_processed_protein_sequence.txt")

        for _, path in entries:
            if path != selected_dir and path.exists():
                shutil.rmtree(path)

    if not args.dry_run:
        for split in SPLITS:
            for directory in sorted((root / split).rglob("*"), reverse=True):
                if directory.is_dir() and not any(directory.iterdir()):
                    directory.rmdir()

        combined_path = root / "combined_activities.csv"
        combined = pd.read_csv(combined_path)
        if "target_name" not in combined or "set_name" not in combined:
            raise ValueError("combined_activities.csv must contain target_name and set_name")
        combined["set_name"] = combined["target_name"].map(assignments)
        if combined["set_name"].isna().any():
            raise ValueError("Some combined rows could not be assigned to a split")
        rows_before = len(combined)
        combined = combined.drop_duplicates().reset_index(drop=True)
        combined.to_csv(combined_path, index=False)
        print(f"Removed duplicate combined rows: {rows_before - len(combined)}")

    print(f"Duplicate targets: {duplicate_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
