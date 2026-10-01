# -*- coding: utf-8 -*-
"""FASTA / CSV input-output helpers.

Sequence identifiers
--------------------
The audit needs to tell training and test targets apart inside one merged
FASTA.  We therefore use two ID namespaces with fixed prefixes:

* ``train_0000, train_0001, ...`` for unique training sequences
* ``test_0000,  test_0001,  ...`` for unique test sequences

IDs are assigned after lexicographically sorting the *de-duplicated* sequence
strings, so the mapping is deterministic across machines and runs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

TRAIN_PREFIX = "train_"
TEST_PREFIX = "test_"


def write_fasta(seq2id: Dict[str, str], path: Path, wrap: int = 60) -> None:
    """Write ``{sequence: identifier}`` to a FASTA file (LF line endings)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        for seq, sid in seq2id.items():
            f.write(f">{sid}\n")
            for i in range(0, len(seq), wrap):
                f.write(seq[i : i + wrap] + "\n")


def write_fasta_pairs(records: Sequence[Tuple[str, str]], path: Path,
                      wrap: int = 60) -> None:
    """Write ``[(identifier, sequence), ...]`` preserving the given order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        for sid, seq in records:
            f.write(f">{sid}\n")
            for i in range(0, len(seq), wrap):
                f.write(seq[i : i + wrap] + "\n")


def read_fasta(path: Path) -> List[Tuple[str, str]]:
    """Parse a FASTA file into ``[(identifier, sequence), ...]`` in file order."""
    records: List[Tuple[str, str]] = []
    name = None
    chunks: List[str] = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(chunks)))
                name, chunks = line[1:].split()[0], []
            else:
                chunks.append(line)
    if name is not None:
        records.append((name, "".join(chunks)))
    return records


def sequences_from_csv(csv_path: Path, seq_column: str) -> List[str]:
    """Read and de-duplicate protein sequences from a CSV column."""
    try:
        df = pd.read_csv(csv_path, usecols=[seq_column])
    except ValueError:
        available = list(pd.read_csv(csv_path, nrows=0).columns)
        raise ValueError(
            f"column {seq_column!r} not found in {csv_path}; "
            f"available columns: {available}"
        ) from None
    seqs = sorted({str(s).strip() for s in df[seq_column].dropna() if str(s).strip()})
    if not seqs:
        raise ValueError(f"no non-empty sequences in column {seq_column!r} of {csv_path}")
    return seqs


def build_id_maps(train_seqs: Sequence[str], test_seqs: Sequence[str]
                  ) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Deterministic ``sequence -> train_/test_ ID`` maps (sorted, zero-padded)."""
    width = max(4, len(str(max(len(train_seqs), len(test_seqs)) - 1)))
    train_map = {s: f"{TRAIN_PREFIX}{i:0{width}d}" for i, s in enumerate(sorted(train_seqs))}
    test_map = {s: f"{TEST_PREFIX}{i:0{width}d}" for i, s in enumerate(sorted(test_seqs))}
    return train_map, test_map


def parse_float_list(text: str, cast=float) -> list:
    """Parse ``"0.4,0.6,0.8"`` / ``"40,60,80"`` into a list of floats."""
    if text is None or not str(text).strip():
        return []
    return [cast(float(x)) for x in str(text).replace(";", ",").split(",") if x.strip()]
