# -*- coding: utf-8 -*-
"""Adapters for common DTA benchmark data formats.

The audit core only needs *two lists of protein sequences* (train / test).
This module turns the formats people actually have into the normalized pair
of CSVs (one column ``target_sequence``) that :func:`homology_audit.audit.
run_audit` consumes.

Supported sources for ``--data``
--------------------------------

``split directory``
    A folder containing ``<name>_train.csv`` + ``<name>_test.csv`` (or
    ``train.fasta`` + ``test.fasta``).  Passed through after a sequence-
    column sanity check.

``single long CSV`` (incl. preprocessed Davis/KIBA releases)
    One CSV with a sequence column.  Recognized sequence columns (first
    match wins): explicit ``--seq-column``, ``target_sequence``,
    ``BindingDB_Target_Chain_Sequence`` (BindingDB TSV/CSV export),
    ``target_seq``, ``Sequence``, ``seq``.  Splitting:

    * a split/set column (``split``, ``set``, ``subset``, ``train_test``)
      whose values look like train/test → rows are routed accordingly
      (``val``/``validation`` rows go to test only if no explicit test rows
      exist);
    * otherwise all rows are training data and a separate ``--test-data``
      (CSV/FASTA) must be provided, or ``--test-targets`` (a file listing
      test target sequences or IDs, one per line) selects the test rows
      from the same CSV.

``FASTA``
    A single FASTA is training data (pair with ``--test-data``); a pair of
    FASTAs via ``--data train.fasta --test-data test.fasta``.

``DAVIS/KIBA affinity matrix``
    ``--data affinity.mat`` with ``--targets-file`` (one target name per
    line) and ``--sequences-fasta`` (``>name`` headers matching target
    names).  Produces a long table of targets that have any measured
    affinity; optional ``--test-targets`` (one test target name per line)
    performs a target-disjoint split, otherwise the matrix is exported as
    training data and ``--test-data`` completes the pair.

Outputs are written to ``<out>/<name>_train.csv`` / ``<out>/<name>_test.csv``
plus a small ``prepare_report.txt``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .io_utils import read_fasta

SEQ_COLUMN = "target_sequence"
DEFAULT_SEQ_COLUMNS = ("target_sequence", "BindingDB_Target_Chain_Sequence",
                       "target_seq", "Sequence", "seq", "protein_sequence")
DEFAULT_SPLIT_COLUMNS = ("split", "set", "subset", "train_test", "group")

TRAIN_FILE_FMT = "{name}_train.csv"
TEST_FILE_FMT = "{name}_test.csv"
PREPARE_REPORT = "prepare_report.txt"

_TEST_RE = re.compile(r"test|holdout|hold_out", re.I)
_TRAIN_RE = re.compile(r"train", re.I)
_VAL_RE = re.compile(r"val", re.I)


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------
def detect_format(data: Path) -> str:
    """Classify ``--data`` into one of the supported layout strings."""
    data = Path(data)
    if data.is_dir():
        return "split_dir"
    suffix = data.suffix.lower()
    if suffix == ".mat":
        return "affinity_mat"
    if suffix in (".fasta", ".fa", ".faa", ".fna", ".txt") and _looks_fasta(data):
        return "fasta"
    if suffix in (".csv", ".tsv", ".txt"):
        return "single_csv"
    return "unknown"


def _looks_fasta(path: Path) -> bool:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                return s.startswith(">")
    except OSError:
        return False
    return False


def _read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _resolve_seq_column(df: pd.DataFrame, seq_column: Optional[str]) -> str:
    if seq_column:
        if seq_column not in df.columns:
            raise ValueError(f"sequence column {seq_column!r} not found; "
                             f"available: {list(df.columns)}")
        return seq_column
    for cand in DEFAULT_SEQ_COLUMNS:
        if cand in df.columns:
            return cand
    raise ValueError(f"no sequence column detected; available: {list(df.columns)}; "
                     f"pass --seq-column explicitly")


def _find_split_column(df: pd.DataFrame, split_col: Optional[str]) -> Optional[str]:
    if split_col:
        if split_col not in df.columns:
            raise ValueError(f"split column {split_col!r} not found; "
                             f"available: {list(df.columns)}")
        return split_col
    for cand in DEFAULT_SPLIT_COLUMNS:
        if cand in df.columns:
            return cand
    return None


def _split_frame(df: pd.DataFrame, split_col: str
                 ) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    vals = df[split_col].astype(str)
    is_test = vals.str.contains(_TEST_RE)
    is_train = vals.str.contains(_TRAIN_RE) & ~is_test
    is_val = vals.str.contains(_VAL_RE) & ~is_test & ~is_train
    if not is_test.any() and not is_train.any():
        return df, df.head(0).copy(), "no-train-test-values"
    if not is_test.any() and is_val.any():
        # route validation rows to test only when no explicit test rows exist
        return df[is_train], df[is_val], "train+val(test=validation)"
    if not is_train.any():
        return df[~is_test], df[is_test], "test-only"
    return df[is_train], df[is_test], "train+test"


# ---------------------------------------------------------------------------
# source loaders
# ---------------------------------------------------------------------------
def _from_split_dir(data: Path, name: str, seq_column: Optional[str],
                    out_dir: Path) -> Tuple[Path, Path, List[str]]:
    notes: List[str] = []
    train_csv = data / TRAIN_FILE_FMT.format(name=name)
    test_csv = data / TEST_FILE_FMT.format(name=name)
    if train_csv.exists() and test_csv.exists():
        for p in (train_csv, test_csv):
            df = _read_table(p)
            _resolve_seq_column(df, seq_column)
            if seq_column and seq_column != SEQ_COLUMN:
                df = df.rename(columns={seq_column: SEQ_COLUMN})
                df.to_csv(p, index=False)
                notes.append(f"renamed {seq_column} -> {SEQ_COLUMN} in {p.name}")
        return train_csv, test_csv, notes
    train_fa, test_fa = data / "train.fasta", data / "test.fasta"
    if train_fa.exists() and test_fa.exists():
        tc, xc = _from_fasta_pair(train_fa, test_fa, out_dir, name)
        return tc, xc, ["converted FASTA pair from split directory"]
    raise FileNotFoundError(
        f"{data} does not contain {train_csv.name}/{test_csv.name} or "
        f"train.fasta/test.fasta")


def _from_fasta_pair(train_fa: Path, test_fa: Path, out_dir: Path, name: str
                     ) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv, test_csv = (out_dir / TRAIN_FILE_FMT.format(name=name),
                           out_dir / TEST_FILE_FMT.format(name=name))
    for src, dst in ((train_fa, train_csv), (test_fa, test_csv)):
        recs = read_fasta(src)
        pd.DataFrame({SEQ_COLUMN: [s for _, s in recs]}).to_csv(dst, index=False)
    return train_csv, test_csv


def _from_single_csv(data: Path, seq_column: Optional[str],
                     split_col_arg: Optional[str], test_data: Optional[Path],
                     test_targets: Optional[Path], out_dir: Path, name: str,
                     notes: List[str]) -> Tuple[Path, Path]:
    df = _read_table(data)
    seq_col = _resolve_seq_column(df, seq_column)
    if seq_col != SEQ_COLUMN:
        df = df.rename(columns={seq_col: SEQ_COLUMN})
        notes.append(f"sequence column {seq_col!r} normalized to "
                     f"{SEQ_COLUMN!r}")

    split_col = _find_split_column(df, split_col_arg)
    if split_col:
        train_df, test_df, how = _split_frame(df, split_col)
        notes.append(f"split by column {split_col!r} ({how}): "
                     f"{len(train_df)} train rows / {len(test_df)} test rows")
        if len(test_df) == 0:
            raise ValueError(
                f"split column {split_col!r} produced an empty test set; "
                f"provide --test-data/--test-targets instead")
    elif test_targets is not None:
        keys = [ln.strip() for ln in
                Path(test_targets).read_text(encoding="utf-8").splitlines()
                if ln.strip()]
        seq_set = set(df[SEQ_COLUMN].astype(str))
        id_col = next((c for c in ("target_id", "uniprot", "UniProt",
                                   "target", "Target_ID", "kinase")
                       if c in df.columns), None)
        if keys and all(k in seq_set for k in keys):
            mask = df[SEQ_COLUMN].astype(str).isin(keys)
            basis = "target sequences"
        elif id_col is not None:
            mask = df[id_col].astype(str).isin(keys)
            basis = f"target ids via column {id_col!r}"
        else:
            raise ValueError("--test-targets entries match neither the "
                             "sequence column nor any id-like column")
        train_df, test_df = df[~mask], df[mask]
        notes.append(f"split by --test-targets ({basis}): "
                     f"{len(test_df)} test rows")
    elif test_data is not None:
        train_df = df
        tp = Path(test_data)
        if _looks_fasta(tp):
            recs = read_fasta(tp)
            test_df = pd.DataFrame({SEQ_COLUMN: [s for _, s in recs]})
        else:
            tdf = _read_table(tp)
            tcol = _resolve_seq_column(tdf, seq_column)
            test_df = tdf.rename(columns={tcol: SEQ_COLUMN})[[SEQ_COLUMN]]
        notes.append(f"train from {data.name}, test from {tp.name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        train_csv = out_dir / TRAIN_FILE_FMT.format(name=name)
        test_csv = out_dir / TEST_FILE_FMT.format(name=name)
        train_df[[SEQ_COLUMN]].to_csv(train_csv, index=False)
        test_df[[SEQ_COLUMN]].to_csv(test_csv, index=False)
        return train_csv, test_csv
    else:
        raise ValueError(
            "single CSV without a split column: provide --test-data, "
            "--test-targets, or name a split column via --split-column")

    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv = out_dir / TRAIN_FILE_FMT.format(name=name)
    test_csv = out_dir / TEST_FILE_FMT.format(name=name)
    train_df[[SEQ_COLUMN]].to_csv(train_csv, index=False)
    test_df[[SEQ_COLUMN]].to_csv(test_csv, index=False)
    return train_csv, test_csv


def _from_affinity_mat(data: Path, targets_file: Optional[Path],
                       sequences_fasta: Optional[Path],
                       test_targets: Optional[Path],
                       affinity_key: str, out_dir: Path, name: str,
                       notes: List[str]) -> Tuple[Path, Path]:
    if targets_file is None or sequences_fasta is None:
        raise ValueError("affinity-matrix input requires --targets-file and "
                         "--sequences-fasta")
    from scipy.io import loadmat  # local import: only needed for .mat
    mat = loadmat(str(data))
    key = affinity_key if affinity_key in mat else next(
        (k for k in mat if not k.startswith("__")), None)
    if key is None:
        raise ValueError(f"no affinity array found in {data}")
    aff = np.asarray(mat[key], float)
    targets = [ln.strip() for ln in
               Path(targets_file).read_text(encoding="utf-8").splitlines()
               if ln.strip()]

    # orient the matrix so axis 0 == targets
    if aff.shape[0] == len(targets):
        axis_is_targets = True
    elif aff.shape[1] == len(targets):
        aff = aff.T
        axis_is_targets = True
    else:
        axis_is_targets = False
    if not axis_is_targets:
        raise ValueError(f"affinity matrix shape {aff.shape} matches neither "
                         f"axis with the {len(targets)} targets listed in "
                         f"{targets_file}")

    seqs: Dict[str, str] = {}
    for hdr, seq in read_fasta(sequences_fasta):
        # accept both bare names and 'name extra...' headers
        seqs[hdr.split()[0]] = seq
        seqs[hdr] = seq

    rows, missing = [], []
    for i, tgt in enumerate(targets):
        vec = aff[i]
        valid = vec[np.isfinite(vec)]
        if len(valid) == 0:
            continue
        seq = seqs.get(tgt) or seqs.get(tgt.split()[0]) or seqs.get(
            tgt.split("_")[0])
        if seq is None:
            missing.append(tgt)
            continue
        rows.append({"target_id": tgt, SEQ_COLUMN: seq,
                     "n_compounds": int(len(valid)),
                     "affinity_min": float(np.min(valid)),
                     "affinity_max": float(np.max(valid))})
    if missing:
        notes.append(f"{len(missing)} targets lack a sequence in "
                     f"{sequences_fasta.name} and were skipped")
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        raise ValueError("no targets with both measured affinities and a "
                         "known sequence")

    test_set = set()
    if test_targets is not None:
        test_set = {ln.strip() for ln in
                    Path(test_targets).read_text(encoding="utf-8").splitlines()
                    if ln.strip()}
        test_df = long_df[long_df["target_id"].isin(test_set)]
        train_df = long_df[~long_df["target_id"].isin(test_set)]
        notes.append(f"target-disjoint split from --test-targets: "
                     f"{len(test_df)} test targets / {len(train_df)} train "
                     f"targets")
    else:
        train_df, test_df = long_df, long_df.head(0)
        notes.append("no --test-targets given: all targets exported as "
                     "training rows; provide --test-data to complete the pair")

    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv = out_dir / TRAIN_FILE_FMT.format(name=name)
    test_csv = out_dir / TEST_FILE_FMT.format(name=name)
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    return train_csv, test_csv


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def prepare_inputs(data: Path, out_dir: Path, name: str,
                   seq_column: Optional[str] = None,
                   split_col: Optional[str] = None,
                   test_data: Optional[Path] = None,
                   test_targets: Optional[Path] = None,
                   targets_file: Optional[Path] = None,
                   sequences_fasta: Optional[Path] = None,
                   affinity_key: str = "affinity",
                   verbose: bool = True) -> Dict[str, object]:
    """Normalize any supported source into ``<name>_{train,test}.csv``.

    Returns ``{"train_csv", "test_csv", "format", "notes"}``.
    """
    data, out_dir = Path(data), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = detect_format(data)
    notes: List[str] = []

    if fmt == "split_dir":
        if test_data is not None:
            notes.append("--test-data ignored for a split directory")
        train_csv, test_csv, extra = _from_split_dir(data, name, seq_column,
                                                     out_dir)
        notes += extra
    elif fmt == "affinity_mat":
        train_csv, test_csv = _from_affinity_mat(
            data, targets_file, sequences_fasta, test_targets, affinity_key,
            out_dir, name, notes)
    elif fmt == "fasta" and test_data is not None and _looks_fasta(test_data):
        train_csv, test_csv = _from_fasta_pair(data, test_data, out_dir, name)
        notes.append("FASTA pair converted to CSV")
    elif fmt in ("single_csv", "fasta"):
        train_csv, test_csv = _from_single_csv(
            data, seq_column, split_col, test_data, test_targets, out_dir,
            name, notes)
    else:
        raise ValueError(f"cannot determine data layout of {data}")

    for p in (train_csv, test_csv):
        df = _read_table(p)
        if SEQ_COLUMN not in df.columns:
            raise ValueError(f"{p} lacks the {SEQ_COLUMN!r} column")
        n_unique = df[SEQ_COLUMN].dropna().astype(str).str.strip().replace(
            "", np.nan).dropna().nunique()
        if n_unique == 0:
            raise ValueError(f"{p} contains no non-empty sequences")

    lines = ["prepare_inputs report", "=" * 50,
             f"format={fmt}", f"train_csv={train_csv}",
             f"test_csv={test_csv}", ""] + [f"- {n}" for n in notes]
    (out_dir / PREPARE_REPORT).write_text("\n".join(lines), encoding="utf-8")
    if verbose:
        print("\n".join(lines))
    return {"train_csv": train_csv, "test_csv": test_csv, "format": fmt,
            "notes": notes}
