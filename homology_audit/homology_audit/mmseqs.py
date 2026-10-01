# -*- coding: utf-8 -*-
"""MMseqs2 binary discovery and subprocess wrappers.

Binary resolution order
-----------------------
1. explicit ``mmseqs_path`` argument / ``--mmseqs`` CLI option
2. ``HOMOLOGY_AUDIT_MMSEQS`` environment variable (full path to the binary)
3. ``mmseqs`` / ``mmseqs.exe`` on ``PATH``

Windows note
------------
The native Windows build of MMseqs2 ships Cygwin helper DLLs / ``busybox`` next
to ``mmseqs.exe``; the directory containing the binary must be on ``PATH`` for
the ``easy-cluster`` / ``easy-search`` workflows to run.  This module takes
care of that automatically.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

INSTALL_HINT = (
    "MMseqs2 binary not found. Install it with one of:\n"
    "  conda install -c conda-forge mmseqs2\n"
    "  brew install mmseqs2        (macOS)\\n"
    "or download a static binary from https://github.com/soedinglab/MMseqs2/releases\n"
    "Then either put it on PATH, set HOMOLOGY_AUDIT_MMSEQS to its full path, "
    "or pass --mmseqs."
)


def resolve_mmseqs(mmseqs_path: Optional[Path] = None) -> Path:
    """Return a usable MMseqs2 executable path or raise a clear error."""
    candidates: List[Path] = []
    if mmseqs_path:
        candidates.append(Path(mmseqs_path))
    env_bin = os.environ.get("HOMOLOGY_AUDIT_MMSEQS")
    if env_bin:
        candidates.append(Path(env_bin))
    for name in ("mmseqs", "mmseqs.exe"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(INSTALL_HINT)


def build_env(mmseqs_bin: Path) -> Dict[str, str]:
    """Process environment with the MMseqs2 directory prepended to PATH."""
    env = os.environ.copy()
    bindir = str(Path(mmseqs_bin).parent)
    env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
    return env


def run_mmseqs(mmseqs_bin: Path, args: Sequence[str], cwd: Path,
               env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    """Run an MMseqs2 command, raising ``RuntimeError`` with the log on failure."""
    proc = subprocess.run(
        [str(mmseqs_bin), *[str(a) for a in args]],
        cwd=str(Path(cwd).resolve()),
        env=env or build_env(mmseqs_bin),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        tail = lambda s: (s or "")[-2000:]  # noqa: E731
        raise RuntimeError(
            "MMseqs2 command failed:\n  "
            + " ".join(str(a) for a in args)
            + f"\n--- stdout ---\n{tail(proc.stdout)}\n--- stderr ---\n{tail(proc.stderr)}"
        )
    return proc


def easy_cluster(mmseqs_bin: Path, fasta: Path, prefix: Path, tmp: Path,
                 min_seq_id: float, coverage: float, cov_mode: int = 0,
                 cluster_mode: int = 0, threads: int = 8) -> Path:
    """Run ``mmseqs easy-cluster``; return the ``*_cluster.tsv`` path.

    Skips execution when the TSV already exists (resumable).
    """
    tsv = Path(str(prefix) + "_cluster.tsv")
    if not tsv.exists():
        run_mmseqs(
            mmseqs_bin,
            ["easy-cluster", fasta, prefix, tmp,
             "--min-seq-id", str(min_seq_id), "-c", str(coverage),
             "--cov-mode", str(cov_mode), "--cluster-mode", str(cluster_mode),
             "--threads", str(threads)],
            cwd=prefix.parent,
        )
    shutil.rmtree(tmp, ignore_errors=True)
    return tsv


def easy_search(mmseqs_bin: Path, query_fasta: Path, target_fasta: Path,
                out_m8: Path, tmp: Path, min_seq_id: float = 0.0,
                coverage: float = 0.8, cov_mode: int = 0,
                threads: int = 8) -> Path:
    """Run ``mmseqs easy-search`` (m8 output); return the m8 path."""
    if out_m8.exists():
        out_m8.unlink()
    run_mmseqs(
        mmseqs_bin,
        ["easy-search", query_fasta, target_fasta, out_m8, tmp,
         "--min-seq-id", str(min_seq_id), "-c", str(coverage),
         "--cov-mode", str(cov_mode), "--threads", str(threads)],
        cwd=out_m8.parent,
    )
    shutil.rmtree(tmp, ignore_errors=True)
    return out_m8
