# -*- coding: utf-8 -*-
"""YAML configuration for the ``audit-benchmark`` pipeline.

A config file is optional — every key has a default.  CLI arguments take
precedence over config values, which take precedence over :data:`DEFAULTS`.

Minimal template::

    dataset:
      name: my_dataset
      data: data/my_dataset.csv   # any format supported by adapters
    audit:
      thresholds: [0.4, 0.6, 0.8]
    mirror:
      enabled: auto               # auto | true | false
    evaluation:
      predictions: null           # optional predictions.csv
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULTS: Dict[str, Any] = {
    "dataset": {
        "name": "dataset",
        "data": None,              # path; required (or --data on the CLI)
        "test_data": None,
        "test_targets": None,
        "targets_file": None,      # affinity-matrix inputs only
        "sequences_fasta": None,   # affinity-matrix inputs only
        "affinity_key": "affinity",
        "seq_column": None,        # auto-detect by default
        "split_column": None,      # auto-detect by default
    },
    "audit": {
        "thresholds": [0.4, 0.6, 0.8],
        "coverage": 0.8,
        "cov_mode": 0,
        "cluster_mode": 0,
        "threads": 8,
        "mmseqs": None,
    },
    "mirror": {
        "enabled": "auto",         # auto: run when required columns exist
        "complex_col": "complex_id",
        "split_col": "split",
        "mutation_col": "mutation",
        "ddg_col": "ddG",
        "aff_mut_col": None,
        "aff_wt_col": None,
        "temp_col": None,
        "default_temp": 298.15,
        "min_shared_mutations": 3,
        "min_sites_ks": 3,
        "thresholds": [0.5, 0.7],
        "target_id_col": None,     # join column for per_target_audit.csv
    },
    "evaluation": {
        "predictions": None,
        "identity_tag": 40,
        "metrics": ["R2", "EF@1%", "ECE"],
    },
    "degradation": {
        "warn_cost_pct": 5.0,
        "fail_cost_pct": 10.0,
    },
}

CONFIG_VERSION = 1


def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: Optional[Path] = None,
                overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load a YAML config and deep-merge it over :data:`DEFAULTS`.

    ``overrides`` (CLI-derived) wins last.  Returns a fresh dict each call.
    """
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PyYAML is required for --config (pip install pyyaml)") from exc
        user = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(user, dict):
            raise ValueError(f"config root must be a mapping, got "
                             f"{type(user).__name__}")
        cfg = _deep_merge(cfg, user)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    """Fail fast on obviously invalid configuration values."""
    audit = cfg["audit"]
    thr = audit["thresholds"]
    if not thr or not all(0.0 < float(t) < 1.0 for t in thr):
        raise ValueError(f"audit.thresholds must be fractions in (0,1), got {thr}")
    if float(audit["coverage"]) <= 0 or float(audit["coverage"]) > 1:
        raise ValueError(f"audit.coverage must be in (0,1], got {audit['coverage']}")
    if int(audit["cluster_mode"]) not in (0, 1, 2):
        raise ValueError("audit.cluster_mode must be 0/1/2 (MMseqs2 modes)")
    mirror = cfg["mirror"]
    if mirror["enabled"] not in ("auto", True, False, "true", "false"):
        raise ValueError("mirror.enabled must be auto/true/false")
    for name in ("min_shared_mutations", "min_sites_ks"):
        if int(mirror[name]) < 2:
            raise ValueError(f"mirror.{name} must be >= 2")
    data = cfg["dataset"]
    for key in ("data", "test_data", "test_targets", "targets_file",
                "sequences_fasta"):
        if data.get(key) is not None:
            data[key] = Path(data[key])
    if cfg["evaluation"].get("predictions") is not None:
        cfg["evaluation"]["predictions"] = Path(
            cfg["evaluation"]["predictions"])
    if audit.get("mmseqs") is not None:
        audit["mmseqs"] = Path(audit["mmseqs"])
