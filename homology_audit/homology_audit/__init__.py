# -*- coding: utf-8 -*-
"""homology_audit — sequence-homology audit toolkit for target-disjoint DTA
benchmarking.

Public API
----------
- :func:`homology_audit.audit.run_audit`
- :func:`homology_audit.router.run_router_validation`
- :func:`homology_audit.sensitivity.run_grid`
- :func:`homology_audit.sensitivity.run_algorithm_check`
- :func:`homology_audit.negative.run_negative_control`
- :func:`homology_audit.evaluate.run_evaluation`
- :func:`homology_audit.band_lodo.run_band_lodo`
- :func:`homology_audit.mirror.run_mirror` — binding-spectrum redundancy
- :func:`homology_audit.degradation.run_degradation` — severity verdicts
- :func:`homology_audit.adapters.prepare_inputs` — benchmark format adapters
- :func:`homology_audit.pipeline.run_audit_benchmark` — one-command pipeline

Command-line entry points:

- ``homology-audit`` (subcommands; see ``homology-audit --help``)
- ``audit-benchmark`` (one-command pipeline; see ``audit-benchmark --help``)
"""
from __future__ import annotations

__version__ = "1.2.0"

__all__ = ["__version__"]
