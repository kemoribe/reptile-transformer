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

The command-line entry point is ``homology-audit`` (see ``homology-audit --help``).
"""
from __future__ import annotations

__version__ = "1.1.0"

__all__ = ["__version__"]
