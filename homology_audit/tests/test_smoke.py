# -*- coding: utf-8 -*-
"""End-to-end smoke tests.

Pure-Python tests always run; MMseqs2-dependent tests are skipped when no
MMseqs2 binary is available (set HOMOLOGY_AUDIT_MMSEQS or pass the path via the
environment variable HOMOLOGY_AUDIT_TEST_MMSEQS).

Run from the toolkit root:

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from homology_audit.band_lodo import fit_band, in_band  # noqa: E402
from homology_audit.io_utils import build_id_maps, read_fasta  # noqa: E402
from homology_audit.metrics import pooled_metrics, ranks  # noqa: E402
from homology_audit.sensitivity import cohen_kappa  # noqa: E402

MMSEQS = os.environ.get("HOMOLOGY_AUDIT_TEST_MMSEQS") or os.environ.get(
    "HOMOLOGY_AUDIT_MMSEQS") or shutil.which("mmseqs") or shutil.which("mmseqs.exe")
HAS_MMSEQS = bool(MMSEQS)


class PurePythonTests(unittest.TestCase):
    def test_id_maps_deterministic(self):
        tr, te = build_id_maps(["ZZZ", "AAA"], ["MMM"])
        self.assertEqual(tr["AAA"], "train_0000")
        self.assertEqual(tr["ZZZ"], "train_0001")
        self.assertEqual(te["MMM"], "test_0000")

    def test_ranks_direction_and_ties(self):
        self.assertTrue((ranks([0.9, 0.1, 0.5], "higher") == [1, 3, 2]).all())
        self.assertTrue((ranks([0.1, 0.9, 0.5], "lower") == [1, 3, 2]).all())
        tied = ranks([0.9, 0.9, 0.1], "higher")
        self.assertEqual(tied.tolist(), [1, 1, 3])

    def test_metrics_basic(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        perfect = pooled_metrics(y, y)
        self.assertAlmostEqual(perfect["R2"], 1.0, places=9)
        self.assertAlmostEqual(perfect["ECE"], 0.0, places=9)
        bad = pooled_metrics(y, y[::-1])
        self.assertLess(bad["R2"], 0.0)

    def test_ef_skipped_on_tiny_panels(self):
        m = pooled_metrics(np.arange(4.0), np.arange(4.0))
        self.assertTrue(np.isnan(m["EF@1%"]))  # k = int(4*0.01) = 0

    def test_cohen_kappa(self):
        # agreement 3/4, chance agreement 0.5 -> kappa = (0.75-0.5)/0.5 = 0.5
        a = {"x": True, "y": True, "z": False, "w": False}
        k, agree, n = cohen_kappa(
            a, {"x": True, "y": True, "z": True, "w": False})
        self.assertEqual((agree, n), (3, 4))
        self.assertAlmostEqual(k, 0.5, places=6)
        self.assertEqual(cohen_kappa(a, a)[0], 1.0)

    def test_band_rule(self):
        # positive point 0.13 with negatives 0.12 and 0.25
        band = fit_band([(0.12, 0), (0.13, 1), (0.25, 0)])
        self.assertAlmostEqual(band[0], 0.125)
        self.assertAlmostEqual(band[1], 0.19)
        self.assertEqual(in_band(0.13, band), 1)
        self.assertEqual(in_band(0.30, band), 0)
        self.assertEqual(fit_band([(0.1, 0), (0.2, 0)]), ("undef", "undef"))


@unittest.skipUnless(HAS_MMSEQS, "MMseqs2 binary not available")
class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="homology_audit_test_"))
        cls.env = dict(os.environ, HOMOLOGY_AUDIT_MMSEQS=MMSEQS)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _run(self, *args):
        proc = subprocess.run(
            [sys.executable, "-m", "homology_audit.cli", *args],
            cwd=str(ROOT), env=self.env, capture_output=True, text=True,
            encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            self.fail(f"command failed: {args}\nSTDOUT:{proc.stdout[-3000:]}\n"
                      f"STDERR:{proc.stderr[-3000:]}")
        return proc.stdout

    def test_01_full_pipeline(self):
        audit = self.tmp / "audit"
        out = self._run(
            "audit-batch", "--data-root", str(ROOT / "examples" / "data"),
            "--datasets", "demo,demo2", "--out", str(audit),
            "--thresholds", "0.4,0.6,0.8")
        self.assertIn("cross_cluster_ratio_pct", out)
        summary = pd.read_csv(audit / "audit_summary.csv")
        demo40 = summary.query("dataset == 'demo' and identity_threshold_pct == 40")
        self.assertEqual(int(demo40["n_test_targets_cross_cluster"].iloc[0]), 2)
        labels = pd.read_csv(audit / "demo" / "per_target_audit.csv")
        self.assertIn("target_sequence", labels.columns)
        # FASTA round-trip
        self.assertEqual(len(read_fasta(audit / "demo" / "sequences_all.fasta")), 7)

        self._run("router", "--labels-root", str(audit),
                  "--datasets", "demo,demo2", "--out", str(self.tmp / "router"),
                  "--grid", "25,60,2.5")
        self.assertTrue((self.tmp / "router" / "router_lodo.csv").exists())

        self._run("sensitivity", "--dataset-dir", str(audit / "demo"),
                  "--out", str(self.tmp / "sens"), "--coverages", "0.8",
                  "--identities", "0.4,0.6", "--reference",
                  str(audit / "audit_summary.csv"), "--name", "demo")
        check = pd.read_csv(self.tmp / "sens" / "reproduction_check.csv")
        self.assertTrue(check["match"].all())

        self._run("algorithm-check", "--dataset-dir", str(audit / "demo"),
                  "--out", str(self.tmp / "algo"), "--identities", "0.4",
                  "--name", "demo")
        comp = pd.read_csv(self.tmp / "algo" / "cluster_algorithm_comparison.csv")
        self.assertGreaterEqual(comp["cohen_kappa"].iloc[0], 0.0)
        self.assertLessEqual(comp["cohen_kappa"].iloc[0], 1.0)

        self._run("negative-control", "--dataset-dir", str(audit / "demo"),
                  "--out", str(self.tmp / "neg"), "--seeds", "0,1",
                  "--thresholds", "0.4,0.8")
        neg = pd.read_csv(self.tmp / "neg" / "negative_control_results.csv")
        self.assertEqual(int((neg["cross_cluster_ratio_pct"] == 100).sum()), 4)

        self._run("evaluate",
                  "--predictions", str(ROOT / "examples" / "predictions" /
                                       "demo_predictions.csv"),
                  "--labels-root", str(audit), "--datasets", "demo",
                  "--out", str(self.tmp / "eval"))
        ranking = pd.read_csv(self.tmp / "eval" / "rank_comparison.csv")
        r2 = ranking[ranking.metric == "R2"].iloc[0]
        self.assertTrue(bool(r2["winner_reversed"]))
        self.assertGreater(r2["misselection_cost"], 0.0)

    def test_02_band_lodo_builtin(self):
        out = self.tmp / "band"
        self._run("band-lodo", "--out", str(out))
        import json
        summary = json.loads((out / "band_lodo_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["accuracy_band_rule"], 0.5)
        self.assertEqual(summary["majority_baseline"], 0.75)
        self.assertFalse(summary["band_supports_deployment"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
