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


class V120MirrorTests(unittest.TestCase):
    """Binding-spectrum mirror audit (pure python)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="ha_mirror_"))
        rng = np.random.default_rng(7)
        mut_names = [f"M{i:02d}" for i in range(12)]
        train_prof = {m: float(1.2 + rng.normal(0, 0.35)) for m in mut_names}
        other_prof = {m: float(2.0 + rng.normal(0, 0.35)) for m in mut_names}
        rows = []
        for cx, prof in (("TRAIN_1", train_prof), ("TRAIN_2", other_prof)):
            rows += [{"complex_id": cx, "split": "train", "mutation": m,
                      "ddG": v} for m, v in prof.items()]
        # mirror pair: same mutations, nearly same ddG ordering
        for m in mut_names:
            rows.append({"complex_id": "T_MIRROR", "split": "test",
                         "mutation": m,
                         "ddG": train_prof[m] + rng.normal(0, 0.1)})
        # distribution-only mirror: no shared mutation names, similar spread
        for i in range(12):
            rows.append({"complex_id": "T_KS", "split": "test",
                         "mutation": f"N{i:02d}",
                         "ddG": other_prof[f"M{i:02d}"] + rng.normal(0, 0.2)})
        # independent: different scale, different names
        for i in range(10):
            rows.append({"complex_id": "T_IND", "split": "test",
                         "mutation": f"Z{i:02d}",
                         "ddG": float(-1.5 + rng.normal(0, 1.2))})
        cls.data = cls.tmp / "mut.csv"
        pd.DataFrame(rows).to_csv(cls.data, index=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_mirror_scores(self):
        from homology_audit.mirror import run_mirror
        res = run_mirror(self.data, self.tmp / "out", "demo", verbose=False)
        pc = res["per_complex"].set_index("test_complex")
        self.assertGreater(pc.loc["T_MIRROR", "max_spearman_rho"], 0.8)
        # 12-point samples make the KS statistic coarse (1/12 resolution)
        self.assertGreater(pc.loc["T_MIRROR", "max_ks_similarity"], 0.7)
        # no shared mutations -> Spearman not comparable, NaN
        self.assertTrue(np.isnan(pc.loc["T_KS", "max_spearman_rho"]))
        self.assertGreater(pc.loc["T_KS", "max_ks_similarity"], 0.7)
        self.assertTrue(pd.isna(pc.loc["T_KS", "best_train_spearman"]))
        self.assertLess(pc.loc["T_IND", "max_ks_similarity"], 0.7)
        s07 = res["summary"].query("threshold == 0.7").iloc[0]
        self.assertEqual(int(s07["n_ks_redundant"]), 2)
        self.assertTrue((self.tmp / "out" / "mirror_report.txt").exists())

    def test_mirror_ddg_from_affinity(self):
        from homology_audit.mirror import compute_ddg, GAS_CONSTANT
        df = pd.DataFrame({"aff_mut": [10.0, 100.0, 1.0],
                           "aff_wt": [100.0, 100.0, 10.0]})
        ddg = compute_ddg(df, None, "aff_mut", "aff_wt", None, 298.15)
        expected = GAS_CONSTANT * 298.15 * np.log(
            df["aff_wt"] / df["aff_mut"])
        np.testing.assert_allclose(ddg.to_numpy(), expected.to_numpy())


class V120DegradationTests(unittest.TestCase):
    def _frames(self):
        per_model = pd.DataFrame([
            # FAIL: winner reversed with 20% relative cost (R2, higher=better)
            {"dataset": "d", "metric": "R2", "model": "A",
             "standard_value": 0.80, "audited_value": 0.50,
             "standard_rank": 1, "audited_rank": 2},
            {"dataset": "d", "metric": "R2", "model": "B",
             "standard_value": 0.60, "audited_value": 0.60,
             "standard_rank": 2, "audited_rank": 1},
            # WARN: winner not reversed but cost 7% (RMSE, lower=better)
            {"dataset": "d", "metric": "RMSE", "model": "A",
             "standard_value": 1.00, "audited_value": 1.20,
             "standard_rank": 1, "audited_rank": 1},
            {"dataset": "d", "metric": "RMSE", "model": "B",
             "standard_value": 1.50, "audited_value": 1.30,
             "standard_rank": 2, "audited_rank": 2},
        ])
        ranking = pd.DataFrame([
            {"dataset": "d", "metric": "R2", "n_models": 2,
             "standard_winner": "A", "audited_winner": "B",
             "winner_reversed": True, "misselection_cost": 0.10,
             "relative_cost_pct": 20.0},
            {"dataset": "d", "metric": "RMSE", "n_models": 2,
             "standard_winner": "A", "audited_winner": "A",
             "winner_reversed": False, "misselection_cost": 0.20,
             "relative_cost_pct": 7.0},
        ])
        return per_model, ranking

    def test_severity_and_flips(self):
        from homology_audit.degradation import detect_degradation
        per_model, ranking = self._frames()
        rep = detect_degradation(per_model, ranking)
        by_metric = rep.set_index("metric")
        self.assertEqual(by_metric.loc["R2", "severity"], "FAIL")
        self.assertEqual(by_metric.loc["R2", "rank_flips"], 1)
        self.assertEqual(by_metric.loc["R2", "n_degraded_models"], 1)
        self.assertEqual(by_metric.loc["RMSE", "severity"], "WARN")
        # lower-is-better direction: A degraded, B improved
        self.assertEqual(by_metric.loc["RMSE", "n_degraded_models"], 1)

    def test_count_rank_flips(self):
        from homology_audit.degradation import _count_rank_flips
        # only the (2,3) pair flips order
        self.assertEqual(_count_rank_flips([1, 2, 3], [1, 3, 2]), 1)
        self.assertEqual(_count_rank_flips([1, 2], [1, 2]), 0)
        self.assertEqual(_count_rank_flips([3, 2, 1], [1, 2, 3]), 3)


class V120AdapterConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="ha_adapters_"))
        rng = np.random.default_rng(0)
        AA = "ACDEFGHIKLMNPQRSTVWY"

        def seq(n=80):
            return "".join(rng.choice(list(AA), n))

        cls.s_tr = [seq() for _ in range(4)]
        cls.s_te = [seq() for _ in range(3)]
        cls.rng = rng

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_long_csv_split_column(self):
        from homology_audit.adapters import prepare_inputs
        df = pd.DataFrame({
            "drug": [f"D{i%5}" for i in range(14)],
            "target_sequence": [self.s_tr[i % 4] for i in range(7)]
                               + [self.s_te[i % 3] for i in range(7)],
            "affinity": self.rng.normal(6, 1, 14).round(3),
            "split": ["train"] * 7 + ["test"] * 7,
        })
        p = self.tmp / "long.csv"
        df.to_csv(p, index=False)
        r = prepare_inputs(p, self.tmp / "o1", "ds", verbose=False)
        n_tr = pd.read_csv(r["train_csv"]).target_sequence.nunique()
        n_te = pd.read_csv(r["test_csv"]).target_sequence.nunique()
        self.assertEqual((n_tr, n_te), (4, 3))

    def test_bindingdb_column_autodetect(self):
        from homology_audit.adapters import prepare_inputs
        df = pd.DataFrame({
            "BindingDB_Target_Chain_Sequence": self.s_tr + self.s_te,
            "set": ["train"] * 4 + ["test"] * 3,
        })
        p = self.tmp / "bdb.csv"
        df.to_csv(p, index=False)
        r = prepare_inputs(p, self.tmp / "o2", "bdb", verbose=False)
        self.assertEqual(pd.read_csv(r["test_csv"]).shape[0], 3)

    def test_single_csv_without_test_info_raises(self):
        from homology_audit.adapters import prepare_inputs
        p = self.tmp / "all.csv"
        pd.DataFrame({"target_sequence": self.s_tr + self.s_te}).to_csv(
            p, index=False)
        with self.assertRaises(ValueError):
            prepare_inputs(p, self.tmp / "o3", "bad", verbose=False)

    def test_fasta_pair(self):
        from homology_audit.adapters import prepare_inputs

        def wf(path, seqs):
            with open(path, "w", encoding="utf-8") as f:
                for i, s in enumerate(seqs):
                    f.write(f">s{i}\n{s}\n")
        wf(self.tmp / "tr.fa", self.s_tr)
        wf(self.tmp / "te.fa", self.s_te)
        r = prepare_inputs(self.tmp / "tr.fa", self.tmp / "o4", "fa",
                           test_data=self.tmp / "te.fa", verbose=False)
        self.assertEqual(pd.read_csv(r["test_csv"]).shape[0], 3)

    def test_affinity_mat_adapter(self):
        from scipy.io import savemat
        from homology_audit.adapters import prepare_inputs
        targets = [f"KIN{i}" for i in range(5)]
        (self.tmp / "targets.txt").write_text("\n".join(targets),
                                              encoding="utf-8")
        with open(self.tmp / "kinases.fa", "w", encoding="utf-8") as f:
            for t in targets:
                f.write(f">{t}\n{self.rng.choice(list('AGCT'), 20)}\n")
        aff = self.rng.normal(6, 1, (5, 8)).round(3)
        aff[4, :] = np.nan                      # one target w/o data
        savemat(str(self.tmp / "aff.mat"), {"affinity": aff})
        (self.tmp / "test_kin.txt").write_text("KIN1\nKIN3\n",
                                               encoding="utf-8")
        r = prepare_inputs(self.tmp / "aff.mat", self.tmp / "o5", "davislike",
                           targets_file=self.tmp / "targets.txt",
                           sequences_fasta=self.tmp / "kinases.fa",
                           test_targets=self.tmp / "test_kin.txt",
                           verbose=False)
        te = pd.read_csv(r["test_csv"])
        tr = pd.read_csv(r["train_csv"])
        self.assertEqual(set(te.target_id), {"KIN1", "KIN3"})
        self.assertEqual(sorted(tr.target_id), ["KIN0", "KIN2"])
        # transposed orientation gives the same answer
        savemat(str(self.tmp / "affT.mat"), {"affinity": aff.T})
        r2 = prepare_inputs(self.tmp / "affT.mat", self.tmp / "o6", "davisT",
                            targets_file=self.tmp / "targets.txt",
                            sequences_fasta=self.tmp / "kinases.fa",
                            test_targets=self.tmp / "test_kin.txt",
                            verbose=False)
        self.assertEqual(set(pd.read_csv(r2["test_csv"]).target_id),
                         {"KIN1", "KIN3"})

    def test_config_merge_and_validate(self):
        from homology_audit.config import load_config, validate_config
        cfgp = self.tmp / "cfg.yaml"
        cfgp.write_text(
            "dataset:\n  name: cfgds\naudit:\n  thresholds: [0.4, 0.8]\n"
            "  coverage: 0.9\nmirror:\n  enabled: false\n",
            encoding="utf-8")
        cfg = load_config(cfgp, overrides={"audit": {"threads": 4}})
        validate_config(cfg)
        self.assertEqual(cfg["dataset"]["name"], "cfgds")
        self.assertEqual(cfg["audit"]["threads"], 4)      # override wins
        self.assertEqual(cfg["audit"]["coverage"], 0.9)   # config wins
        self.assertEqual(cfg["audit"]["thresholds"], [0.4, 0.8])
        self.assertEqual(cfg["mirror"]["complex_col"], "complex_id")
        bad = load_config(None)
        bad["audit"]["thresholds"] = [1.5]
        with self.assertRaises(ValueError):
            validate_config(bad)


@unittest.skipUnless(HAS_MMSEQS, "MMseqs2 binary not available")
class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="homology_audit_test_"))
        cls.env = dict(os.environ, HOMOLOGY_AUDIT_MMSEQS=MMSEQS)
        # regenerate the synthetic data so tests never depend on stale files
        subprocess.run([sys.executable, str(ROOT / "examples" /
                                                "make_example_data.py")],
                       cwd=str(ROOT), env=cls.env, check=True,
                       capture_output=True, text=True)

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

    def test_03_mirror_degradation_pipeline(self):
        audit = self.tmp / "audit"
        self._run("audit-batch", "--data-root", str(ROOT / "examples" / "data"),
                  "--datasets", "demo", "--out", str(audit),
                  "--thresholds", "0.4,0.6,0.8")

        # mirror audit with cross-cluster intersection
        self._run("mirror", "--data",
                  str(ROOT / "examples" / "data" / "demo_mutation_ddg.csv"),
                  "--out", str(self.tmp / "mirror"), "--name", "demo",
                  "--per-target-audit",
                  str(audit / "demo" / "per_target_audit.csv"),
                  "--target-id-col", "target_id")
        pc = pd.read_csv(self.tmp / "mirror" / "per_complex_mirror.csv")
        self.assertEqual(len(pc), 4)
        t1 = pc[pc.test_complex == "TEST_T1"].iloc[0]
        self.assertGreater(t1["max_spearman_rho"], 0.8)
        self.assertGreater(t1["max_ks_similarity"], 0.8)
        summary = pd.read_csv(self.tmp / "mirror" / "mirror_summary.csv")
        s07 = summary[summary.threshold == 0.7].iloc[0]
        self.assertEqual(int(s07["n_ks_redundant"]), 2)

        # degradation from evaluation outputs
        self._run("evaluate", "--predictions", str(ROOT / "examples" /
                                                   "predictions" /
                                                   "demo_predictions.csv"),
                  "--labels-root", str(audit), "--datasets", "demo",
                  "--out", str(self.tmp / "eval"))
        self._run("degradation", "--evaluation-dir", str(self.tmp / "eval"),
                  "--out", str(self.tmp / "degradation"))
        rep = pd.read_csv(self.tmp / "degradation" / "degradation_report.csv")
        self.assertIn("severity", rep.columns)
        self.assertIn(rep["severity"].isin(["FAIL", "WARN"]).any(), (True,))
        self.assertIn("Overall verdict: FAIL",
                      (self.tmp / "degradation" /
                       "degradation_report.txt").read_text(encoding="utf-8"))

        # one-command pipeline via the second console entry point
        out = self.tmp / "pipeline"
        proc = subprocess.run(
            [sys.executable, "-m", "homology_audit.pipeline",
             "--data", str(ROOT / "examples" / "data"),
             "--output", str(out), "--name", "demo", "--predictions",
             str(ROOT / "examples" / "predictions" / "demo_predictions.csv")],
            cwd=str(ROOT), env=self.env, capture_output=True, text=True,
            encoding="utf-8", errors="replace")
        self.assertEqual(proc.returncode, 0,
                         f"pipeline failed:\n{proc.stderr[-3000:]}")
        text = (out / "audit-benchmark_summary.txt").read_text(encoding="utf-8")
        self.assertIn("stages run: prepare -> audit -> evaluate -> degradation",
                      text)
        self.assertTrue((out / "audit" / "demo" / "per_target_audit.csv").exists())
        self.assertTrue((out / "degradation" /
                         "degradation_report.csv").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
