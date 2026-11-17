# -*- coding: utf-8 -*-
"""按 morgan_descriptors 正确屏蔽 MACCS，重生成 ChEMBL 5-seed 的 raw 指标。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\lht")
sys.path.insert(0, str(ROOT))
import run_chembl_reptile_morgan_5seeds as runner
import recompute_si_tables as rc

rows = []
for seed in runner.SEEDS:
    out = runner.OUT_ROOT / f"seed_{seed}"
    rc.make_raw_chembl_predictions(out, mask_maccs=True, force=True)
    r = rc.row_from_run(out, "chembl")
    r["seed"] = seed
    rows.append(r)
    print(f"seed {seed}: R2={r['R2']:.4f} RMSE={r['RMSE']:.4f} EF={r['EF']:.3f} "
          f"AUPR={r['AUPR']:.4f} ECE={r['ECE']:.4f}")

for k in ["R2", "RMSE", "EF", "AUPR", "ECE"]:
    vals = [r[k] for r in rows]
    print(f"{k}: {np.mean(vals):.6f} ± {np.std(vals, ddof=1):.6f}")

with open(runner.OUT_ROOT / "summary.csv", "w", encoding="utf-8-sig") as f:
    f.write("seed,R2,RMSE,EF@1%,AUPR,ECE\n")
    for r in rows:
        f.write(f"{r['seed']},{r['R2']:.6f},{r['RMSE']:.6f},{r['EF']:.6f},"
                f"{r['AUPR']:.6f},{r['ECE']:.6f}\n")
print("[saved]", runner.OUT_ROOT / "summary.csv")
