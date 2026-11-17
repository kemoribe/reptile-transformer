# -*- coding: utf-8 -*-
"""
第三轴 × 性能：配体骨架覆盖是否与逐靶点性能相关（5-seed 汇总）
================================================================
预测配置：
- full Reptile-Transformer，morgan_descriptors，bidirectional
- ranking=0.3, contrastive=0.05, consistency=0.1
- 220 epoch, batch 2048, inner steps 3
- seeds 11/22/33/44/55

汇总方式：
1. 每个 seed 单独计算逐靶点 R²、EF@1%、EF@5%；
2. 按靶点对 5 个 seed 取平均；
3. 用 seed 平均后的逐靶点指标做 Spearman / 偏 Spearman / bootstrap；
4. 同时输出每个 seed 的 Spearman，报告 per-seed 范围。

输入：
- ligand_scaffold_axis/per_target.csv
- 已有 5-seed Reptile 预测（KIBA / ChEMBL / Davis / BindingDB）

输出：
- per_target_metrics.csv       seed 平均后的逐靶点指标
- per_seed_metrics.csv         每个 seed 的逐靶点指标
- per_seed_correlations.csv    每个 seed 的 Spearman
- correlations.csv             seed 平均后的 Spearman / 偏 Spearman / bootstrap CI
- coverage_groups.csv          coverage>=0.5 vs <0.5 的 Mann-Whitney
- REPORT.md

注意：
- KIBA / Davis 的 coverage 恒为 1，无法算相关，标为 no variance。
- 不需要 GPU。

用法：
    python scaffold_vs_performance.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, rankdata, pearsonr

ROOT = Path(r"D:\lht")
SCAF = ROOT / "ligand_scaffold_axis" / "per_target.csv"
OUT = ROOT / "scaffold_performance"
SEEDS = [11, 22, 33, 44, 55]
B = 2000

CASES = {
    "kiba": dict(
        feat=ROOT / "transformer_ablation_output_kiba" / "precomputed_features.npz",
        base=ROOT / "修改supplement_output" / "revision2_experiments" / "e5_repeats" / "kiba",
        pred_name="predictions.npz",
    ),
    "chembl": dict(
        feat=ROOT / "transformer_ablation_output_chembl" / "precomputed_features.npz",
        base=ROOT / "chembl_morgan_5seeds_b2048",
        pred_name="predictions_raw.npz",
    ),
    "davis": dict(
        feat=ROOT / "davis_panel_morgan" / "_features" / "precomputed_features.npz",
        base=ROOT / "davis_panel_morgan" / "reptile",
        pred_name="predictions_raw.npz",
    ),
    "bindingdb": dict(
        feat=ROOT / "bindingdb_panel_morgan" / "_features" / "precomputed_features.npz",
        base=ROOT / "bindingdb_panel_morgan" / "reptile",
        pred_name="predictions_raw.npz",
    ),
}

import sys
sys.path.insert(0, str(ROOT))
from recompute_s18_s19 import ef_at  # noqa


def per_target_metrics(dataset, cfg, seed):
    f = np.load(cfg["feat"], mmap_mode="r", allow_pickle=True)
    sp = f["splits"][:]
    tn = f["target_names"][:][sp == "test"]
    yt_all = f["y"][:][sp == "test"]
    pred_path = cfg["base"] / f"seed_{seed}" / cfg["pred_name"]
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    yp_all = np.load(pred_path, allow_pickle=True)["y_pred"]
    if len(tn) != len(yp_all):
        raise RuntimeError(f"{dataset} seed {seed}: len mismatch {len(tn)} vs {len(yp_all)}")
    rows = []
    for target in np.unique(tn):
        m = tn == target
        yt, yp = yt_all[m], yp_all[m]
        n = len(yt)
        if n < 2:
            continue
        sst = float(np.sum((yt - yt.mean()) ** 2))
        r2 = 1 - float(np.sum((yt - yp) ** 2)) / sst if sst > 0 else np.nan
        ef1 = ef_at(yt, yp, 1) if n >= 100 else np.nan
        ef5 = ef_at(yt, yp, 5) if n >= 20 else np.nan
        rows.append(dict(dataset=dataset, target=target, seed=seed, n=n,
                         R2=r2, EF1=ef1, EF5=ef5))
    return pd.DataFrame(rows)


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)

    def resid(a, b):
        A = np.column_stack([np.ones_like(b), b])
        beta = np.linalg.lstsq(A, a, rcond=None)[0]
        return a - A @ beta

    return float(pearsonr(resid(rx, rz), resid(ry, rz))[0])


def boot_rho(x, y, seed=0, b=B):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    vals = []
    for _ in range(b):
        j = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(x[j])) < 2 or len(np.unique(y[j])) < 2:
            continue
        vals.append(spearmanr(x[j], y[j]).correlation)
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scaf = pd.read_csv(SCAF)

    all_seed = []
    for name, cfg in CASES.items():
        for seed in SEEDS:
            all_seed.append(per_target_metrics(name, cfg, seed))
    raw = pd.concat(all_seed, ignore_index=True)
    raw = raw.merge(scaf, on=["dataset", "target"], how="inner")
    raw.to_csv(OUT / "per_seed_metrics.csv", index=False, encoding="utf-8-sig")

    # 每个 seed 的 Spearman
    seed_rows = []
    for dataset, g in raw.groupby("dataset"):
        for cov in ["exact_cov", "generic_cov"]:
            for metric in ["R2", "EF1", "EF5"]:
                for seed, gs in g.groupby("seed"):
                    sub = gs[[cov, metric, "n"]].dropna()
                    if (len(sub) < 3 or sub[cov].nunique() < 2
                            or sub[metric].nunique() < 2):
                        rho = np.nan
                    else:
                        rho = float(spearmanr(sub[cov], sub[metric]).correlation)
                    seed_rows.append(dict(dataset=dataset, cov=cov, metric=metric,
                                          seed=seed, n_targets=len(sub), rho=rho))
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(OUT / "per_seed_correlations.csv", index=False, encoding="utf-8-sig")

    # seed 平均后的逐靶点指标
    avg = (raw.groupby(["dataset", "target"])
              .agg(n=("n", "first"),
                   R2=("R2", "mean"), EF1=("EF1", "mean"), EF5=("EF5", "mean"),
                   exact_cov=("exact_cov", "first"),
                   generic_cov=("generic_cov", "first"),
                   cluster5_cov=("cluster5_cov", "first"),
                   n_seeds=("seed", "nunique"))
              .reset_index())
    avg.to_csv(OUT / "per_target_metrics.csv", index=False, encoding="utf-8-sig")

    # seed 平均后的相关
    corr_rows, group_rows = [], []
    for dataset, g in avg.groupby("dataset"):
        for cov in ["exact_cov", "generic_cov"]:
            for metric in ["R2", "EF1", "EF5"]:
                sub = g[[cov, metric, "n"]].dropna()
                if len(sub) < 3:
                    corr_rows.append(dict(dataset=dataset, cov=cov, metric=metric,
                                          n_targets=len(sub), rho=np.nan, ci_low=np.nan,
                                          ci_high=np.nan, partial_rho=np.nan,
                                          n_seeds=0, rho_seed_min=np.nan,
                                          rho_seed_max=np.nan, note="n<3"))
                    continue
                if sub[cov].nunique() < 2 or sub[metric].nunique() < 2:
                    corr_rows.append(dict(dataset=dataset, cov=cov, metric=metric,
                                          n_targets=len(sub), rho=np.nan, ci_low=np.nan,
                                          ci_high=np.nan, partial_rho=np.nan,
                                          n_seeds=0, rho_seed_min=np.nan,
                                          rho_seed_max=np.nan, note="no variance"))
                    continue
                rho = float(spearmanr(sub[cov], sub[metric]).correlation)
                lo, hi = boot_rho(sub[cov].values, sub[metric].values)
                prho = partial_spearman(sub[cov].values, sub[metric].values,
                                        np.log10(sub["n"].values))
                ss = seed_df[(seed_df["dataset"] == dataset) & (seed_df["cov"] == cov)
                             & (seed_df["metric"] == metric)]["rho"].dropna()
                corr_rows.append(dict(dataset=dataset, cov=cov, metric=metric,
                                      n_targets=len(sub), rho=rho, ci_low=lo,
                                      ci_high=hi, partial_rho=prho,
                                      n_seeds=int(len(ss)),
                                      rho_seed_min=float(ss.min()) if len(ss) else np.nan,
                                      rho_seed_max=float(ss.max()) if len(ss) else np.nan,
                                      note=""))

        if g["exact_cov"].nunique() >= 2:
            hi = g[g["exact_cov"] >= 0.5]
            lo = g[g["exact_cov"] < 0.5]
            for metric in ["R2", "EF1", "EF5"]:
                a = hi[metric].dropna()
                b = lo[metric].dropna()
                p = (float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
                     if len(a) >= 3 and len(b) >= 3 else np.nan)
                group_rows.append(dict(
                    dataset=dataset, metric=metric,
                    n_high=len(a), n_low=len(b),
                    high_mean=float(a.mean()) if len(a) else np.nan,
                    low_mean=float(b.mean()) if len(b) else np.nan,
                    high_median=float(a.median()) if len(a) else np.nan,
                    low_median=float(b.median()) if len(b) else np.nan,
                    p_mannwhitney=p, n_seeds=int(g["n_seeds"].max())))

    corr = pd.DataFrame(corr_rows)
    groups = pd.DataFrame(group_rows)
    corr.to_csv(OUT / "correlations.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(OUT / "coverage_groups.csv", index=False, encoding="utf-8-sig")

    lines = ["# 第三轴 × 性能（5-seed 汇总）", ""]
    lines.append("预测：full Reptile-Transformer，morgan_descriptors，bidirectional；"
                 "ranking=0.3, contrastive=0.05, consistency=0.1；220 epoch, batch 2048, inner steps 3。")
    lines.append("汇总：每个 seed 单独算逐靶点 R²/EF，再按靶点对 5 个 seed 取平均；"
                 "Spearman/偏 Spearman 用 seed 平均后的指标，bootstrap 按靶点重采样（B=2000）。")
    lines.append("")
    lines.append("## Spearman / 偏 Spearman（seed 平均）")
    lines.append(corr.round(4).to_string(index=False))
    lines.append("")
    lines.append("## 每个 seed 的 Spearman")
    lines.append(seed_df.round(4).to_string(index=False))
    lines.append("")
    lines.append("## coverage >= 0.5 vs < 0.5（seed 平均）")
    lines.append(groups.round(4).to_string(index=False) if len(groups) else "无方差，无法分组。")
    lines.append("")
    lines.append("## 说明")
    lines.append("- KIBA / Davis 的逐靶点 coverage 恒为 1，无法计算相关，标为 no variance。")
    lines.append("- ChEMBL / BindingDB 是主要可检验数据集；per-seed ρ 范围已列在表中。")
    lines.append("- 这是关联分析，不是因果推断。")
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("[saved]", OUT)
    print(corr.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
