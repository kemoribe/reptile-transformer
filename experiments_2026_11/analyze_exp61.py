# -*- coding: utf-8 -*-
"""
6.1 分析：预算匹配对照 + 完整 Reptile 族逐 seed 差值 + 融合/consistency 对照
==========================================================================
输入（全部已有或由 run_budget_matched_control.py 生成）：
  - full Reptile: 修改supplement_output/revision2_experiments/e5_repeats/kiba/seed_*/
  - 预算匹配对照: control_budget_matched/seed_*/
  - 模块删除: reptile_module_ablation/{variant}/seed_*/

输出：
  D:\\lht\\exp61_analysis\\
    per_seed.csv        逐 seed 指标 + 预算/耗时/最佳 epoch
    budget.csv          每种配置的更新次数、样本访问量、meta 更新、耗时
    deltas.csv          相对 full 的逐 seed 差值（control 5 对；消融 3 对）
    bootstrap.csv       共同靶点配对 bootstrap（ΔR2 / ΔEF1 的 95% CI）
    REPORT.md

用法：
    python analyze_exp61.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\lht")
E5 = ROOT / "修改supplement_output" / "revision2_experiments" / "e5_repeats" / "kiba"
CONTROL = ROOT / "control_budget_matched"
ABL = ROOT / "reptile_module_ablation"
OUT = ROOT / "exp61_analysis"
FEAT = ROOT / "transformer_ablation_output_kiba" / "precomputed_features.npz"

import sys
sys.path.insert(0, str(ROOT))
import run_table2_kiba_baselines as tb  # noqa
from recompute_s18_s19 import macro_ef  # noqa

SEEDS_FULL = [11, 22, 33, 44, 55]
SEEDS_ABL = [11, 22, 33]
VARIANTS = {
    "reptile_no_ranking": "ranking=0",
    "reptile_no_contrastive": "contrastive=0",
    "reptile_no_consistency": "consistency=0",
    "reptile_no_bidir_fusion": "concat + consistency=0",
}
B = 2000


def test_targets():
    d = np.load(FEAT, mmap_mode="r", allow_pickle=True)
    sp = d["splits"][:]
    tn = d["target_names"][:]
    tr = sp == "train"
    return tn[sp == "test"], tn[tr]


def metrics(pred_path, targets):
    yt, yp = tb.load_pred(pred_path)
    g = tb.global_metrics(yt, yp)
    mm = tb.macro_metrics(yt, yp, targets)
    ef5, used5, _ = macro_ef(yt, yp, targets, 5, 20)
    return dict(R2=g["R2"], RMSE=g["RMSE"], ECE=g["ECE"],
                EF1=mm["EF@1%"], EF5=ef5, AUPR=mm["AUPR"],
                n_ef1=mm["EF_targets"], n_ef5=used5), yt, yp


def best_epoch_from_history(fp):
    if not fp.exists():
        return None
    h = json.loads(fp.read_text(encoding="utf-8"))
    if not h:
        return None
    best = max(h, key=lambda r: float(r.get("val_r2", -1e9)))
    return int(best.get("epoch", 0)) + 1


def budget_row(name, epochs, counts, walltime, best_epoch, control=False):
    n_tasks = len(counts)
    inner_updates = epochs * n_tasks * 3
    samples = int(epochs * int(np.minimum(counts, 2048).sum()) * 3)
    return dict(config=name, epochs=epochs, n_train_targets=n_tasks,
                inner_updates=inner_updates,
                meta_updates=0 if control else epochs,
                samples_seen=samples,
                walltime_s=walltime, best_epoch=best_epoch,
                control_mode="direct_supervised" if control else "reptile")


def bootstrap_pair(yt, yp_a, yp_b, targets, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(targets)
    idx = {t: np.where(targets == t)[0] for t in uniq}
    # 逐靶点 EF1（仅 n>=100）
    ef_idx = [idx[t] for t in uniq if len(idx[t]) >= 100]
    ef_a = np.array([tb.ef_at(yt[i], yp_a[i], 1) for i in ef_idx])
    ef_b = np.array([tb.ef_at(yt[i], yp_b[i], 1) for i in ef_idx])
    ok = np.isfinite(ef_a) & np.isfinite(ef_b)
    ef_a, ef_b = ef_a[ok], ef_b[ok]

    d_r2, d_ef = [], []
    for _ in range(B):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx[t] for t in samp])
        y = yt[ii]
        r2a = 1 - np.sum((y - yp_a[ii]) ** 2) / np.sum((y - y.mean()) ** 2)
        r2b = 1 - np.sum((y - yp_b[ii]) ** 2) / np.sum((y - y.mean()) ** 2)
        d_r2.append(r2b - r2a)
        if len(ef_a) > 0:
            j = rng.integers(0, len(ef_a), size=len(ef_a))
            d_ef.append(float(np.mean(ef_b[j] - ef_a[j])))
    out = {}
    for key, vals in [("dR2", d_r2), ("dEF1", d_ef)]:
        if not vals:
            out[key] = (np.nan, np.nan, np.nan, np.nan)
            continue
        v = np.array(vals)
        out[key] = (float(v.mean()), float(np.percentile(v, 2.5)),
                    float(np.percentile(v, 97.5)), float(np.mean(v > 0)))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    targets, train_targets = test_targets()
    counts = np.array([int((train_targets == t).sum()) for t in np.unique(train_targets)])

    per_seed, budget, deltas = [], [], []
    full_metrics = {}

    # full Reptile（5 seeds）
    for s in SEEDS_FULL:
        d = E5 / f"seed_{s}"
        if not (d / "predictions.npz").exists():
            continue
        m, _, _ = metrics(d / "predictions.npz", targets)
        wall = float((d / "walltime_sec.txt").read_text().strip())
        be = best_epoch_from_history(d / "training_history.json")
        per_seed.append(dict(family="full_reptile", variant="full_reptile", seed=s,
                             **m, walltime_s=wall, best_epoch=be, epochs=220))
        full_metrics[s] = m
        budget.append(budget_row("full_reptile", 220, counts, wall, be, control=False))

    # 预算匹配 direct_supervised（5 seeds）
    ctrl_metrics = {}
    for s in SEEDS_FULL:
        d = CONTROL / f"seed_{s}"
        if not (d / "predictions.npz").exists():
            continue
        m, _, _ = metrics(d / "predictions.npz", targets)
        b = json.loads((d / "training_budget.json").read_text(encoding="utf-8")) \
            if (d / "training_budget.json").exists() else {}
        epochs_run = int(b.get("epochs_run", 220))
        seg_inner = b.get("inner_updates")
        seg_samples = b.get("samples_seen")
        total_inner = epochs_run * len(counts) * 3
        total_samples = int(epochs_run * int(np.minimum(counts, 2048).sum()) * 3)
        resumed = seg_inner is not None and seg_inner < total_inner
        per_seed.append(dict(family="control", variant="direct_supervised", seed=s,
                             **m, walltime_s=b.get("walltime_s"),
                             best_epoch=b.get("best_epoch"), epochs=epochs_run,
                             inner_updates_total=total_inner,
                             inner_updates_segment=seg_inner,
                             samples_seen_total=total_samples,
                             samples_seen_segment=seg_samples,
                             resumed=resumed))
        ctrl_metrics[s] = m
        budget.append(dict(config="direct_supervised", epochs=epochs_run,
                           n_train_targets=len(counts),
                           inner_updates=total_inner, meta_updates=0,
                           samples_seen=total_samples,
                           walltime_s=b.get("walltime_s"),
                           best_epoch=b.get("best_epoch"),
                           control_mode="direct_supervised",
                           resumed=resumed,
                           segment_inner_updates=seg_inner,
                           segment_samples_seen=seg_samples))

    # 完整 Reptile 族删除（3 seeds）
    abl_metrics = {}
    for v in VARIANTS:
        for s in SEEDS_ABL:
            d = ABL / v / f"seed_{s}"
            if not (d / "predictions.npz").exists():
                continue
            m, _, _ = metrics(d / "predictions.npz", targets)
            wall = None
            ps = ABL / "per_seed.csv"
            if ps.exists():
                df = pd.read_csv(ps)
                r = df[(df["variant"] == v) & (df["seed"] == s)]
                if len(r):
                    wall = float(r["walltime_s"].iloc[0])
            be = best_epoch_from_history(d / "training_history.json")
            per_seed.append(dict(family="ablation", variant=v, seed=s,
                                 **m, walltime_s=wall, best_epoch=be, epochs=220))
            abl_metrics[(v, s)] = m
            budget.append(budget_row(v, 220, counts, wall, be, control=False))

    # 逐 seed 差值
    for s in SEEDS_FULL:
        if s in ctrl_metrics and s in full_metrics:
            a, b = full_metrics[s], ctrl_metrics[s]
            deltas.append(dict(comparison="direct_supervised - full", seed=s,
                               dR2=b["R2"] - a["R2"], dEF1=b["EF1"] - a["EF1"],
                               dEF5=b["EF5"] - a["EF5"], dAUPR=b["AUPR"] - a["AUPR"],
                               dECE=b["ECE"] - a["ECE"]))
    for v in VARIANTS:
        for s in SEEDS_ABL:
            if (v, s) in abl_metrics and s in full_metrics:
                a, b = full_metrics[s], abl_metrics[(v, s)]
                deltas.append(dict(comparison=f"{v} - full", seed=s,
                                   dR2=b["R2"] - a["R2"], dEF1=b["EF1"] - a["EF1"],
                                   dEF5=b["EF5"] - a["EF5"], dAUPR=b["AUPR"] - a["AUPR"],
                                   dECE=b["ECE"] - a["ECE"]))
    # 融合 × consistency（两者 cons=0）
    for s in SEEDS_ABL:
        a = abl_metrics.get(("reptile_no_consistency", s))
        b = abl_metrics.get(("reptile_no_bidir_fusion", s))
        if a and b:
            deltas.append(dict(comparison="concat - bidirectional (both cons=0)", seed=s,
                               dR2=b["R2"] - a["R2"], dEF1=b["EF1"] - a["EF1"],
                               dEF5=b["EF5"] - a["EF5"], dAUPR=b["AUPR"] - a["AUPR"],
                               dECE=b["ECE"] - a["ECE"]))

    df_seed = pd.DataFrame(per_seed)
    df_budget = pd.DataFrame(budget).drop_duplicates()
    df_delta = pd.DataFrame(deltas)
    df_seed.to_csv(OUT / "per_seed.csv", index=False, encoding="utf-8-sig")
    df_budget.to_csv(OUT / "budget.csv", index=False, encoding="utf-8-sig")
    df_delta.to_csv(OUT / "deltas.csv", index=False, encoding="utf-8-sig")

    # 配对统计：单样本 t 检验（H0: 逐 seed 差值均值 = 0）
    from scipy import stats as sstats
    stats_rows = []
    for comp, g in df_delta.groupby("comparison"):
        for metric in ["dR2", "dEF1", "dEF5", "dAUPR", "dECE"]:
            vals = g[metric].dropna().values.astype(float)
            if len(vals) < 2:
                continue
            t, p = sstats.ttest_1samp(vals, 0)
            try:
                _, pw = sstats.wilcoxon(vals)
            except Exception:
                pw = np.nan
            stats_rows.append(dict(comparison=comp, metric=metric, n=len(vals),
                                   mean=vals.mean(), sd=vals.std(ddof=1),
                                   t=t, p_t=p, p_wilcoxon=pw))
    df_stats = pd.DataFrame(stats_rows)
    df_stats.to_csv(OUT / "stats.csv", index=False, encoding="utf-8-sig")

    # bootstrap：control vs full；每个消融 vs full；融合对照
    rows = []
    yt_full, yp_full = tb.load_pred(E5 / "seed_11" / "predictions.npz")

    def add_boot(name, yp_a, yp_b, seed=0):
        r = bootstrap_pair(yt_full, yp_a, yp_b, targets, seed=seed)
        for k, vals in r.items():
            rows.append(dict(comparison=name, metric=k, mean=vals[0],
                             ci_low=vals[1], ci_high=vals[2], p_gt0=vals[3], B=B))

    # control：先做 seed 11 的代表性 bootstrap；其余 seed 同法
    if (CONTROL / "seed_11" / "predictions.npz").exists():
        _, yp_c = tb.load_pred(CONTROL / "seed_11" / "predictions.npz")
        add_boot("direct_supervised - full (seed11)", yp_full, yp_c)
    for v in VARIANTS:
        if (ABL / v / "seed_11" / "predictions.npz").exists():
            _, yp_v = tb.load_pred(ABL / v / "seed_11" / "predictions.npz")
            add_boot(f"{v} - full (seed11)", yp_full, yp_v)
    a = ABL / "reptile_no_consistency" / "seed_11" / "predictions.npz"
    b = ABL / "reptile_no_bidir_fusion" / "seed_11" / "predictions.npz"
    if a.exists() and b.exists():
        _, yp_a = tb.load_pred(a)
        _, yp_b = tb.load_pred(b)
        add_boot("concat - bidirectional (both cons=0, seed11)", yp_a, yp_b)
    pd.DataFrame(rows).to_csv(OUT / "bootstrap.csv", index=False, encoding="utf-8-sig")

    lines = ["# 6.1 预算匹配对照与模块贡献", ""]
    lines.append("## 预算")
    lines.append(df_budget.to_string(index=False))
    lines.append("")
    lines.append("## 逐 seed 差值（相对 full Reptile，配对 seed）")
    if len(df_delta):
        summ = df_delta.groupby("comparison")[["dR2", "dEF1", "dEF5", "dAUPR", "dECE"]].agg(["mean", "std"])
        lines.append(summ.to_string())
    lines.append("")
    lines.append("## 配对统计（单样本 t 检验，H0: Δ=0；Wilcoxon 仅 n>=5 时稳定）")
    lines.append(df_stats.round(4).to_string(index=False))
    lines.append("")
    lines.append("## 融合 × consistency")
    lines.append("consistency loss 在 BidirectionalCrossAttention 内计算；concat 模式下 consistency 恒为 0。")
    lines.append("因此用 reptile_no_consistency（bidirectional, cons=0）与 reptile_no_bidir_fusion（concat, cons=0）比较，")
    lines.append("两者只差融合方式；不额外为 concat 设计 consistency。")
    lines.append("")
    lines.append("## 说明")
    lines.append("- direct_supervised seed 11 从 epoch 80 续训，最终仍为 220 epoch；总预算按 220 epoch 计，")
    lines.append("  段内计数器（training_budget.json）只覆盖后 140 epoch；per_seed.csv 中同时保留 segment 计数。")
    lines.append("- 3 seeds 的消融为探索性结果；若要把很小的模块收益作为核心结论，再扩展到 5 seeds。")
    lines.append("- 预测配对 bootstrap 回答“换一批靶点是否还成立”，与跨 seed 波动不是同一个问题。")
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("[saved]", OUT)


if __name__ == "__main__":
    main()
