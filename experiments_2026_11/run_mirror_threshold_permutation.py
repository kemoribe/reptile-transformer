# -*- coding: utf-8 -*-
"""
结合谱镜像审计：共享化合物阈值敏感性 + 置换对照
================================================
主分析阈值：>=10 共享化合物
敏感性：>=3, >=5, >=20

置换设计：
- 对每个训练靶点，在其“已测配体”内部置换 affinity 值；
- 测试靶点配体集合、训练靶点缺失结构、test-train 共享配体数量均保持不变；
- 每次置换后重新计算每个测试靶点的 max Spearman ρ（对同一训练 panel 取 max）。

输出：
  D:\\lht\\mirror_threshold_permutation\\
    summary.csv
    per_target.csv
    null_{dataset}.npz
    REPORT.md

用法（PyCharm 终端，工作目录 D:\\lht）：
    python run_mirror_threshold_permutation.py
    python run_mirror_threshold_permutation.py --permutations 1000 --bootstrap 2000 --jobs 4
    python run_mirror_threshold_permutation.py --datasets davis,kiba,chembl,bindingdb

说明：本实验是 CPU 置换检验，不使用 GPU；可与 GPU 上的 ChEMBL 训练并行，提高整机利用率。
"""
import argparse
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.stats import rankdata

RDLogger.DisableLog("rdApp.*")

ROOT = Path(r"D:\lht")
GD = ROOT / "GrapthDTA" / "data"
PC = ROOT / "修改supplement_output" / "protein_cluster"
OUT = ROOT / "mirror_threshold_permutation"
CACHE = ROOT / "mirror_threshold_permutation" / "cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

DATASETS = ["davis", "kiba", "chembl", "bindingdb"]
THRESHOLDS = [3, 5, 10, 20]
RHO_T = 0.7


def canon(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m, isomericSmiles=True) if m is not None else None


def build_profiles(df):
    out = []
    for seq, g in df.groupby("target_sequence"):
        vals = g[["smiles", "affinity"]].dropna()
        vals = vals[vals["smiles"].notna()]
        p = vals.groupby("smiles")["affinity"].mean()
        if len(p) == 0:
            continue
        out.append((seq, list(p.index), p.values.astype(float)))
    return out


def load_or_build(dataset: str):
    cache = CACHE / f"{dataset}_profiles.pkl"
    if cache.exists():
        print(f"[cache] {dataset}")
        return pickle.load(open(cache, "rb"))

    print(f"[build] {dataset}: reading/canonicalizing SMILES ...", flush=True)
    tr = pd.read_csv(GD / f"{dataset}_train.csv")
    te = pd.read_csv(GD / f"{dataset}_test.csv")
    tr["smiles"] = [canon(s) for s in tr["compound_iso_smiles"]]
    te["smiles"] = [canon(s) for s in te["compound_iso_smiles"]]
    tr = tr.dropna(subset=["smiles", "target_sequence", "affinity"])
    te = te.dropna(subset=["smiles", "target_sequence", "affinity"])
    train = build_profiles(tr)
    test = sorted(build_profiles(te), key=lambda x: x[0])
    pid = pd.read_csv(PC / dataset / "per_test_target_identity.csv")
    tid2same = dict(zip(pid["target_id"], pid["same_cluster_40"].astype(bool)))
    tid2ident = dict(zip(pid["target_id"], pid["nearest_train_identity_pct"]))
    obj = {
        "train": train,
        "test": test,
        "tid2same": tid2same,
        "tid2ident": tid2ident,
    }
    pickle.dump(obj, open(cache, "wb"), protocol=4)
    print(f"[cache] saved {cache}")
    return obj


def centered_rank(x):
    r = rankdata(x)
    r = r - r.mean()
    return r, float(np.dot(r, r))


def rho_from_rank(ra, ssa, b):
    rb, ssb = centered_rank(b)
    if ssb <= 1e-12:
        return np.nan
    return float(np.dot(ra, rb) / np.sqrt(ssa * ssb))


def build_pairs(test_item, train, thr):
    t_seq, t_lig, t_val = test_item
    t_map = {lig: i for i, lig in enumerate(t_lig)}
    pairs = []
    for j, (tr_seq, tr_lig, tr_val) in enumerate(train):
        tr_set = set(tr_lig)
        shared = [lig for lig in t_lig if lig in tr_set]
        if len(shared) < thr:
            continue
        idx = np.array([tr_lig.index(lig) for lig in shared], dtype=np.int64)
        vt = np.array([t_val[t_map[lig]] for lig in shared], dtype=float)
        if np.std(vt) <= 1e-12 or np.std(tr_val[idx]) <= 1e-12:
            continue
        ra, ssa = centered_rank(vt)
        pairs.append((j, idx, ra, ssa, len(shared)))
    return pairs


def observed_max(pairs, train):
    best = -2.0
    best_n = 0
    for j, idx, ra, ssa, n_shared in pairs:
        rho = rho_from_rank(ra, ssa, train[j][2][idx])
        if np.isfinite(rho) and rho > best:
            best = rho
            best_n = n_shared
    return (best if best > -2 else np.nan), best_n


def process_dataset(args):
    dataset, thresholds, B = args
    print(f"\n===== {dataset} =====", flush=True)
    obj = load_or_build(dataset)
    train = obj["train"]
    test = obj["test"]
    tid2same = obj["tid2same"]
    tid2ident = obj["tid2ident"]
    n_total = len(test)
    target_ids = [f"test_{i:04d}" for i in range(n_total)]
    stratum = []
    for tid in target_ids:
        if tid in tid2same:
            stratum.append("same" if tid2same[tid] else "cross")
        else:
            stratum.append("unknown")

    rng = np.random.default_rng(20261113)
    per_target_rows = []
    null_store = {"target_ids": np.array(target_ids), "thresholds": np.array(thresholds)}

    for thr in thresholds:
        print(f"[{dataset}] threshold >= {thr}: building pairs", flush=True)
        pairs_by_target = [build_pairs(t, train, thr) for t in test]
        n_eval = sum(1 for p in pairs_by_target if len(p) > 0)
        print(f"[{dataset}] thr={thr}: evaluable {n_eval}/{n_total}, pairs={sum(len(p) for p in pairs_by_target)}", flush=True)

        obs_rho = np.full(n_total, np.nan)
        obs_mirror = np.zeros(n_total, dtype=bool)
        obs_nshared = np.zeros(n_total, dtype=int)
        obs_ncmp = np.array([len(p) for p in pairs_by_target], dtype=int)
        for i, pairs in enumerate(pairs_by_target):
            if not pairs:
                continue
            rho, nsh = observed_max(pairs, train)
            obs_rho[i] = rho
            obs_nshared[i] = nsh
            obs_mirror[i] = bool(np.isfinite(rho) and rho >= RHO_T)

        null_rho = np.full((B, n_total), np.nan, dtype=np.float32)
        null_mirror = np.zeros((B, n_total), dtype=bool)
        for b in range(B):
            perm_values = [rng.permutation(tr[2]) for tr in train]
            for i, pairs in enumerate(pairs_by_target):
                if not pairs:
                    continue
                best = -2.0
                for j, idx, ra, ssa, _ in pairs:
                    rho = rho_from_rank(ra, ssa, perm_values[j][idx])
                    if np.isfinite(rho) and rho > best:
                        best = rho
                if best > -2:
                    null_rho[b, i] = best
                    null_mirror[b, i] = bool(best >= RHO_T)
            if (b + 1) % 100 == 0:
                print(f"[{dataset}] thr={thr}: permutation {b+1}/{B}", flush=True)

        null_store[f"null_rho_{thr}"] = null_rho
        null_store[f"null_mirror_{thr}"] = null_mirror
        null_store[f"obs_rho_{thr}"] = obs_rho
        null_store[f"obs_mirror_{thr}"] = obs_mirror

        for i, tid in enumerate(target_ids):
            if obs_ncmp[i] == 0:
                p_t = np.nan
                null_mean = np.nan
            else:
                nv = null_rho[:, i]
                null_mean = float(np.nanmean(nv)) if np.isfinite(nv).any() else np.nan
                if np.isfinite(obs_rho[i]):
                    p_t = float((1 + np.sum(nv >= obs_rho[i])) / (B + 1))
                else:
                    p_t = np.nan
            per_target_rows.append({
                "dataset": dataset, "threshold": thr, "target_id": tid,
                "stratum": stratum[i],
                "n_ligands": len(test[i][1]),
                "n_comparable_train": int(obs_ncmp[i]),
                "n_shared_best": int(obs_nshared[i]),
                "max_rho": float(obs_rho[i]) if np.isfinite(obs_rho[i]) else np.nan,
                "mirror": bool(obs_mirror[i]),
                "null_mean_rho": null_mean,
                "null_p": p_t,
                "nearest_train_identity_pct": tid2ident.get(tid, np.nan),
            })

    np.savez_compressed(OUT / f"null_{dataset}.npz", **null_store)
    print(f"[{dataset}] saved null_{dataset}.npz", flush=True)
    return dataset, per_target_rows


def summarize_dataset(dataset, per_target, B, n_boot):
    df = pd.DataFrame(per_target)
    rng = np.random.default_rng(20261114)
    rows = []
    for thr in sorted(df["threshold"].unique()):
        sub = df[df["threshold"] == thr].copy()
        for strat in ["all", "same", "cross"]:
            s = sub if strat == "all" else sub[sub["stratum"] == strat]
            total = len(s)
            evaluable = s[s["n_comparable_train"] > 0]
            n_eval = len(evaluable)
            n_lig_ge = int((s["n_ligands"] >= thr).sum())
            if n_eval == 0:
                rows.append({
                    "dataset": dataset, "threshold": thr, "stratum": strat,
                    "n_total_targets": total, "n_ligands_ge_thr": n_lig_ge,
                    "n_evaluable": 0, "coverage": 0.0,
                    "n_mirror": 0, "observed_rate": np.nan,
                    "null_rate_mean": np.nan, "delta": np.nan,
                    "delta_ci_low": np.nan, "delta_ci_high": np.nan,
                    "perm_p": np.nan, "B": B, "note": "no evaluable targets",
                })
                continue
            obs = evaluable["mirror"].astype(float).values
            obs_rate = float(obs.mean())
            n_mirror = int(obs.sum())

            # 从 null npz 读取该 threshold 的 null_mirror
            z = np.load(OUT / f"null_{dataset}.npz")
            null_mirror = z[f"null_mirror_{thr}"]
            idx_all = np.array([int(t.replace("test_", "")) for t in evaluable["target_id"]])
            nm = null_mirror[:, idx_all]
            null_rates = nm.mean(axis=1)
            null_mean = float(null_rates.mean())
            delta = obs_rate - null_mean
            perm_p = float((1 + np.sum(null_rates >= obs_rate)) / (B + 1))

            # target + permutation bootstrap
            deltas = []
            for _ in range(n_boot):
                ii = rng.integers(0, n_eval, n_eval)
                bb = rng.integers(0, B)
                deltas.append(float(obs[ii].mean() - nm[bb, ii].mean()))
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            rows.append({
                "dataset": dataset, "threshold": thr, "stratum": strat,
                "n_total_targets": total, "n_ligands_ge_thr": n_lig_ge,
                "n_evaluable": n_eval, "coverage": n_eval / total if total else np.nan,
                "n_mirror": n_mirror, "observed_rate": obs_rate,
                "null_rate_mean": null_mean, "delta": delta,
                "delta_ci_low": float(lo), "delta_ci_high": float(hi),
                "perm_p": perm_p, "B": B,
                "note": "low coverage" if n_eval < 10 else "",
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--permutations", type=int, default=1000)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--thresholds", default="3,5,10,20")
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    thresholds = [int(x) for x in args.thresholds.split(",") if x.strip()]

    print("datasets:", datasets)
    print("thresholds:", thresholds)
    print("permutations:", args.permutations, "bootstrap:", args.bootstrap, "jobs:", args.jobs)

    tasks = [(ds, thresholds, args.permutations) for ds in datasets]
    all_per_target = []
    if args.jobs > 1:
        with Pool(processes=min(args.jobs, len(tasks))) as pool:
            for ds, rows in pool.imap_unordered(process_dataset, tasks):
                all_per_target.extend(rows)
                print(f"[done] {ds}")
    else:
        for t in tasks:
            ds, rows = process_dataset(t)
            all_per_target.extend(rows)
            print(f"[done] {ds}")

    per_target = pd.DataFrame(all_per_target)
    per_target.to_csv(OUT / "per_target.csv", index=False, encoding="utf-8-sig")

    summary = []
    for ds in datasets:
        sub = per_target[per_target["dataset"] == ds]
        summary.extend(summarize_dataset(ds, sub.to_dict("records"), args.permutations, args.bootstrap))
    summ = pd.DataFrame(summary)
    summ.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")

    # Markdown report
    lines = ["# 结合谱镜像审计：阈值敏感性与置换对照", ""]
    lines.append(f"置换次数 B={args.permutations}；bootstrap={args.bootstrap}；镜像阈值 ρ≥{RHO_T}。")
    lines.append("置换设计：在每个训练靶点的已测配体内部置换 affinity；测试配体集合、训练缺失结构、共享配体数保持不变。")
    lines.append("")
    for ds in datasets:
        lines.append(f"## {ds}")
        s = summ[summ["dataset"] == ds]
        for thr in thresholds:
            for strat in ["all", "same", "cross"]:
                r = s[(s["threshold"] == thr) & (s["stratum"] == strat)]
                if r.empty:
                    continue
                r = r.iloc[0]
                if r["n_evaluable"] == 0:
                    lines.append(f"- ≥{thr} {strat}: 无可评估靶点")
                    continue
                lines.append(
                    f"- ≥{thr} {strat}: {int(r['n_mirror'])}/{int(r['n_evaluable'])} "
                    f"({100*r['observed_rate']:.1f}%), null {100*r['null_rate_mean']:.1f}%, "
                    f"Δ={100*r['delta']:+.1f} pp "
                    f"[{100*r['delta_ci_low']:+.1f}, {100*r['delta_ci_high']:+.1f}], "
                    f"p={r['perm_p']:.4f}, coverage={100*r['coverage']:.1f}%")
        lines.append("")
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 80)
    print(summ.to_string(index=False))
    print("=" * 80)
    print("[saved]", OUT / "summary.csv")
    print("[saved]", OUT / "per_target.csv")
    print("[saved]", OUT / "REPORT.md")


if __name__ == "__main__":
    main()
