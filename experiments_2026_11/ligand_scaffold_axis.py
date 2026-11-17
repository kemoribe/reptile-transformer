# -*- coding: utf-8 -*-
"""
审计第三轴：配体骨架覆盖（ligand scaffold coverage）
====================================================
回答的问题：测试集的配体骨架，有多少在训练集里出现过？

口径：
- 骨架：RDKit Bemis-Murcko scaffold
  * exact：保留原子类型（MurckoScaffoldSmiles）
  * generic：把原子/键泛化（MakeScaffoldGeneric）
- 覆盖率：
  * compound-level（主口径）：测试集唯一 SMILES 中，骨架在训练集出现过的比例
  * pair-level（辅助）：测试集 (target, SMILES) 对中，骨架在训练集出现过的比例
  * cluster5：训练集中至少有 5 个不同化合物的骨架，测试集覆盖率
- 无环分子 Murcko 骨架为空，按“未覆盖”计入；表中同时报告 acyclic_pct。
- 只做描述统计，不训练模型，不需要 GPU。

用法（PyCharm 终端，工作目录 D:\\lht）：
    python ligand_scaffold_axis.py --datasets pdbbind --workers 4     # 快速自测
    python ligand_scaffold_axis.py --datasets kiba,chembl,davis,bindingdb
    python ligand_scaffold_axis.py --datasets all --workers 16
    python ligand_scaffold_axis.py --datasets all --skip-scaffold     # 复用缓存

输出：
    D:\\lht\\ligand_scaffold_axis\\
      scaffold_cache.csv      每个 SMILES 的 exact/generic 骨架
      dataset_summary.csv     每个数据集的覆盖率汇总
      per_target.csv          逐靶点覆盖率
      REPORT.md
"""
import argparse
import glob
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path(r"D:\lht")
OUT = ROOT / "ligand_scaffold_axis"
CACHE = OUT / "scaffold_cache.csv"

KIBA = ROOT / "3_kiba_preprocessed" / "combined_activities.csv"
CHEMBL = ROOT / "3_all_data_chembl_targets_preprocessed" / "combined_activities.csv"
PANELS = {
    "davis": ROOT / "davis_panel_preprocessed",
    "bindingdb": ROOT / "bindingdb_panel_preprocessed",
}
P2 = {
    "pdbbind": ROOT / "p2_c3_experiments" / "data" / "pdbbind_pairs.csv",
    "metz": ROOT / "p2_c3_experiments" / "data" / "metz_pairs.csv",
    "papyrus": ROOT / "p2_c3_experiments" / "data" / "papyrus_pairs.csv",
}
ALL_DATASETS = ["kiba", "chembl", "davis", "bindingdb", "pdbbind", "metz", "papyrus"]


def scaffold_one(smiles):
    """返回 (smiles, exact, generic, valid)；无环分子骨架为空字符串。"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles, "", "", False
        exact = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf.GetNumAtoms() == 0:
            generic = ""
        else:
            generic = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(scaf))
        return smiles, exact, generic, True
    except Exception:
        return smiles, "", "", False


def load_kiba():
    d = pd.read_csv(KIBA, usecols=["smiles", "protein_id", "split"])
    d["split"] = d["split"].map({"train_set": "train", "test_set": "test"})
    d = d[d["split"].isin(["train", "test"])]
    return pd.DataFrame({"dataset": "kiba", "target": d["protein_id"],
                         "smiles": d["smiles"], "split": d["split"]})


def load_chembl():
    d = pd.read_csv(CHEMBL, usecols=["smiles", "target_name", "set_name"])
    d["split"] = d["set_name"].map({"train_set": "train", "test_set": "test"})
    d = d[d["split"].isin(["train", "test"])]
    return pd.DataFrame({"dataset": "chembl", "target": d["target_name"],
                         "smiles": d["smiles"], "split": d["split"]})


def load_panel(name, root):
    rows = []
    for split, sub in [("train", "train_set"), ("test", "test_set")]:
        for p in glob.glob(str(root / sub / "**" / "*_processed_activities.csv"), recursive=True):
            target = Path(p).parent.name
            try:
                d = pd.read_csv(p, usecols=["smiles"])
            except Exception:
                continue
            if d.empty:
                continue
            d = d[["smiles"]].copy()
            d["dataset"] = name
            d["target"] = target
            d["split"] = split
            rows.append(d)
    if not rows:
        raise FileNotFoundError(f"{name}: no *_processed_activities.csv under {root}")
    return pd.concat(rows, ignore_index=True)


def load_p2(name, path):
    d = pd.read_csv(path, usecols=["target_id", "smiles", "split"])
    d = d[d["split"].isin(["train", "test"])]
    return pd.DataFrame({"dataset": name, "target": d["target_id"],
                         "smiles": d["smiles"], "split": d["split"]})


def load_all(names):
    frames = []
    for n in names:
        if n == "kiba":
            frames.append(load_kiba())
        elif n == "chembl":
            frames.append(load_chembl())
        elif n in PANELS:
            frames.append(load_panel(n, PANELS[n]))
        elif n in P2:
            frames.append(load_p2(n, P2[n]))
        else:
            raise ValueError(n)
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["smiles", "target"])
    df["smiles"] = df["smiles"].astype(str).str.strip()
    df = df[df["smiles"] != ""]
    # 同一 (dataset,target,smiles,split) 只保留一次，避免重复测量放大覆盖计数
    df = df.drop_duplicates(["dataset", "target", "smiles", "split"])
    return df


def build_scaffolds(df, workers, skip=False):
    OUT.mkdir(parents=True, exist_ok=True)
    uniq = pd.unique(df["smiles"]).tolist()
    cache = None
    if CACHE.exists():
        cache = pd.read_csv(CACHE)
        cache = cache.drop_duplicates("smiles")
    if skip and cache is not None:
        print(f"[cache] {len(cache)} SMILES")
    else:
        done = set(cache["smiles"]) if cache is not None else set()
        todo = [s for s in uniq if s not in done]
        print(f"[scaffold] unique={len(uniq)} cached={len(done)} todo={len(todo)} workers={workers}")
        if todo:
            rows = []
            with Pool(processes=workers) as pool:
                for i, r in enumerate(pool.imap_unordered(scaffold_one, todo, chunksize=200), 1):
                    rows.append(r)
                    if i % 50000 == 0:
                        print(f"  {i}/{len(todo)}", flush=True)
            new = pd.DataFrame(rows, columns=["smiles", "scaffold_exact", "scaffold_generic", "valid"])
            cache = pd.concat([cache, new], ignore_index=True) if cache is not None else new
            cache = cache.drop_duplicates("smiles")
            cache.to_csv(CACHE, index=False, encoding="utf-8-sig")
            print(f"[saved] {CACHE} ({len(cache)} rows)")
    if cache is None:
        raise RuntimeError("no scaffold cache")
    return cache


def coverage_for_dataset(name, d, scaf):
    d = d.merge(scaf, on="smiles", how="left")
    valid = d[d["valid"] == True].copy()  # noqa: E712
    n_invalid = int((d["valid"] != True).sum())  # noqa: E712
    tr = valid[valid["split"] == "train"]
    te = valid[valid["split"] == "test"]
    if tr.empty or te.empty:
        return None, None

    tr_exact = set(tr.loc[tr["scaffold_exact"] != "", "scaffold_exact"])
    tr_generic = set(tr.loc[tr["scaffold_generic"] != "", "scaffold_generic"])

    # 训练集中至少 5 个不同化合物的 exact 骨架
    cnt = tr[tr["scaffold_exact"] != ""].drop_duplicates(["scaffold_exact", "smiles"]) \
        .groupby("scaffold_exact")["smiles"].nunique()
    tr_cluster5 = set(cnt[cnt >= 5].index)

    te_comp = te.drop_duplicates("smiles").copy()
    te_comp["seen_exact"] = te_comp["scaffold_exact"].isin(tr_exact)
    te_comp["seen_generic"] = te_comp["scaffold_generic"].isin(tr_generic)
    te_comp["seen_cluster5"] = te_comp["scaffold_exact"].isin(tr_cluster5)

    pair_exact = te["scaffold_exact"].isin(tr_exact).mean()
    pair_generic = te["scaffold_generic"].isin(tr_generic).mean()
    comp_exact = te_comp["seen_exact"].mean()
    comp_generic = te_comp["seen_generic"].mean()
    comp_cluster5 = te_comp["seen_cluster5"].mean()
    acyclic = (te_comp["scaffold_exact"] == "").mean()

    # 逐靶点（compound-level）：先在每个靶点内部按 SMILES 去重，
    # 不能全局按 SMILES 去重，否则共享化合物面板（Davis/Metz）会只剩第一个靶点
    te_pt = te.drop_duplicates(["target", "smiles"]).copy()
    te_pt["seen_exact"] = te_pt["scaffold_exact"].isin(tr_exact)
    te_pt["seen_generic"] = te_pt["scaffold_generic"].isin(tr_generic)
    te_pt["seen_cluster5"] = te_pt["scaffold_exact"].isin(tr_cluster5)
    pt_rows = []
    for target, g in te_pt.groupby("target", sort=False):
        pt_rows.append({
            "dataset": name, "target": target,
            "n_test_ligands": int(g["smiles"].nunique()),
            "exact_cov": float(g["seen_exact"].mean()),
            "generic_cov": float(g["seen_generic"].mean()),
            "cluster5_cov": float(g["seen_cluster5"].mean()),
        })
    pt = pd.DataFrame(pt_rows)
    vals = pt["exact_cov"].values if len(pt) else np.array([np.nan])

    summary = {
        "dataset": name,
        "n_train_pairs": int(len(tr)),
        "n_test_pairs": int(len(te)),
        "n_train_compounds": int(tr["smiles"].nunique()),
        "n_test_compounds": int(te_comp["smiles"].nunique()),
        "n_test_targets": int(len(pt)),
        "invalid_smiles": n_invalid,
        "exact_pair_cov": float(pair_exact),
        "generic_pair_cov": float(pair_generic),
        "exact_compound_cov": float(comp_exact),
        "generic_compound_cov": float(comp_generic),
        "cluster5_compound_cov": float(comp_cluster5),
        "novel_exact_pct": float(1 - comp_exact),
        "novel_generic_pct": float(1 - comp_generic),
        "acyclic_pct": float(acyclic),
        "per_target_exact_median": float(np.nanmedian(vals)),
        "per_target_exact_q25": float(np.nanpercentile(vals, 25)),
        "per_target_exact_q75": float(np.nanpercentile(vals, 75)),
        "per_target_exact_min": float(np.nanmin(vals)),
        "per_target_exact_max": float(np.nanmax(vals)),
    }
    return summary, pt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="kiba,chembl,davis,bindingdb",
                    help="逗号分隔；all = 7 个数据集")
    ap.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 8) - 1)))
    ap.add_argument("--skip-scaffold", action="store_true", help="复用已有 scaffold_cache.csv")
    args = ap.parse_args()

    names = ALL_DATASETS if args.datasets.strip().lower() == "all" else \
        [x.strip() for x in args.datasets.split(",") if x.strip()]
    OUT.mkdir(parents=True, exist_ok=True)

    print("[load] datasets:", names)
    df = load_all(names)
    print(f"[load] rows={len(df)} datasets={df['dataset'].nunique()} "
          f"targets={df.groupby('dataset')['target'].nunique().to_dict()}")

    scaf = build_scaffolds(df, args.workers, skip=args.skip_scaffold)

    summaries, per_targets = [], []
    for name in names:
        s, pt = coverage_for_dataset(name, df[df["dataset"] == name], scaf)
        if s is None:
            print(f"[skip] {name}: no train/test")
            continue
        summaries.append(s)
        per_targets.append(pt)
        print(f"[{name}] exact_compound_cov={s['exact_compound_cov']:.3f} "
              f"generic={s['generic_compound_cov']:.3f} novel={s['novel_exact_pct']:.3f} "
              f"targets={s['n_test_targets']}")

    summ = pd.DataFrame(summaries)
    pt = pd.concat(per_targets, ignore_index=True) if per_targets else pd.DataFrame()
    summ.to_csv(OUT / "dataset_summary.csv", index=False, encoding="utf-8-sig")
    pt.to_csv(OUT / "per_target.csv", index=False, encoding="utf-8-sig")

    lines = ["# 审计第三轴：配体骨架覆盖", ""]
    lines.append("骨架定义：RDKit Bemis-Murcko scaffold；exact 保留原子类型，generic 做原子/键泛化。")
    lines.append("主口径为 compound-level：测试集唯一 SMILES 中骨架在训练集出现过的比例。")
    lines.append("无环分子骨架为空，按未覆盖计入；acyclic_pct 单独报告。")
    lines.append("")
    lines.append(summ.round(4).to_string(index=False))
    lines.append("")
    lines.append("## 说明")
    lines.append("- pair-level 会被重复测量放大，主文建议用 compound-level。")
    lines.append("- cluster5 只统计训练集中至少 5 个不同化合物的骨架，用于排除通用骨架的影响。")
    lines.append("- 这是描述性审计轴，不表示骨架重叠与性能之间存在因果关系。")
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("[saved]", OUT)


if __name__ == "__main__":
    main()
