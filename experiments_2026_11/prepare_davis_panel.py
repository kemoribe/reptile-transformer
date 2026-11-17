# -*- coding: utf-8 -*-
"""
构建 Davis 面板数据（morgan_descriptors 统一口径）
=================================================
来源：
  D:\\lht\\3_davis_preprocessed\\combined_activities.csv
    columns: standardized_smiles, paffinity, affinity_type, gene_name, set_name
  D:\\lht\\Davis\\proteins.csv
    columns: Protein_Index, Accession_Number, Gene_Name, Sequence

输出 D:\\lht\\davis_panel_preprocessed：
  {train_set,val_set,test_set}/default/DAVIS_{gene_name}/
      DAVIS_{gene_name}_processed_activities.csv   (smiles, paffinity, ...)
      DAVIS_{gene_name}_processed_protein_sequence.txt

校验：
  - train/val/test 行数 19,668 / 4,224 / 4,290
  - 靶点数 298 / 64 / 65
  - test 的 65 条序列与 GrapthDTA/data/davis_test.csv 完全对应
  - test 的 4,290 个 (canonical SMILES, sequence) 与 GNN CSV 完全对应，
    paffinity 与 GNN affinity 差异 < 1e-5

用法（PyCharm 终端，工作目录 D:\\lht）：
    python prepare_davis_panel.py
    python prepare_davis_panel.py --force
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

ROOT = Path(r"D:\lht")
COMBINED = ROOT / "3_davis_preprocessed" / "combined_activities.csv"
PROTEINS = ROOT / "Davis" / "proteins.csv"
GNN_CSV = ROOT / "GrapthDTA" / "data" / "davis_test.csv"
DST = ROOT / "davis_panel_preprocessed"

EXPECTED = {
    "train_set": (19668, 298),
    "val_set": (4224, 64),
    "test_set": (4290, 65),
}


def canon(smiles):
    m = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(m) if m is not None else None


def safe_name(gene):
    return str(gene).replace("/", "_").replace("\\", "_")


def build():
    proteins = pd.read_csv(PROTEINS)
    gene2seq = {}
    for r in proteins.itertuples():
        gene2seq[str(r.Gene_Name)] = str(r.Sequence)
    print(f"[proteins] {len(gene2seq)} 条序列")

    acts = pd.read_csv(COMBINED)
    if acts["standardized_smiles"].isna().any():
        raise RuntimeError("combined_activities.csv 中存在空 SMILES")
    print(f"[activities] {len(acts)} 行，set 分布: "
          f"{dict(Counter(acts['set_name']))}")

    stats = {}
    for set_name, (exp_rows, exp_targets) in EXPECTED.items():
        sub = acts[acts["set_name"] == set_name]
        if len(sub) != exp_rows:
            raise RuntimeError(f"{set_name} 行数 {len(sub)} != {exp_rows}")
        targets = sorted(sub["gene_name"].unique())
        if len(targets) != exp_targets:
            raise RuntimeError(f"{set_name} 靶点数 {len(targets)} != {exp_targets}")

        n_rows = 0
        for gene in targets:
            if gene not in gene2seq:
                raise RuntimeError(f"proteins.csv 缺少序列: {gene}")
            g = sub[sub["gene_name"] == gene]
            tname = f"DAVIS_{safe_name(gene)}"
            odir = DST / set_name / "default" / tname
            odir.mkdir(parents=True, exist_ok=True)

            out = pd.DataFrame({
                "smiles": g["standardized_smiles"].values,
                "paffinity": g["paffinity"].values.astype(float),
                "affinity_type": g["affinity_type"].values,
                "gene_name": g["gene_name"].values,
            })
            out.to_csv(odir / f"{tname}_processed_activities.csv",
                       index=False, encoding="utf-8-sig")
            (odir / f"{tname}_processed_protein_sequence.txt").write_text(
                gene2seq[gene], encoding="utf-8")
            n_rows += len(out)

        stats[set_name] = dict(rows=n_rows, targets=len(targets))
        print(f"[build] {set_name}: {n_rows} 行 / {len(targets)} 靶点")

    # 与 GNN 的 Davis test 对齐校验
    gnn = pd.read_csv(GNN_CSV)
    test_acts = acts[acts["set_name"] == "test_set"].copy()
    test_acts["seq"] = test_acts["gene_name"].map(gene2seq)
    test_seqs = set(test_acts["seq"].dropna())
    if len(test_seqs & set(gnn["target_sequence"])) != 65:
        raise RuntimeError("Davis test 序列与 GrapthDTA/davis_test.csv 不一致")

    gmap = {(canon(s), t): float(a) for s, t, a in
            zip(gnn["compound_iso_smiles"], gnn["target_sequence"], gnn["affinity"])}
    pmap = {(canon(s), t): float(a) for s, t, a in
            zip(test_acts["standardized_smiles"], test_acts["seq"], test_acts["paffinity"])}
    if set(gmap) != set(pmap):
        raise RuntimeError(f"test 配对不一致: gnn={len(gmap)} panel={len(pmap)}")
    max_diff = max(abs(gmap[k] - pmap[k]) for k in gmap)
    if max_diff > 1e-5:
        raise RuntimeError(f"paffinity 与 GNN affinity 最大差异 {max_diff}")

    manifest = dict(
        combined=str(COMBINED),
        proteins=str(PROTEINS),
        gnn_test_csv=str(GNN_CSV),
        stats=stats,
        test_pairs=len(gmap),
        max_label_diff=max_diff,
        feature_config="morgan_descriptors",
    )
    (DST / "panel_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[saved]", DST)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    manifest = DST / "panel_manifest.json"
    if manifest.exists() and not args.force:
        print("[skip] 输出已存在:", DST)
        print(manifest.read_text(encoding="utf-8"))
        return
    if DST.exists() and args.force:
        print(f"[clean] 删除 {DST}")
        shutil.rmtree(DST)
    DST.mkdir(parents=True, exist_ok=True)
    build()


if __name__ == "__main__":
    main()
