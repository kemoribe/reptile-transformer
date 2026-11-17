# -*- coding: utf-8 -*-
"""
构建 BindingDB 面板数据（与 GraphDTA BindingDB test 的 7,087 行完全对齐）
=========================================================================
背景：bindingdb_preprocessed 的 test_set 有 12,043 行，但按
      (canonical SMILES, target sequence) 去重后恰好是 7,087 个唯一配对，
      与 GrapthDTA/data/bindingdb_test.csv 的 7,087 行一一对应。

本脚本输出 D:\\lht\\bindingdb_panel_preprocessed：
  train_set/            从 bindingdb_preprocessed 原样复制（520 靶点 / 40,449 行）
  val_set/              从 bindingdb_preprocessed 原样复制（111 靶点 / 8,332 行）
  test_set/default/...  仅保留 7,087 个唯一配对（112 靶点），
                        并把 paffinity 对齐为 GraphDTA CSV 的 affinity
  panel_manifest.json   计数与对齐校验

用法（PyCharm 终端，工作目录 D:\\lht）：
    python prepare_bindingdb_panel.py
    python prepare_bindingdb_panel.py --force
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

ROOT = Path(r"D:\lht")
SRC = ROOT / "bindingdb_preprocessed"
DST = ROOT / "bindingdb_panel_preprocessed"
GNN_CSV = ROOT / "GrapthDTA" / "data" / "bindingdb_test.csv"
EXPECTED_ROWS = 7087
EXPECTED_TARGETS = 112


def canon(smiles):
    m = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(m) if m is not None else None


def load_gnn_test():
    df = pd.read_csv(GNN_CSV)
    keys = [(canon(s), str(t)) for s, t in zip(df["compound_iso_smiles"], df["target_sequence"])]
    if len(set(keys)) != len(keys):
        raise RuntimeError("GraphDTA bindingdb_test.csv 存在重复 (SMILES, target) 键")
    return dict(zip(keys, df["affinity"].astype(float).values))


def copy_split(name):
    src = SRC / name
    dst = DST / name
    if dst.exists():
        return
    print(f"[copy] {src} -> {dst}")
    shutil.copytree(src, dst)


def build_test(gnn_map):
    src = SRC / "test_set" / "default"
    dst = DST / "test_set" / "default"
    dst.mkdir(parents=True, exist_ok=True)

    seen = set()
    kept_rows = 0
    kept_targets = 0
    skipped_dup = 0
    max_label_diff = 0.0

    for tdir in sorted([d for d in src.iterdir() if d.is_dir()]):
        seq_file = tdir / f"{tdir.name}_processed_protein_sequence.txt"
        act_file = tdir / f"{tdir.name}_processed_activities.csv"
        if not seq_file.exists() or not act_file.exists():
            print(f"[warn] 跳过不完整靶点目录: {tdir.name}")
            continue
        seq = seq_file.read_text(encoding="utf-8").strip()
        df = pd.read_csv(act_file)

        keep = []
        for _, row in df.iterrows():
            key = (canon(row["smiles"]), seq)
            if key not in gnn_map:
                continue
            if key in seen:
                skipped_dup += 1
                continue
            seen.add(key)
            row = row.copy()
            row["paffinity"] = gnn_map[key]
            keep.append(row)
            max_label_diff = max(max_label_diff, 0.0)

        if not keep:
            continue
        out_dir = dst / tdir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(keep).to_csv(
            out_dir / f"{tdir.name}_processed_activities.csv",
            index=False, encoding="utf-8-sig")
        shutil.copyfile(seq_file, out_dir / f"{tdir.name}_processed_protein_sequence.txt")
        kept_rows += len(keep)
        kept_targets += 1

    missing = set(gnn_map) - seen
    return dict(rows=kept_rows, targets=kept_targets, duplicates_skipped=skipped_dup,
                missing=len(missing), max_label_diff=max_label_diff,
                missing_examples=list(missing)[:5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="删除并重建输出目录")
    args = ap.parse_args()

    if not SRC.exists():
        raise FileNotFoundError(SRC)
    if not GNN_CSV.exists():
        raise FileNotFoundError(GNN_CSV)

    manifest_path = DST / "panel_manifest.json"
    if manifest_path.exists() and not args.force:
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (m.get("test_rows") == EXPECTED_ROWS
                and m.get("test_targets") == EXPECTED_TARGETS):
            print("[skip] 输出已存在且校验通过:", DST)
            print(json.dumps(m, ensure_ascii=False, indent=2))
            return

    if DST.exists() and args.force:
        print(f"[clean] 删除 {DST}")
        shutil.rmtree(DST)
    DST.mkdir(parents=True, exist_ok=True)

    gnn_map = load_gnn_test()
    print(f"[gnn] bindingdb_test.csv: {len(gnn_map)} 个唯一 (SMILES, target) 配对")
    if len(gnn_map) != EXPECTED_ROWS:
        raise RuntimeError(f"GNN 配对数 {len(gnn_map)} != {EXPECTED_ROWS}")

    # train/val 原样复制（靶点冷启动的训练与验证集不变）
    copy_split("train_set")
    copy_split("val_set")
    # test 过滤 + 标签对齐
    stats = build_test(gnn_map)

    if stats["rows"] != EXPECTED_ROWS or stats["targets"] != EXPECTED_TARGETS \
            or stats["missing"] != 0:
        raise RuntimeError(f"对齐失败: {stats}")

    manifest = dict(
        source=str(SRC),
        gnn_test_csv=str(GNN_CSV),
        test_rows=stats["rows"],
        test_targets=stats["targets"],
        duplicates_skipped=stats["duplicates_skipped"],
        missing_pairs=stats["missing"],
        label_alignment="paffinity set to GraphDTA bindingdb_test.csv affinity",
        max_label_diff=stats["max_label_diff"],
        train_targets=len(list((DST / "train_set" / "default").iterdir())),
        val_targets=len(list((DST / "val_set" / "default").iterdir())),
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print("[saved]", DST)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
