# -*- coding: utf-8 -*-
"""
ChEMBL GNN 重建 + 重训 + 导出预测（补齐 S18/S19 的 ChEMBL GNN 四格）
=====================================================================
背景：GrapthDTA/model_*_chembl.model、checkpoint_*_chembl.pt、
      data/processed/chembl_{train,val,test}.pt 均为 0 字节，无法直接推理。

本脚本：
  1) --stage data : 从 data/chembl_{train,val,test}.csv 重建 PyG .pt
                    （324,727 个 SMILES 建图；smile_graph 缓存为 pkl）
  2) --stage train: 依次训练 GINConvNet / GATNet / GAT_GCN / GCNNet（GraphDTA 协议），
                    并把 predictions_*_chembl.npz 复制到 gnn_preds/
  3) --stage all  : 先 data 后 train

用法（PyCharm 终端，工作目录 D:\\lht）：
    python run_chembl_gnn_pipeline.py --stage data
    python run_chembl_gnn_pipeline.py --stage train
    python run_chembl_gnn_pipeline.py --stage all
    python run_chembl_gnn_pipeline.py --stage train --only GATNet --force

注意：
- data 阶段内存峰值较高（34GB 内存，建议关闭其他程序）。
- train 阶段每个模型最多 1000 epoch、早停 patience=50；旧结果 best epoch 为 2–29。
- ChEMBL GNN test = 30,180 对 / 26 靶点，与非 GNN 的 31,132 对不同；
  写入 S18/S19 时需保留该 caveat。
"""
import argparse
import gc
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(r"D:\lht")
GD = ROOT / "GrapthDTA"
GNN_DIR = ROOT / "修改supplement_output" / "paper_revision修改V1" / "gnn_preds"
LOG_DIR = GD / "batch_chembl_logs"
RETRAIN_DIR = ROOT / "chembl_gnn_retrain"
PROCESSED = GD / "data" / "processed"
EXPECTED_N = 30180

SPLITS = ["test", "val", "train"]  # 先小后大，便于尽早暴露问题
MODELS = [("GINConvNet", 0), ("GATNet", 1), ("GAT_GCN", 2), ("GCNNet", 3)]

os.chdir(GD)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(GD))

from rdkit import Chem  # noqa: E402
from torch_geometric import data as DATA  # noqa: E402
from utils import TestbedDataset  # noqa: E402

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
_SEQ_CACHE = {}


# ---------------------------------------------------------------- GraphDTA 建图（与 create_data.py 一致）
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return np.array(
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
             'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
             'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
             'Pt', 'Hg', 'Pb', 'Unknown'])
        + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + [atom.GetIsAromatic()]
    )


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return 0, [], []
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        f = atom_features(atom)
        features.append((f / sum(f)).astype(np.float32))
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
    return c_size, features, edge_index


def seq_cat(prot):
    if prot in _SEQ_CACHE:
        return _SEQ_CACHE[prot]
    x = np.zeros(1000, dtype=np.float32)
    for i, ch in enumerate(prot[:1000]):
        if ch in seq_dict:
            x[i] = seq_dict[ch]
    _SEQ_CACHE[prot] = x
    return x


# ---------------------------------------------------------------- data 阶段
def build_smile_graph(force=False):
    cache = GD / "data" / "chembl_smile_graph.pkl"
    if cache.exists() and not force:
        print(f"[cache] 加载 {cache} ({cache.stat().st_size / 1e9:.2f} GB)")
        with open(cache, "rb") as f:
            return pickle.load(f)

    smiles = set()
    for sp in SPLITS:
        df = pd.read_csv(f"data/chembl_{sp}.csv", usecols=["compound_iso_smiles"])
        smiles.update(df["compound_iso_smiles"].values)
        del df
    print(f"[graph] 构建 {len(smiles)} 个 SMILES 的分子图（CPU，预计 20–60 分钟）...")

    sg = {}
    for k, s in enumerate(smiles, 1):
        c, feat, ei = smile_to_graph(s)
        if c > 0 and len(feat) > 0:
            sg[s] = (c, feat, ei)
        if k % 20000 == 0:
            print(f"  {k}/{len(smiles)}", flush=True)
    print(f"[graph] 有效分子图 {len(sg)}")
    with open(cache, "wb") as f:
        pickle.dump(sg, f, protocol=4)
    print(f"[cache] 已保存 {cache} ({cache.stat().st_size / 1e9:.2f} GB)")
    return sg


def build_split(split, sg, force=False):
    out = PROCESSED / f"chembl_{split}.pt"
    if out.exists() and out.stat().st_size > 50_000_000 and not force:
        print(f"[skip] {out.name} 已存在 ({out.stat().st_size / 1e9:.2f} GB)")
        return

    csv = GD / "data" / f"chembl_{split}.csv"
    df = pd.read_csv(csv)
    drugs = df["compound_iso_smiles"].values
    prots = df["target_sequence"].values
    y = df["affinity"].values.astype(np.float32)
    print(f"[build] {split}: {len(df)} 行，编码蛋白序列...", flush=True)
    XT = np.array([seq_cat(t) for t in prots], dtype=np.float32)

    data_list = []
    skipped = 0
    for i in range(len(df)):
        s = drugs[i]
        g = sg.get(s)
        if g is None:
            skipped += 1
            continue
        c_size, features, edge_index = g
        if len(edge_index) == 0:
            ei = torch.zeros((2, 1), dtype=torch.long)
        else:
            ei = torch.LongTensor(edge_index).transpose(1, 0)
        d = DATA.Data(x=torch.Tensor(features), edge_index=ei,
                      y=torch.FloatTensor([float(y[i])]))
        d.target = torch.as_tensor(XT[i], dtype=torch.long).unsqueeze(0)
        d.__setitem__("c_size", torch.LongTensor([c_size]))
        data_list.append(d)
        if (i + 1) % 50000 == 0:
            print(f"  {i + 1}/{len(df)}", flush=True)

    print(f"[collate] {split}: kept={len(data_list)} skipped={skipped}", flush=True)
    data, slices = TestbedDataset.collate(data_list)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    torch.save((data, slices), out)
    print(f"[saved] {out} ({out.stat().st_size / 1e9:.2f} GB)", flush=True)

    del data_list, data, slices, XT, df, drugs, prots, y
    gc.collect()


def stage_data(force=False):
    sg = build_smile_graph(force=force)
    for sp in SPLITS:
        build_split(sp, sg, force=force)
    print("\n[data] 完成。")
    for sp in ["train", "val", "test"]:
        p = PROCESSED / f"chembl_{sp}.pt"
        print(f"  {p.name}: {p.stat().st_size / 1e9:.2f} GB")


# ---------------------------------------------------------------- train 阶段
def env_for_run():
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    return env


def run_stream(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80)
    print("$", " ".join(str(c) for c in cmd))
    print("=" * 80, flush=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        p = subprocess.Popen(
            [str(c) for c in cmd], cwd=str(GD), env=env_for_run(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace")
        for line in p.stdout:
            print(line, end="")
            log.write(line)
        p.wait()
    return p.returncode


def validate_predictions(pred_path):
    d = np.load(pred_path)
    yt = d["y_true"].astype(np.float64)
    yp = d["y_pred"].astype(np.float64)
    ref = pd.read_csv(GD / "data" / "chembl_test.csv")
    if len(yt) != len(ref):
        raise RuntimeError(f"{pred_path.name}: N={len(yt)} != {len(ref)}")
    max_diff = float(np.max(np.abs(yt - ref["affinity"].values.astype(np.float64))))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    ss_res = float(np.sum((yt - yp) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))

    from recompute_s18_s19 import macro_ef
    targets = ref["target_sequence"].values
    ef1, used, total = macro_ef(yt, yp, targets, 1, 100)
    return dict(n=len(yt), max_label_diff=max_diff, R2=r2, RMSE=rmse,
                EF1=ef1, n_used=used, n_total=total)


def check_pt_files():
    for sp in ["train", "val", "test"]:
        p = PROCESSED / f"chembl_{sp}.pt"
        if not p.exists() or p.stat().st_size < 50_000_000:
            raise RuntimeError(
                f"{p} 缺失或仍为 0 字节。请先运行："
                f"python run_chembl_gnn_pipeline.py --stage data")


def stage_train(only=None, force=False):
    check_pt_files()
    GNN_DIR.mkdir(parents=True, exist_ok=True)
    RETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    only = set(only) if only else None

    for model, idx in MODELS:
        if only and model not in only:
            continue
        dst = GNN_DIR / f"{model}_chembl.npz"
        if dst.exists() and not force:
            try:
                if len(np.load(dst)["y_true"]) == EXPECTED_N:
                    print(f"[skip] {dst.name} 已存在且 N={EXPECTED_N}")
                    continue
            except Exception:
                pass

        # 清掉 0 字节或旧产物，避免 training.py 末尾加载空 model 文件时报 EOFError
        for fn in [f"model_{model}_chembl.model", f"checkpoint_{model}_chembl.pt",
                   f"result_{model}_chembl.json", f"predictions_{model}_chembl.npz"]:
            p = GD / fn
            if p.exists():
                p.unlink()
                print(f"[clean] {fn}")

        log = LOG_DIR / f"chembl_{model}_{time.strftime('%Y%m%d_%H%M%S')}.log"
        t0 = time.time()
        rc = run_stream([sys.executable, "training.py", "0", str(idx), "0"], log)
        wall = time.time() - t0
        if rc != 0:
            raise SystemExit(f"{model} 训练失败 returncode={rc}，日志：{log}")

        pred = GD / f"predictions_{model}_chembl.npz"
        if not pred.exists():
            raise RuntimeError(f"{model} 训练结束但没有 {pred}")
        info = validate_predictions(pred)
        if info["max_label_diff"] > 1e-3:
            raise RuntimeError(f"{model} y_true 与 chembl_test.csv 不一致 "
                               f"(max diff={info['max_label_diff']})")
        shutil.copyfile(pred, dst)

        best_epoch = None
        rj = GD / f"result_{model}_chembl.json"
        if rj.exists():
            with open(rj, encoding="utf-8") as f:
                best_epoch = json.load(f).get("best_epoch")

        rows.append(dict(model=model, n=info["n"], R2=info["R2"], RMSE=info["RMSE"],
                         EF1=info["EF1"], n_used=info["n_used"], n_total=info["n_total"],
                         best_epoch=best_epoch, walltime_s=wall, log=str(log)))
        print(f"[done] {model}: R2={info['R2']:.4f} RMSE={info['RMSE']:.4f} "
              f"EF@1%={info['EF1']:.3f} ({info['n_used']}/{info['n_total']} targets) "
              f"wall={wall / 60:.1f} min", flush=True)

    if rows:
        csv = RETRAIN_DIR / "summary.csv"
        pd.DataFrame(rows).to_csv(csv, index=False, encoding="utf-8-sig")
        print("\n[saved]", csv)

        lines = ["# ChEMBL GNN 重训汇总", ""]
        lines.append("ChEMBL GraphDTA split: test 30,180 对 / 26 靶点；"
                     "与非 GNN 的 31,132 对不是同一测试集。")
        lines.append("")
        for r in rows:
            lines.append(f"- {r['model']}: R2={r['R2']:.4f}, RMSE={r['RMSE']:.4f}, "
                         f"EF@1%={r['EF1']:.3f} ({r['n_used']}/{r['n_total']}), "
                         f"best_epoch={r['best_epoch']}, wall={r['walltime_s'] / 60:.1f} min")
        (RETRAIN_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
        print("[saved]", RETRAIN_DIR / "REPORT.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["data", "train", "all"], default="all")
    ap.add_argument("--only", default=None, help="只训练指定模型，逗号分隔")
    ap.add_argument("--force", action="store_true", help="强制重跑/覆盖")
    args = ap.parse_args()

    only = [s.strip() for s in args.only.split(",")] if args.only else None
    if args.stage in ("data", "all"):
        stage_data(force=args.force)
    if args.stage in ("train", "all"):
        stage_train(only=only, force=args.force)


if __name__ == "__main__":
    main()
