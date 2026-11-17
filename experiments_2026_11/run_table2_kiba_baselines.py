# -*- coding: utf-8 -*-
"""
Table 2 KIBA 18,478 统一重评估
================================
1. 在 3_kiba_preprocessed（靶点冷启动 70/15/15）上重训 MLP 与 MSE-only Transformer；
2. 用现有 best_model.pt 生成 composite Transformer 的测试集预测；
3. 用同一个评估函数从四组预测重算指标：
   - R²、RMSE、ECE：全局 pooled；
   - EF@1%、AUPR：逐靶点计算后宏平均（与 Reptile 行现有口径一致）。

用法（PyCharm 终端，工作目录 D:\\lht）：
    python run_table2_kiba_baselines.py
    python run_table2_kiba_baselines.py --force          # 忽略已有 final_results，全部重训
    python run_table2_kiba_baselines.py --skip-train     # 只重算指标，不训练
    python run_table2_kiba_baselines.py --resume         # 从 checkpoint 续训

输出：
    D:\\lht\\table2_kiba_rerun\\mlp\\
    D:\\lht\\table2_kiba_rerun\\transformer_mse\\
    D:\\lht\\table2_kiba_rerun\\table2_corrected.csv
    D:\\lht\\table2_kiba_rerun\\table2_summary.txt
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "3_kiba_preprocessed"
FEATURE_SRC = ROOT / "transformer_ablation_output_kiba" / "precomputed_features.npz"
OUT_ROOT = ROOT / "table2_kiba_rerun"
MLP_OUT = OUT_ROOT / "mlp"
MSE_OUT = OUT_ROOT / "transformer_mse"
COMPOSITE_DIR = ROOT / "transformer_ablation_output_kiba" / "morgan_descriptors"
COMPOSITE_BEST = COMPOSITE_DIR / "best_model.pt"
COMPOSITE_PRED = COMPOSITE_DIR / "predictions.npz"
REPTILE_DIR = (ROOT / "修改supplement_output" / "revision2_experiments" / "e5_repeats" / "kiba")

EXPECTED_N = 18478


def env_for_run():
    env = dict(os.environ)
    env["PREPROCESSED_DIR"] = str(DATA_DIR)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    return env


def link_features(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "precomputed_features.npz"
    if dst.exists():
        return dst
    if not FEATURE_SRC.exists():
        raise FileNotFoundError(f"找不到 18,478 对 KIBA 特征文件: {FEATURE_SRC}")
    try:
        os.link(FEATURE_SRC, dst)
        print(f"[link] {dst}")
    except OSError:
        print(f"[copy] {dst}  (硬链接失败，改为复制)")
        shutil.copyfile(FEATURE_SRC, dst)
    return dst


def run_command(cmd):
    print("\n" + "=" * 80)
    print("$", " ".join(str(c) for c in cmd))
    print("=" * 80, flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), env=env_for_run())
    if r.returncode != 0:
        raise SystemExit(f"\n命令失败，returncode={r.returncode}\n{cmd}")
    return r.returncode


def has_results(out_dir: Path):
    return (out_dir / "final_results.json").exists() and (out_dir / "predictions.npz").exists()


def train_script(script: Path, out_dir: Path, extra_args, force=False, resume=False):
    link_features(out_dir)
    if has_results(out_dir) and not force:
        print(f"[skip] 已完成: {out_dir}")
        return
    cmd = [sys.executable, script,
           "--data_dir", DATA_DIR,
           "--output_dir", out_dir,
           "--batch_size", "2048",
           "--epochs", "300",
           "--gpu", "0"] + list(extra_args)
    if force:
        cmd.append("--force")
    elif resume and (out_dir / "checkpoint.pt").exists():
        cmd.append("--resume")
    elif (out_dir / "checkpoint.pt").exists():
        cmd.append("--resume")
    else:
        cmd.append("--force")
    run_command(cmd)


def generate_composite_predictions(force=False):
    if COMPOSITE_PRED.exists() and not force:
        print(f"[skip] 已有 composite 预测: {COMPOSITE_PRED}")
        return
    if not COMPOSITE_BEST.exists():
        raise FileNotFoundError(f"找不到 composite best_model.pt: {COMPOSITE_BEST}")

    os.environ["PREPROCESSED_DIR"] = str(DATA_DIR)
    import torch
    from data_preprocessing import LazySubset, TargetScaler
    from reptile_transformer_model import ReptileTransformer

    print("\n生成 composite Transformer 预测（morgan_descriptors 配置）...")
    data = np.load(FEATURE_SRC, mmap_mode="r", allow_pickle=True)
    splits = data["splits"][:]
    target_names = data["target_names"][:]
    train_idx = np.where(splits == "train")[0]
    test_idx = np.where(splits == "test")[0]
    if len(test_idx) != EXPECTED_N:
        raise RuntimeError(f"测试集样本数 {len(test_idx)} != {EXPECTED_N}")

    scaler = TargetScaler()
    scaler.fit(data["y"][train_idx], target_names[train_idx])
    test_data = LazySubset(data, test_idx, scaler)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ReptileTransformer()
    model.load_state_dict(torch.load(COMPOSITE_BEST, map_location=device))
    model.to(device)
    model.eval()

    ys, ps = [], []
    with torch.no_grad():
        for i in range(0, len(test_data), 2048):
            bi = np.arange(i, min(i + 2048, len(test_data)))
            b = test_data.get_batch(bi)
            morgan = torch.from_numpy(b["morgan"]).to(device)
            maccs = torch.zeros_like(torch.from_numpy(b["maccs"])).to(device)  # 屏蔽 MACCS
            desc = torch.from_numpy(b["descriptors"]).to(device)
            protein = torch.from_numpy(b["protein"]).to(device)
            if torch.cuda.is_available():
                with torch.amp.autocast("cuda"):
                    preds, _, _ = model(morgan, maccs, desc, protein)
            else:
                preds, _, _ = model(morgan, maccs, desc, protein)
            ys.append(np.asarray(b["y"], dtype=np.float64))
            ps.append(scaler.inverse_transform(preds.float().cpu().numpy()))
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    np.savez(COMPOSITE_PRED, y_true=y_true, y_pred=y_pred)
    print(f"[saved] {COMPOSITE_PRED}  N={len(y_true)}")


def load_pred(path: Path):
    d = np.load(path)
    return d["y_true"].astype(np.float64), d["y_pred"].astype(np.float64)


def load_test_targets():
    data = np.load(FEATURE_SRC, mmap_mode="r", allow_pickle=True)
    splits = data["splits"][:]
    target_names = data["target_names"][:]
    test_idx = np.where(splits == "test")[0]
    if len(test_idx) != EXPECTED_N:
        raise RuntimeError(f"测试集样本数 {len(test_idx)} != {EXPECTED_N}")
    return np.asarray(target_names[test_idx])


def ece_global(y_true, y_pred):
    n = len(y_true)
    a = min(float(y_true.min()), float(y_pred.min()))
    c = max(float(y_true.max()), float(y_pred.max()))
    if c == a:
        return 0.0
    yt = (y_true - a) / (c - a)
    yp = (y_pred - a) / (c - a)
    bins = np.digitize(yp, np.linspace(0, 1, 11)[1:-1])
    ece = 0.0
    for b in range(10):
        m = bins == b
        if m.sum() == 0:
            continue
        ece += abs(float(yp[m].mean()) - float(yt[m].mean())) * int(m.sum()) / n
    return float(ece)


def global_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - np.sum((y_true - y_pred) ** 2) / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"R2": float(r2), "RMSE": rmse, "ECE": ece_global(y_true, y_pred)}


def ef_at(y_true, y_pred, pct=1):
    """与 reptile_training / baseline 脚本一致的逐靶点 EF。"""
    n = len(y_true)
    if n < 10:
        return float("nan")
    active_count = max(5, int(n * 0.2))
    thr = np.sort(y_true)[::-1][min(active_count - 1, n - 1)]
    n_top = int(n * pct / 100)
    if n_top == 0:
        return float("nan")
    n_active_top = np.sum(y_true[np.argsort(y_pred)[::-1][:n_top]] >= thr)
    return float((n_active_top / n_top) / (active_count / n))


def aupr_at(y_true, y_pred):
    """逐靶点 average precision，活性=该靶点 top 20%。"""
    n = len(y_true)
    if n < 5:
        return float("nan")
    n_pos = max(1, int(n * 0.2))
    thr = np.sort(y_true)[::-1][min(n_pos - 1, n - 1)]
    is_pos = (y_true >= thr).astype(np.int64)
    n_positive = int(is_pos.sum())
    if n_positive == 0:
        return 0.0
    pos_sorted = is_pos[np.argsort(y_pred)[::-1]]
    tp = np.cumsum(pos_sorted)
    fp = np.cumsum(1 - pos_sorted)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / n_positive
    recall_prev = np.zeros_like(recall)
    recall_prev[1:] = recall[:-1]
    return float(np.sum((recall - recall_prev) * precision))


def macro_metrics(y_true, y_pred, targets):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    targets = np.asarray(targets)
    if len(y_true) != len(targets):
        raise RuntimeError(f"预测数 {len(y_true)} != target 数 {len(targets)}")
    ef_vals, aupr_vals = [], []
    for t in np.unique(targets):
        m = targets == t
        n = int(m.sum())
        if n >= 100:
            v = ef_at(y_true[m], y_pred[m], 1)
            if np.isfinite(v):
                ef_vals.append(v)
        if n >= 5:
            v = aupr_at(y_true[m], y_pred[m])
            if np.isfinite(v):
                aupr_vals.append(v)
    return {
        "EF@1%": float(np.mean(ef_vals)) if ef_vals else float("nan"),
        "AUPR": float(np.mean(aupr_vals)) if aupr_vals else float("nan"),
        "EF_targets": len(ef_vals),
        "AUPR_targets": len(aupr_vals),
    }


def reptile_macro_from_json(path: Path):
    d = json.loads(path.read_text(encoding="utf-8"))
    ef_vals, aupr_vals = [], []
    for t, v in d.items():
        n = int(float(v.get("samples", 0)))
        if n >= 100:
            val = float(v.get("EF@1%", float("nan")))
            if np.isfinite(val):
                ef_vals.append(val)
        if n >= 5:
            val = float(v.get("AUPR", float("nan")))
            if np.isfinite(val):
                aupr_vals.append(val)
    return {
        "EF@1%": float(np.mean(ef_vals)) if ef_vals else float("nan"),
        "AUPR": float(np.mean(aupr_vals)) if aupr_vals else float("nan"),
        "EF_targets": len(ef_vals),
        "AUPR_targets": len(aupr_vals),
    }


def fmt(v, nd=4):
    return f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="忽略已有 final_results，全部重训")
    ap.add_argument("--resume", action="store_true", help="从 checkpoint 续训")
    ap.add_argument("--skip-train", action="store_true", help="只重算指标，不训练")
    ap.add_argument("--regenerate-composite", action="store_true", help="强制重生成 composite 预测")
    args = ap.parse_args()

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"找不到数据目录: {DATA_DIR}")
    if not FEATURE_SRC.exists():
        raise FileNotFoundError(f"找不到 18,478 对特征文件: {FEATURE_SRC}")

    print(f"数据目录: {DATA_DIR}")
    print(f"特征文件: {FEATURE_SRC}")
    print(f"输出目录: {OUT_ROOT}")

    if not args.skip_train:
        train_script(ROOT / "run_baseline_mlp.py", MLP_OUT,
                     ["--ablation", "morgan_descriptors"],
                     force=args.force, resume=args.resume)
        train_script(ROOT / "run_transformer_morgan_mse.py", MSE_OUT,
                     [], force=args.force, resume=args.resume)
        generate_composite_predictions(force=args.regenerate_composite)

    test_targets = load_test_targets()

    paths = {
        "Descriptor MLP": MLP_OUT / "predictions.npz",
        "Transformer, MSE only": MSE_OUT / "predictions.npz",
        "Transformer, composite loss": COMPOSITE_PRED,
    }
    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"{name} 缺少预测文件: {p}\n先运行训练，或检查路径。")

    rows = []
    ref_yt = None
    for name, p in paths.items():
        yt, yp = load_pred(p)
        if len(yt) != EXPECTED_N:
            raise RuntimeError(f"{name}: N={len(yt)} != {EXPECTED_N}")
        if ref_yt is None:
            ref_yt = np.sort(yt)
        elif not np.allclose(np.sort(yt), ref_yt, rtol=0, atol=1e-6):
            raise RuntimeError(f"{name}: 测试集 y_true 与参考不一致，可能不是同一 18,478 对")
        g = global_metrics(yt, yp)
        mm = macro_metrics(yt, yp, test_targets)
        rows.append({"Model": name, **g, **mm, "N": len(yt), "Note": "single run"})
        print(f"[eval] {name}: R2={fmt(g['R2'])} RMSE={fmt(g['RMSE'])} "
              f"AUPR={fmt(mm['AUPR'])} EF@1%={fmt(mm['EF@1%'],3)} ECE={fmt(g['ECE'])} "
              f"(EF targets={mm['EF_targets']}, AUPR targets={mm['AUPR_targets']})")

    # Reptile 五个 seed：全局指标来自 predictions.npz；EF/AUPR 来自 per_target_results.json
    reptile_dirs = sorted([d for d in REPTILE_DIR.glob("seed_*") if d.is_dir()])
    if len(reptile_dirs) != 5:
        raise FileNotFoundError(f"Reptile seed 目录应为 5 个，实际 {len(reptile_dirs)}: {REPTILE_DIR}")
    rep_rows = []
    for d in reptile_dirs:
        pred = d / "predictions.npz"
        pt = d / "per_target_results.json"
        if not pred.exists() or not pt.exists():
            raise FileNotFoundError(f"Reptile 缺少文件: {pred} 或 {pt}")
        yt, yp = load_pred(pred)
        if len(yt) != EXPECTED_N:
            raise RuntimeError(f"{pred}: N={len(yt)} != {EXPECTED_N}")
        if not np.allclose(np.sort(yt), ref_yt, rtol=0, atol=1e-6):
            raise RuntimeError(f"{pred}: 测试集 y_true 与参考不一致")
        g = global_metrics(yt, yp)
        mm = reptile_macro_from_json(pt)
        rep_rows.append({**g, **mm})

    rep_mean = {k: float(np.mean([r[k] for r in rep_rows])) for k in rep_rows[0]}
    rep_sd = {k: float(np.std([r[k] for r in rep_rows], ddof=1)) for k in rep_rows[0]}
    rows.append({
        "Model": "Reptile-Transformer",
        "R2": rep_mean["R2"], "RMSE": rep_mean["RMSE"], "AUPR": rep_mean["AUPR"],
        "EF@1%": rep_mean["EF@1%"], "ECE": rep_mean["ECE"],
        "R2_sd": rep_sd["R2"], "RMSE_sd": rep_sd["RMSE"], "AUPR_sd": rep_sd["AUPR"],
        "EF1_sd": rep_sd["EF@1%"], "ECE_sd": rep_sd["ECE"],
        "N": EXPECTED_N, "Note": "five-seed mean ± SD",
    })
    print(f"[eval] Reptile-Transformer: R2={fmt(rep_mean['R2'])}±{fmt(rep_sd['R2'])} "
          f"RMSE={fmt(rep_mean['RMSE'])}±{fmt(rep_sd['RMSE'])} "
          f"AUPR={fmt(rep_mean['AUPR'])}±{fmt(rep_sd['AUPR'])} "
          f"EF@1%={fmt(rep_mean['EF@1%'],3)}±{fmt(rep_sd['EF@1%'],3)} "
          f"ECE={fmt(rep_mean['ECE'])}±{fmt(rep_sd['ECE'])}")

    lines = []
    lines.append("Table 2. KIBA comparison and architecture controls on the same 18,478 test pairs")
    lines.append("Model | R2 | RMSE | AUPR | EF@1% | ECE")
    for r in rows:
        if r["Model"] == "Reptile-Transformer":
            lines.append(
                f"{r['Model']} | {fmt(r['R2'])} ± {fmt(r['R2_sd'])} | "
                f"{fmt(r['RMSE'])} ± {fmt(r['RMSE_sd'])} | "
                f"{fmt(r['AUPR'])} ± {fmt(r['AUPR_sd'])} | "
                f"{fmt(r['EF@1%'],3)} ± {fmt(r['EF1_sd'],3)} | "
                f"{fmt(r['ECE'])} ± {fmt(r['ECE_sd'])}")
        else:
            lines.append(
                f"{r['Model']} | {fmt(r['R2'])} | {fmt(r['RMSE'])} | "
                f"{fmt(r['AUPR'])} | {fmt(r['EF@1%'],3)} | {fmt(r['ECE'])}")
    lines.append("")
    lines.append("Metric definitions: R2/RMSE/ECE pooled over all test pairs; EF@1%/AUPR macro-averaged over targets")
    lines.append("(EF: targets with n>=100; AUPR: targets with n>=5).")
    lines.append("NOTE: legacy Table 2 mixed global-pooled EF/AUPR (baselines) with macro EF/AUPR (Reptile),")
    lines.append("and raw-scale ECE (composite/MSE-only) with joint-minmax ECE (MLP/Reptile).")
    lines.append("The values below are the first fully common-definition comparison.")
    lines.append("")
    lines.append("Label-variance check: RMSE^2 / (1 - R2)  (should all be ~0.7153 for KIBA 18,478)")
    for r in rows:
        var = r["RMSE"] ** 2 / (1 - r["R2"]) if r["R2"] < 1 else float("nan")
        lines.append(f"  {r['Model']}: {var:.6f}")

    mlp = rows[0]
    comp = rows[2]
    rep = rows[3]
    lines.append("")
    lines.append("Relative changes vs Descriptor MLP:")
    lines.append(f"  R2:   {(rep['R2'] / mlp['R2'] - 1) * 100:+.1f}%")
    lines.append(f"  RMSE: {(rep['RMSE'] / mlp['RMSE'] - 1) * 100:+.1f}%")
    lines.append(f"  AUPR: {(rep['AUPR'] / mlp['AUPR'] - 1) * 100:+.1f}%")
    lines.append(f"  EF@1%:{(rep['EF@1%'] / mlp['EF@1%'] - 1) * 100:+.1f}%")
    lines.append(f"  ECE:  {(rep['ECE'] / mlp['ECE'] - 1) * 100:+.1f}%")
    lines.append("")
    lines.append("Relative changes composite -> Reptile:")
    lines.append(f"  R2:   {(rep['R2'] / comp['R2'] - 1) * 100:+.1f}%")
    lines.append(f"  ECE:  {(rep['ECE'] / comp['ECE'] - 1) * 100:+.1f}%")

    text = "\n".join(lines)
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "table2_summary.txt").write_text(text, encoding="utf-8")

    csv_lines = ["Model,R2,RMSE,AUPR,EF1,ECE,N,Note"]
    for r in rows:
        if r["Model"] == "Reptile-Transformer":
            csv_lines.append(
                f"Reptile-Transformer,{r['R2']:.6f},{r['RMSE']:.6f},{r['AUPR']:.6f},"
                f"{r['EF@1%']:.6f},{r['ECE']:.6f},{r['N']},five-seed mean")
            csv_lines.append(
                f"Reptile-Transformer SD,{r['R2_sd']:.6f},{r['RMSE_sd']:.6f},{r['AUPR_sd']:.6f},"
                f"{r['EF1_sd']:.6f},{r['ECE_sd']:.6f},{r['N']},five-seed SD")
        else:
            csv_lines.append(
                f"{r['Model']},{r['R2']:.6f},{r['RMSE']:.6f},{r['AUPR']:.6f},"
                f"{r['EF@1%']:.6f},{r['ECE']:.6f},{r['N']},single run")
    (OUT_ROOT / "table2_corrected.csv").write_text("\n".join(csv_lines), encoding="utf-8")
    print(f"\n[saved] {OUT_ROOT / 'table2_summary.txt'}")
    print(f"[saved] {OUT_ROOT / 'table2_corrected.csv'}")


if __name__ == "__main__":
    main()
