# -*- coding: utf-8 -*-
"""
汇总 2026-11-13 已完成实验，输出一个多工作表 Excel。
输出：D:\\lht\\today_experiment_results.xlsx
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from scipy import stats as sstats

ROOT = Path(r"D:\lht")
sys.path.insert(0, str(ROOT))
import run_table2_kiba_baselines as tb  # noqa
import run_panel_chembl as pc  # noqa

OUT = ROOT / "today_experiment_results.xlsx"
KIBA_FEAT = ROOT / "transformer_ablation_output_kiba" / "precomputed_features.npz"
CHEMBL_FEAT = ROOT / "transformer_ablation_output_chembl" / "precomputed_features.npz"

HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
TITLE_FILL = PatternFill("solid", fgColor="DDEBF7")
ALT_FILL = PatternFill("solid", fgColor="F2F2F2")
GOOD_FILL = PatternFill("solid", fgColor="E2EFDA")
WARN_FILL = PatternFill("solid", fgColor="FFF2CC")
BAD_FILL = PatternFill("solid", fgColor="FCE4D6")
WHITE_BOLD = Font(color="FFFFFF", bold=True)
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def agg(rows, keys=("R2", "RMSE", "AUPR", "EF@1%", "ECE")):
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows if r.get(k) is not None and np.isfinite(r[k])], float)
        out[f"{k}_mean"] = float(vals.mean()) if len(vals) else np.nan
        out[f"{k}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
    out["n_runs"] = len(rows)
    return out


def agg_family(df, keys=("R2", "RMSE", "EF1", "EF5")):
    rows = []
    for fam, g in df.groupby("family", sort=False):
        row = {"Model": fam, "Runs": len(g)}
        for k in keys:
            vals = g[k].astype(float)
            row[f"{k}_mean"] = float(vals.mean())
            row[f"{k}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def eval_pred(path, targets):
    yt, yp = tb.load_pred(path)
    return {**tb.global_metrics(yt, yp), **tb.macro_metrics(yt, yp, targets)}


def write_table(ws, df, title=None, note=None, start_row=1, pct_cols=(), pp_cols=(), num4=(), num3=()):
    r = start_row
    if title:
        ws.cell(row=r, column=1, value=title).font = Font(bold=True, size=13, color="1F4E78")
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=max(len(df.columns), 1))
        ws.cell(row=r, column=1).fill = TITLE_FILL
        r += 2
    header_row = r
    for j, col in enumerate(df.columns, 1):
        c = ws.cell(row=r, column=j, value=str(col))
        c.fill = HEADER_FILL
        c.font = WHITE_BOLD
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = BORDER
    r += 1
    for i, (_, row) in enumerate(df.iterrows()):
        for j, col in enumerate(df.columns, 1):
            val = row[col]
            if isinstance(val, (np.floating, float)) and (pd.isna(val)):
                val = ""
            c = ws.cell(row=r, column=j, value=val)
            c.border = BORDER
            c.alignment = Alignment(vertical="center", wrap_text=False)
            if i % 2 == 1:
                c.fill = ALT_FILL
            if col in num4:
                c.number_format = "0.0000"
            elif col in num3:
                c.number_format = "0.000"
            elif col in pct_cols and isinstance(val, (int, float)):
                c.number_format = "0.0%"
            elif col in pp_cols and isinstance(val, (int, float)):
                c.number_format = "+0.0;-0.0"
        r += 1
    if note:
        ws.cell(row=r + 1, column=1, value=note).font = Font(italic=True, size=9, color="7F7F7F")
        ws.merge_cells(start_row=r + 1, start_column=1, end_row=r + 1, end_column=max(len(df.columns), 1))
    ws.freeze_panes = ws.cell(row=header_row + 1, column=1)
    for j, col in enumerate(df.columns, 1):
        width = max(10, min(46, max(len(str(col)), max([len(str(x)) for x in df[col].head(30)] + [0]) + 2)))
        ws.column_dimensions[get_column_letter(j)].width = width
    return r + 2


def main():
    wb = Workbook()
    wb.remove(wb.active)

    # ---------- 00 总览 ----------
    ws = wb.create_sheet("00_总览")
    status = pd.DataFrame([
        ["Table 2 / KIBA", "已完成", "18,478 对统一重评；MLP 5 runs、Reptile 5 seeds",
         "Reptile 回归略低/持平，AUPR/EF 不占优；旧表提升百分比作废",
         "D:\\lht\\table2_kiba_rerun"],
        ["ChEMBL panel screen", "已完成", "morgan_descriptors + full 两套配置",
         "full: Reptile 赢 R²/RMSE/ECE、输 AUPR/EF；morgan: 只赢 ECE",
         "D:\\lht\\panel_screen"],
        ["ChEMBL 5-seed (morgan)", "已完成", "batch 2048/220；5 seeds；MACCS 正确屏蔽后 raw-scale 重评",
         "R² 0.2136±0.0112；MLP 0.2178±0.0489；Reptile 仅 ECE 更低",
         "D:\\lht\\chembl_morgan_5seeds_b2048"],
        ["S23–S30 重算", "已完成", "预测已存在，统一口径重算",
         "S24/S27/S28/S29/S30 与旧表一致；S23 ChEMBL R² 显著下调",
         "D:\\lht\\table_audit\\recomputed"],
        ["S8/S9 逐靶点归属", "已完成", "GNN/Davis/BindingDB 精确复现；KIBA Reptile 行不可用",
         "KIBA Reptile split 与审计 split 只有 3/35 UniProt 重叠；S9 KIBA 改为 4 GNN",
         "D:\\lht\\table_audit\\s8_s9"],
        ["镜像阈值 + 置换", "已完成", "≥3/5/10/20；B=1000；保持缺失结构",
         "≥10 主分析成立；≥3 被小样本 null 高估；≥20 稳健但 coverage 下降",
         "D:\\lht\\mirror_threshold_permutation"],
        ["模块删除消融（两族）", "已完成", "3 seeds；监督族 + 完整 Reptile 族；KIBA 18,478",
         "完整 Reptile 中 ranking/contrastive 无可检测贡献；consistency 小幅影响回归/富集；Reptile 本身是主贡献",
         "D:\\lht\\module_ablation + reptile_module_ablation"],
        ["BindingDB 面板（morgan）", "已完成", "7,087 对 / 112 靶点；MLP/Transformer/Reptile 各 5 次",
         "Transformer R² 最高（0.262），MLP EF@1% 最高（3.96），Reptile R² 最低（0.153）但 EF@5% 略高（3.75）",
         "D:\\lht\\bindingdb_panel_morgan"],
        ["Davis 面板（morgan）", "已完成", "4,290 对 / 65 靶点；各 5 次",
         "Transformer R² 0.321 > MLP 0.291 > Reptile 0.223；EF@1% Transformer 4.24 最高",
         "D:\\lht\\davis_panel_morgan"],
        ["ChEMBL GNN 重训", "已完成", "4 个 GNN；30,180 对 / 26 靶点",
         "GNN R² -0.05–0.09、EF@1% 1.29–1.60，明显低于非 GNN（R²≈0.21）",
         "D:\\lht\\chembl_gnn_retrain"],
        ["S18/S19 EF 重算", "已完成", "7 模型 × 4 数据集 = 28 格，无 NA",
         "非 GNN 全部 morgan_descriptors；GNN 按 GraphDTA split 标注；Davis EF@1% pooled",
         "D:\\lht\\s18_s19_recomputed"],
        ["6.1 预算匹配对照", "已完成",
         "KIBA；full vs direct_supervised 各 5 seeds；220 epoch / 105,600 inner updates / 55.9M samples",
         "仅去掉 Reptile 插值：ΔR² -0.119（p=0.015）、ΔAUPR -0.018（p=0.025）、ΔECE +0.005（p=0.040）；EF 不显著 → 增益来自 Reptile 更新而非预算",
         "D:\\lht\\exp61_analysis + control_budget_matched"],
        ["审计第三轴（配体骨架覆盖）", "已完成",
         "7 数据集；Murcko exact/generic；compound-level",
         "KIBA/Davis/Metz 100% 覆盖、novel=0；BindingDB 35.5%、Papyrus 53.2%、PDBbind 57.6%、ChEMBL 71.3%",
         "D:\\lht\\ligand_scaffold_axis"],
        ["第三轴 × 性能", "已完成",
         "主 4 数据集；full Reptile 5 seeds；逐靶点指标按 seed 平均；偏 Spearman 控制 n；B=2000",
         "BindingDB exact coverage vs R² ρ=0.605（CI 0.47–0.72）、EF5 ρ=0.579；ChEMBL R² ρ=0.383、EF5 ρ=0.413；KIBA/Davis 饱和无法检验",
         "D:\\lht\\scaffold_performance"],
    ], columns=["实验", "状态", "口径", "关键结论", "输出目录"])
    write_table(ws, status, title="2026-11-13 实验总览")
    for row in ws.iter_rows(min_row=4, max_row=3 + len(status), min_col=2, max_col=2):
        for c in row:
            c.fill = GOOD_FILL if c.value == "已完成" else (BAD_FILL if "失败" in str(c.value) else WARN_FILL)

    # ---------- 01 Table 2 ----------
    ws = wb.create_sheet("01_Table2_KIBA")
    tsplits = np.load(KIBA_FEAT, mmap_mode="r", allow_pickle=True)["splits"][:]
    tnames = np.load(KIBA_FEAT, mmap_mode="r", allow_pickle=True)["target_names"][:]
    kiba_targets = tnames[tsplits == "test"]
    t2 = ROOT / "table2_kiba_rerun"
    mlp_paths = [t2 / "mlp" / "predictions.npz"] + [t2 / f"mlp_run{i}" / "predictions.npz" for i in range(2, 6)]
    mlp_rows = [eval_pred(p, kiba_targets) for p in mlp_paths if p.exists()]
    mse_row = eval_pred(t2 / "transformer_mse" / "predictions.npz", kiba_targets)
    comp_row = eval_pred(ROOT / "transformer_ablation_output_kiba" / "morgan_descriptors" / "predictions.npz", kiba_targets)
    rep_rows = []
    for s in [11, 22, 33, 44, 55]:
        d = ROOT / "修改supplement_output" / "revision2_experiments" / "e5_repeats" / "kiba" / f"seed_{s}"
        if (d / "predictions.npz").exists():
            rep_rows.append({**tb.global_metrics(*tb.load_pred(d / "predictions.npz")),
                             **tb.reptile_macro_from_json(d / "per_target_results.json")})
    rows = []
    a = agg(mlp_rows); rows.append(["Descriptor MLP", a["R2_mean"], a["R2_sd"], a["RMSE_mean"], a["RMSE_sd"],
                                   a["AUPR_mean"], a["AUPR_sd"], a["EF@1%_mean"], a["EF@1%_sd"], a["ECE_mean"], a["ECE_sd"], a["n_runs"]])
    rows.append(["Transformer, MSE only", mse_row["R2"], np.nan, mse_row["RMSE"], np.nan, mse_row["AUPR"], np.nan,
                 mse_row["EF@1%"], np.nan, mse_row["ECE"], np.nan, 1])
    rows.append(["Transformer, composite", comp_row["R2"], np.nan, comp_row["RMSE"], np.nan, comp_row["AUPR"], np.nan,
                 comp_row["EF@1%"], np.nan, comp_row["ECE"], np.nan, 1])
    a = agg(rep_rows); rows.append(["Reptile-Transformer", a["R2_mean"], a["R2_sd"], a["RMSE_mean"], a["RMSE_sd"],
                                   a["AUPR_mean"], a["AUPR_sd"], a["EF@1%_mean"], a["EF@1%_sd"], a["ECE_mean"], a["ECE_sd"], a["n_runs"]])
    t2df = pd.DataFrame(rows, columns=["Model", "R2_mean", "R2_sd", "RMSE_mean", "RMSE_sd", "AUPR_mean", "AUPR_sd",
                                       "EF_mean", "EF_sd", "ECE_mean", "ECE_sd", "Runs"])
    write_table(ws, t2df, title="Table 2. KIBA 18,478 统一口径（morgan_descriptors）",
                note="R2/RMSE/ECE 全局 pooled；EF@1%/AUPR 逐靶点 macro。MLP 为 5 次独立运行，Reptile 为 5 seeds；其余单次。",
                num4=("R2_mean", "R2_sd", "RMSE_mean", "RMSE_sd", "AUPR_mean", "AUPR_sd", "ECE_mean", "ECE_sd"),
                num3=("EF_mean", "EF_sd"))
    rel = pd.DataFrame([
        ["MLP → Reptile", 100 * (a["R2_mean"] / t2df.iloc[0]["R2_mean"] - 1), 100 * (a["RMSE_mean"] / t2df.iloc[0]["RMSE_mean"] - 1),
         100 * (a["AUPR_mean"] / t2df.iloc[0]["AUPR_mean"] - 1), 100 * (a["EF@1%_mean"] / t2df.iloc[0]["EF_mean"] - 1),
         100 * (a["ECE_mean"] / t2df.iloc[0]["ECE_mean"] - 1)],
        ["composite → Reptile", 100 * (a["R2_mean"] / comp_row["R2"] - 1), 100 * (a["RMSE_mean"] / comp_row["RMSE"] - 1),
         100 * (a["AUPR_mean"] / comp_row["AUPR"] - 1), 100 * (a["EF@1%_mean"] / comp_row["EF@1%"] - 1),
         100 * (a["ECE_mean"] / comp_row["ECE"] - 1)],
    ], columns=["Comparison", "ΔR2 (%)", "ΔRMSE (%)", "ΔAUPR (%)", "ΔEF (%)", "ΔECE (%)"])
    write_table(ws, rel, title="相对变化（按显示均值计算）", start_row=ws.max_row + 2, num3=("ΔR2 (%)", "ΔRMSE (%)", "ΔAUPR (%)", "ΔEF (%)", "ΔECE (%)"))

    stat_rows = []
    for k in ["R2", "RMSE", "AUPR", "EF@1%", "ECE"]:
        a = np.array([r[k] for r in mlp_rows], float)
        b = np.array([r[k] for r in rep_rows], float)
        t, p = sstats.ttest_ind(a, b, equal_var=False)
        stat_rows.append([k, a.mean(), b.mean(), a.mean() - b.mean(), t, p])
    stats_df = pd.DataFrame(stat_rows, columns=["Metric", "MLP_mean", "Reptile_mean", "Diff (MLP-Reptile)", "Welch_t", "p_value"])
    write_table(wb.create_sheet("16_Table2_Stats"), stats_df,
                title="Table 2 统计比较：MLP 5 runs vs Reptile 5 seeds（Welch t-test）",
                num4=("MLP_mean", "Reptile_mean", "Diff (MLP-Reptile)", "p_value"), num3=("Welch_t",))

    # ---------- 02 ChEMBL panel ----------
    ws = wb.create_sheet("02_ChEMBL_Panel")
    # morgan
    cm = ROOT / "panel_screen" / "chembl_morgan_descriptors"
    morgan_targets = pc.load_targets()
    mlp_m = [pc.load_pred(cm / f"mlp_run{i}" / "predictions.npz") for i in range(1, 6)]
    mlp_m_rows = []
    for yt, yp in mlp_m:
        mlp_m_rows.append({**pc.global_metrics(yt, yp), **pc.macro_metrics(yt, yp, morgan_targets)})
    comp_m = pc.load_pred(cm / "composite_raw.npz")
    comp_m_row = {**pc.global_metrics(*comp_m), **pc.macro_metrics(*comp_m, morgan_targets)}
    rep_m_rows = []
    for s in [11, 22, 33, 44, 55]:
        p = ROOT / "chembl_morgan_5seeds_b2048" / f"seed_{s}" / "predictions_raw.npz"
        if p.exists():
            yt, yp = pc.load_pred(p)
            rep_m_rows.append({**pc.global_metrics(yt, yp), **pc.macro_metrics(yt, yp, morgan_targets)})
    # full
    cf = ROOT / "panel_screen" / "chembl"
    full_targets = tnames_full = np.load(CHEMBL_FEAT, mmap_mode="r", allow_pickle=True)["target_names"][:]
    full_splits = np.load(CHEMBL_FEAT, mmap_mode="r", allow_pickle=True)["splits"][:]
    full_targets = full_targets[full_splits == "test"]
    mlp_f_rows = [eval_pred(cf / f"mlp_run{i}" / "predictions.npz", full_targets) for i in range(1, 6)]
    comp_f_row = eval_pred(cf / "composite_raw.npz", full_targets)
    rep_f_rows = []
    for s in [11, 22, 33, 44, 55]:
        p = ROOT / "修改supplement_output" / "revision2_experiments" / "e5_repeats" / "chembl" / f"seed_{s}" / "predictions_raw.npz"
        if p.exists():
            rep_f_rows.append(eval_pred(p, full_targets))
    rows = []
    for label, a, n in [
        ("morgan_descriptors / MLP", agg(mlp_m_rows), len(mlp_m_rows)),
        ("morgan_descriptors / composite", agg([comp_m_row]), 1),
        ("morgan_descriptors / Reptile", agg(rep_m_rows), len(rep_m_rows)),
        ("full / MLP", agg(mlp_f_rows), len(mlp_f_rows)),
        ("full / composite", agg([comp_f_row]), 1),
        ("full / Reptile", agg(rep_f_rows), len(rep_f_rows)),
    ]:
        rows.append([label, a["R2_mean"], a["R2_sd"], a["RMSE_mean"], a["RMSE_sd"], a["AUPR_mean"], a["AUPR_sd"],
                     a["EF@1%_mean"], a["EF@1%_sd"], a["ECE_mean"], a["ECE_sd"], n])
    cdf = pd.DataFrame(rows, columns=["Setting / Model", "R2_mean", "R2_sd", "RMSE_mean", "RMSE_sd", "AUPR_mean", "AUPR_sd",
                                      "EF_mean", "EF_sd", "ECE_mean", "ECE_sd", "Runs"])
    write_table(ws, cdf, title="ChEMBL panel: morgan_descriptors vs full（31,132 测试对）",
                note="morgan_descriptors 与 full 的 Reptile 均为 5 seeds。EF@1%/AUPR 为逐靶点 macro。",
                num4=("R2_mean", "R2_sd", "RMSE_mean", "RMSE_sd", "AUPR_mean", "AUPR_sd", "ECE_mean", "ECE_sd"),
                num3=("EF_mean", "EF_sd"))
    cstat = []
    for k in ["R2", "RMSE", "AUPR", "EF@1%", "ECE"]:
        a = np.array([r[k] for r in mlp_m_rows], float)
        b = np.array([r[k] for r in rep_m_rows], float)
        t, p = sstats.ttest_ind(a, b, equal_var=False)
        cstat.append([k, a.mean(), b.mean(), a.mean() - b.mean(), t, p])
    write_table(wb.create_sheet("18_ChEMBL_Stats"),
                pd.DataFrame(cstat, columns=["Metric", "MLP_mean", "Reptile_mean", "Diff (MLP-Reptile)", "Welch_t", "p_value"]),
                title="ChEMBL morgan_descriptors：MLP 5 runs vs Reptile 5 seeds（Welch t-test）",
                num4=("MLP_mean", "Reptile_mean", "Diff (MLP-Reptile)", "p_value"), num3=("Welch_t",))

    # ---------- S tables ----------
    s23 = pd.read_csv(ROOT / "table_audit" / "recomputed" / "S23_chembl_lambda.csv")
    write_table(wb.create_sheet("03_S23_ChEMBL_lambda"), s23, title="S23. ChEMBL λ sensitivity（统一 raw scale）",
                num4=("R2", "RMSE", "MAE", "ECE", "AUPR"), num3=("EF",))
    s24 = pd.read_csv(ROOT / "table_audit" / "recomputed" / "S24_davis_kiba_lambda.csv")
    write_table(wb.create_sheet("04_S24_Davis_KIBA_lambda"), s24, title="S24. Davis / KIBA λ sensitivity",
                num4=("R2", "RMSE", "MAE", "ECE", "AUPR"), num3=("EF",))
    s27 = pd.read_csv(ROOT / "table_audit" / "recomputed" / "S27_esm2_freeze.csv")
    write_table(wb.create_sheet("05_S27_ESM2"), s27, title="S27. ESM2 freeze ablation",
                num4=("R2", "RMSE", "MAE", "ECE", "AUPR"), num3=("EF",))
    s28 = pd.read_csv(ROOT / "table_audit" / "recomputed" / "S28_encoder_capacity.csv")
    write_table(wb.create_sheet("06_S28_Encoder"), s28, title="S28. Protein encoder capacity")
    s29 = pd.read_csv(ROOT / "table_audit" / "recomputed" / "S29_inner_steps.csv")
    write_table(wb.create_sheet("07_S29_InnerSteps"), s29, title="S29. Inner steps sweep",
                num4=("R2", "RMSE", "MAE", "ECE", "AUPR"), num3=("EF",))
    s30 = pd.read_csv(ROOT / "table_audit" / "recomputed" / "S30_maml_vs_reptile.csv")
    write_table(wb.create_sheet("08_S30_MAML"), s30, title="S30. MAML vs Reptile",
                num4=("R2", "RMSE", "MAE", "ECE", "AUPR"), num3=("EF",))
    s8 = pd.read_csv(ROOT / "table_audit" / "s8_s9" / "S8_recomputed.csv")
    write_table(wb.create_sheet("09_S8_HomologyFree"), s8, title="S8. Homology-free subset ablation（KIBA Reptile 行不可用）",
                num4=("full_R2", "full_RMSE", "full_MAE", "full_ECE", "sub_R2", "sub_RMSE", "sub_MAE", "sub_ECE"),
                num3=("full_EF@1%", "full_EF@5%", "full_EF@10%", "sub_EF@1%", "sub_EF@5%", "sub_EF@10%"))
    s9 = pd.read_csv(ROOT / "table_audit" / "s8_s9" / "S9_recomputed.csv")
    write_table(wb.create_sheet("10_S9_Audited"), s9, title="S9. Audited vs unaudited winner consistency",
                num4=("Kendall_tau", "Spearman_rho", "mis_selection_cost", "relative_cost_pct"))

    # ---------- Mirror ----------
    ms = pd.read_csv(ROOT / "mirror_threshold_permutation" / "summary.csv")
    ms2 = ms.copy()
    for c in ["coverage", "observed_rate", "null_rate_mean", "delta", "delta_ci_low", "delta_ci_high"]:
        ms2[c] = ms2[c].astype(float)
    ms2["delta_pp"] = ms2["delta"] * 100
    ms2["delta_ci_low_pp"] = ms2["delta_ci_low"] * 100
    ms2["delta_ci_high_pp"] = ms2["delta_ci_high"] * 100
    write_table(wb.create_sheet("11_Mirror_Threshold"), ms2,
                title="镜像审计：共享化合物阈值敏感性 + 置换对照（B=1000）",
                note="observed/null/delta 为比例；delta_pp 为百分点。CI 为 target+permutation bootstrap 95%。",
                pct_cols=("coverage", "observed_rate", "null_rate_mean"),
                num3=("delta_pp", "delta_ci_low_pp", "delta_ci_high_pp", "perm_p"))
    key = ms[(ms["threshold"] == 10)].copy()
    key["observed"] = key["n_mirror"].astype(str) + "/" + key["n_evaluable"].astype(str)
    key["observed_pct"] = key["observed_rate"] * 100
    key["null_pct"] = key["null_rate_mean"] * 100
    key["delta_pp"] = key["delta"] * 100
    key["ci"] = key["delta_ci_low"].map(lambda x: f"{100*x:+.1f}") + " to " + key["delta_ci_high"].map(lambda x: f"{100*x:+.1f}")
    key = key[["dataset", "stratum", "observed", "observed_pct", "null_pct", "delta_pp", "ci", "perm_p", "coverage"]]
    write_table(wb.create_sheet("12_Mirror_Key"), key,
                title="镜像审计主结果（≥10 共享化合物）",
                note="coverage = 可评估靶点 / 该数据集全部测试靶点。",
                pct_cols=("coverage",), num3=("observed_pct", "null_pct", "delta_pp", "perm_p"))

    # ---------- Module ablation ----------
    mod = pd.read_csv(ROOT / "module_ablation" / "summary.csv")
    meta = {
        "full_reptile": dict(family="reptile", reptile="yes", rank_w=0.3, con_w=0.05, cons_w=0.10,
                             fusion="bidirectional", inner_steps=3, epochs=220,
                             samples_seen=220 * 84641, model_selection="best val checkpoint, early stop",
                             walltime="not logged"),
        "no_reptile": dict(family="supervised", reptile="no", rank_w=0.3, con_w=0.05, cons_w=0.10,
                           fusion="bidirectional", inner_steps=0, epochs="<=300 (early stop)",
                           samples_seen="not logged", model_selection="best val MSE, patience 50",
                           walltime="measured"),
        "no_ranking": dict(family="supervised", reptile="no", rank_w=0.0, con_w=0.05, cons_w=0.10,
                           fusion="bidirectional", inner_steps=0, epochs="<=300 (early stop)",
                           samples_seen="not logged", model_selection="best val MSE, patience 50",
                           walltime="measured"),
        "no_contrastive": dict(family="supervised", reptile="no", rank_w=0.3, con_w=0.0, cons_w=0.10,
                               fusion="bidirectional", inner_steps=0, epochs="<=300 (early stop)",
                               samples_seen="not logged", model_selection="best val MSE, patience 50",
                               walltime="measured"),
        "no_consistency": dict(family="supervised", reptile="no", rank_w=0.3, con_w=0.05, cons_w=0.0,
                               fusion="bidirectional", inner_steps=0, epochs="<=300 (early stop)",
                               samples_seen="not logged", model_selection="best val MSE, patience 50",
                               walltime="measured"),
        "no_bidir_fusion": dict(family="supervised", reptile="no", rank_w=0.3, con_w=0.05, cons_w=0.0,
                                fusion="concat", inner_steps=0, epochs="<=300 (early stop)",
                                samples_seen="not logged", model_selection="best val MSE, patience 50",
                                walltime="measured"),
        "mse_only": dict(family="supervised", reptile="no", rank_w=0.0, con_w=0.0, cons_w=0.0,
                         fusion="bidirectional", inner_steps=0, epochs="<=300 (early stop)",
                         samples_seen="not logged", model_selection="best val MSE, patience 50",
                         walltime="measured"),
    }
    mod["family"] = mod["variant"].map(lambda x: meta[x]["family"])
    mod["reptile"] = mod["variant"].map(lambda x: meta[x]["reptile"])
    mod["rank_w"] = mod["variant"].map(lambda x: meta[x]["rank_w"])
    mod["con_w"] = mod["variant"].map(lambda x: meta[x]["con_w"])
    mod["cons_w"] = mod["variant"].map(lambda x: meta[x]["cons_w"])
    mod["fusion"] = mod["variant"].map(lambda x: meta[x]["fusion"])
    mod["inner_steps"] = mod["variant"].map(lambda x: meta[x]["inner_steps"])
    mod["epochs"] = mod["variant"].map(lambda x: meta[x]["epochs"])
    mod["samples_seen"] = mod["variant"].map(lambda x: meta[x]["samples_seen"])
    mod["model_selection"] = mod["variant"].map(lambda x: meta[x]["model_selection"])
    mod["walltime_s"] = mod["variant"].map(lambda x: meta[x]["walltime"] if meta[x]["walltime"] == "not logged" else round(float(mod.loc[mod["variant"] == x, "walltime_mean_s"].iloc[0]), 1))
    mod = mod[["variant", "family", "reptile", "rank_w", "con_w", "cons_w", "fusion", "inner_steps", "epochs",
               "samples_seen", "model_selection", "n_seeds", "R2_mean", "R2_sd", "RMSE_mean", "RMSE_sd",
               "AUPR_mean", "AUPR_sd", "EF@1%_mean", "EF@1%_sd", "ECE_mean", "ECE_sd", "walltime_s"]]
    write_table(wb.create_sheet("13_Module_Ablation"), mod,
                title="模块删除消融（KIBA 18,478，3 seeds）",
                note="no_ranking/no_contrastive/no_consistency/no_bidir_fusion 是 supervised 族（从 no_reptile 删除），不是从 full Reptile 删除；两者不可混读。",
                num4=("R2_mean", "R2_sd", "RMSE_mean", "RMSE_sd", "AUPR_mean", "AUPR_sd", "ECE_mean", "ECE_sd"),
                num3=("EF@1%_mean", "EF@1%_sd", "walltime_s"))
    base = mod[mod["variant"] == "no_reptile"].iloc[0]
    rel_rows = []
    for _, r in mod.iterrows():
        rel_rows.append([r["variant"],
                         100*(r["R2_mean"]/base["R2_mean"]-1),
                         100*(r["RMSE_mean"]/base["RMSE_mean"]-1),
                         100*(r["AUPR_mean"]/base["AUPR_mean"]-1),
                         100*(r["EF@1%_mean"]/base["EF@1%_mean"]-1),
                         100*(r["ECE_mean"]/base["ECE_mean"]-1)])
    rel_mod = pd.DataFrame(rel_rows, columns=["variant", "ΔR2 (%)", "ΔRMSE (%)", "ΔAUPR (%)", "ΔEF (%)", "ΔECE (%)"])
    rel_mod = rel_mod[rel_mod["variant"] != "full_reptile"].reset_index(drop=True)
    write_table(wb.create_sheet("17_Module_Relative"), rel_mod,
                title="监督族相对变化（以 no_reptile 为基准）",
                note="本表只解释 supervised 模型内部的损失/融合作用；full_reptile 是 reptile 族参考，不在此表，不能与 supervised 删除混读。",
                num3=("ΔR2 (%)", "ΔRMSE (%)", "ΔAUPR (%)", "ΔEF (%)", "ΔECE (%)"))
    per = pd.read_csv(ROOT / "module_ablation" / "per_seed.csv")
    write_table(wb.create_sheet("14_Module_PerSeed"), per, title="监督族模块消融逐 seed",
                num4=("R2", "RMSE", "ECE", "AUPR"), num3=("EF@1%",))

    # ---------- Reptile 族模块消融 ----------
    rm = pd.read_csv(ROOT / "reptile_module_ablation" / "summary.csv")
    full_ref = mod[mod["variant"] == "full_reptile"].iloc[0].to_dict()
    full_ref["variant"] = "full_reptile (e5 reference)"
    full_ref["n_seeds"] = 3
    full_ref["walltime_mean_s"] = np.nan
    full_ref["EF_mean"] = full_ref["EF@1%_mean"]
    full_ref["EF_sd"] = full_ref["EF@1%_sd"]
    rm = pd.concat([pd.DataFrame([full_ref]), rm], ignore_index=True)
    meta_r = {
        "full_reptile (e5 reference)": dict(rank=0.3, con=0.05, cons=0.10, fusion="bidirectional"),
        "reptile_no_ranking": dict(rank=0.0, con=0.05, cons=0.10, fusion="bidirectional"),
        "reptile_no_contrastive": dict(rank=0.3, con=0.0, cons=0.10, fusion="bidirectional"),
        "reptile_no_consistency": dict(rank=0.3, con=0.05, cons=0.0, fusion="bidirectional"),
        "reptile_no_bidir_fusion": dict(rank=0.3, con=0.05, cons=0.0, fusion="concat"),
    }
    rm["rank_w"] = rm["variant"].map(lambda x: meta_r[x]["rank"])
    rm["con_w"] = rm["variant"].map(lambda x: meta_r[x]["con"])
    rm["cons_w"] = rm["variant"].map(lambda x: meta_r[x]["cons"])
    rm["fusion"] = rm["variant"].map(lambda x: meta_r[x]["fusion"])
    rm = rm[["variant", "rank_w", "con_w", "cons_w", "fusion", "n_seeds", "R2_mean", "R2_sd", "RMSE_mean", "RMSE_sd",
             "AUPR_mean", "AUPR_sd", "EF_mean", "EF_sd", "ECE_mean", "ECE_sd", "walltime_mean_s"]]
    write_table(wb.create_sheet("20_Reptile_Module"), rm,
                title="完整 Reptile 模型的模块删除消融（KIBA 18,478，3 seeds）",
                note="全部从 full Reptile 删除；full_reptile 为 e5 参考，walltime 未记录。",
                num4=("R2_mean", "R2_sd", "RMSE_mean", "RMSE_sd", "AUPR_mean", "AUPR_sd", "ECE_mean", "ECE_sd"),
                num3=("EF_mean", "EF_sd", "walltime_mean_s"))
    base_r = rm.iloc[0]
    rr = []
    for _, r in rm.iterrows():
        rr.append([r["variant"],
                   100*(r["R2_mean"]/base_r["R2_mean"]-1),
                   100*(r["RMSE_mean"]/base_r["RMSE_mean"]-1),
                   100*(r["AUPR_mean"]/base_r["AUPR_mean"]-1),
                   100*(r["EF_mean"]/base_r["EF_mean"]-1),
                   100*(r["ECE_mean"]/base_r["ECE_mean"]-1)])
    write_table(wb.create_sheet("21_Reptile_Relative"),
                pd.DataFrame(rr, columns=["variant", "ΔR2 (%)", "ΔRMSE (%)", "ΔAUPR (%)", "ΔEF (%)", "ΔECE (%)"]),
                title="完整 Reptile 族相对变化（以 full_reptile 为基准）",
                num3=("ΔR2 (%)", "ΔRMSE (%)", "ΔAUPR (%)", "ΔEF (%)", "ΔECE (%)"))

    # ---------- ChEMBL 5seed ----------
    c5 = pd.read_csv(ROOT / "chembl_morgan_5seeds_b2048" / "summary.csv")
    c5_df = c5.rename(columns={"seed": "Seed"})
    mean_row = {c: "" for c in c5_df.columns}
    mean_row["Seed"] = "mean"
    sd_row = {c: "" for c in c5_df.columns}
    sd_row["Seed"] = "SD"
    for c in ["R2", "RMSE", "EF@1%", "AUPR", "ECE"]:
        mean_row[c] = float(c5[c].mean())
        sd_row[c] = float(c5[c].std(ddof=1))
    c5_df = pd.concat([c5_df, pd.DataFrame([mean_row, sd_row])], ignore_index=True)
    write_table(wb.create_sheet("15_ChEMBL_5Seed"), c5_df,
                title="ChEMBL 5-seed Reptile（morgan_descriptors，batch 2048/220，raw-scale）",
                note="MACCS 在训练和推理时均屏蔽。mean/SD 为 5 seeds。",
                num4=("R2", "RMSE", "AUPR", "ECE"), num3=("EF@1%",))

    # ---------- 消融设计说明 ----------
    design = pd.DataFrame([
        ["Supervised 族", "no_reptile（完整 composite loss，无 Reptile 内循环）",
         "no_ranking / no_contrastive / no_consistency / no_bidir_fusion / mse_only",
         "监督模型内部的损失与融合作用", "已完成（3 seeds）"],
        ["Reptile 族", "full_reptile（Reptile + composite loss + 双向融合）",
         "reptile_no_ranking / reptile_no_contrastive / reptile_no_consistency / reptile_no_bidir_fusion",
         "完整 Reptile 模型内部的模块贡献", "已完成（3 seeds；full_reptile 为 e5 参考）"],
    ], columns=["Family", "Base", "Variants", "What it isolates", "Status"])
    write_table(wb.create_sheet("19_Ablation_Design"), design,
                title="模块消融的两族设计（不可混读）",
                note="表 13 的 no_ranking 等是 Supervised 族；要证明完整 Reptile 的模块贡献，必须看 Reptile 族。")

    # ---------- S18 / S19 EF 矩阵（统一口径重算） ----------
    s18_dir = ROOT / "s18_s19_recomputed"
    s18_long = pd.read_csv(s18_dir / "S18_long.csv")
    s19_long = pd.read_csv(s18_dir / "S19_long.csv")
    for sheet, stem, long_df, title, cov_note in [
        ("22_S18_EF1", "S18", s18_long,
         "S18. EF@1% 矩阵（统一口径重算；空白=NA）",
         "EF@1%：KIBA/ChEMBL/BindingDB 为逐靶点 macro，Davis 为 pooled（每靶点 66 个化合物，"
         "per-target EF@1% 未定义）。各数据集覆盖率与缺失原因见下方明细表的 status 列。"),
        ("23_S19_EF5", "S19", s19_long,
         "S19. EF@5% 矩阵（统一口径重算；空白=NA）",
         "EF@5%：KIBA/ChEMBL/BindingDB/Davis 均为逐靶点 macro。"
         "各数据集覆盖率与缺失原因见下方明细表的 status 列。"),
    ]:
        ws = wb.create_sheet(sheet)
        mat = pd.read_csv(s18_dir / f"{stem}_matrix.csv")
        write_table(ws, mat, title=title, note=cov_note, num3=("BindingDB", "ChEMBL", "Davis", "KIBA"))
        cov = long_df[["Dataset", "Model", "Aggregation", "n_targets_used", "n_targets_total",
                       "coverage", "n_pairs", "runs", "status"]].copy()
        write_table(ws, cov, title=f"{stem}. 覆盖率与聚合口径明细", start_row=ws.max_row + 2,
                    pct_cols=("coverage",))

    # ---------- 新补齐面板（morgan_descriptors） ----------
    for sheet, path, title, note in [
        ("24_BindingDB_Panel", ROOT / "bindingdb_panel_morgan" / "summary.csv",
         "BindingDB 面板（morgan_descriptors，7,087 对 / 112 靶点）",
         "EF@1% 为逐靶点 macro（10/112 靶点满足 n>=100）；EF@5% 为逐靶点 macro（76/112）。"
         "旧 p2_c3 full 预测不再使用。"),
        ("25_Davis_Panel", ROOT / "davis_panel_morgan" / "summary.csv",
         "Davis 面板（morgan_descriptors，4,290 对 / 65 靶点）",
         "EF@1% 为 pooled（每靶点 66 个化合物，per-target EF@1% 未定义）；EF@5% 为逐靶点 macro。"
         "旧 panel_screen/davis full 结果不再使用。"),
    ]:
        ws = wb.create_sheet(sheet)
        df = pd.read_csv(path)
        summ = agg_family(df)
        write_table(ws, summ, title=title, note=note,
                    num4=("R2_mean", "R2_sd", "RMSE_mean", "RMSE_sd"),
                    num3=("EF1_mean", "EF1_sd", "EF5_mean", "EF5_sd"))
        write_table(ws, df, title="逐 run / seed", start_row=ws.max_row + 2,
                    num4=("R2", "RMSE"), num3=("EF1", "EF5"))

    ws = wb.create_sheet("26_ChEMBL_GNN")
    gnn_df = pd.read_csv(ROOT / "chembl_gnn_retrain" / "summary.csv")
    gnn_df = gnn_df.drop(columns=["log"], errors="ignore")
    write_table(ws, gnn_df, title="ChEMBL GNN 重训（GraphDTA split：30,180 对 / 26 靶点）",
                note="与非 GNN 的 31,132 对不是同一测试集；EF@1% 为逐靶点 macro（24/26 靶点满足 n>=100）。",
                num4=("R2", "RMSE"), num3=("EF1",))

    # ---------- 6.1 预算匹配对照 ----------
    exp61 = ROOT / "exp61_analysis"
    if (exp61 / "budget.csv").exists():
        b = pd.read_csv(exp61 / "budget.csv")
        ws = wb.create_sheet("27_Exp61_Budget")
        bsum = (b.groupby("config", sort=False)
                  .agg(n=("epochs", "size"), epochs=("epochs", "first"),
                       n_train_targets=("n_train_targets", "first"),
                       inner_updates=("inner_updates", "first"),
                       meta_updates=("meta_updates", "first"),
                       samples_seen=("samples_seen", "first"),
                       walltime_mean_s=("walltime_s", "mean"),
                       walltime_sd_s=("walltime_s", lambda x: x.std(ddof=1)),
                       best_epoch_mean=("best_epoch", "mean"))
                  .reset_index())
        write_table(ws, bsum, title="6.1 预算匹配（KIBA 18,478；220 epoch）",
                    note="full Reptile 与 direct_supervised 的 inner updates / samples seen 相同；"
                         "control 的 meta updates=0。seed 11 为续训，segment 计数见明细。",
                    num4=("walltime_mean_s", "walltime_sd_s"), num3=("best_epoch_mean",))
        write_table(ws, b, title="逐 seed 预算明细", start_row=ws.max_row + 2,
                    num4=("walltime_s",), num3=("best_epoch",))

        d = pd.read_csv(exp61 / "deltas.csv")
        ws = wb.create_sheet("28_Exp61_Deltas")
        dsum = d.groupby("comparison", sort=False)[["dR2", "dEF1", "dEF5", "dAUPR", "dECE"]].agg(["mean", "std"])
        dsum.columns = [f"{a}_{b}" for a, b in dsum.columns]
        dsum = dsum.reset_index()
        write_table(ws, dsum, title="6.1 逐 seed 配对差值汇总（相对 full Reptile）",
                    note="3 seeds 的模块消融为探索性；direct_supervised 为 5 seeds。",
                    num4=("dR2_mean", "dR2_std", "dEF1_mean", "dEF1_std", "dEF5_mean", "dEF5_std",
                          "dAUPR_mean", "dAUPR_std", "dECE_mean", "dECE_std"))
        write_table(ws, d, title="逐 seed 差值明细", start_row=ws.max_row + 2,
                    num4=("dR2", "dEF1", "dEF5", "dAUPR", "dECE"))

        ws = wb.create_sheet("29_Exp61_Stats")
        s = pd.read_csv(exp61 / "stats.csv")
        write_table(ws, s, title="6.1 配对统计（H0: Δ=0）",
                    note="n=5 时 Wilcoxon 双侧最小 p=0.0625；n=3 的消融只作探索性。",
                    num4=("mean", "sd", "t", "p_t", "p_wilcoxon"))

        ws = wb.create_sheet("30_Exp61_Bootstrap")
        bt = pd.read_csv(exp61 / "bootstrap.csv")
        write_table(ws, bt, title="6.1 共同靶点配对 bootstrap（seed 11，B=2000）",
                    note="CI 为 2.5–97.5 百分位；p_gt0 为 Δ>0 的 bootstrap 比例。",
                    num4=("mean", "ci_low", "ci_high", "p_gt0"))

        ws = wb.create_sheet("31_Exp61_PerSeed")
        ps = pd.read_csv(exp61 / "per_seed.csv")
        write_table(ws, ps, title="6.1 逐 seed 指标与预算",
                    note="seed 11 control 为续训；inner_updates_segment 只覆盖后 140 epoch，"
                         "inner_updates_total 按 220 epoch 计。",
                    num4=("R2", "RMSE", "EF1", "EF5", "AUPR", "ECE", "walltime_s"),
                    num3=("best_epoch",))

    # ---------- 审计第三轴：配体骨架覆盖 ----------
    scaf_dir = ROOT / "ligand_scaffold_axis"
    if (scaf_dir / "dataset_summary.csv").exists():
        ss = pd.read_csv(scaf_dir / "dataset_summary.csv")
        ws = wb.create_sheet("32_Scaffold_Summary")
        pct = ("exact_pair_cov", "generic_pair_cov", "exact_compound_cov",
               "generic_compound_cov", "cluster5_compound_cov", "novel_exact_pct",
               "novel_generic_pct", "acyclic_pct", "per_target_exact_median",
               "per_target_exact_q25", "per_target_exact_q75",
               "per_target_exact_min", "per_target_exact_max")
        write_table(ws, ss, title="审计第三轴：配体骨架覆盖（Murcko，compound-level）",
                    note="exact=保留原子类型；generic=原子/键泛化；无环骨架按未覆盖计入。"
                         "KIBA/Davis 为共享化合物面板，覆盖率饱和；BindingDB 含较多新骨架。",
                    pct_cols=pct)

        ws = wb.create_sheet("33_Scaffold_PerTarget")
        pt = pd.read_csv(scaf_dir / "per_target.csv")
        write_table(ws, pt, title="第三轴：逐靶点配体骨架覆盖",
                    note="先在每个靶点内部按 SMILES 去重；Davis 为共享 66 化合物面板，所有靶点覆盖率为 1。",
                    pct_cols=("exact_cov", "generic_cov", "cluster5_cov"))

    # ---------- 第三轴 × 性能 ----------
    perf_dir = ROOT / "scaffold_performance"
    if (perf_dir / "correlations.csv").exists():
        ws = wb.create_sheet("34_Scaffold_Perf_Corr")
        corr = pd.read_csv(perf_dir / "correlations.csv")
        write_table(ws, corr, title="第三轴 × 性能：Spearman / 偏 Spearman（5-seed 平均）",
                    note="每个 seed 单独算逐靶点指标，再按靶点平均；偏 Spearman 控制 log10(n_test_ligands)；"
                         "Bootstrap B=2000 按靶点重采样。KIBA/Davis 的 coverage 恒为 1，标为 no variance。",
                    num4=("rho", "ci_low", "ci_high", "partial_rho", "rho_seed_min", "rho_seed_max"))

        ws = wb.create_sheet("35_Scaffold_Perf_Groups")
        groups = pd.read_csv(perf_dir / "coverage_groups.csv")
        write_table(ws, groups, title="第三轴 × 性能：coverage>=0.5 vs <0.5（5-seed 平均）",
                    note="R² 的均值会被小样本靶点的极端负值拉低，主文建议看 median 与 Mann-Whitney。",
                    num4=("high_mean", "low_mean", "high_median", "low_median", "p_mannwhitney"))

        ws = wb.create_sheet("36_Scaffold_Perf_Target")
        ptm = pd.read_csv(perf_dir / "per_target_metrics.csv")
        write_table(ws, ptm, title="第三轴 × 性能：逐靶点指标（5-seed 平均）",
                    note="R²/EF 由 raw-scale predictions 逐靶点计算后按 seed 平均；"
                         "EF@1% 要求 n>=100，EF@5% 要求 n>=20。",
                    num4=("R2", "EF1", "EF5"), pct_cols=("exact_cov", "generic_cov", "cluster5_cov"))

        ws = wb.create_sheet("37_Scaffold_Perf_Seeds")
        pss = pd.read_csv(perf_dir / "per_seed_correlations.csv")
        write_table(ws, pss, title="第三轴 × 性能：每个 seed 的 Spearman",
                    note="用于说明 seed 间稳定性；KIBA/Davis 的 coverage 恒为 1，rho 为空。",
                    num4=("rho",))

    wb.save(OUT)
    print("[saved]", OUT)
    print("sheets:", wb.sheetnames)


if __name__ == "__main__":
    main()
