# -*- coding: utf-8 -*-
"""
生成《审计第三轴：配体骨架覆盖》主文 + 补充材料整合稿 Word 文档。
同时生成补充材料草案图（PNG + TIFF，300 dpi，蓝绿色系）。
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(r"D:\lht")
SCAF = ROOT / "ligand_scaffold_axis"
PERF = ROOT / "scaffold_performance"
FIG = ROOT / "第三轴_补充材料图"
OUT_DOCX = ROOT / "第三轴_配体骨架覆盖_主文与补充材料.docx"

TEAL = "#2A9D8F"
BLUE = "#457B9D"
LIGHT = "#A8DADC"
DARK = "#1F4E78"
HEAD_FILL = "D9EFEA"
BOX_FILL = "EAF6F3"
GREY = "#7F7F7F"
DISP = {"kiba": "KIBA", "chembl": "ChEMBL", "davis": "Davis",
        "bindingdb": "BindingDB", "pdbbind": "PDBbind", "metz": "Metz", "papyrus": "Papyrus"}
MET = {"R2": "R²", "EF1": "EF@1%", "EF5": "EF@5%"}


def hex6(c):
    return str(c).lstrip("#")


def fmt_p(p):
    if pd.isna(p):
        return "—"
    if p >= 0.001:
        return f"{p:.3f}"
    exp = int(np.floor(np.log10(p)))
    mant = p / (10 ** exp)
    sup = str(exp).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{mant:.1f} × 10{sup}"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), color)
    tcPr.append(shd)


def set_cell_text(cell, text, bold=False, size=8, color=None, align=WD_ALIGN_PARAGRAPH.CENTER):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    r = p.add_run(str(text))
    r.font.size = Pt(size)
    r.font.bold = bold
    if color:
        r.font.color.rgb = RGBColor.from_string(hex6(color))


def add_table(doc, headers, rows, widths=None, font=8):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(headers):
        c = t.rows[0].cells[j]
        set_cell_text(c, h, bold=True, size=font)
        shade(c, HEAD_FILL)
    for row in rows:
        cells = t.add_row().cells
        for j, v in enumerate(row):
            set_cell_text(cells[j], v, size=font)
    if widths:
        for row in t.rows:
            for j, w in enumerate(widths):
                row.cells[j].width = Cm(w)
    return t


def add_para(doc, text, bold=False, italic=False, size=10.5, color=None,
             align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=4):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.2
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    if color:
        r.font.color.rgb = RGBColor.from_string(hex6(color))
    return p


def add_bullet(doc, text, size=10.5):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    r = p.add_run(text)
    r.font.size = Pt(size)
    return p


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(text)
    r.font.size = Pt(8.5)
    r.font.italic = True
    r.font.color.rgb = RGBColor.from_string(hex6(GREY))
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT


def add_note(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.font.size = Pt(8.5)
    r.font.color.rgb = RGBColor.from_string(hex6(GREY))
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT


def add_box(doc, text, size=10.5):
    t = doc.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    c = t.rows[0].cells[0]
    c.text = ""
    p = c.paragraphs[0]
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.font.size = Pt(size)
    shade(c, BOX_FILL)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_code(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.left_indent = Cm(0.3)
    r = p.add_run(text)
    r.font.name = "Consolas"
    r.font.size = Pt(8.5)
    return p


def add_page_number_footer(doc):
    section = doc.sections[0]
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("审计第三轴 · 主文与补充材料整合稿 · ")
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor.from_string(hex6(GREY))
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    r = OxmlElement("w:r")
    t = OxmlElement("w:t")
    t.text = "1"
    r.append(t)
    fld.append(r)
    p._p.append(fld)


# ---------------------------------------------------------------- figures
def make_figures():
    FIG.mkdir(parents=True, exist_ok=True)
    pt = pd.read_csv(PERF / "per_target_metrics.csv")
    b = pt[pt["dataset"] == "bindingdb"].copy()
    corr = pd.read_csv(PERF / "correlations.csv")

    def ann(metric):
        r = corr[(corr["dataset"] == "bindingdb") & (corr["cov"] == "exact_cov")
                 & (corr["metric"] == metric)].iloc[0]
        return (f"Spearman ρ = {r['rho']:.3f} (95% CI {r['ci_low']:.3f}–{r['ci_high']:.3f}); "
                f"partial ρ = {r['partial_rho']:.3f}")

    def scatter(ycol, ylabel, fname, ylim, hline, rho_text):
        fig, ax = plt.subplots(figsize=(3.5, 3.2))
        x = b["exact_cov"].values
        y = b[ycol].values
        ax.scatter(x, y, s=24, color=TEAL, edgecolor="white", linewidth=0.5, alpha=0.85)
        ax.axvline(0.5, color=GREY, ls="--", lw=0.8)
        ax.axhline(hline, color=GREY, ls="--", lw=0.8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Exact Murcko scaffold coverage")
        ax.set_ylabel(ylabel)
        ax.set_title("BindingDB", fontsize=10, color=DARK)
        fig.text(0.5, 0.005, rho_text, ha="center", va="bottom", fontsize=8, color="#333333")
        fig.subplots_adjust(bottom=0.22, left=0.18, right=0.97, top=0.9)
        fig.savefig(FIG / f"{fname}.png", dpi=300, bbox_inches="tight")
        fig.savefig(FIG / f"{fname}.tiff", dpi=300, bbox_inches="tight",
                    pil_kwargs={"compression": "tiff_lzw"})
        plt.close(fig)

    scatter("R2", "Per-target R²", "figS29a_bindingdb_r2",
            (-2.05, 1.05), 0.0, ann("R2"))
    scatter("EF5", "Per-target EF@5%", "figS29b_bindingdb_ef5",
            (0.0, max(6.0, float(np.nanmax(b["EF5"])) * 1.08)), 1.0, ann("EF5"))

    ss = pd.read_csv(SCAF / "dataset_summary.csv")
    order = ["kiba", "chembl", "davis", "bindingdb", "pdbbind", "metz", "papyrus"]
    ss = ss.set_index("dataset").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(ss))
    w = 0.36
    ax.bar(x - w / 2, ss["exact_compound_cov"] * 100, w, color=TEAL, label="Exact Murcko")
    ax.bar(x + w / 2, ss["generic_compound_cov"] * 100, w, color=BLUE, label="Generic Murcko")
    for i in range(len(ss)):
        ax.text(x[i] - w / 2, ss["exact_compound_cov"].iloc[i] * 100 + 1.5,
                f"{ss['exact_compound_cov'].iloc[i]*100:.0f}", ha="center", fontsize=7, color=TEAL)
        ax.text(x[i] + w / 2, ss["generic_compound_cov"].iloc[i] * 100 + 1.5,
                f"{ss['generic_compound_cov'].iloc[i]*100:.0f}", ha="center", fontsize=7, color=BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels([DISP.get(s, s) for s in ss["dataset"]], rotation=0, fontsize=8)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Compound coverage (%)")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "figS30_dataset_scaffold_coverage.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "figS30_dataset_scaffold_coverage.tiff", dpi=300, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("[figures]", FIG)


# ---------------------------------------------------------------- docx
def build_doc():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.left_margin = Cm(2.0)
    sec.right_margin = Cm(2.0)
    sec.top_margin = Cm(2.0)
    sec.bottom_margin = Cm(2.0)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    for name, size, color in [("Heading 1", 15, DARK), ("Heading 2", 12, "2A6F6B"),
                              ("Heading 3", 10.5, "2A6F6B")]:
        st = styles[name]
        st.font.name = "Times New Roman"
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor.from_string(hex6(color))
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")

    add_page_number_footer(doc)

    # Title
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("审计第三轴：配体骨架覆盖")
    r.font.size = Pt(20)
    r.font.bold = True
    r.font.color.rgb = RGBColor.from_string(hex6(DARK))
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("主文与补充材料整合稿")
    r.font.size = Pt(13)
    r.font.color.rgb = RGBColor.from_string("2A6F6B")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("2026-11-14 · 7 个数据集 · Murcko exact/generic · 5-seed 汇总 · CPU-only")
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor.from_string(hex6(GREY))
    doc.add_paragraph()

    add_box(doc,
            "一句话结论：第三轴补齐了前两轴看不到的配体化学新颖性。KIBA、Davis、Metz 的测试配体骨架 100% "
            "在训练集出现过；BindingDB、Papyrus、PDBbind 仍分别有 64.5%、46.8%、42.4% 的新骨架；"
            "在 BindingDB 中，低覆盖靶点的 R² 与 EF@5% 显著更差（5-seed 平均后中位数 0.099 vs -1.636；4.52 vs 1.35）。",
            size=10.5)

    # 1
    doc.add_heading("1. 这组实验补了什么", level=1)
    add_para(doc, "审计包原本有两条可操作轴：序列相似性（靶点侧泄漏）和结合谱冗余（配体侧标签泄漏）。"
                  "第三轴回答一个前两轴看不到的问题：测试集的配体骨架，有多少在训练集里出现过？"
                  "它衡量的是配体化学新颖性，而不是靶点同源性或标签冗余。")
    add_bullet(doc, "靶点冷拆分（target-cold）不等于配体新颖：KIBA、Davis、Metz 的测试配体骨架 100% 见过。")
    add_bullet(doc, "配体骨架有变化的场景：BindingDB、Papyrus、PDBbind 的新骨架比例为 64.5%、46.8%、42.4%；ChEMBL 为 28.7%。")
    add_bullet(doc, "在可检验的数据集里，覆盖率与性能正相关：BindingDB 最强，ChEMBL 方向一致但较弱；KIBA/Davis 因饱和无法检验。")

    # 2
    doc.add_heading("2. 方法与口径（可直接改写进补充材料）", level=1)
    doc.add_heading("2.1 骨架定义", level=2)
    add_para(doc, "使用 RDKit 2022.09.5 计算 Bemis–Murcko scaffold，并给出两套定义：exact 保留原子类型；"
                  "generic 对原子和键类型做泛化。无环分子的 Murcko 骨架为空，按“未覆盖”计入，并在表中单独报告 acyclic 比例。")
    doc.add_heading("2.2 覆盖定义", level=2)
    add_bullet(doc, "compound-level（主口径）：测试集唯一 SMILES 中，骨架在训练集出现过的比例。")
    add_bullet(doc, "pair-level（辅助）：测试集唯一 (target, ligand) 对中，骨架在训练集出现过的比例；会被重复测量放大。")
    add_bullet(doc, "cluster5：训练集中至少出现于 5 个不同化合物的骨架，测试集覆盖率；用于排除通用骨架的影响。")
    add_bullet(doc, "逐靶点：先在每个靶点内部按 SMILES 去重，再算覆盖率；报告 median 与 IQR。")
    doc.add_heading("2.3 数据集与测试规模", level=2)
    ss = pd.read_csv(SCAF / "dataset_summary.csv")
    rows = []
    for _, r in ss.iterrows():
        rows.append([DISP.get(r["dataset"], r["dataset"]),
                     f"{int(r['n_train_pairs']):,}", f"{int(r['n_test_pairs']):,}",
                     f"{int(r['n_test_compounds']):,}", f"{int(r['n_test_targets']):,}"])
    add_table(doc, ["Dataset", "Train pairs", "Test pairs", "Test compounds", "Test targets"],
              rows, widths=[3.0, 3.0, 3.0, 3.2, 3.0], font=8.5)
    add_note(doc, "注：pairs 为去重后的 target–ligand 对；PDBbind 的 target_id 为复合物级，逐靶点统计不建议使用。")
    doc.add_heading("2.4 与性能的关联", level=2)
    add_para(doc, "逐靶点 R²、EF@1%、EF@5% 由 raw-scale predictions 计算；EF@1% 要求 n≥100，EF@5% 要求 n≥20。"
                  "每个 seed 单独计算逐靶点指标，再按靶点对 5 个 seed（11/22/33/44/55）取平均；"
                  "Spearman ρ、控制 log10(n_test_ligands) 的偏 Spearman ρ 和 bootstrap 95% CI（B=2,000，按靶点重采样）"
                  "均在 seed 平均后的指标上计算，并同时报告 per-seed ρ 范围。"
                  "另按 coverage ≥0.5 与 <0.5 分组，用双侧 Mann–Whitney U 检验；由于小靶点的 R² 可能出现极端负值，主文看 median，不看 mean。")
    doc.add_heading("2.5 软件与运行", level=2)
    add_para(doc, "Python 3.10、RDKit 2022.09.5、SciPy、pandas；全部为 CPU 计算，不需要 GPU。"
                  "骨架缓存位于 ligand_scaffold_axis/scaffold_cache.csv。")

    # 3
    doc.add_heading("3. 主文需要放什么", level=1)
    doc.add_heading("3.1 建议插入位置", level=2)
    add_para(doc, "放在 Results 的“审计三轴”段落：先讲序列相似性轴，再讲结合谱冗余轴，最后用本段补第三轴；"
                  "在 C3 退化检测与 C4 路由之前。主文不需要新增方法小节，也不需要新增主图；"
                  "引用 Table S42、Table S43、Figure S29 即可。")
    doc.add_heading("3.2 可直接使用的英文段落", level=2)
    main_par = (
        "Third, we asked whether target-cold splits also remove ligand-side novelty. Using Bemis–Murcko "
        "scaffolds, we computed the fraction of test ligands whose scaffold was present in the training panel. "
        "Coverage was saturated in KIBA, Davis, and Metz (100%; no novel scaffolds), whereas BindingDB, Papyrus, "
        "and PDBbind retained substantial novel-scaffold fractions (64.5%, 46.8%, and 42.4%, respectively), with "
        "ChEMBL intermediate (28.7%). Using five seeds (11–55) and averaging per-target metrics across seeds, "
        "targets with lower exact-scaffold coverage performed markedly worse in BindingDB "
        "(median R² 0.099 vs −1.636; Mann–Whitney p = 1.9 × 10⁻⁷; median EF@5% 4.52 vs 1.35, p = 2.5 × 10⁻⁵), "
        "and coverage remained associated with R² after controlling for target size "
        "(partial Spearman ρ = 0.50; per-seed ρ = 0.58–0.63). ChEMBL showed the same direction but weaker "
        "evidence (ρ = 0.38, 95% CI −0.07 to 0.73; EF@5% ρ = 0.41, 95% CI 0.05 to 0.67). These results "
        "identify ligand-scaffold coverage as a third, non-redundant audit axis: a target-cold split does not "
        "guarantee chemical novelty, particularly for shared-compound panels."
    )
    add_box(doc, main_par, size=10)
    add_para(doc, "若主文篇幅紧张，可只保留前两句 + 最后一句；把 BindingDB 的统计细节放到补充材料。", size=9.5, italic=True)
    doc.add_heading("3.3 主文是否需要新表/新图", level=2)
    add_bullet(doc, "推荐：不新增主文表/图，只加一段正文 + 引用补充材料；主文已有 9 图 7 表，空间有限。")
    add_bullet(doc, "若审稿人要求主文可见证据：可加一个 4 行小表（KIBA/ChEMBL/BindingDB/Davis × exact coverage / novel % / 可检验性），放在补充材料而非主文。")
    doc.add_heading("3.4 与现有结论的衔接", level=2)
    add_para(doc, "这段的作用不是推翻前两轴，而是说明三轴互补：序列轴发现靶点侧泄漏，镜像轴发现结合谱冗余，"
                  "骨架轴发现配体侧新颖性不足。它也为 C3 的“退化检测”提供第三个解释维度：低骨架覆盖的靶点可能天然更难。")

    # 4
    doc.add_heading("4. 补充材料需要放什么", level=1)
    doc.add_heading("4.1 Table S42：数据集级骨架覆盖", level=2)
    add_caption(doc, "Table S42. Dataset-level ligand-scaffold coverage. Coverage is compound-level (unique test SMILES) "
                     "unless indicated otherwise. Exact and generic refer to Bemis–Murcko scaffolds with or without "
                     "atom/bond-type generalization. Cluster5 denotes scaffolds occurring in at least five distinct "
                     "training compounds. Acyclic compounds have empty Murcko scaffolds and are counted as uncovered. "
                     "For PDBbind, target_id is complex-level.")
    rows = []
    for _, r in ss.iterrows():
        rows.append([
            DISP.get(r["dataset"], r["dataset"]),
            f"{int(r['n_train_pairs']):,}", f"{int(r['n_test_pairs']):,}",
            f"{int(r['n_train_compounds']):,}", f"{int(r['n_test_compounds']):,}",
            f"{int(r['n_test_targets']):,}",
            f"{r['exact_pair_cov']*100:.1f}", f"{r['generic_pair_cov']*100:.1f}",
            f"{r['exact_compound_cov']*100:.1f}", f"{r['generic_compound_cov']*100:.1f}",
            f"{r['novel_exact_pct']*100:.1f}", f"{r['cluster5_compound_cov']*100:.1f}",
            f"{r['acyclic_pct']*100:.1f}",
        ])
    add_table(doc, ["Dataset", "Train pairs", "Test pairs", "Train comp.", "Test comp.", "Targets",
                    "Exact pair %", "Generic pair %", "Exact comp. %", "Generic comp. %",
                    "Novel exact %", "Cluster5 %", "Acyclic %"],
              rows, widths=[2.1, 1.35, 1.3, 1.3, 1.3, 1.05, 1.2, 1.3, 1.25, 1.35, 1.25, 1.1, 1.05], font=7.5)
    add_note(doc, "数据来源：ligand_scaffold_axis/dataset_summary.csv。百分比为 0–100，四舍五入到 1 位小数。")

    doc.add_heading("4.2 Table S43：覆盖率与逐靶点性能的关联", level=2)
    add_caption(doc, "Table S43. Association between scaffold coverage and per-target performance. Per-target metrics were "
                     "averaged across five seeds (11–55); Spearman ρ with 95% bootstrap percentile intervals (B=2,000, "
                     "target-level resampling) was computed on the seed-averaged metrics, and the per-seed ρ range is "
                     "reported. Partial ρ controls for log10(n_test_ligands). KIBA and Davis had coverage 1 for all "
                     "targets (no variance) and were not testable. EF@1% requires n≥100; EF@5% requires n≥20.")
    corr = pd.read_csv(PERF / "correlations.csv")
    corr = corr[corr["dataset"].isin(["bindingdb", "chembl"])]
    rows = []
    for _, r in corr.iterrows():
        ci = "—" if pd.isna(r["ci_low"]) else f"{r['ci_low']:.3f} to {r['ci_high']:.3f}"
        rng = ("—" if pd.isna(r["rho_seed_min"])
               else f"{r['rho_seed_min']:.3f}–{r['rho_seed_max']:.3f}")
        rows.append([DISP.get(r["dataset"], r["dataset"]),
                     "Exact" if r["cov"] == "exact_cov" else "Generic",
                     MET.get(r["metric"], r["metric"]), int(r["n_targets"]),
                     f"{r['rho']:.3f}", ci, f"{r['partial_rho']:.3f}",
                     int(r["n_seeds"]) if pd.notna(r["n_seeds"]) else "—", rng])
    add_table(doc, ["Dataset", "Coverage", "Metric", "n targets", "Spearman ρ", "95% CI",
                    "Partial ρ", "Seeds", "ρ range"],
              rows, widths=[2.1, 1.4, 1.2, 1.2, 1.6, 2.8, 1.4, 1.0, 2.1], font=8)
    add_note(doc, "KIBA 与 Davis 的 coverage 恒为 1，无法计算相关；在表中以 no variance 标注，未列入本表。"
                  "Generic 结果与 exact 方向一致。")

    doc.add_heading("4.3 Table S44：高/低覆盖分组", level=2)
    add_caption(doc, "Table S44. Per-target performance stratified by exact scaffold coverage (≥0.5 vs <0.5). Values are "
                     "medians of seed-averaged per-target metrics (5 seeds); p-values from two-sided Mann–Whitney U tests. "
                     "Medians are preferred over means because per-target R² can be extreme for small targets.")
    groups = pd.read_csv(PERF / "coverage_groups.csv")
    rows = []
    for _, r in groups.iterrows():
        rows.append([DISP.get(r["dataset"], r["dataset"]), MET.get(r["metric"], r["metric"]),
                     int(r["n_high"]), int(r["n_low"]),
                     f"{r['high_median']:.3f}", f"{r['low_median']:.3f}", fmt_p(r["p_mannwhitney"])])
    add_table(doc, ["Dataset", "Metric", "n (≥0.5)", "n (<0.5)", "Median (≥0.5)", "Median (<0.5)", "p (Mann–Whitney)"],
              rows, widths=[2.3, 1.5, 1.8, 1.8, 2.6, 2.6, 2.8], font=8)
    add_note(doc, "BindingDB 的 R² 均值会被小靶点极端负值拉低，因此表中使用 median；主文也应引用 median 与秩检验。")

    doc.add_heading("4.4 Figure S29：BindingDB 的 coverage–performance 关系", level=2)
    add_caption(doc, "Figure S29. BindingDB per-target exact scaffold coverage versus (a) R² and (b) EF@5%. Each point is "
                     "one target; dashed lines mark coverage = 0.5 and R² = 0 (or EF@5% = 1). Spearman ρ and bootstrap "
                     "95% CI are shown outside the plotting area. R² values below −2 are clipped for display.")
    doc.add_picture(str(FIG / "figS29a_bindingdb_r2.png"), width=Cm(10.5))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_picture(str(FIG / "figS29b_bindingdb_ef5.png"), width=Cm(10.5))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_note(doc, "源图：第三轴_补充材料图/figS29a_bindingdb_r2.{png,tiff}、figS29b_bindingdb_ef5.{png,tiff}，300 dpi。")

    doc.add_heading("4.5 Figure S30：7 个数据集的骨架覆盖", level=2)
    add_caption(doc, "Figure S30. Compound-level exact and generic scaffold coverage across seven benchmarks. Values are "
                     "percentages of unique test compounds whose scaffold occurs in the training panel.")
    doc.add_picture(str(FIG / "figS30_dataset_scaffold_coverage.png"), width=Cm(15.5))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_note(doc, "源图：第三轴_补充材料图/figS30_dataset_scaffold_coverage.{png,tiff}，300 dpi。")

    doc.add_heading("4.6 补充材料 Methods 可用文字", level=2)
    methods = [
        ("Ligand-scaffold coverage.",
         "For every ligand, we computed the Bemis–Murcko scaffold with RDKit 2022.09.5. Two definitions were used: "
         "exact Murcko scaffolds (retaining atom types) and generic Murcko scaffolds (atom and bond types generalized). "
         "Acyclic molecules have an empty Murcko scaffold and were treated as uncovered. For each dataset, we defined "
         "compound-level coverage as the fraction of unique test-set SMILES whose scaffold occurred in at least one "
         "training ligand; pair-level coverage was computed as an auxiliary metric over unique target–ligand pairs. "
         "To reduce the influence of ubiquitous cores, we additionally report cluster5 coverage, defined as the fraction "
         "of test compounds whose scaffold occurred in at least five distinct training compounds. Per-target coverage "
         "was computed after deduplicating SMILES within each target."),
        ("Association with performance.",
         "We related per-target exact-scaffold coverage to per-target R² and EF@k% computed from raw-scale predictions. "
         "Per-target metrics were computed separately for five seeds (11–55) and averaged across seeds before "
         "correlation analysis. We report Spearman ρ with 95% bootstrap percentile intervals (B=2,000, resampling "
         "targets), partial Spearman ρ controlling for log10(n_test_ligands), and the per-seed ρ range. Targets were "
         "also stratified into coverage ≥0.5 and <0.5 and compared with two-sided Mann–Whitney U tests; medians are "
         "reported because per-target R² can be extreme for small targets. EF@1% was computed only for targets with "
         "n≥100 and EF@5% for n≥20. KIBA and Davis had coverage 1 for all targets and therefore could not be tested."),
        ("Software.",
         "All analyses were CPU-only and used Python 3.10, RDKit 2022.09.5, SciPy, and pandas. Scripts: "
         "ligand_scaffold_axis.py and scaffold_vs_performance.py."),
    ]
    for head, text in methods:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(head + " ")
        r.font.bold = True
        r.font.size = Pt(9.5)
        r = p.add_run(text)
        r.font.size = Pt(9.5)

    # 5
    doc.add_heading("5. 结果解读：可写什么、不要写什么", level=1)
    doc.add_heading("5.1 可写的三句话", level=2)
    add_bullet(doc, "第三轴与前两轴不冗余：target-cold split 不保证配体化学新颖性；KIBA/Davis/Metz 的骨架覆盖率为 100%。")
    add_bullet(doc, "在 BindingDB 中，低骨架覆盖靶点的 R² 与 EF@5% 显著更差，且控制靶点规模后关联仍在（偏 ρ≈0.50–0.58；per-seed ρ=0.58–0.63）。")
    add_bullet(doc, "ChEMBL 方向一致但证据较弱（26 个靶点，CI 接近 0）；KIBA/Davis 因饱和无法检验。这是“数据集相关”，不是普遍规律。")
    doc.add_heading("5.2 不要写", level=2)
    add_bullet(doc, "不要写“scaffold coverage determines performance”——这是关联，不是因果。")
    add_bullet(doc, "不要写“KIBA/Davis 含新骨架”——它们的 coverage 是 100%。")
    add_bullet(doc, "不要用 mean R² 讲 BindingDB 分组差异——小靶点的极端负值会主导均值；用 median + Mann–Whitney。")
    add_bullet(doc, "不要把 PDBbind 的 target_id 当蛋白靶点——它是复合物级；PDBbind 只做数据集级覆盖，不做逐靶点推断。")

    # 6
    doc.add_heading("6. 审稿人可能追问", level=1)
    qa = [
        ("为什么用 Murcko scaffold？",
         "Bemis–Murcko 是化学信息学中最常用、可解释的骨架定义；同时给 exact 与 generic 两套，并用 cluster5 排除通用骨架。"),
        ("为什么按 0.5 分组？",
         "0.5 只是描述性分层，不是调出来的阈值；连续 Spearman 与偏 Spearman 也一并报告，避免“挑阈值”的质疑。"),
        ("KIBA/Davis 全是 100%，第三轴是不是没用？",
         "恰恰相反：这说明它们是 shared-compound panel，target-cold split 不能测试化学新颖性；第三轴揭示了前两轴看不到的饱和。"),
        ("ChEMBL 为什么不显著？",
         "只有 26 个测试靶点，检验功效有限。5-seed 平均后 R² 的 ρ=0.38（per-seed 0.34–0.43），"
         "EF@5% 的 ρ=0.41（95% CI 0.05–0.67，p=0.041）；方向与 BindingDB 一致，"
         "但 R² 的 CI 跨 0，只能作为支持性证据，不能当主结论。"),
        ("这是因果吗？",
         "不是。骨架覆盖是审计诊断量，不是部署规则；它说明“低覆盖场景更难”，不说明“提高覆盖就能提高性能”。"),
    ]
    for q, a in qa:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run("Q: " + q)
        r.font.bold = True
        r.font.size = Pt(10)
        add_para(doc, "A: " + a, size=10)

    # 7
    doc.add_heading("7. 复现与文件", level=1)
    add_code(doc, "python ligand_scaffold_axis.py --datasets all --skip-scaffold --workers 16\n"
                  "python scaffold_vs_performance.py")
    add_para(doc, "输出：ligand_scaffold_axis/{dataset_summary.csv, per_target.csv, scaffold_cache.csv}；"
                  "scaffold_performance/{per_target_metrics.csv, per_seed_metrics.csv, per_seed_correlations.csv, "
                  "correlations.csv, coverage_groups.csv}。"
                  "图：第三轴_补充材料图/figS29a_bindingdb_r2.{png,tiff}、figS29b_bindingdb_ef5.{png,tiff}、"
                  "figS30_dataset_scaffold_coverage.{png,tiff}（300 dpi）。", size=9.5)
    add_para(doc, "总表：today_experiment_results.xlsx 的 32–37 号 sheet（Scaffold Summary / PerTarget / Perf Corr / "
                  "Perf Groups / Perf Target / Perf PerSeed）。", size=9.5)

    # 8
    doc.add_heading("8. 待办清单", level=1)
    todo = [
        "把 3.2 的英文段落插入主文“审计三轴”位置，确认前后引用编号。",
        "补充材料加入 Table S42、S43、S44，Figure S29a/b、S30，并使用 4.1–4.5 的 caption。",
        "补充材料 Methods 加入 4.6 的三段文字；确认与现有 scaffold 定义不冲突。",
        "检查主文/补充材料编号连续性；更新 Response to Reviewers 中对应条目。",
        "确认图件为 300 dpi TIFF/PNG、蓝绿色系；表格使用三线表或现有期刊模板。",
        "若期刊不允许新增补充图，可把 Figure S30 合并进 Table S42 的脚注。",
        "确认 5-seed 平均、per-seed ρ 范围已写入 Table S43 与主文；不要留下“seed-11 exploratory”的旧表述。",
    ]
    for x in todo:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(x)
        r.font.size = Pt(10)

    # 9
    doc.add_heading("9. 附录：ECE、双向注意力与 ranking loss 的精确方法文字", level=1)
    add_para(doc, "以下三段为可直接粘贴进 Methods / Supplementary Methods 的英文文字；"
                  "同时列出必须同步修改的现有表述，避免方法与代码不一致。", size=9.5)

    doc.add_heading("9.1 Expected calibration error（ECE）", level=2)
    ece_text = (
        "Expected calibration error (ECE) was computed after jointly rescaling predictions and targets "
        "to [0,1]. For an evaluation set with targets y and predictions ŷ, we defined "
        "a = min(min(y), min(ŷ)) and c = max(max(y), max(ŷ)); both vectors were transformed as "
        "(· − a)/(c − a). Predictions were assigned to 10 equal-width bins defined by [0,0.1), …, [0.9,1.0]. "
        "ECE was then computed as Σ_{b=1}^{10} (n_b/n) |mean(ŷ_b) − mean(y_b)|, where n_b is the number of "
        "samples in bin b. This is a pooled global ECE; per-target ECE values stored in per-target files use "
        "target-local min–max scaling and are not used for the global tables. No sigmoid or logit transform "
        "was applied."
    )
    add_box(doc, ece_text, size=9.5)
    add_code(doc,
             "a = min(min(y_true), min(y_pred))\n"
             "c = max(max(y_true), max(y_pred))\n"
             "yt = (y_true - a) / (c - a)\n"
             "yp = (y_pred - a) / (c - a)\n"
             "bins = digitize(yp, linspace(0, 1, 11)[1:-1])\n"
             "ECE = Σ_b (n_b / n) * |mean(yp_b) - mean(yt_b)|")
    add_note(doc, "需要同步修改：若现有方法段写的是“per-target macro ECE”或“先逐靶点归一化再平均”，"
                  "改为“pooled global ECE”。BindingDB/Davis 面板当前没有 ECE 列；若要补，使用同一函数。")

    doc.add_heading("9.2 双向注意力的 token 数与张量形状", level=2)
    attn_text = (
        "Bidirectional interaction was implemented with one pooled token per modality. The molecular encoder "
        "maps Morgan fingerprints (2048), MACCS keys (167), and physicochemical descriptors (10) to 512-, "
        "256-, and 128-dimensional vectors, respectively; these are concatenated (896) and projected to a "
        "single 512-dimensional drug token. The protein branch projects the mean-pooled ESM2 embedding (128) "
        "to a single 512-dimensional protein token. Both tokens are unsqueezed to sequence length one, giving "
        "drug and protein tensors of shape (B, 1, 512). The bidirectional block contains two layers, each "
        "with two 8-head cross-attention directions; with sequence length one, the attention weights are "
        "trivially (B, 8, 1, 1) = 1, so the block acts as a head-wise linear projection rather than a "
        "token-alignment mechanism. The consistency loss is the mean over layers of the MSE between the two "
        "512-dimensional outputs, not a comparison of attention maps. We therefore describe this component as "
        "a single-token bidirectional interaction and do not claim atom- or residue-level alignment."
    )
    add_box(doc, attn_text, size=9.5)
    add_table(doc,
              ["Component", "Input shape", "Output shape", "Token count"],
              [["Morgan projection", "(B, 2048)", "(B, 512)", "—"],
               ["MACCS projection", "(B, 167)", "(B, 256)", "—"],
               ["Descriptor projection", "(B, 10)", "(B, 128)", "—"],
               ["Drug fusion", "(B, 896)", "(B, 512)", "1"],
               ["Protein projection", "(B, 128)", "(B, 512)", "1"],
               ["Cross-attention Q/K/V", "(B, 1, 512)", "(B, 8, 1, 64)", "1 each"],
               ["Attention scores", "(B, 8, 1, 1)", "weights = 1", "—"],
               ["Fusion / prediction", "(B, 1024) → (B, 512)", "(B, 1) → (B,)", "—"]],
              widths=[4.2, 4.2, 4.6, 2.6], font=8)
    add_note(doc, "需要同步修改：若现有方法段写“atom-level / residue-level cross-attention”或“token alignment”，"
                  "改为“single-token bidirectional interaction”；并在 Limitations 中明确单 token 退化。"
                  "真正的多 token 注意力需要把 Morgan 子结构/ESM2 残基作为 token 并重新训练，会改变全部结果。")

    doc.add_heading("9.3 Ranking loss 的权重与定义", level=2)
    rank_text = (
        "Ranking was optimized with a listwise ListNet-style loss applied within each target-specific "
        "mini-batch. For a batch of n ≥ 2 pairs with targets y_i and predictions ŷ_i, we defined "
        "q_i = exp(y_i)/Σ_j exp(y_j) and p_i = exp(ŷ_i)/Σ_j exp(ŷ_j). The ranking term was "
        "L_rank = KL(q||p) = (1/n) Σ_i q_i (log q_i − log p_i), implemented as "
        "F.kl_div(log p, q, reduction='batchmean'). The total objective was "
        "L = L_MSE + 0.3 L_rank + 0.05 L_contrastive + 0.1 L_consistency. No temperature scaling or pairwise "
        "margin was used; batches with n < 2 were skipped. The same weights were used for the full Reptile "
        "model, the budget-matched direct-supervised control, and the panel models; module-deletion variants "
        "set the corresponding weight to zero."
    )
    add_box(doc, rank_text, size=9.5)
    add_code(doc,
             "p_i = softmax(pred_i)        # over the target mini-batch\n"
             "q_i = softmax(y_i)           # over the target mini-batch\n"
             "L_rank = KL(q || p) = (1/n) Σ_i q_i (log q_i - log p_i)\n"
             "L = L_MSE + 0.3 * L_rank + 0.05 * L_contrastive + 0.1 * L_consistency")
    add_note(doc, "需要同步修改：若现有材料只写“ranking loss”或“pairwise ranking”，补上 ListNet/KL、权重 0.3、"
                  "逐靶点 mini-batch、softmax 维度、batchmean 与 n<2 跳过。")

    doc.add_heading("9.4 论文中需要同步替换/补充的表述", level=2)
    add_para(doc, "（a）MACCS 配置（morgan_descriptors）", bold=True, size=9.5)
    maccs_text = (
        "In the morgan_descriptors configuration, MACCS inputs were set to zero during both training and "
        "inference, while the 167→256 projection branch was retained; the branch therefore contributes a "
        "constant offset and the fusion input remains 896-dimensional. We therefore describe this setting as "
        "MACCS-masked rather than MACCS-removed."
    )
    add_box(doc, maccs_text, size=9.5)
    add_note(doc, "如果现有方法/图表写“MACCS removed”“w/o MACCS branch”或“fusion input 640”，需要改为"
                  "“MACCS input masked; 167→256 branch retained; fusion input 896”。真正的分支删除是新模型变体，"
                  "需要重新训练。")

    add_para(doc, "（b）蛋白 128 维特征", bold=True, size=9.5)
    protein_text = (
        "Protein inputs were precomputed once: the final-layer ESM-2 representation (480 dimensions) was "
        "mean-pooled over tokens and passed through a fixed, randomly initialized linear projection to 128 "
        "dimensions. ESM-2 and this projection were frozen and not optimized during training. The "
        "128-dimensional vector was then projected to 512 dimensions inside the model."
    )
    add_box(doc, protein_text, size=9.5)
    add_note(doc, "如果现有材料写“learned projection”“fine-tuned ESM-2”或“raw 480-d input”，需要改为"
                  "“fixed random linear projection of frozen ESM-2; 128-d precomputed feature”。")

    add_para(doc, "（c）其他三处需要同步修改的表述", bold=True, size=9.5)
    for x in [
        "ECE：把“per-target macro ECE / 逐靶点归一化后平均”改为“pooled global ECE；联合 min–max；10 个等宽分箱”。",
        "注意力：把“atom-level / residue-level cross-attention / token alignment”改为“single-token bidirectional interaction”，"
        "并在 Limitations 中说明单 token 退化。",
        "Ranking：把“pairwise ranking”或未定义权重改为“ListNet-style KL(q||p), weight 0.3, per-target mini-batch, batchmean”。",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(x)
        r.font.size = Pt(10)

    doc.save(OUT_DOCX)
    print("[saved]", OUT_DOCX)


if __name__ == "__main__":
    make_figures()
    build_doc()
