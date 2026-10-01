# 论文 V3 补充材料：P0 稳健性实验 + 论文附录归档

本目录归档论文 **V3_P0稳健性验证版**（`Paper_V3_P0_robustness_zh.docx`，
§3.6.5、Table 6、Tables S26–S31、Figure S11）对应的全部 P0 优先级稳健性
实验脚本、数值结果、论文写回脚本，以及论文附录表格（S2–S12）与补充图（S1–S11）。

> **可复用版本请使用 [`../homology_audit/`](../homology_audit/) 工具包。**
> `p0_experiments/` 下的脚本是论文实验的原始运行记录（含本机绝对路径、
> 中文输出），归档用于数值溯源；工具包是这些逻辑的通用化、文档化实现，
> 命令与论文条目的对应关系见工具包 README 第 7 节。

## 目录内容

```
Paper_V3_P0_robustness_zh.docx # 写回 P0 内容后的论文（新增 §3.6.5 等；中文原稿）
gen_paper_v3_p0.py             # 论文写回脚本（provenance；含本机路径）
p0_experiments/
  p0_exp1_tau_lodo.py          # P0-1 τ≈0.13 阈值 LODO → Table S31
  p0_exp2_router_lodo.py       # P0-2 family-remote 路由器 LODO → Table S26
  p0_exp3a_mmseqs_grid.py      # P0-3a 参数敏感性 18 格扫描 → Table S27 / Figure S11
  p0_exp3b_cluster_algorithm.py# P0-3b set-cover vs CD-HIT 式贪心 → Table S28
  p0_exp4_audit_vs_standard.py # P0-4 审计 vs 标准评估+误选代价 → Table S29
  p0_exp5_negative_control.py  # P0-5 随机置乱负对照 → Table S30
  plot_exp3_heatmap.py         # Figure S11 出图（300 dpi PNG/TIFF）
  *.csv / *.json / *_report.txt# 全部数值结果与文字报告
  param_sensitivity/           # 网格与算法对照结果表
  negative_control/            # 负对照汇总结果表
protein_cluster_labels/
  cross_cluster_summary.csv/.json     # 四数据集 40/60/80% 跨簇汇总（审计主结果）
  <dataset>/per_test_target_identity.csv  # 逐测试靶点标签（26/65/34/112 靶点）
appendix_tables/               # 论文附录 Table S2–S12（CSV，实验最终汇总表）
figures/                       # 论文补充图 Figure S1–S8（300 dpi PNG；TIFF 见 Release 附件）
```

注：MMseqs2 的 FASTA/TSV 中间产物（`clu_*`、`sequences_*.fasta` 等，约 60 MB）
可由发布数据集经 `homology-audit audit-batch` 完整重生成，故不纳入仓库；
300 dpi TIFF 版 Figure S11 及其他补充图 TIFF 见 GitHub Release 附件
`figure-s11-300dpi-v1.1.0.zip`。GNN 逐对预测（`gnn_preds/*.npz`）属二进制中间
产物，可由模型权重复现，不纳入。

## 五项实验结论摘要

| 实验 | 结论 | 对论文声称的动作 |
|---|---|---|
| P0-1 τ LODO | 带规则准确率 0.50（2/4）≤ 多数类基线 0.75；下界 0.130 稳定，上界 0.194–0.249 | τ≈0.13 **降级为 descriptive band**，不作部署阈值 |
| P0-2 路由器 LODO | 四折 95.5–97.1%，FNR 全 0，每折自选阈值均为 40% | 泛化声称成立，保留 validated 表述 |
| P0-3a 参数网格 | c=0.8×40/60/80% 共 12/12 精确复现原审计；全网格跨簇比例 26.1–100% | 审计不依赖单组参数 |
| P0-3b 算法对照 | 标签一致率 92.3–100%，Cohen κ 0.843–1.000，最大偏差 7.7 pp | 审计不依赖特定聚类算法 |
| P0-4 评估口径 | R² Kendall τb：davis 0.00 / bindingdb 0.33 / kiba 0.80；R² winner 2/3 易主；误选 ΔR² 0.0144–0.0154 | 量化不审计的模型误选代价 |
| P0-5 负对照 | 4 数据集×3 种子×3 阈值 = 36/36 设置跨簇 100%，最近一致性 max/mean 0% | 跨簇判定捕捉真实同源信号 |

## 复现方式

```bash
# 等价的通用实现（推荐）
pip install -e ../homology_audit
homology-audit audit-batch --data-root <数据目录> \
    --datasets chembl,davis,kiba,bindingdb --out audit/
homology-audit router --labels-root audit/ --datasets chembl,davis,kiba,bindingdb --out router/
# sensitivity / algorithm-check / negative-control / evaluate / band-lodo 见工具包 README
```

数据源口径注意：τ 与 ΔR² 使用论文 Table S3 权威值
（`p0_exp1` 内置：ChEMBL 0.122/−0.2041，Davis 0.137/+0.1412，KIBA 0.250/−0.0989，
BindingDB 0.360/−0.1340）。
