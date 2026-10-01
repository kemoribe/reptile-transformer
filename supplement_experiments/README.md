# 补充实验脚本与结果（revision2 / remaining）

本目录归档论文**除 P0 同源性稳健性实验之外**的所有补充实验脚本与汇总结果。
P0 审计稳健性实验见 [`../paper_supplement/`](../paper_supplement/)，
可复用的通用审计工具见 [`../homology_audit/`](../homology_audit/)。

## 收纳原则

* **脚本**：全部收录（`scripts/`，均为文本 `.py`）。
* **结果**：只收录**汇总表与图**（CSV / JSON / TXT / PNG，单文件 < 5 MB）。
* **不收录（可由脚本 + 发布数据/权重复现，避免仓库膨胀）**：
  模型权重（`*.pt`/`*.pth`/检查点）、逐对预测（`*.npz`）、预计算特征
  （`*.npy`/`features.npz`）、MMseqs2 中间产物（`clu_*.fasta/tsv`）、
  300 dpi **TIFF** 图（PNG 版已在本目录，TIFF 见 Release 附件）。
  对应数据/权重下载地址见根目录 [`DATA.md`](../DATA.md) 与
  [`docs/EXPERIMENT_ARTIFACTS_UPLOAD.md`](../docs/EXPERIMENT_ARTIFACTS_UPLOAD.md)。

## 实验索引

| 编号 | 实验 | 脚本 | 结果目录 | 论文位置 |
|---|---|---|---|---|
| E1 | MAML inner steps 数量消融 | `supplement_e1_inner_steps.py` | `results/revision2_e1/e1_results.csv` | Table S14 |
| E2 | λ（同源性惩罚）OFAT 敏感性 | `supplement_e2_lambda_ofat.py` | `results/revision2_e2/` | Table S17, Figure S7 |
| E3 | 梯度冲突（GNN vs Transformer） | `supplement_e3_grad_conflict.py` | `results/revision2_e3/` | §3.5.2 |
| E4 | ESM-2 冻结层数消融 | `supplement_e4_esm2_freeze.py` | （仅权重/预测，无独立汇总表） | Table S15 |
| E5 | 五种子重复 + Wilcoxon/Bootstrap | `supplement_e5_repeats.py`, `supplement_e5_stats.py` | `results/revision2_e5/` | Table S13, Figure S10 |
| E6 | MAML vs Reptile 元学习器对照 | `supplement_e6_maml.py` | （仅权重/预测） | Table S16 |
| E7 | τ 阈值敏感性（ECFP4 口径） | `run_e7_tau_sensitivity.py` | `results/revision2_e7/` | Table S3（补充口径，论文主口径见 exp6） |
| E8 | 40% 同源自由子集消融（审计 vs 标准） | `run_e8_homology_free_ablation.py` | `results/revision2_e8/homology_free_ablation_results.csv` | §3.4, Table S21 |
| exp2 | McNemar / Fisher 显著性 | `supplement_remaining_experiments.py` | `results/remaining_exp2/exp2_mcnemar_fisher_report.txt` | §3.5.1 |
| exp3 | 结合谱镜像审计 + Fisher 分层 | `supplement_remaining_experiments.py` | `results/remaining_exp3/` | §3.3, Table S24, Figure S9 |
| exp6 | 概念框架（τ / ΔR²） | `supplement_remaining_experiments.py` | `results/remaining_exp6/` | §3.2.1, Figure 6, Table S3 |
| exp8 | 多重比较 p 值热图 | `supplement_remaining_experiments.py` | `results/remaining_exp8/` | Figure S10 |
| exp9 | 标准化（TargetScaler）消融 | `supplement_remaining_experiments.py` | `results/remaining_exp9/exp9_std_report.txt` | Table S18 |
| exp12 | 蛋白质语言模型（ESM-1b / ProtT5）消融 | `exp12_protein_lm_ablation.py` | `results/remaining_exp12/` | Table S19 |
| exp14 | family-remote 路由器阈值扫描 | `supplement_remaining_experiments.py` | `results/remaining_exp14/` | Table S25 |
| λ | λ 敏感性主结果（与 E2 互补） | `supplement_s9_lambda_sensitivity.py` | `results/lambda_sensitivity/lambda_results.csv` | Table S9 |
| GNN | GNN 可解释性（Grad-CAM/attribution） | `supplement_gnn_explain.py`, `supplement_s4_attribution.py` | `results/gnn_explain/{davis,kiba}/` | Figure 8, Table S23 |

辅助脚本：`supplement_exp_common.py`（共享工具）、
`supplement_revision2_master.py`（revision2 总控）、
`supplement_revision_pack.py`（打包）、
`supplement_ef_summary.py`（EF 汇总）、
`supplement_merge_final.py`、`supplement_recommendation_tables.py`、
`supplement_s5_lambda_heatmap.py`、`supplement_s6_calibration.py`、
`supplement_s8_cost.py`、`supplement_target_clustering.py`（同源性聚类，
已被 `homology_audit` 工具包通用化，此处保留原始版本）。

## 复现

1. 按根 README 配置环境并下载四个数据集与模型权重。
2. 进入本目录 `scripts/`，按上表运行对应脚本（部分脚本含本机绝对路径，
   运行前按你机器的路径修改顶部 `BASE` 常量；`homology_audit` 工具包已去掉
   此依赖，新工作优先用工具包）。
3. 汇总结果会写入脚本顶部 `OUT` 指向的目录，与本目录 `results/` 中的文件对照
   即可验证数值一致。

## 与 P0 工具包的关系

* E8（同源自由消融）的**通用化版本**即 `homology_audit evaluate` 命令；
  E14（路由器）的通用化版本即 `homology_audit router`；
  `supplement_target_clustering.py` 的通用化版本即 `homology_audit audit`。
* 本目录脚本是论文当时的运行记录，保留用于数值溯源；
  新研究请直接使用 [`homology_audit/`](../homology_audit/)。
