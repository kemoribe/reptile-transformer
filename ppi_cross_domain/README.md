# P1: SKEMPI v2 跨领域同源性审计与结合谱镜像验证

本目录包含 P1 实验（论文 §3.6.5 的跨领域延伸）的全部脚本、结果数据和图表。
P1 将同源性审计框架从 DTA（药物-靶点亲和力）数据集（Davis/KIBA/BindingDB，
P0 实验）扩展到 PPI（蛋白质-蛋白质相互作用）数据集（SKEMPI v2），验证：

1. **PPI 数据集在序列层面天然干净**（SKEMPI Pr/PI hold-out 跨簇 98.55%，
   仅 1/69 同簇泄漏——1EAW_A，与 DTA ~59% 泄漏率形成"脏 vs 净"对照）
2. **但序列干净 ≠ 功能干净**：结合谱镜像审计发现 59 个序列级跨簇复合物中
   11 个（18.6%）存在 ΔΔG 分布层面隐性冗余（KS ≥ 0.7）
3. **审计能精确识别泄漏受益者**：模型动物园实验证明纯身份记忆探针
   （ESM2-mean+MLP）审计后 ρ 跌 24%，中游排名反转（site+TF ↔ RF），
   冠军 site+MLP 稳健

## 目录结构

```
ppi_cross_domain/
├── README.md                 # 本文件
├── scripts/                  # 实验脚本（按流程顺序）
│   ├── fetch_sequences.py        # 1. RCSB FASTA 抓取（252 PDB）
│   ├── fix_missing_mmcif.py      # 2. mmCIF entity_poly 兜底解析
│   ├── fetch_new_chains.py      # 3. 扩展训练集后补抓新 PDB
│   ├── make_audit_inputs.py     # 4. 生成审计输入 train.fasta/test.fasta/labels.csv
│   ├── p1_1_numbering.py         # 5. mmCIF 编号映射（pdbx_poly_seq_scheme）
│   ├── p1_1b_resolve_sites.py    # 6. 突变位点解析（双路覆盖率扫描）
│   ├── p1_2_esm2_embed.py        # 7. ESM2 embedding 预计算
│   ├── p1_3_train_eval.py        # 8. 模型动物园训练评估（7 模型 × 3 seeds × 3 评估集）
│   ├── action3_binding_spectrum.py  # 9. 结合谱镜像审计
│   └── plot_p1_figures.py        # 10. 三张出版图绘制
├── results/                  # 实验结果
│   ├── p1_model_results.csv      # 模型动物园评估结果（Spearman 等指标 × 3 评估集）
│   ├── sites_resolved.json       # 突变位点解析记录（4864 行）
│   ├── p1_3_log.txt              # 训练日志
│   ├── p1_3_console.txt          # 控制台输出
│   ├── resolution_summary.txt    # 位点解析率汇总
│   ├── audit_vs_standard_results.csv   # 审计 vs 标准评估对比
│   └── binding_spectrum_mirror_results.csv  # 结合谱镜像审计结果
├── figures/                  # 出版图（PNG 300dpi）
│   ├── fig1_cross_cluster_dta_vs_ppi.png  # DTA（脏）vs PPI（净）跨簇比例
│   ├── fig2_ppi_mirror_redundancy.png     # PPI 结合谱镜像冗余（KS ≥ 0.7）
│   └── fig3_rank_reversal_arrows.png      # 标准 vs 审计评估排名变化
├── audit_results/            # MMseqs2 聚类审计结果
│   └── skempi_prpi_expanded/      # 扩展版审计（40%/60%/80% 阈值）
└── skempi_prpi/              # 审计输入 + 原始数据
    ├── train.fasta              # 训练集序列（557 链 / 469 去重）
    ├── test.fasta               # 测试集序列（120 链 / 69 去重）
    ├── labels.csv               # 审计标签
    └── skempi_v2.csv            # SKEMPI v2 原始数据（7085 行）
```

## 复现步骤

```bash
# 前提：已安装 homology_audit 工具包（pip install -e ../homology_audit）
#       MMseqs2 可执行（HOMOLOGY_AUDIT_MMSEQS 环境变量或 PATH）
#       ESM2 模型（facebook/esm2_t33_650M_UR50D，需 GPU）

cd scripts

# 1-4. 生成审计输入（已预跑，结果在 skempi_prpi/）
# python make_audit_inputs.py

# 5-6. 突变位点解析（已预跑，结果在 results/sites_resolved.json）
# python p1_1_numbering.py    # 下载 mmCIF + 建编号映射
# python p1_1b_resolve_sites.py  # 双路覆盖率扫描解析突变位点

# 7. ESM2 embedding 预计算（需 GPU，~408 MB npy 不含在本目录中）
# python p1_2_esm2_embed.py    # 输出 esm2_emb/*.npy

# 8. 模型动物园训练评估
# python p1_3_train_eval.py    # 输出 p1_model_results.csv

# 9. 结合谱镜像审计
# python action3_binding_spectrum.py  # 输出 binding_spectrum_mirror_results.csv

# 10. 绘图
# python plot_p1_figures.py   # 输出 figures/*.png + *.tif
```

## 核心结果

| 指标 | 数值 |
|---|---|
| 跨簇比例 @40%（Pr/PI hold-out） | 98.55%（68/69） |
| 唯一序列级泄漏 | 1EAW_A（99.5% 同一性，AB/AG × Pr/PI 交叉污染） |
| 结合谱镜像冗余（KS≥0.7，跨簇子集） | 18.6%（11/59） |
| 冠军模型（site+MLP）Spearman | 0.345 → 0.348（审计前后稳健） |
| 身份记忆探针（mean+MLP）Spearman | 0.105 → 0.080（审计后跌 24%） |
| 排名反转 | site+TF 3↔4 RF（Spearman） |

## 与 homology_audit 工具包的关系

P1 的 `action3_binding_spectrum.py` 已泛化为工具包中的
`homology_audit.mirror` 模块（CLI: `homology-audit mirror`），
支持任意突变级 ΔΔG 数据的镜像审计。
