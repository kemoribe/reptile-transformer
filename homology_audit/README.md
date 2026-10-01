# homology-audit — 靶点冷启动 DTA 的序列同源性审计工具包

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![MMseqs2](https://img.shields.io/badge/MMseqs2-required-success)]()

药物–靶点亲和力（DTA）模型常用的"靶点冷启动"划分，测试靶点仍可能与训练靶点
属于同一蛋白家族（序列同一性 ≥40%），由此产生的同源泄漏会让基准分数虚高、
模型排名失真。本工具包基于 [MMseqs2](https://github.com/soedinglab/MMseqs2)
把整套审计流程封装为**一条命令链**，并附论文所要求的稳健性验证：

| 命令 | 功能 | 对应论文内容 |
|---|---|---|
| `audit` / `audit-batch` | 训练/测试靶点聚类、最近同一性、跨簇标签、跨簇比例 | §3.1 同源性审计；Tables S27 口径 |
| `router` | family-remote 部署路由器 + 留一数据集（LODO）验证 | Table S26；§3.6 路由器 |
| `sensitivity` | 覆盖率 × 同一性参数网格扫描 + 原审计复现校验 | Table S27；Figure S11 |
| `algorithm-check` | set-cover 与 CD-HIT 式贪心增量两种聚类算法对照 | Table S28 |
| `negative-control` | 残基置乱随机序列负对照（保持长度/组成） | Table S30 |
| `evaluate` | 审计子集 vs 标准全集评估：排名、winner 易主、误选代价 | Table S29 |
| `band-lodo` | 一维适用性带/阈值的留一面板验证（含 1-NN、多数类基线） | Table S31（τ≈0.13 降级依据） |

---

## 1. 安装

### 1.1 Python 依赖

需要 Python 3.8+。建议在虚拟环境中安装：

```bash
pip install -r requirements.txt        # numpy / pandas / scipy
pip install -e .                       # 安装 homology-audit 命令（可选，推荐）
```

不安装也可直接在本目录用 `python -m homology_audit.cli ...` 运行。

### 1.2 MMseqs2（必需，用于聚类和比对）

三种方式任选其一：

* **Linux / conda（最省事）**：`conda install -c conda-forge mmseqs2`
* **macOS**：`brew install mmseqs2`
* **Windows / 手动安装**：从
  <https://github.com/soedinglab/MMseqs2/releases> 下载 `mmseqs-win64.zip`，
  解压后目录内含 `mmseqs.exe` 与 Cygwin/busybox 运行时，**整个目录要保留在
  一起**。

工具包按以下顺序寻找二进制：

1. 命令行参数 `--mmseqs /完整路径/mmseqs[.exe]`
2. 环境变量 `HOMOLOGY_AUDIT_MMSEQS`
3. `PATH` 上的 `mmseqs` / `mmseqs.exe`

Windows 下无需手动配置 PATH——工具会自动把 `mmseqs.exe` 所在目录加入子进程
PATH（busybox 依赖需要）。

验证：

```bash
mmseqs version          # 或 mmseqs.exe version
python -m homology_audit.cli --version
```

---

## 2. 五分钟快速上手（内置合成示例）

仓库 `examples/` 下附带一个**完全合成、可直接运行**的小数据集
（`make_example_data.py` 以固定随机种子生成，两个面板 `demo`/`demo2`，
序列长 160 aa；无需任何外部数据）：

```bash
# Linux / macOS
bash examples/run_example.sh

# Windows PowerShell
.\examples\run_example.ps1 -Mmseqs "D:\tools\mmseqs\bin\mmseqs.exe"
```

或逐步执行（以下命令在工具包根目录运行）：

```bash
# 0) 生成示例数据（examples/data 与 examples/predictions）
python examples/make_example_data.py

# 1) 批量审计：demo / demo2，同一性 40/60/80%，双向覆盖 0.8
python -m homology_audit.cli audit-batch \
    --data-root examples/data --datasets demo,demo2 \
    --out examples/output/audit --thresholds 0.4,0.6,0.8

# 2) 路由器 LODO 验证（阈值在其余面板上选择，留出面板测试）
python -m homology_audit.cli router \
    --labels-root examples/output/audit --datasets demo,demo2 \
    --out examples/output/router

# 3) 参数敏感性网格 + 与原审计的复现校验
python -m homology_audit.cli sensitivity \
    --dataset-dir examples/output/audit/demo \
    --reference examples/output/audit/audit_summary.csv \
    --out examples/output/sens/demo --name demo

# 4) 聚类算法对照（MMseqs2 set-cover vs CD-HIT 式贪心增量）
python -m homology_audit.cli algorithm-check \
    --dataset-dir examples/output/audit/demo \
    --out examples/output/sens/demo

# 5) 随机序列负对照（3 个种子，40/60/80%）
python -m homology_audit.cli negative-control \
    --dataset-dir examples/output/audit/demo \
    --out examples/output/negative/demo

# 6) 审计子集 vs 标准全集评估（排名、winner 易主、误选代价）
python -m homology_audit.cli evaluate \
    --predictions examples/predictions/demo_predictions.csv \
    --labels-root examples/output/audit --datasets demo \
    --out examples/output/eval

# 7) 适用性阈值带的留一面板验证（默认内置论文 Table S3 四点）
python -m homology_audit.cli band-lodo --out examples/output/band
```

示例的预期结果（合成数据，固定种子）：

* `audit`：demo 在 40/60/80% 同一性下跨簇比例 = **50% / 75% / 100%**；
* `router`：两折 LODO 选中阈值均为 40%，准确率 100%（示例设计如此）；
* `negative-control`：所有种子/阈值下跨簇比例 = **100%**，最近同一性 0%；
* `evaluate`：R² 的标准口径 winner 与审计口径 winner **发生易主**
  （`GraphLikeModel` → `RemoteExpert`，Kendall τ = −1），演示误选代价；
* `band-lodo`：带规则 LODO 准确率 0.50 ≤ 多数类基线 0.75，结论为
  *descriptive region*（与论文 τ≈0.13 的处理一致）。

自动化测试（含端到端）：

```bash
# Linux/macOS（mmseqs 在 PATH 上时全部测试；不在时自动跳过 MMseqs2 测试）
python -m unittest discover -s tests -v

# Windows PowerShell
$env:HOMOLOGY_AUDIT_TEST_MMSEQS="D:\tools\mmseqs\bin\mmseqs.exe"
python -m unittest discover -s tests -v
```

---

## 3. 输入文件规范

### 3.1 审计输入（两种方式二选一）

**方式 A — CSV（推荐，GraphDTA 风格）**：训练/测试各一个 CSV，至少包含
蛋白序列列（默认列名 `target_sequence`，可用 `--seq-column` 修改）：

```
compound_iso_smiles,target_sequence,target_id,...
CC...,MKTLLLTL... ,AAAAA
```

批量模式要求文件名符合 `<数据集名>_train.csv` / `<数据集名>_test.csv`
（模板可用 `--train-template` / `--test-template` 自定义）。
序列在文件内自动去重、按字典序排序后赋予确定性 ID：
`train_0000, train_0001, ...` 与 `test_0000, test_0001, ...`。

**方式 B — FASTA**：`audit --train-fasta ... --test-fasta ...`。
序列同样被去重并重新赋予 `train_#### / test_####` 命名空间 ID
（聚类标签依赖此前缀区分训练/测试成员）。

### 3.2 `evaluate` 的预测文件

长表 CSV，必需列：`dataset, model, y_true, y_pred`，以及下列任一连接键：

* `target_sequence`（推荐，直接与审计标签按序列连接），或
* `target_id`（与 `per_target_audit.csv` 中的 ID 一致）

```csv
dataset,model,target_sequence,y_true,y_pred
davis,GCNNet,MKTLLLTL...,5.0,4.82
```

预测行数与化合物数不受限制，但**每个模型必须覆盖相同的测试靶点集合**，
否则排名无意义（工具不强制，请自行保证）。

### 3.3 `band-lodo` 的点表（可选）

CSV 列：`name,x,y`（可选 `effect` 列展示效应量）。`y=1` 表示模型家族在该
面板占优。不给 `--points-csv` 时使用内置论文 Table S3 四点。

---

## 4. 输出文件说明

```
<out>/
├── audit_summary.csv                  # audit-batch 汇总（每个数据集×阈值一行）
├── audit_report.txt
└── <dataset>/
    ├── sequences_train.fasta          # 去重后的训练序列（train_#### ID）
    ├── sequences_test.fasta
    ├── sequences_all.fasta            # 训练+测试合并（聚类输入）
    ├── search_test_vs_train.m8        # 每个测试靶点→训练集的全部比对命中
    ├── clu_40_cluster.tsv             # MMseqs2 簇（代表<TAB>成员）
    ├── clu_60_cluster.tsv / clu_80_*
    ├── clu_40_rep_seq.fasta / *_all_seqs.fasta
    ├── per_target_audit.csv           # ★ 逐测试靶点标签（后续命令的输入）
    └── audit_summary_dataset.csv
```

`per_target_audit.csv` 关键列：

| 列 | 含义 |
|---|---|
| `target_id` | 确定性 ID（`test_####`） |
| `target_sequence` | 蛋白序列（便于与预测按序列连接） |
| `nearest_train_identity_pct` | 双向覆盖 0.8 下到训练集的最高序列同一性（%） |
| `same_cluster_40/60/80` | 在该同一性阈值下是否与任一训练靶点同簇 |

**跨簇比例（cross-cluster ratio）** = 不与任何训练靶点同簇的测试靶点占比，
即真正的同源自由（family-remote）比例。

其他命令的输出：

| 命令 | 主要产物 |
|---|---|
| `router` | `router_lodo.csv`（LODO 选阈值）、`router_fixed_threshold.csv`（固定预注册阈值对照）、`router_threshold_scan.csv`、`router_summary.json`、`router_report.txt` |
| `sensitivity` | `param_grid.csv`、`reproduction_check.csv`（与原审计逐格比对） |
| `algorithm-check` | `cluster_algorithm_comparison.csv`（簇数、跨簇比例、标签一致率、Cohen's κ） |
| `negative-control` | `negative_control_results.csv`、`negative_control_summary.csv`、`*_report.txt` |
| `evaluate` | `audited_evaluation_per_model.csv`（逐模型两口径值与名次）、`rank_comparison.csv`（Kendall/Spearman、winner、误选代价）、报告 |
| `band-lodo` | `band_lodo_folds.csv`、`band_lodo_boundaries.csv`、`band_lodo_summary.json` |

所有聚类命令**可断点续跑**：目标 `*_cluster.tsv` 已存在时自动跳过 MMseqs2
计算（删除该文件即可强制重算）。

---

## 5. 在你自己的 DTA 基准上使用（推荐流程）

```bash
# (1) 审计全部数据集
python -m homology_audit.cli audit-batch --data-root data/ \
    --datasets chembl,davis,kiba,bindingdb --out audit/ \
    --thresholds 0.4,0.6,0.8 --coverage 0.8 --threads 8

# (2) 用部署路由器检验“最近同一性 < 40% 即走 family-remote 分支”的跨面板泛化
python -m homology_audit.cli router --labels-root audit/ \
    --datasets chembl,davis,kiba,bindingdb --out router/ \
    --grid 25,60,0.5 --prereg 40

# (3) 逐数据集做参数稳健性、算法对照、负对照
for ds in chembl davis kiba bindingdb; do
  python -m homology_audit.cli sensitivity --dataset-dir audit/$ds \
      --reference audit/audit_summary.csv --out robustness/$ds/sens --name $ds
  python -m homology_audit.cli algorithm-check --dataset-dir audit/$ds \
      --out robustness/$ds/sens
  python -m homology_audit.cli negative-control --dataset-dir audit/$ds \
      --out robustness/$ds/neg --seeds 0,1,2
done

# (4) 用 40% 同源自由子集重评所有模型，量化“不审计的代价”
python -m homology_audit.cli evaluate --predictions all_predictions.csv \
    --labels-root audit/ --datasets davis,kiba,bindingdb \
    --identity-tag 40 --metrics "R2,EF@1%,ECE" --out eval/
```

解读原则：

* `router`：关注留出折准确率、FNR（漏判 remote 的代价更高）以及数据自选阈值
  是否与预注册阈值一致；
* `sensitivity`：`reproduction_check.csv` 必须全为 `True`；跨簇比例在合理参数
  范围内不应发生结论性翻转；
* `algorithm-check`：κ ≥ 0.8 可认为审计结论不依赖特定聚类算法；
* `negative-control`：期望 100% 跨簇、最近同一性显著低于审计阈值；
* `evaluate`：`winner_reversed=True` 的指标即存在"标准评估选错模型"，
  `misselection_cost` 为审计子集上的绝对性能损失。

---

## 6. 指标口径（evaluate）

与论文同源自由消融协议一致，在**被评估面板内的全部（靶点, 化合物）对上池化**：

* **R² / RMSE / MAE**：面板内池化计算；
* **EF@k%**：以面板内活性最高的 20%（不少于 5 个）为活性集，前 k% 预测中的
  活性富集倍数。面板太小导致 `int(n·k%) = 0` 时返回 NaN（论文 Davis EF@1%
  退化面板即此情况，报告时需注明样本量）；
* **ECE**：对面板内 y_true/y_pred 联合 min-max 归一化后分 10 个等宽箱计算。

注意：这里的 EF 是**面板内池化**口径，与模型训练阶段使用的"逐靶点 EF 再平均"
是两种不同约定，不可混用。

---

## 7. 论文 P0 稳健性实验的复现

本工具包即论文新增 §3.6.5 / Tables S26–S31 / Figure S11 的实现来源。
在四个基准数据集（GraphDTA 风格 CSV）上：

| 论文条目 | 命令 / 数据 |
|---|---|
| Table S26 路由器 LODO | `router`（四折；论文结果 95.5–97.1%，FNR=0，每折自选阈值=40%） |
| Table S27 参数网格 | `sensitivity`（cov 0.7/0.8/0.9 × id 30–80%；c=0.8 的 40/60/80% 12/12 精确复现） |
| Table S28 算法对照 | `algorithm-check`（论文 κ=0.843–1.000，跨簇比例最大偏差 7.7 pp） |
| Table S29 审计 vs 标准 | `evaluate`（R² winner 2/3 易主；误选代价 ΔR² 0.0144–0.0154） |
| Table S30 负对照 | `negative-control`（36/36 设置跨簇 100%，最近一致性 max/mean=0%） |
| Table S31 τ≈0.13 LODO | `band-lodo`（带规则 0.50、1-NN 0.50、多数类 0.75 → 降级为 descriptive band） |

Figure S11 的热力图可直接用 `sensitivity` 输出的 `param_grid.csv` 绘制
（交叉簇比例透视：行 `dataset/coverage`、列 `identity_pct`）。

---

## 8. Python API

所有命令都有同名 Python 函数，可嵌入自己的流程：

```python
from pathlib import Path
from homology_audit.audit import run_batch
from homology_audit.router import run_router_validation, discover_label_paths
from homology_audit.evaluate import run_evaluation

run_batch(data_root=Path("data"), datasets=["davis", "kiba"],
          out_root=Path("audit"), thresholds=(0.4, 0.6, 0.8),
          mmseqs_path=Path("/usr/local/bin/mmseqs"))
run_router_validation(discover_label_paths(Path("audit"), ["davis", "kiba"]),
                      out_dir=Path("router"))
run_evaluation(Path("preds.csv"), Path("audit"), ["davis", "kiba"],
               out_dir=Path("eval"))
```

---

## 9. 常见问题

**Q1：Windows 报 `Input xxx.fasta does not exist` 或 busybox 相关错误？**
请把 `--mmseqs` 指到**解压目录里**的 `mmseqs.exe`，不要只拷贝单独的 exe；
工具会自动将其同目录加入 PATH。使用相对路径即可（内部已转为绝对路径）。

**Q2：负对照的最近同一性为什么是 0%？**
双向覆盖（`-c 0.8`）要求两条序列 80% 长度相互覆盖。长蛋白经置乱后只可能出现
短的随机局部相似，会被覆盖度过滤——0% 命中是该算法设置下的预期行为，正是阴性
对照要证明的性质。

**Q3：序列长度/数量对结果有影响吗？**
审计按**去重后的靶点序列**计数，不按（靶点, 化合物）行数。同一性阈值、覆盖度
均显式可配；`audit_summary.csv` 记录了全部参数，便于回溯。

**Q4：EF@1% 出现 NaN？**
面板化合物数不足 100 时前 1% 位置数为 0，该指标在该面板上不可估，报告时需注明
面板大小（不要填 0）。

**Q5：想强制重算？**
删除对应的 `clu_*_cluster.tsv`（或负对照/网格输出目录）后重跑即可；
m8 搜索每次都会重新执行。

**Q6：支持其他聚类软件吗？**
`algorithm-check` 用 MMseqs2 `--cluster-mode 2`（longest-seed 贪心增量，
CD-HIT 的算法对应物）给出对照；本工具不直接依赖 CD-HIT。

---

## 10. 引用

如使用本工具包，请引用论文对同源性审计与 P0 稳健性实验的相应章节
（§3.1、§3.6.5，Tables S26–S31，Figure S11），以及
[MMseqs2](https://github.com/soedinglab/MMseqs2)（Steinegger & Söding, 2017）。

## 许可

MIT（MMseqs2 遵循其各自许可）。
