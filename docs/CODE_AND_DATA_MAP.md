# 代码与数据位置总表

本文档用于确认哪些内容进入 GitHub 仓库、哪些内容作为 GitHub Release
附件，以及每个模型和预处理步骤对应的文件。

## 七个模型

| 模型 | 模型定义/训练入口 | 说明 |
|---|---|---|
| GCN | `graphdta/models/gcn.py`、`graphdta/training.py` | GraphDTA 分子图基线 |
| GAT | `graphdta/models/gat.py`、`graphdta/training.py` | GraphDTA 分子图基线 |
| GIN | `graphdta/models/ginconv.py`、`graphdta/training.py` | GraphDTA 分子图基线 |
| GAT-GCN | `graphdta/models/gat_gcn.py`、`graphdta/training.py` | GraphDTA 混合图网络基线 |
| MLP | `run_baseline_mlp.py` | 固定长度药物与蛋白特征 |
| Transformer | `run_transformer_baseline.py` | 无元学习的 Transformer 基线 |
| Reptile-Transformer | `reptile_transformer_model.py`、`reptile_training.py`、`run_reptile_transformer.py` | Reptile 元学习模型 |

GraphDTA 数据转换和加载还依赖：

- `graphdta/convert_to_graphdta.py`
- `graphdta/convert_kiba.py`
- `graphdta/convert_bindingdb_to_graphdta.py`
- `graphdta/create_data.py`
- `graphdta/utils.py`

MLP、Transformer 和 Reptile-Transformer 共同依赖
`data_preprocessing.py`。特征消融入口为：

- `run_transformer_baseline_ablation.py`
- `run_reptile_transformer_ablation.py`

发布检查入口：

- `scripts/validate_data.py`：检查数据行数、SHA256 和目标隔离；
- `scripts/smoke_test_models.py`：用合成输入检查七个模型的前向传播；
- `scripts/check_esm2_model.py`：确认本地权重由 Hugging Face
  `transformers.AutoModel` 加载并输出 480 维嵌入；
- `scripts/package_release.ps1`：生成代码包、四个数据包和校验文件；
- `scripts/verify_transfer.ps1`：在目标电脑验证传输包。

## 最终特征配置

最终 MLP、Transformer 和 Reptile-Transformer 使用：

- 药物：Morgan fingerprint，`radius=2`、`nBits=2048`；
- 药物：10 个 RDKit 理化描述符；
- 蛋白质：`facebook/esm2_t12_35M_UR50D` 的 480 维平均池化嵌入；
- MACCS 分支为保持网络输入结构而保留，但在
  `--ablation morgan_descriptors` 下置零，不属于最终有效特征。

README 或论文中可使用以下英文表述：

> Unless otherwise stated, the MLP, Transformer, and
> Reptile-Transformer models use a 2,048-bit Morgan fingerprint
> (radius 2) concatenated with 10 RDKit physicochemical descriptors
> for compound representation, together with a 480-dimensional
> mean-pooled ESM-2 (`facebook/esm2_t12_35M_UR50D`) protein
> embedding. The MACCS branch is retained only for architectural
> compatibility and is zero-masked in the final
> `morgan_descriptors` configuration.

GCN、GAT、GIN 和 GAT-GCN 不使用该固定向量组合。它们使用 RDKit
分子图和整数编码蛋白质序列，保持 GraphDTA 原始输入形式。

## 四个数据集

| 数据集 | 预处理代码 | 处理后发布包 |
|---|---|---|
| ChEMBL | `preprocessing/3_all_data_chembl_targets_preprocessed.py`、`preprocessing/repair_chembl_target_split.py` | `data-processed-chembl-v1.0.0.zip` |
| Davis | `preprocessing/preprocess_davis.py` | `data-processed-davis-v1.0.0.zip` |
| KIBA | `preprocessing/preprocess_kiba.py` | `data-processed-kiba-v1.0.0.zip` |
| BindingDB | `preprocessing/prepare_bindingdb.py` | `data-processed-bindingdb-v1.0.0.zip` |

处理后数据是论文实验的直接输入，应作为 GitHub Release 或 Zenodo
附件发布。原始数据库文件不应直接提交到 Git 历史；应提供官方来源、
版本、引用和预处理脚本。完整说明见 `DATA.md`。

## 上传位置

### 放入普通 GitHub 仓库

- 所有 `.py` 源代码；
- `README.md`、`DATA.md` 和 `docs/`；
- `requirements.txt`、`environment.yml`；
- `LICENSE`、`CITATION.cff`；
- `data/data_manifest.json`；
- 最终轻量结果 CSV/JSON 和画图所需预测值。

### 只放 GitHub Release 或 Zenodo

- 四个 `data-processed-*.zip`；
- `SHA256SUMS.txt`；
- 必须公开时的大型模型权重或预测文件。

### 不上传

- `.venv/`、`__pycache__/`、`*.pyc`；
- Hugging Face 缓存和本机下载的 ESM-2 权重；
- `precomputed_features.npz`、`*.npy` 和其他可重建缓存；
- 训练中间检查点、日志、临时输出；
- 令牌、密码、私人路径或许可不允许再分发的原始数据。

## 转移到另一台电脑

只需要复制整个 `D:\lht\release_packages` 文件夹。目标电脑先阅读
其中的 `README-FIRST.txt`，再运行 `verify_transfer.ps1`。验证通过后，
把代码 ZIP 解压为 Git 仓库目录；四个数据 ZIP 不执行 `git add`，
仅作为 Release 附件上传。

本机权重目录通过命令行指定，例如：

```powershell
python scripts/check_esm2_model.py --model_dir "D:\lht\esm2_model"
python run_baseline_mlp.py --esm2_model "D:\lht\esm2_model" --rebuild_features --force
```

路径只作为运行参数使用，不会写死在仓库代码中。旧特征缓存没有修正版元数据
时会被拒绝，必须加 `--rebuild_features --force` 重新生成并训练。
