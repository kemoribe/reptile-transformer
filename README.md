# Target-disjoint Drug-Target Affinity Prediction

本仓库包含目标冷启动（target-disjoint）药物-靶点亲和力预测实验的代码、数据处理流程和数据完整性清单。模型包括 GCN、GAT、GIN、GAT-GCN、MLP、Transformer 和 Reptile-Transformer。

第一次发布请先阅读 [START_HERE.md](START_HERE.md)，其中给出了不使用 Git
命令、完全通过 GitHub 网页上传的步骤。

> **发布状态说明**  
> 本发布副本已修正数据目录解析和 ESM-2 权重加载方式。旧实验目录中的指标与模型权重是在修正前产生的，因此未放入本仓库。正式提交论文结果前，必须使用此版本代码重新训练并核对指标。详见 [发布前检查表](PUBLISH_CHECKLIST.md)。

## 最终特征配置

MLP、Transformer 和 Reptile-Transformer 最终采用的输入组合为：

**Morgan fingerprint + 10 个 RDKit 理化描述符 + ESM-2 蛋白质嵌入**

| 模态 | 配置 | 维度 |
|---|---|---:|
| Morgan fingerprint | radius=2, nBits=2048 | 2048 |
| RDKit descriptors | MolWt、LogP、HBD、HBA、RotatableBonds、RingCount、TPSA、FractionCSP3、HeavyAtomCount、AromaticRings | 10 |
| ESM-2 | `facebook/esm2_t12_35M_UR50D`，去除特殊 token 后平均池化 | 480 |

在训练脚本中，该组合对应 `--ablation morgan_descriptors`。为保持网络结构一致，167 维 MACCS 分支仍存在，但输入被置零；MACCS 不属于最终有效特征。消融实验的四种配置见两个 `*_ablation.py` 脚本。

四个 GraphDTA 基线不使用上述固定向量组合。它们按原 GraphDTA 结构输入 RDKit 分子图和整数编码的蛋白质序列，以保证 GCN、GAT、GIN 和 GAT-GCN 比较的模型定义清晰。

## 数据集

训练使用按蛋白质靶点划分的 70%/15%/15% 冷启动数据。训练、验证和测试集之间的靶点 ID 与蛋白序列交集均为 0。

| 数据集 | Train 行/靶点 | Val 行/靶点 | Test 行/靶点 | 总行数 |
|---|---:|---:|---:|---:|
| ChEMBL | 389,593 / 123 | 69,164 / 41 | 31,132 / 26 | 489,889 |
| Davis | 19,668 / 298 | 4,224 / 64 | 4,290 / 65 | 28,182 |
| KIBA | 84,641 / 160 | 15,135 / 34 | 18,478 / 35 | 118,254 |
| BindingDB | 40,449 / 520 | 8,332 / 111 | 12,043 / 112 | 60,824 |

处理后数据不直接提交到 Git 历史，而作为 GitHub Release/Zenodo 附件发布。数据来源、格式、预处理和许可注意事项见 [DATA.md](DATA.md)，校验值见 [data/data_manifest.json](data/data_manifest.json)。

## 仓库结构

```text
.
|-- START_HERE.md
|-- data_preprocessing.py
|-- reptile_transformer_model.py
|-- reptile_training.py
|-- run_baseline_mlp.py
|-- run_transformer_baseline.py
|-- run_reptile_transformer.py
|-- run_transformer_baseline_ablation.py
|-- run_reptile_transformer_ablation.py
|-- graphdta/
|   |-- models/                 # GCN, GAT, GIN, GAT-GCN
|   |-- convert_*.py
|   |-- create_data.py
|   `-- training.py
|-- preprocessing/             # 四个数据集的预处理脚本
|-- scripts/
|   |-- check_esm2_model.py
|   |-- validate_data.py
|   |-- package_release.ps1
|   `-- verify_transfer.ps1
|-- analysis/
|-- data/data_manifest.json
|-- DATA.md
`-- PUBLISH_CHECKLIST.md
```

七个模型、四个数据集、预处理脚本及上传位置的逐项对应关系见
[代码与数据位置总表](docs/CODE_AND_DATA_MAP.md)。尚需作者本人确认的信息见
[正式发布信息模板](docs/FINAL_RELEASE_INFO_TEMPLATE.md)。

## 环境安装

推荐 Python 3.9、CUDA 兼容的 PyTorch 2.8 和至少 20 GB 可用磁盘空间。ChEMBL 的未压缩预计算特征约为 5 GB。

Conda：

```bash
conda env create -f environment.yml
conda activate dta-reptile
```

`environment.yml` 使用 pip 安装 PyTorch，便于跨机器创建基础环境。正式使用
GPU 训练前，应按 <https://pytorch.org/get-started/locally/> 给出的当前命令，
重新安装与目标电脑 CUDA/驱动匹配的 PyTorch 2.8 构建。

也可直接使用 pip：

```bash
python -m pip install -r requirements.txt
```

首次提取蛋白质特征时，Transformers 会下载 `facebook/esm2_t12_35M_UR50D`。离线环境可将 `ESM2_MODEL` 指向本地 Hugging Face 模型目录。

当前电脑上的模型可先这样验证：

```powershell
python scripts/check_esm2_model.py --model_dir "D:\lht\esm2_model"
```

输出必须包含 `"status": "ok"`、`"loader": "transformers.AutoModel"` 和
`"hidden_size": 480`。代码使用的是 Hugging Face 格式；不要改回
`fair-esm`，也不要把 `esm2_model/` 权重目录上传到 GitHub。

## 放置与校验数据

从 Release 下载四个 `data-processed-*-v1.0.0.zip`，分别解压到 `data/processed/`。最终目录应为：

```text
data/processed/
|-- chembl/
|-- davis/
|-- kiba/
`-- bindingdb/
```

然后验证数据：

```bash
python scripts/validate_data.py
```

只有输出中的 `all_target_disjoint` 为 `true`，且 `matches_split_rows` 均为 `true` 时才继续训练。

无需下载 ESM-2 权重即可检查七个模型的基本前向传播：

```bash
python scripts/smoke_test_models.py
```

## 运行最终特征配置

以下示例使用 ChEMBL；把 `chembl` 替换为 `davis`、`kiba` 或 `bindingdb` 即可。

MLP：

```powershell
python run_baseline_mlp.py --data_dir data/processed/chembl --output_dir outputs/mlp/chembl --ablation morgan_descriptors --esm2_model "D:\lht\esm2_model" --rebuild_features --force
```

Transformer：

```powershell
python run_transformer_baseline.py --data_dir data/processed/chembl --output_dir outputs/transformer/chembl --ablation morgan_descriptors --esm2_model "D:\lht\esm2_model" --rebuild_features --force
```

Reptile-Transformer：

```powershell
python run_reptile_transformer.py --data_dir data/processed/chembl --output_dir outputs/reptile_transformer/chembl --ablation morgan_descriptors --esm2_model "D:\lht\esm2_model" --rebuild_features --force
```

`--rebuild_features` 会删除该输出目录中的旧特征和旧训练结果，然后使用当前
Hugging Face ESM-2 重新生成。首次修正重跑必须保留此参数；后续确认缓存正确
后可以去掉。换电脑时只需把 `--esm2_model` 后面的路径改成新电脑上的位置。

完整消融实验：

```powershell
python run_transformer_baseline_ablation.py --all --data_dir data/processed/chembl --esm2_model "D:\lht\esm2_model" --rebuild_features
python run_reptile_transformer_ablation.py --all --data_dir data/processed/chembl --esm2_model "D:\lht\esm2_model" --rebuild_features
```

可用 `--gpu`、`--batch_size` 和 `--epochs` 调整资源与训练轮数。各脚本的全部参数可通过 `python <script> --help` 查看。

## 运行 GraphDTA 基线

先将四个处理后数据集转换为 GraphDTA 格式：

```bash
python graphdta/convert_to_graphdta.py chembl data/processed/chembl --output-root graphdta/data
python graphdta/convert_to_graphdta.py davis data/processed/davis --output-root graphdta/data
python graphdta/convert_kiba.py data/processed/kiba --output-root graphdta/data
python graphdta/convert_bindingdb_to_graphdta.py data/processed/bindingdb --output-root graphdta/data
cd graphdta
python create_data.py chembl davis kiba bindingdb
```

训练命令格式为：

```bash
python training.py <dataset_index> <model_index> <gpu_index>
```

数据集索引：`0=chembl`、`1=davis`、`2=kiba`、`3=bindingdb`。模型索引：`0=GIN`、`1=GAT`、`2=GAT-GCN`、`3=GCN`。例如：

```bash
python training.py 0 0 0
python training.py 0 1 0
python training.py 0 2 0
python training.py 0 3 0
```

## 从原始数据重新预处理

```bash
python preprocessing/preprocess_davis.py --input data/raw/davis --output data/processed/davis
python preprocessing/preprocess_kiba.py --input data/raw/kiba --output data/processed/kiba
python preprocessing/prepare_bindingdb.py --output data/processed/bindingdb
python preprocessing/3_all_data_chembl_targets_preprocessed.py --input data/raw/chembl/2_all_data_chembl_targets --output data/processed/chembl
python preprocessing/repair_chembl_target_split.py data/processed/chembl
python scripts/validate_data.py
```

ChEMBL 脚本的输入是已经按靶点组织的中间目录，不是原始 ChEMBL 数据库转储。公开前需要在 [DATA.md](DATA.md) 中补充该中间数据的生成来源或下载地址。

## 引用与许可

代码使用 MIT License。使用本项目时请引用对应论文、原始数据集和 ESM-2；发布前请先填写 [CITATION.cff](CITATION.cff) 中的作者与仓库地址占位符。

完整 GitHub 上传流程见 [docs/GITHUB_UPLOAD_GUIDE.md](docs/GITHUB_UPLOAD_GUIDE.md)，
发布页说明可直接使用
[docs/GITHUB_RELEASE_TEMPLATE.md](docs/GITHUB_RELEASE_TEMPLATE.md)。本地验证范围和
结果见 [docs/VERIFICATION.md](docs/VERIFICATION.md)。
