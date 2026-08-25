# Data

## 发布策略

建议在 GitHub 仓库中提交预处理代码、数据说明和校验清单；处理后的数据作为 GitHub Release 附件或 Zenodo 数据集发布。不要把原始数据库转储、5 GB 级特征缓存或模型检查点直接提交到 Git 历史。

本项目的发布数据分为四个压缩包：

- `data-processed-chembl-v1.0.0.zip`
- `data-processed-davis-v1.0.0.zip`
- `data-processed-kiba-v1.0.0.zip`
- `data-processed-bindingdb-v1.0.0.zip`

解压后目录应为 `data/processed/<dataset>/`。压缩包 SHA256 写在发布包目录的 `SHA256SUMS.txt` 中；核心汇总 CSV 的 SHA256 写在 `data/data_manifest.json` 中。

## 为什么发布处理后数据

处理后数据保存了本实验实际使用的：

- SMILES 标准化结果；
- pKd、pKi 或 pIC50 活性值；
- 蛋白质序列；
- 固定的目标冷启动 train/validation/test 划分；
- 每个靶点对应的样本文件。

这使他人能够复现训练输入。原始数据仍应通过官方来源获取，并由 `preprocessing/` 下的脚本重建。发布任何原始或派生数据前，仓库维护者必须核对对应数据源的许可与再分发条款。

## 数据来源

| 数据集 | 建议引用/获取位置 | 本项目预处理入口 |
|---|---|---|
| ChEMBL | <https://www.ebi.ac.uk/chembl/> | `preprocessing/3_all_data_chembl_targets_preprocessed.py` |
| Davis | Davis et al., Nature Biotechnology (2011), DOI: `10.1038/nbt.1990`; DeepDTA 数据格式 | `preprocessing/preprocess_davis.py` |
| KIBA | Tang et al., Journal of Chemical Information and Modeling (2014), DOI: `10.1021/ci400709d`; DeepDTA 数据格式 | `preprocessing/preprocess_kiba.py` |
| BindingDB | <https://www.bindingdb.org/>；脚本内记录 Harvard Dataverse 文件 ID | `preprocessing/prepare_bindingdb.py` |

ESM-2 模型：`facebook/esm2_t12_35M_UR50D`，来自 <https://huggingface.co/facebook/esm2_t12_35M_UR50D>。

## 原始输入布局

Davis：

```text
data/raw/davis/
|-- drugs.csv
|-- proteins.csv
`-- drug_protein_affinity.csv
```

KIBA（DeepDTA 格式）：

```text
data/raw/kiba/
|-- ligands_can.txt
|-- proteins.txt
`-- Y
```

BindingDB 原始文件默认下载到 `data/raw/bindingdb/`。

ChEMBL 预处理入口需要一个已按数据集、类别和靶点组织的中间目录：

```text
data/raw/chembl/2_all_data_chembl_targets/
|-- train_set/<category>/<target>/activities.csv
|-- train_set/<category>/<target>/sequence.fasta
|-- val_set/...
`-- test_set/...
```

当前仓库没有生成该 ChEMBL 中间目录的上游抓取脚本。正式公开前必须完成以下二选一：

1. 补充从 ChEMBL 版本化下载文件生成该目录的脚本；或
2. 发布该中间目录并记录 ChEMBL 版本、查询条件、下载日期和许可信息。

## 处理后目录格式

```text
data/processed/<dataset>/
|-- train_set/<category>/<target>/
|   |-- *_processed_activities.csv
|   `-- *_processed_protein_sequence.txt
|-- val_set/<category>/<target>/
|-- test_set/<category>/<target>/
`-- combined_activities.csv
```

Davis 的靶点目录使用 `activities.csv`、`sequence.fasta` 和 `target_info.json`，加载器兼容该格式。

## 划分与校验

Davis、KIBA 和 BindingDB 使用 `seed=42` 按靶点划分。ChEMBL 保留上游划分，并使用 `repair_chembl_target_split.py` 合并跨集合重复靶点，优先级为 test > validation > train。

运行：

```bash
python scripts/validate_data.py
```

校验器检查：

- 三个集合是否存在；
- 活性记录数和靶点数；
- 靶点 ID 是否跨集合重叠；
- 蛋白序列是否跨集合重叠；
- 是否缺少蛋白序列；
- `combined_activities.csv` 行数是否等于三个集合之和；
- 汇总 CSV 的 SHA256。

## 不应上传的内容

- 原始数据库完整转储，除非许可明确允许；
- `precomputed_features.npz` 和 `_*.npy` 特征缓存；
- `*.pt`、`*.pth`、`*.model` 检查点；
- 虚拟环境、Hugging Face 缓存、日志和临时文件；
- 含本机绝对路径、账号、令牌或未脱敏数据的配置文件。
