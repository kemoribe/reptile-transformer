# 发布前检查表

带 `[ ]` 的项目仍需仓库维护者确认；不要在未完成关键项时把仓库标记为论文最终版本。

## 必须完成

- [ ] 使用修正后的 Hugging Face ESM-2 加载方式重新生成特征并重跑最终实验。
- [ ] 确认论文、表格和图中的指标来自修正版代码，而不是旧缓存或旧检查点。
- [ ] 记录每个模型、数据集的随机种子、epoch、batch size、学习率和 GPU 环境。
- [ ] 将最终的轻量结果文件加入 `results/`，至少包括指标 JSON/CSV 和生成表图所需预测值。
- [ ] 补充 ChEMBL 中间输入 `2_all_data_chembl_targets` 的来源、ChEMBL 版本、查询条件和生成步骤。
- [ ] 核对 ChEMBL、Davis、KIBA、BindingDB 派生数据的再分发条款。
- [ ] 将 `CITATION.cff` 中的姓名和 GitHub 地址占位符替换为真实信息。
- [ ] 将 `LICENSE` 中的版权人改为真实姓名或团队名称。
- [ ] 为论文、数据和代码补齐正式引用；有 DOI 后更新 README 和 `CITATION.cff`。

## 数据检查

- [x] 四个处理后数据集已整理为统一目录。
- [x] train/validation/test 的靶点 ID 交集为 0。
- [x] train/validation/test 的蛋白序列交集为 0。
- [x] ChEMBL 汇总文件已去除修复划分时产生的重复行。
- [x] `data/data_manifest.json` 可记录行数和 SHA256。
- [x] 在最终压缩后核对 `SHA256SUMS.txt`。
- [ ] 从 Release 实际下载一次并重新运行 `python scripts/validate_data.py`。

## 代码检查

- [x] 包含 GCN、GAT、GIN、GAT-GCN。
- [x] 包含 MLP、Transformer、Reptile-Transformer。
- [x] 包含特征消融脚本和四个数据集的预处理入口。
- [x] 默认数据目录统一为 `data/processed/<dataset>`。
- [x] 本地 `D:\lht\esm2_model` 已通过 Hugging Face ESM-2 加载自检。
- [x] 旧特征缓存缺少修正版元数据时会被拒绝，入口支持 `--rebuild_features`。
- [x] 排除虚拟环境、缓存、检查点和大体积预计算特征。
- [ ] 在一台干净电脑上创建环境并执行所有入口脚本的 `--help`。
- [x] 至少完成一次小规模 smoke test，并记录命令与输出（见 `docs/VERIFICATION.md`）。

## GitHub 发布检查

- [ ] 仓库名称、简介、topics 和可见性设置正确。
- [ ] Git 首次提交中没有数据压缩包、模型权重、缓存或隐私信息。
- [ ] `main` 分支已推送，网页可正常显示 README。
- [ ] 创建 `v1.0.0` tag 和 GitHub Release。
- [ ] 上传四个数据压缩包及 `SHA256SUMS.txt` 到 Release。
- [ ] 从新目录执行 `git clone`，验证 README 中的命令和相对路径。
- [ ] 如需 DOI，在 Zenodo 中归档正式 release，并把 DOI 回填到仓库。
