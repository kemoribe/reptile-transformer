# 正式发布前待填写信息

以下信息无法从代码中可靠推断。请在正式公开仓库或创建 `v1.0.0`
之前填写，不能使用猜测内容。

## 作者与仓库

- 作者姓（对应 `CITATION.cff` 的 `family-names`）：
- 作者名（对应 `given-names`）：
- GitHub 用户名或组织名：
- 仓库名，建议 `target-disjoint-dta`：
- 联系邮箱（可选，建议 GitHub noreply 邮箱）：
- LICENSE 版权人姓名或团队名：

## 论文与 DOI

- 论文正式标题：
- 作者列表：
- 期刊/会议或预印本平台：
- 论文 DOI 或 URL：
- 代码 Zenodo DOI（获得后填写）：
- 数据 Zenodo DOI（获得后填写）：

## ChEMBL 来源

- ChEMBL release/version：
- 下载日期：
- 下载入口或文件名：
- 查询条件：
- 单位和活性类型筛选规则：
- 去重、标准化和异常值处理规则：
- `2_all_data_chembl_targets` 中间目录的生成命令或脚本：

## 数据许可

- ChEMBL 派生数据是否允许按当前形式再分发：
- Davis 派生数据是否允许按当前形式再分发：
- KIBA 派生数据是否允许按当前形式再分发：
- BindingDB 派生数据是否允许按当前形式再分发：
- 若不能再分发，改为提供的官方下载链接或重建步骤：

## 最终实验

对 7 个模型和 4 个数据集分别填写
`results/experiment_manifest_template.csv`：

- 随机种子；
- epoch；
- batch size；
- learning rate；
- GPU 型号和显存；
- Python、PyTorch、CUDA 版本；
- 指标文件和预测文件路径；
- 最终结果表、指标文件和权重文件之间的对应关系；
- 论文表格/图片与结果文件的对应关系。

完成后：

1. 修改 `CITATION.cff`、`LICENSE` 和 README；
2. 将确认后的轻量结果放入 `results/`；
3. 更新 `PUBLISH_CHECKLIST.md`；
4. 运行 `scripts/package_release.ps1` 和
   `scripts/package_experiment_artifacts.ps1`；
5. 重新校验 SHA256，再创建正式 tag 和 Release。
