# 发布包验证记录

验证日期：2026-08-25

## 验证环境

- Windows 10
- Python 3.9.13
- PyTorch 2.8.0+cu129
- CUDA runtime 12.9
- NVIDIA GeForce RTX 4090，24 GiB
- PyTorch Geometric 2.6.1
- Transformers 4.57.6

## 已通过项目

1. `python -m compileall -q`：全部 Python 文件编译通过。
2. MLP、Transformer、Reptile-Transformer、两个消融脚本、
   数据校验脚本、五个预处理脚本和三个 GraphDTA 转换脚本的
   `--help` 均以状态码 0 结束。
3. `python scripts/smoke_test_models.py`：7 个模型入口均完成合成
   前向传播。
4. `python scripts/check_esm2_model.py --model_dir D:\lht\esm2_model`：
   本地 checkpoint 由 `transformers.AutoModel` 加载为 `EsmModel`，
   `hidden_size=480`，测试序列得到有限的 480 维嵌入。
5. 五个向量模型训练入口均支持 `--esm2_model` 和
   `--rebuild_features`；缺少修正版元数据的旧特征缓存会被拒绝。
6. `python scripts/validate_data.py`：四个数据集均满足
   `target_disjoint=true`，全部 `matches_split_rows=true`。
7. `scripts/package_release.ps1`：成功生成一个代码包和四个数据包。
8. `verify_transfer.ps1`：五个 ZIP 的 SHA256 和压缩包目录结构全部通过。
9. 将最终代码 ZIP 解压到新目录后，再次执行编译、三个主要入口的
   `--help`、模型 smoke test 和数据校验，全部通过。

## 数据校验摘要

| 数据集 | 总记录数 | 目标隔离 |
|---|---:|---|
| ChEMBL | 489,889 | 通过 |
| Davis | 28,182 | 通过 |
| KIBA | 118,254 | 通过 |
| BindingDB | 60,824 | 通过 |

压缩包的最终 SHA256 记录在转移目录的 `SHA256SUMS.txt` 中。

## 尚未验证

- 尚未在一台全新电脑上从零创建 Conda/pip 环境；
- 尚未从真实 GitHub Release 下载附件后复验；
- 尚未执行修正版代码的完整训练；
- 已验证本地 ESM-2 加载和短序列分词，但尚未生成四个数据集的完整新版
  特征缓存；
- 当前电脑没有可用的 Git 命令，因此尚未执行 `git init`、
  commit、push、tag 或 GitHub Release 发布。

上述未验证项不影响代码包和数据包的传输完整性，但正式论文结果和
`v1.0.0` 发布必须按 `PUBLISH_CHECKLIST.md` 继续完成。
