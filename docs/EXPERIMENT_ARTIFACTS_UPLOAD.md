# 实验结果与权重上传说明

实验结果和模型权重作为 GitHub Release 附件发布，不提交到 Git 仓库历史。

## 附件

- `experiment-results-v1.0.0.zip`
  - 三张最终结果 Excel；
  - `metrics_all_tables.csv`；
  - 各实验的指标、预测值、逐靶点结果、训练历史和 scaler。
- `model-weights-mlp-v1.0.0.zip`
- `model-weights-transformer-v1.0.0.zip`
- `model-weights-reptile-transformer-v1.0.0.zip`
- `model-weights-graphdta-v1.0.0.zip`
- `SHA256SUMS-EXPERIMENT-ARTIFACTS.txt`

每个压缩包都包含 `MANIFEST.csv`，记录原始相对路径、压缩包内路径、文件大小和
SHA256。权重包只包含非空的最佳模型文件，不包含特征缓存、优化器状态或逐轮
checkpoint。

## GitHub 操作

1. 打开仓库的 `Releases` 页面。
2. 编辑现有 pre-release，或创建正式 `v1.0.0` Release。
3. 保留已经上传的四个处理后数据 ZIP 和数据校验文件。
4. 上传上面列出的五个实验附件 ZIP。
5. 上传 `SHA256SUMS-EXPERIMENT-ARTIFACTS.txt`。
6. 将 `docs/GITHUB_RELEASE_TEMPLATE.md` 的内容粘贴到 Release 说明。
7. 等待所有附件上传完成后再点击发布。

不上传以下内容：

- `precomputed_features.npz`；
- `checkpoint_epoch_*.pt` 或包含优化器状态的 checkpoint；
- `esm2_model/`；
- `.venv/`、缓存和临时日志。

## 本地校验

```powershell
Get-Content .\SHA256SUMS-EXPERIMENT-ARTIFACTS.txt
Get-FileHash .\experiment-results-v1.0.0.zip -Algorithm SHA256
```

下载后还应解压 `experiment-results-v1.0.0.zip`，确认其中包含三张 Excel、
`metrics_all_tables.csv` 和 `MANIFEST.csv`。
