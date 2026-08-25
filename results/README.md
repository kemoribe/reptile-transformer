# Results

项目维护者已确认 `tables/` 中的三张 Excel 为最终实验结果表。
`metrics_all_tables.csv` 是三张表中有效模型行的统一机器可读导出。

大型结果和权重不提交到 Git 历史，而作为 GitHub Release 附件发布：

- `experiment-results-v1.0.0.zip`：结果表、指标、预测值、逐靶点结果和训练历史；
- `model-weights-mlp-v1.0.0.zip`；
- `model-weights-transformer-v1.0.0.zip`；
- `model-weights-reptile-transformer-v1.0.0.zip`；
- `model-weights-graphdta-v1.0.0.zip`。

每个附件内含 `MANIFEST.csv` 和 SHA256。打包方式见
`scripts/package_experiment_artifacts.ps1`，上传步骤见
`docs/EXPERIMENT_ARTIFACTS_UPLOAD.md`。

大型 `precomputed_features.npz`、逐轮 checkpoint、虚拟环境及第三方 ESM-2
权重不属于发布结果。
