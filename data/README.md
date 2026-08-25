# Data directory

处理后数据通过 GitHub Release 或 Zenodo 单独发布，不进入 Git 历史。

下载四个数据压缩包后，将它们解压到本目录，形成：

```text
data/processed/{chembl,davis,kiba,bindingdb}
```

随后从仓库根目录运行：

```bash
python scripts/validate_data.py
```

数据来源、格式和预处理说明见仓库根目录的 `DATA.md`。
