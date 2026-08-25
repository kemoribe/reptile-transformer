# 第一次手动上传 GitHub：从这里开始

你只需要处理 `D:\lht\release_packages` 中的文件，不要从原始
`D:\lht` 工作目录随意挑文件上传。

## 先分清两类文件

| 文件 | 上传位置 | 是否先解压 |
|---|---|---|
| `dta-reptile-code-v1.0.0.zip` | GitHub 仓库代码区 | 是 |
| 四个 `data-processed-*.zip` | GitHub Release 附件 | 否 |
| `experiment-results-v1.0.0.zip` | GitHub Release 附件 | 否 |
| 四个 `model-weights-*.zip` | GitHub Release 附件 | 否 |
| `SHA256SUMS.txt` | GitHub Release 附件 | 否 |
| `SHA256SUMS-EXPERIMENT-ARTIFACTS.txt` | GitHub Release 附件 | 否 |
| `esm2_model/` | 不上传 | 不适用 |

ESM-2 权重约数百 MB，而且可以从 Hugging Face 重新获得，所以不要上传。
代码通过 `--esm2_model` 接收本机模型目录。

## 第一步：先验证传输包

打开 PowerShell：

```powershell
cd D:\lht\release_packages
powershell -ExecutionPolicy Bypass -File .\verify_transfer.ps1
```

看到 `All release packages passed verification.` 才继续。

## 第二步：解压代码包

```powershell
New-Item -ItemType Directory -Force D:\github-upload\target-disjoint-dta
Expand-Archive .\dta-reptile-code-v1.0.0.zip D:\github-upload\target-disjoint-dta
```

打开 `D:\github-upload\target-disjoint-dta`，应直接看到 `README.md`、
`.gitignore`、`run_baseline_mlp.py`、`graphdta` 等内容。不要再多套一层目录。

## 第三步：填写本人信息

上传前至少修改：

1. `CITATION.cff`：姓名、GitHub 用户名和仓库名。
2. `LICENSE`：把版权人改为本人姓名或团队名。
3. `docs/FINAL_RELEASE_INFO_TEMPLATE.md`：能确认的项目先填写。

本发布中的最终结果表由项目维护者确认，结果和权重附件按
`docs/EXPERIMENT_ARTIFACTS_UPLOAD.md` 上传。

## 第四步：在 GitHub 网页创建仓库

1. 登录 <https://github.com/>，右上角点 `+`，选择 `New repository`。
2. Repository name 填 `target-disjoint-dta`。
3. 第一次可选 `Private`，确认无误后再改成 `Public`。
4. 不勾选 README、`.gitignore`、License。
5. 点击 `Create repository`。

## 第五步：通过网页上传代码

1. 在空仓库页面点击 `uploading an existing file`；若仓库不是空的，则点
   `Add file` -> `Upload files`。
2. 把解压后目录**里面的全部文件和子目录**拖入页面。不要上传代码 ZIP。
3. 等待文件列表加载完成，确认至少能看到 `README.md`、`.gitignore`、
   `data_preprocessing.py`、`graphdta/`、`preprocessing/` 和 `scripts/`。
4. Commit message 填 `Initial reproducible release`。
5. 点击 `Commit changes`。

代码文件数量低于网页单次上传限制，且没有大文件。若浏览器没有保留子目录，
不要逐个重建目录，改用 `docs/GITHUB_UPLOAD_GUIDE.md` 中的 Git 命令方案。

## 第六步：创建 Release 并上传数据、结果和权重

1. 打开仓库首页右侧 `Releases`，点击 `Create a new release`。
2. 点击 `Choose a tag`，输入版本号并选择 `Create new tag`。
3. Release title 填 `v1.0.0 - Code, data, results, and model weights`。
4. 说明内容参考 `docs/GITHUB_RELEASE_TEMPLATE.md`。
5. 上传四个数据 ZIP、一个结果 ZIP、四个权重 ZIP 和两个 SHA256 文件。
6. 等待 11 个附件全部上传完成，再点击 `Publish release`。

## 第七步：网页检查

- 仓库首页能正常显示 README。
- `graphdta/models/` 下能看到 GCN、GAT、GIN、GAT-GCN。
- 仓库代码区没有数据 ZIP、ESM-2 权重、`.venv`、缓存和检查点。
- Release 页面能看到四个数据 ZIP、结果 ZIP、四个权重 ZIP 和两个 SHA256 文件。
- `CITATION.cff` 和 `LICENSE` 不再含占位内容。

更完整的命令行方案和故障处理见
[`docs/GITHUB_UPLOAD_GUIDE.md`](docs/GITHUB_UPLOAD_GUIDE.md)。
