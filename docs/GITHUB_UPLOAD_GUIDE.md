# GitHub 上传指南

本流程采用“代码进入 Git 仓库，处理后数据作为 Release 附件”的方式。这样不会把 200 MB 数据和数千个小文件写入 Git 历史，也不需要 Git LFS。

第一次使用 GitHub、希望完全通过网页上传时，直接按
[`START_HERE.md`](../START_HERE.md) 操作。本文第 4 至 7 节是更适合后续维护的
Git 命令方案，两种方式选择一种即可。

## 1. 在当前电脑生成传输包

在 `D:\lht` 中运行：

```powershell
cd D:\lht\github_release
powershell -ExecutionPolicy Bypass -File .\scripts\package_release.ps1 `
  -DataRoot ..\github_release_data\processed `
  -OutputRoot ..\release_packages
```

生成目录：

```text
D:\lht\release_packages\
|-- dta-reptile-code-v1.0.0.zip
|-- data-processed-chembl-v1.0.0.zip
|-- data-processed-davis-v1.0.0.zip
|-- data-processed-kiba-v1.0.0.zip
|-- data-processed-bindingdb-v1.0.0.zip
|-- experiment-results-v1.0.0.zip
|-- model-weights-mlp-v1.0.0.zip
|-- model-weights-transformer-v1.0.0.zip
|-- model-weights-reptile-transformer-v1.0.0.zip
|-- model-weights-graphdta-v1.0.0.zip
|-- SHA256SUMS.txt
|-- SHA256SUMS-EXPERIMENT-ARTIFACTS.txt
|-- verify_transfer.ps1
`-- README-FIRST.txt
```

把整个 `release_packages` 文件夹复制到移动硬盘、网盘或另一台电脑。
目标电脑上先打开 `README-FIRST.txt`，不要只复制其中某一个 ZIP。

## 2. 在目标电脑校验和解压

先运行自动校验。它会检查代码、数据、结果和权重 ZIP 的 SHA256，确认代码包
中没有混入大型数据/缓存/权重，并检查各压缩包的目录结构：

```powershell
cd D:\transfer\release_packages
powershell -ExecutionPolicy Bypass -File .\verify_transfer.ps1
```

必须看到 `All release packages passed verification.` 才继续。也可手工复核：

```powershell
Get-FileHash .\*.zip -Algorithm SHA256
Get-Content .\SHA256SUMS.txt
Get-Content .\SHA256SUMS-EXPERIMENT-ARTIFACTS.txt
```

然后创建项目目录并解压：

```powershell
New-Item -ItemType Directory -Force D:\projects\dta-reptile
Expand-Archive .\dta-reptile-code-v1.0.0.zip D:\projects\dta-reptile
cd D:\projects\dta-reptile
New-Item -ItemType Directory -Force .\data\processed
Expand-Archive D:\transfer\release_packages\data-processed-chembl-v1.0.0.zip .\data\processed
Expand-Archive D:\transfer\release_packages\data-processed-davis-v1.0.0.zip .\data\processed
Expand-Archive D:\transfer\release_packages\data-processed-kiba-v1.0.0.zip .\data\processed
Expand-Archive D:\transfer\release_packages\data-processed-bindingdb-v1.0.0.zip .\data\processed
```

确认存在 `data\processed\chembl`、`davis`、`kiba` 和 `bindingdb`。

## 3. 填写发布者信息

发布前修改：

1. `CITATION.cff` 中的 `family-names`、`given-names` 和 `repository-code`。
2. `LICENSE` 中的版权人；个人项目可写本人姓名，团队项目可写团队或机构名称。
3. README 中论文标题、论文链接、数据 DOI（获得后填写）。
4. `PUBLISH_CHECKLIST.md` 中尚未完成的关键项目。

需要收集的全部信息已经列在
`docs/FINAL_RELEASE_INFO_TEMPLATE.md`。模型、脚本和数据位置见
`docs/CODE_AND_DATA_MAP.md`。

检查是否仍有占位符或本机绝对路径：

```powershell
Get-ChildItem -Recurse -File | Select-String -Pattern "REPLACE_WITH|C:\\Users\\|TODO|FIXME"
```

该命令应找到尚未填写的 `CITATION.cff`；完成作者和仓库信息替换后再运行
一次，结果应为空。指南中的示例路径 `D:\lht` 和 `D:\projects` 不需要替换。

## 4. 安装并配置 Git

Windows 没有 Git 时可从 <https://git-scm.com/download/win> 安装，或运行：

```powershell
winget install --id Git.Git -e --source winget
```

安装后关闭并重新打开 PowerShell，再确认 Git 可用：

```powershell
git --version
```

首次使用 Git 时设置提交身份：

```powershell
git config --global user.name "你的姓名"
git config --global user.email "你的GitHub邮箱"
```

建议在 GitHub 的 **Settings > Emails** 中使用 `noreply` 邮箱，避免公开个人邮箱。

## 5. 在 GitHub 网页创建空仓库

1. 登录 <https://github.com/>。
2. 右上角选择 **New repository**。
3. 输入仓库名，例如 `target-disjoint-dta`。
4. 填写简介，例如 `Target-disjoint drug-target affinity prediction with GNN, Transformer, and Reptile meta-learning`。
5. 选择 `Public` 或 `Private`。
6. 不要勾选自动创建 README、`.gitignore` 或 License，因为本地已经有这些文件。
7. 点击 **Create repository**。
8. 建议添加 topics：`drug-target-affinity`、`graph-neural-network`、`transformer`、`meta-learning`、`esm2`。

记下 GitHub 给出的 HTTPS 地址：

```text
https://github.com/<YOUR_GITHUB_USERNAME>/<REPOSITORY_NAME>.git
```

## 6. 首次提交代码

在解压后的项目根目录运行：

```powershell
git init
git add .
git status
```

仔细检查 `git status`。不应出现：

- `data/processed/` 或 `data/raw/`；
- `*.zip`；
- `precomputed_features.npz`；
- `*.pt`、`*.pth`、`*.model`；
- `__pycache__`、`.venv`、日志或账号令牌。

检查暂存文件是否存在超大文件：

```powershell
git ls-files | ForEach-Object {
  if (Test-Path $_) {
    $item = Get-Item $_
    if ($item.Length -gt 50MB) {
      "{0:N1} MB`t{1}" -f ($item.Length / 1MB), $_
    }
  }
}
```

确认无误后提交：

```powershell
git commit -m "Initial reproducible release"
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<REPOSITORY_NAME>.git
git push -u origin main
```

GitHub 若要求认证，浏览器登录授权即可；命令行密码位置不能使用 GitHub 登录密码，需要 personal access token 或 Git Credential Manager。

## 7. 创建版本标签

本地创建并推送标签：

```powershell
git tag -a v1.0.0 -m "Version 1.0.0"
git push origin v1.0.0
```

项目维护者确认最终结果表和附件后，可创建正式 `v1.0.0`。

## 8. 上传数据、结果和权重到 GitHub Release

1. 打开仓库网页。
2. 进入右侧 **Releases**，点击 **Draft a new release**。
3. 选择已经推送的 `v1.0.0` tag。
4. 标题填写 `v1.0.0 - code, data, results, and model weights`。
5. 将 `docs/GITHUB_RELEASE_TEMPLATE.md` 的内容粘贴到说明框并替换方括号占位项。
6. 上传四个 `data-processed-*.zip`、`experiment-results-v1.0.0.zip`、
   四个 `model-weights-*.zip`、`SHA256SUMS.txt` 和
   `SHA256SUMS-EXPERIMENT-ARTIFACTS.txt`。
7. 等待所有文件上传完成后再点击 **Publish release**。

不要把数据压缩包强制 `git add` 到普通提交中。GitHub 单文件限制和 Git 历史膨胀会让仓库难以维护。

四个数据包和模型权重均通过 Release 附件发布，因此无需 Git LFS。不要把大型
模型权重提交到普通 Git 历史。

## 9. 可选：用 Zenodo 生成 DOI

1. 使用 GitHub 账号登录 <https://zenodo.org/>。
2. 在 Zenodo 的 GitHub 设置中授权并启用目标仓库。
3. 在 GitHub 发布正式 release。
4. 等待 Zenodo 归档并生成 DOI。
5. 把 DOI 加到 README 和 `CITATION.cff`。
6. 提交 DOI 更新，并创建后续补丁版本；也可以先在 Zenodo 预留 DOI，再写入首个正式 release。

如果数据许可不适合放在 GitHub Release，可单独建立 Zenodo 数据记录，只在 README 中提供 DOI 和下载链接。

## 10. 从零验证公开版本

不要只检查原目录。另建空目录执行：

```powershell
cd D:\temp
git clone https://github.com/<YOUR_GITHUB_USERNAME>/<REPOSITORY_NAME>.git
cd <REPOSITORY_NAME>
python -m compileall .
```

从 Release 下载并解压数据后运行：

```powershell
python scripts\validate_data.py
python scripts\smoke_test_models.py
python scripts\check_esm2_model.py --model_dir "D:\models\esm2_model"
python run_baseline_mlp.py --help
python run_transformer_baseline.py --help
python run_reptile_transformer.py --help
```

最后确认 GitHub 网页上的 README、引用信息、License、Release 附件和下载链接均可访问。

## 常见错误

`remote origin already exists`：

```powershell
git remote set-url origin https://github.com/<YOUR_GITHUB_USERNAME>/<REPOSITORY_NAME>.git
```

远端仓库不是空仓库而导致 push 被拒绝：最简单的处理方式是在 GitHub 重新创建一个空仓库。若必须保留远端文件，先执行 `git pull --rebase origin main`，解决冲突后再 push。

文件超过 100 MB：从提交中移除该文件，并放到 Release/Zenodo。不要仅删除工作区文件，因为它仍可能留在 Git 历史中；首次提交阶段可执行：

```powershell
git rm --cached path\to\large-file
git commit --amend --no-edit
```
