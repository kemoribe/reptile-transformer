DTA 项目传输包：请先阅读
========================

本目录应包含：

  dta-reptile-code-v1.0.0.zip
  data-processed-chembl-v1.0.0.zip
  data-processed-davis-v1.0.0.zip
  data-processed-kiba-v1.0.0.zip
  data-processed-bindingdb-v1.0.0.zip
  experiment-results-v1.0.0.zip
  model-weights-mlp-v1.0.0.zip
  model-weights-transformer-v1.0.0.zip
  model-weights-reptile-transformer-v1.0.0.zip
  model-weights-graphdta-v1.0.0.zip
  SHA256SUMS.txt
  SHA256SUMS-EXPERIMENT-ARTIFACTS.txt
  verify_transfer.ps1

1. 在 PowerShell 中先验证所有文件：

   powershell -ExecutionPolicy Bypass -File .\verify_transfer.ps1

   只有看到下面这句话才能继续：

   All release packages passed verification.

2. 解压代码包：

   New-Item -ItemType Directory -Force D:\github-upload\target-disjoint-dta
   Expand-Archive .\dta-reptile-code-v1.0.0.zip D:\github-upload\target-disjoint-dta

3. 第一次使用 GitHub，先阅读这个文件：

   D:\github-upload\target-disjoint-dta\START_HERE.md

4. 上传前填写：

   CITATION.cff
   LICENSE
   docs\FINAL_RELEASE_INFO_TEMPLATE.md

5. 代码 ZIP 要先解压，再把里面的内容上传到 GitHub 仓库代码区。
   四个数据 ZIP、结果 ZIP、四个模型权重 ZIP 和两个 SHA256 文件均不解压，
   直接上传到 GitHub Release。
   不要上传 ESM-2 权重、虚拟环境、特征缓存、逐轮 checkpoint 和原始工作目录。
