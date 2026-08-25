# v1.0.0 - Code and processed target-disjoint datasets

## Overview

This release contains the reproducible code and processed datasets for target-disjoint drug-target affinity prediction with GCN, GAT, GIN, GAT-GCN, MLP, Transformer, and Reptile-Transformer.

## Final feature configuration

MLP, Transformer, and Reptile-Transformer use:

- Drug: Morgan fingerprint (`radius=2`, `nBits=2048`)
- Drug: 10 RDKit physicochemical descriptors
- Protein: mean-pooled `facebook/esm2_t12_35M_UR50D` embedding (480 dimensions)
- The MACCS branch is retained for architecture compatibility but zero-masked in the final `morgan_descriptors` configuration.

The four GraphDTA baselines use RDKit molecular graphs and integer-encoded protein sequences.

## Data

The four processed archives contain fixed target-disjoint train/validation/test splits:

| Archive | Records |
|---|---:|
| `data-processed-chembl-v1.0.0.zip` | 489,889 |
| `data-processed-davis-v1.0.0.zip` | 28,182 |
| `data-processed-kiba-v1.0.0.zip` | 118,254 |
| `data-processed-bindingdb-v1.0.0.zip` | 60,824 |

Verify every download against `SHA256SUMS.txt`, then run:

```powershell
python scripts\validate_data.py
```

## Files

- `dta-reptile-code-v1.0.0.zip`: source code and documentation
- `data-processed-*-v1.0.0.zip`: processed datasets
- `experiment-results-v1.0.0.zip`: final tables, metrics, predictions, and training records
- `model-weights-mlp-v1.0.0.zip`: MLP best weights
- `model-weights-transformer-v1.0.0.zip`: Transformer best weights
- `model-weights-reptile-transformer-v1.0.0.zip`: Reptile-Transformer best weights
- `model-weights-graphdta-v1.0.0.zip`: non-empty GraphDTA weights
- `SHA256SUMS.txt`: archive checksums
- `SHA256SUMS-EXPERIMENT-ARTIFACTS.txt`: result and weight archive checksums

## Reproducibility status

**Final:** The project owner designated the result workbooks in `results/tables/`
as the final experimental summaries. Release assets preserve the corresponding
lightweight outputs and non-empty best-model weights with per-file manifests and
SHA256 checksums.

## Citation

Paper: `[PAPER TITLE AND DOI OR URL]`

Code DOI: `[ZENODO DOI, IF AVAILABLE]`

Data DOI: `[ZENODO DATA DOI, IF AVAILABLE]`

See `DATA.md` for source-dataset citations and redistribution notes.
