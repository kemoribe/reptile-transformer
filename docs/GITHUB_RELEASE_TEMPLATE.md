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
- `SHA256SUMS.txt`: archive checksums

## Reproducibility status

Before publishing, replace this paragraph with one of the following:

- **Final:** All reported metrics were regenerated with this release and the corrected Hugging Face ESM-2 loader.
- **Code/data preview:** The loader and data split issues are corrected, but final metrics are still being regenerated. Do not cite the previous metrics as results from this release.

## Citation

Paper: `[PAPER TITLE AND DOI OR URL]`

Code DOI: `[ZENODO DOI, IF AVAILABLE]`

Data DOI: `[ZENODO DATA DOI, IF AVAILABLE]`

See `DATA.md` for source-dataset citations and redistribution notes.
