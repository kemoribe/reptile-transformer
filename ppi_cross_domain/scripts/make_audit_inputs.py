# -*- coding: utf-8 -*-
"""Generate train.fasta / test.fasta / labels.csv for the SKEMPI Pr/PI
cross-domain homology audit.

Split: SKEMPI v2 ``Hold_out_type`` == "Pr/PI" rows form the test set
(protease/inhibitor hold-out); all other rows (NaN, AB/AG, TCR/pMHC,
AB/AG,Pr/PI) form the training set.  Each unique protein chain sequence
is one audit "target".
"""
import json
from pathlib import Path

import pandas as pd

OUT_DIR = Path(r'd:\lht\审计验证与跨领域')
DATA_DIR = OUT_DIR / 'skempi_prpi'
DATA_DIR.mkdir(exist_ok=True)

df = pd.read_csv(OUT_DIR / 'skempi_v2.csv', sep=';')
parts = df['#Pdb'].str.split('_', expand=True)
df['pdb_id'] = parts[0]
df['chain1'] = parts[1]
df['chain2'] = parts[2]
chain_seqs = json.load(open(OUT_DIR / 'pdb_chain_sequences.json'))

train_df = df[df['Hold_out_type'] != 'Pr/PI']
test_df = df[df['Hold_out_type'] == 'Pr/PI']

def collect(sub, split):
    """chain_key -> {pdb_id, chain_id, sequence, n_rows} for one split.
    Skips chains whose sequences are unavailable (obsolete PDB entries)."""
    d = {}
    skipped = 0
    for _, row in sub.iterrows():
        for which in ('chain1', 'chain2'):
            key = f"{row['pdb_id']}_{row[which]}"
            if key not in chain_seqs:
                skipped += 1
                continue
            if key not in d:
                d[key] = {'chain_key': key, 'pdb_id': row['pdb_id'],
                          'chain_id': row[which],
                          'sequence': chain_seqs[key], 'n_rows': 0}
            d[key]['n_rows'] += 1
    if skipped:
        print(f'  [{split}] skipped {skipped} chain lookups (obsolete PDB IDs)')
    return d

train_chains = collect(train_df, 'train')
test_chains = collect(test_df, 'test')
overlap = set(train_chains) & set(test_chains)
assert not overlap, f'train/test chain overlap: {overlap}'

def write_fasta(d, path):
    with open(path, 'w', newline='\n', encoding='utf-8') as f:
        for key in sorted(d):
            seq = d[key]['sequence']
            f.write(f'>{key}\n')
            for i in range(0, len(seq), 60):
                f.write(seq[i:i + 60] + '\n')

write_fasta(train_chains, DATA_DIR / 'train.fasta')
write_fasta(test_chains, DATA_DIR / 'test.fasta')

# labels.csv: chain-level labels + complex-level affinity context
rows = []
for split, d in (('train', train_chains), ('test', test_chains)):
    for key in sorted(d):
        v = d[key]
        rows.append({'chain_key': key, 'pdb_id': v['pdb_id'],
                     'chain_id': v['chain_id'], 'split': split,
                     'seq_len': len(v['sequence']), 'n_skempi_rows': v['n_rows']})
labels = pd.DataFrame(rows)
labels.to_csv(DATA_DIR / 'labels.csv', index=False, encoding='utf-8-sig')

print(f'train chains: {len(train_chains)}, test chains: {len(test_chains)}')
print(f'train unique sequences: {len({v["sequence"] for v in train_chains.values()})}')
print(f'test unique sequences: {len({v["sequence"] for v in test_chains.values()})}')
print(f'labels.csv rows: {len(labels)}')
print(f'written to {DATA_DIR}')
