# -*- coding: utf-8 -*-
"""P1-1b: Resolve SKEMPI mutation sites to sequence indices via per-chain
consensus offset search, then emit the final modelling table.

SKEMPI v2 numbering (from legacy PDB files) can drift from the current
remediated mmCIF author numbering.  For each (pdb, chain) we search an
offset delta in [-60, 60] such that seq[map(pos+delta)-1] == wt for >=90%
of that chain's sites; direct lookup is delta=0.

Outputs:
  p1_out/sites_resolved.json : {row_id: {pdb, chain, seq_index, wt, mut, ddG, split, complex}}
  p1_out/resolution_summary.txt
"""
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

OUT_DIR = Path(r'd:\lht\审计验证与跨领域')
P1_DIR = OUT_DIR / 'p1_out'
R = 1.987e-3

df = pd.read_csv(OUT_DIR / 'skempi_v2.csv', sep=';')
df = df.dropna(subset=['Affinity_mut_parsed', 'Affinity_wt_parsed', 'Mutation(s)_cleaned']).copy()
for c in ('Affinity_mut_parsed', 'Affinity_wt_parsed', 'Temperature'):
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna(subset=['Affinity_mut_parsed', 'Affinity_wt_parsed'])
df = df[(df['Affinity_mut_parsed'] > 0) & (df['Affinity_wt_parsed'] > 0)].copy()
df['T_K'] = df['Temperature'].fillna(298.15)
df['ddG'] = R * df['T_K'] * np.log(df['Affinity_wt_parsed'] / df['Affinity_mut_parsed'])

parts = df['#Pdb'].str.split('_', expand=True)
df['pdb_id'] = parts[0].str.upper()
df['complex'] = df['#Pdb']
df['row_id'] = np.arange(len(df))

MUT_RE = re.compile(r'^([A-Z])([A-Z])(-?\d+)([A-Z])$')

num_map = json.load(open(P1_DIR / 'numbering_map.json'))
pdb_fasta = json.load(open(OUT_DIR / 'pdb_fasta_cache.json'))

AA3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V', 'MSE': 'M', 'SEC': 'U', 'PYL': 'O', 'ASX': 'B',
    'GLX': 'Z', 'UNK': 'X',
}

# Group sites per (pdb, chain)
chain_sites = defaultdict(set)   # (pdb, chain) -> {(pos, wt)}
rows_tokens = []                  # per row: list of (pdb, chain, pos, wt, mut)
for _, row in df.iterrows():
    toks = []
    for m in str(row['Mutation(s)_cleaned']).split(','):
        m = m.strip()
        mm = MUT_RE.match(m)
        if mm:
            wt, chain, pos, mut = mm.groups()
            chain = chain.upper()
            toks.append((row['pdb_id'], chain, int(pos), wt.upper(), mut.upper()))
            chain_sites[(row['pdb_id'], chain)].add((int(pos), wt.upper()))
        else:
            toks.append(None)
    rows_tokens.append(toks)

# Per-chain offset resolution: CIF-based and sequence-based scans, best wins
chain_offset = {}   # (pdb, chain) -> ('cif'|'seq', delta_int) or ('rescue', delta_int)
chain_status = {}
for (pdb, chain), site_set in sorted(chain_sites.items()):
    cm = num_map.get(pdb, {}).get(chain)
    seq = pdb_fasta.get(pdb, {}).get(chain)
    if cm is None or seq is None:
        chain_status[(pdb, chain)] = 'no_map_or_seq'
        continue
    L = len(seq)

    def cov_cif(delta):
        hit = tot = 0
        for pos, wt in site_set:
            ent = cm.get(str(pos + delta))
            if ent is None:
                continue
            tot += 1
            seq_id, mon = ent
            if seq_id - 1 < L and (AA3TO1.get(mon, 'X') == wt or seq[seq_id - 1] == wt):
                hit += 1
        return hit / len(site_set) if site_set else 0.0

    def cov_seq(delta):
        hit = 0
        for pos, wt in site_set:
            i = pos - 1 + delta
            if 0 <= i < L and seq[i] == wt:
                hit += 1
        return hit / len(site_set) if site_set else 0.0

    best = ('cif', 0, -1.0)
    for delta in range(-60, 61):
        c = cov_cif(delta)
        if c > best[2]:
            best = ('cif', delta, c)
        s = cov_seq(delta)
        if s > best[2]:
            best = ('seq', delta, s)
    src, delta, cov = best
    if cov >= 0.9:
        chain_offset[(pdb, chain)] = (src, delta)
        chain_status[(pdb, chain)] = f'ok({src},delta={delta},cov={cov:.2f})'
    elif cov >= 0.15:
        chain_offset[(pdb, chain)] = ('rescue', delta)
        chain_status[(pdb, chain)] = f'rescue({src},best={delta},cov={cov:.2f})'
    else:
        chain_status[(pdb, chain)] = f'low_cov(best={delta},cov={cov:.2f})'

n_ok = sum(1 for v in chain_status.values() if v.startswith('ok'))
print(f'Chains with sites: {len(chain_status)}, resolved: {n_ok}, unresolved: {len(chain_status)-n_ok}')

# Build resolved row table
out_rows = []
for i, (_, row) in enumerate(df.iterrows()):
    toks = rows_tokens[i]
    if any(t is None for t in toks) or len(toks) == 0:
        continue  # multi-mutation or unparsable -> skip (single-mutation rows only)
    # rows with comma-separated tokens are multi-mutations; keep only single
    if ',' in str(row['Mutation(s)_cleaned']):
        continue
    pdb, chain, pos, wt, mut = toks[0]
    status = chain_status.get((pdb, chain), '')
    mode = 'ok' if status.startswith('ok') else ('rescue' if status.startswith('rescue') else None)
    if mode is None:
        continue
    src, delta = chain_offset.get((pdb, chain))
    cm = num_map[pdb][chain]
    seq = pdb_fasta[pdb][chain]
    if mode == 'ok' and src == 'seq':
        seq_idx0 = pos - 1 + delta
        if not (0 <= seq_idx0 < len(seq)) or seq[seq_idx0] != wt:
            continue
    elif mode == 'ok':  # cif
        ent = cm.get(str(pos + delta))
        if ent is None:
            continue
        seq_id, mon = ent
        if seq_id - 1 >= len(seq):
            continue
        if not (AA3TO1.get(mon, 'X') == wt or seq[seq_id - 1] == wt):
            continue
        seq_idx0 = seq_id - 1
    else:  # rescue: nearest same-wt residue around pos-1+best_delta
        target0 = pos - 1 + delta
        best_i, best_d = None, 10**9
        for i in range(max(0, target0 - 60), min(len(seq), target0 + 61)):
            if seq[i] == wt and abs(i - target0) < best_d:
                best_i, best_d = i, abs(i - target0)
        if best_i is None:
            continue
        seq_idx0 = best_i
    out_rows.append({
        'row_id': int(row['row_id']),
        'complex': row['complex'],
        'pdb_id': pdb,
        'chain': chain,
        'pos_auth': pos,
        'pos_seq0': seq_idx0,            # 0-based index into chain sequence
        'wt': wt, 'mut': mut,
        'ddG': float(row['ddG']),
        'split': 'test' if row['Hold_out_type'] == 'Pr/PI' else 'train',
    })

res = pd.DataFrame(out_rows)
res.to_json(P1_DIR / 'sites_resolved.json', orient='records', indent=1)

# Summary
n_train = (res['split'] == 'train').sum()
n_test = (res['split'] == 'test').sum()
tot = len(df)
print(f'\nRows total (single-mut, valid ddG): {tot}')
print(f'Resolved rows: {len(res)} ({100*len(res)/tot:.1f}%)')
print(f'  train: {n_train}, test: {n_test}')
print(f'Unique complexes resolved: {res["complex"].nunique()}')
print(f'Unique chains: {res.groupby(["pdb_id","chain"]).ngroups}')

with open(P1_DIR / 'resolution_summary.txt', 'w', encoding='utf-8') as f:
    f.write(f'rows_total={tot}\nrows_resolved={len(res)}\ntrain={n_train}\ntest={n_test}\n')
    f.write(f'chains_resolved={n_ok}/{len(chain_status)}\n\n')
    for k, v in sorted(chain_status.items()):
        if not v.startswith('ok'):
            f.write(f'{k[0]}_{k[1]}: {v}\n')
print('Saved p1_out/sites_resolved.json')
