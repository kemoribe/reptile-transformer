# -*- coding: utf-8 -*-
"""Fallback: resolve missing chains via mmCIF _entity_poly.pdbx_strand_id."""
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

OUT_DIR = Path(r'd:\lht\审计验证与跨领域')
pdb_cache = json.load(open(OUT_DIR / 'pdb_fasta_cache.json'))
chain_seqs = json.load(open(OUT_DIR / 'pdb_chain_sequences.json'))

df = pd.read_csv(OUT_DIR / 'skempi_v2.csv', sep=';')
parts = df['#Pdb'].str.split('_', expand=True)
df['pdb_id'] = parts[0]
df['chain1'] = parts[1]
df['chain2'] = parts[2]
train_df = df[df['Hold_out_type'].isna()]
test_df = df[df['Hold_out_type'] == 'Pr/PI']
all_chains = set()
for sub in (train_df, test_df):
    for _, row in sub.iterrows():
        all_chains.add((row['pdb_id'], row['chain1']))
        all_chains.add((row['pdb_id'], row['chain2']))

missing_pdbs = sorted({p for p, c in all_chains
                       if f'{p}_{c}' not in chain_seqs})
print(f'PDBs with missing chains: {len(missing_pdbs)}')

def parse_cif_entity_poly(text):
    """Parse mmCIF _entity_poly loop -> {auth_chain: canonical_sequence}."""
    lines = text.splitlines()
    # locate the loop header
    start = None
    for i, l in enumerate(lines):
        if l.strip().startswith('_entity_poly.'):
            start = i
            break
    if start is None:
        return {}
    cols = []
    i = start
    while i < len(lines) and lines[i].strip().startswith('_entity_poly.'):
        cols.append(lines[i].strip())
        i += 1
    idx = {c: k for k, c in enumerate(cols)}
    seq_col = idx.get('_entity_poly.pdbx_seq_one_letter_code_can')
    strand_col = idx.get('_entity_poly.pdbx_strand_id')
    if seq_col is None or strand_col is None:
        return {}
    # data rows follow; mmCIF multiline values use ';' at line start
    rows = []
    cur = []
    ncols = len(cols)
    j = i
    while j < len(lines):
        line = lines[j]
        s = line.strip()
        if s.startswith('_') or s.startswith('#'):
            break
        if line.startswith(';'):
            # multiline value: consume until closing ';' line
            val = []
            line = line[1:]
            while True:
                if line.rstrip().endswith(';'):
                    val.append(line.rstrip()[:-1])
                    break
                val.append(line.rstrip('\n'))
                j += 1
                if j >= len(lines):
                    break
                line = lines[j]
            cur.append(''.join(val))
        else:
            cur.extend(s.split())
        if len(cur) >= ncols:
            rows.append(cur[:ncols])
            cur = cur[ncols:]
        j += 1
    mapping = {}
    for row in rows:
        if len(row) <= max(seq_col, strand_col):
            continue
        seq = row[seq_col].replace('\n', '').replace(' ', '')
        strands = row[strand_col]
        for ch in strands.split(','):
            ch = ch.strip()
            if ch:
                mapping[ch] = seq
    return mapping

fixed = 0
for pdb_id in missing_pdbs:
    try:
        r = requests.get(f'https://files.rcsb.org/download/{pdb_id}.cif', timeout=30)
        if r.status_code != 200:
            print(f'  {pdb_id}: HTTP {r.status_code}')
            continue
        mapping = parse_cif_entity_poly(r.text)
        if mapping:
            # merge into pdb_cache (FASTA entries take precedence)
            cur = pdb_cache.get(pdb_id, {})
            for k, v in mapping.items():
                cur.setdefault(k, v)
            pdb_cache[pdb_id] = cur
            fixed += 1
        else:
            print(f'  {pdb_id}: no entity_poly parsed')
        time.sleep(0.1)
    except Exception as e:
        print(f'  {pdb_id}: ERROR {e}')

print(f'mmCIF fallback fixed: {fixed}/{len(missing_pdbs)}')

# Rebuild chain sequences for previously missing ones
still_missing = []
for pdb_id, chain_id in sorted(all_chains):
    key = f'{pdb_id}_{chain_id}'
    if key in chain_seqs:
        continue
    mapping = pdb_cache.get(pdb_id, {})
    seqs = []
    ok = True
    for cid in chain_id:
        s = mapping.get(cid)
        if s is None:
            ok = False
            break
        seqs.append(s)
    if ok:
        chain_seqs[key] = ''.join(seqs)
    else:
        still_missing.append(key)

with open(OUT_DIR / 'pdb_fasta_cache.json', 'w') as f:
    json.dump(pdb_cache, f)
with open(OUT_DIR / 'pdb_chain_sequences.json', 'w') as f:
    json.dump(chain_seqs, f)
print(f'Total chain sequences: {len(chain_seqs)}, still missing: {len(still_missing)}')
if still_missing:
    print('Still missing:', still_missing)
