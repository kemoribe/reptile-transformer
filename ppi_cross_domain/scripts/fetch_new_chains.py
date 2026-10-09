# -*- coding: utf-8 -*-
"""Fetch sequences for newly added chains from expanded training set."""
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

# Expanded split
train_df = df[df['Hold_out_type'] != 'Pr/PI']
test_df = df[df['Hold_out_type'] == 'Pr/PI']

all_chains = set()
for sub in (train_df, test_df):
    for _, row in sub.iterrows():
        all_chains.add((row['pdb_id'], row['chain1']))
        all_chains.add((row['pdb_id'], row['chain2']))

# Find chains not yet in chain_seqs
missing_chains = sorted({(p, c) for p, c in all_chains
                         if f'{p}_{c}' not in chain_seqs})
missing_pdbs = sorted({p for p, _ in missing_chains})
print(f'New missing chains: {len(missing_chains)}, PDBs: {len(missing_pdbs)}')
print('Missing PDBs:', missing_pdbs)

AUTH_RE = re.compile(r'\[auth ([^\]]+)\]')
CHAINLIST_RE = re.compile(r'\bChains?\s+([A-Za-z0-9]+(?:\s*,\s*[A-Za-z0-9]+)*)')

def parse_rcsb_fasta(text):
    mapping = {}
    header = None
    chunks = []
    def flush():
        if header is None:
            return
        auths = []
        m = AUTH_RE.search(header)
        if m:
            auths = [m.group(1).strip()]
        else:
            m2 = CHAINLIST_RE.search(header)
            if m2:
                auths = [c.strip() for c in m2.group(1).split(',') if c.strip()]
        for auth in auths:
            mapping[auth] = ''.join(chunks)
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            flush()
            header = line
            chunks = []
        else:
            chunks.append(line)
    flush()
    return mapping

def parse_cif_entity_poly(text):
    lines = text.splitlines()
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
    ncols = len(cols)
    rows = []
    cur = []
    j = i
    while j < len(lines):
        line = lines[j]
        s = line.strip()
        if s.startswith('_') or s.startswith('#'):
            break
        if line.startswith(';'):
            val = [line[1:]]
            while True:
                j += 1
                if j >= len(lines):
                    break
                line = lines[j]
                if line.rstrip().endswith(';'):
                    val.append(line.rstrip()[:-1])
                    break
                val.append(line.rstrip('\n'))
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

# Fetch missing PDBs via FASTA first, then mmCIF fallback
fetched_fasta = 0
failed_fasta = []
for pdb_id in missing_pdbs:
    if pdb_id in pdb_cache:
        continue
    try:
        r = requests.get(f'https://www.rcsb.org/fasta/entry/{pdb_id}', timeout=30)
        if r.status_code != 200:
            failed_fasta.append(pdb_id)
            continue
        mapping = parse_rcsb_fasta(r.text)
        if not mapping:
            failed_fasta.append(pdb_id)
            continue
        pdb_cache[pdb_id] = mapping
        fetched_fasta += 1
        time.sleep(0.12)
    except Exception as e:
        print(f'  FASTA ERROR {pdb_id}: {e}')
        failed_fasta.append(pdb_id)

print(f'FASTA fetched: {fetched_fasta}, failed: {len(failed_fasta)}')

# mmCIF fallback for failed
fetched_cif = 0
for pdb_id in failed_fasta:
    try:
        r = requests.get(f'https://files.rcsb.org/download/{pdb_id}.cif', timeout=30)
        if r.status_code != 200:
            print(f'  CIF FAILED {pdb_id}: HTTP {r.status_code}')
            continue
        mapping = parse_cif_entity_poly(r.text)
        if mapping:
            pdb_cache[pdb_id] = mapping
            fetched_cif += 1
        else:
            print(f'  CIF no entity_poly {pdb_id}')
        time.sleep(0.1)
    except Exception as e:
        print(f'  CIF ERROR {pdb_id}: {e}')

print(f'mmCIF fetched: {fetched_cif}')

# Build chain sequences for missing
still_missing = []
for pdb_id, chain_id in missing_chains:
    key = f'{pdb_id}_{chain_id}'
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
