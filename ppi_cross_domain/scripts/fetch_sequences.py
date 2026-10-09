# -*- coding: utf-8 -*-
"""Fetch protein sequences for SKEMPI chains via RCSB FASTA endpoint.

Strategy: download one FASTA per unique PDB ID, parse ``[auth X]`` chain
annotations to map author chain IDs to sequences.  Multi-letter chain IDs
(e.g. ``HL``) are concatenated in order.
"""
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

OUT_DIR = Path(r'd:\lht\审计验证与跨领域')
CACHE_FILE = OUT_DIR / 'pdb_chain_sequences.json'

df = pd.read_csv(OUT_DIR / 'skempi_v2.csv', sep=';')
parts = df['#Pdb'].str.split('_', expand=True)
df['pdb_id'] = parts[0]
df['chain1'] = parts[1]
df['chain2'] = parts[2]

train_df = df[df['Hold_out_type'].isna()]
test_df = df[df['Hold_out_type'] == 'Pr/PI']

train_chains = set()
for _, row in train_df.iterrows():
    train_chains.add((row['pdb_id'], row['chain1']))
    train_chains.add((row['pdb_id'], row['chain2']))
test_chains = set()
for _, row in test_df.iterrows():
    test_chains.add((row['pdb_id'], row['chain1']))
    test_chains.add((row['pdb_id'], row['chain2']))

all_chains = sorted(train_chains | test_chains)
unique_pdbs = sorted({p for p, _ in all_chains})
print(f'Total unique chains: {len(all_chains)}, unique PDBs: {len(unique_pdbs)}')

# Load per-PDB FASTA cache: {pdb_id: {auth_chain: sequence}}
pdb_cache = {}
pdb_cache_file = OUT_DIR / 'pdb_fasta_cache.json'
if pdb_cache_file.exists():
    with open(pdb_cache_file, 'r') as f:
        pdb_cache = json.load(f)
print(f'Cached PDB FASTAs: {len(pdb_cache)}')

AUTH_RE = re.compile(r'\[auth ([^\]]+)\]')
CHAINLIST_RE = re.compile(r'\bChains?\s+([A-Za-z0-9]+(?:\s*,\s*[A-Za-z0-9]+)*)')

def parse_rcsb_fasta(text):
    """Parse RCSB FASTA into {auth_chain: sequence}.
    Tries ``[auth X]`` first, then ``Chain X`` / ``Chains X, Y``.
    """
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

fetched = 0
failed_pdbs = []
for pdb_id in unique_pdbs:
    if pdb_id in pdb_cache:
        continue
    url = f'https://www.rcsb.org/fasta/entry/{pdb_id}'
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            print(f'  FAILED {pdb_id}: HTTP {r.status_code}')
            failed_pdbs.append(pdb_id)
            continue
        mapping = parse_rcsb_fasta(r.text)
        if not mapping:
            print(f'  FAILED {pdb_id}: no auth chains parsed')
            failed_pdbs.append(pdb_id)
            continue
        pdb_cache[pdb_id] = mapping
        fetched += 1
        if fetched % 25 == 0:
            print(f'  fetched {fetched} PDBs...')
            with open(pdb_cache_file, 'w') as f:
                json.dump(pdb_cache, f)
        time.sleep(0.12)
    except Exception as e:
        print(f'  ERROR {pdb_id}: {e}')
        failed_pdbs.append(pdb_id)

with open(pdb_cache_file, 'w') as f:
    json.dump(pdb_cache, f)
print(f'PDB FASTAs fetched: {fetched}, failed: {len(failed_pdbs)}, total cached: {len(pdb_cache)}')
if failed_pdbs:
    print('Failed PDBs:', failed_pdbs)

# Build chain-level sequences
chain_seqs = {}
missing = []
for pdb_id, chain_id in all_chains:
    key = f'{pdb_id}_{chain_id}'
    mapping = pdb_cache.get(pdb_id, {})
    seqs = []
    ok = True
    for cid in chain_id:  # multi-letter chain IDs concatenated
        s = mapping.get(cid)
        if s is None:
            ok = False
            break
        seqs.append(s)
    if ok:
        chain_seqs[key] = ''.join(seqs)
    else:
        missing.append(key)

with open(CACHE_FILE, 'w') as f:
    json.dump(chain_seqs, f)
print(f'Chain sequences resolved: {len(chain_seqs)}, missing: {len(missing)}')
if missing:
    print('Missing chains:', missing[:30])
