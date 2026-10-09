# -*- coding: utf-8 -*-
"""P1-1: Build PDB author-numbering -> sequence-index maps from mmCIF
_pdbx_poly_seq_scheme, and validate SKEMPI mutation sites.

Output:
  p1_out/numbering_map.json : {pdb_id: {auth_chain: {auth_seq_num(str): seq_id(int)}}}
  p1_out/mutation_site_validation.txt : wt-residue consistency report
"""
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

OUT_DIR = Path(r'd:\lht\审计验证与跨领域')
P1_DIR = OUT_DIR / 'p1_out'
P1_DIR.mkdir(exist_ok=True)

df = pd.read_csv(OUT_DIR / 'skempi_v2.csv', sep=';')
df = df.dropna(subset=['Affinity_mut_parsed', 'Affinity_wt_parsed', 'Mutation(s)_cleaned']).copy()
for c in ('Affinity_mut_parsed', 'Affinity_wt_parsed'):
    df[c] = pd.to_numeric(df[c], errors='coerce')
df = df.dropna(subset=['Affinity_mut_parsed', 'Affinity_wt_parsed'])
df = df[(df['Affinity_mut_parsed'] > 0) & (df['Affinity_wt_parsed'] > 0)].copy()

parts = df['#Pdb'].str.split('_', expand=True)
df['pdb_id'] = parts[0]

MUT_RE = re.compile(r'^([A-Z])([A-Z])(-?\d+)([A-Z])$')

# Collect unique (pdb, chain, auth_pos, wt) sites
# SKEMPI v2 format: [WT_aa][CHAIN][position][MUT_aa], e.g. 'MA14A' = chain A, Met14->Ala
sites = set()
unparsed = 0
for mut_str, pdb_id in zip(df['Mutation(s)_cleaned'], df['pdb_id']):
    for m in str(mut_str).split(','):
        m = m.strip()
        mm = MUT_RE.match(m)
        if mm:
            wt, chain, pos, mut = mm.groups()
            sites.add((pdb_id, chain.upper(), int(pos), wt.upper()))
        else:
            unparsed += 1

unique_pdbs = sorted({p for p, _, _, _ in sites})
print(f'Mutation entries: {len(df)}, unparsed mutation tokens: {unparsed}')
print(f'Unique sites: {len(sites)}, unique PDBs: {len(unique_pdbs)}')

# Load existing numbering map if present
num_map_file = P1_DIR / 'numbering_map.json'
num_map = {}
if num_map_file.exists():
    num_map = json.load(open(num_map_file))
print(f'Existing numbering maps: {len(num_map)}')


def parse_poly_seq_scheme(text):
    """Parse _pdbx_poly_seq_scheme loop -> {auth_asym_id: {auth_seq_num: seq_id}}."""
    lines = text.splitlines()
    start = None
    for i, l in enumerate(lines):
        if l.strip().startswith('_pdbx_poly_seq_scheme.'):
            start = i
            break
    if start is None:
        return {}
    cols = []
    i = start
    while i < len(lines) and lines[i].strip().startswith('_pdbx_poly_seq_scheme.'):
        cols.append(lines[i].strip().split('.')[1])
        i += 1
    idx = {c: k for k, c in enumerate(cols)}
    need = ['pdb_strand_id', 'seq_id', 'pdb_seq_num', 'mon_id']
    for c in need:
        if c not in idx:
            return {}
    i_auth = idx['pdb_strand_id']
    i_seq = idx['seq_id']
    i_num = idx['pdb_seq_num']
    i_mon = idx['mon_id']
    i_ins = idx.get('pdb_ins_code')
    i_asym = idx.get('asym_id', i_auth)
    ncols = len(cols)

    mapping = {}
    cur = []
    j = i
    while j < len(lines):
        line = lines[j]
        s = line.strip()
        if s.startswith('_') or s.startswith('#') or s.startswith('loop_'):
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
            row = cur[:ncols]
            cur = cur[ncols:]
            if len(row) <= max(i_auth, i_seq, i_num, i_mon):
                j += 1
                continue
            # pdb_strand_id = author chain; pdb_seq_num = author numbering (SKEMPI uses these)
            auth = row[i_auth].strip().strip('"').strip("'")
            asym = row[i_asym].strip().strip('"').strip("'")
            try:
                seq_id = int(row[i_seq])
            except ValueError:
                j += 1
                continue
            num = row[i_num].strip().strip('"').strip("'")
            try:
                num_i = int(num)
            except ValueError:
                num_i = num  # rare non-numeric
            mon = row[i_mon].strip()
            ins = row[i_ins].strip().strip('"').strip("'") if i_ins is not None else ''
            if ins not in ('', '.', '?'):
                # insertion-coded residues get a distinct key like "45A"
                num_i = f'{num_i}{ins}'
            for key in {auth, asym}:
                if not key or key in ('.', '?'):
                    continue
                d = mapping.setdefault(key, {})
                if num_i not in d:
                    d[num_i] = (seq_id, mon)
        j += 1
    return mapping


failed = []
for k, pdb_id in enumerate(unique_pdbs):
    if pdb_id in num_map:
        continue
    try:
        r = requests.get(f'https://files.rcsb.org/download/{pdb_id}.cif', timeout=60)
        if r.status_code != 200:
            print(f'  {pdb_id}: HTTP {r.status_code}')
            failed.append(pdb_id)
            continue
        mapping = parse_poly_seq_scheme(r.text)
        if not mapping:
            print(f'  {pdb_id}: no poly_seq_scheme')
            failed.append(pdb_id)
            continue
        # strip mon -> keep (seq_id, mon)
        num_map[pdb_id] = {ch: {str(num): v for num, v in d.items()}
                           for ch, d in mapping.items()}
        if (k + 1) % 25 == 0:
            print(f'  processed {k+1}/{len(unique_pdbs)}')
            json.dump(num_map, open(num_map_file, 'w'))
        time.sleep(0.1)
    except Exception as e:
        print(f'  {pdb_id}: ERROR {e}')
        failed.append(pdb_id)

json.dump(num_map, open(num_map_file, 'w'))
print(f'Numbering maps built: {len(num_map)}/{len(unique_pdbs)}, failed: {len(failed)}')

# Validate mutation sites: sequence[seq_id-1] == wt (3->1 letter)
AA3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V', 'MSE': 'M', 'SEC': 'U', 'PYL': 'O', 'UNK': 'X',
}
pdb_fasta = json.load(open(OUT_DIR / 'pdb_fasta_cache.json'))

ok = bad = no_map = no_pos = 0
bad_examples = []
for pdb_id, chain, pos, wt in sorted(sites):
    pm = num_map.get(pdb_id, {})
    cm = pm.get(chain)
    if cm is None:
        # try uppercase/lower variants
        cm = pm.get(chain.upper()) or pm.get(chain.lower())
    if cm is None:
        no_map += 1
        continue
    ent = cm.get(str(pos))
    if ent is None:
        no_pos += 1
        continue
    seq_id, mon = ent
    seq = pdb_fasta.get(pdb_id, {}).get(chain)
    if seq is None:
        no_map += 1
        continue
    one = AA3TO1.get(mon, 'X')
    if seq_id - 1 < len(seq) and (one == wt or seq[seq_id - 1] == wt):
        ok += 1
    else:
        bad += 1
        if len(bad_examples) < 15:
            bad_examples.append((pdb_id, chain, pos, wt, mon, seq_id,
                                 seq[seq_id - 1] if seq_id - 1 < len(seq) else 'OOR'))

print(f'\nSite validation: ok={ok}, mismatch={bad}, no_chain_map={no_map}, no_pos={no_pos}')
if bad_examples:
    print('Mismatch examples (pdb, chain, pos, skempi_wt, cif_mon, seq_id, seq_res):')
    for b in bad_examples:
        print('  ', b)
