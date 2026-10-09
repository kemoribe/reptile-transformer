# -*- coding: utf-8 -*-
"""P1-2: Precompute per-chain ESM2 (esm2_t33_650M_UR50D) final-layer residue
embeddings for all chains referenced by resolved SKEMPI mutation sites.

Constraints honored:
  - torch.no_grad() inference; embeddings frozen (never trained)
  - float32 storage (fp16 compute on GPU)
  - chains longer than 1022 aa processed in overlapping chunks

Output:
  p1_out/esm2_emb/{pdb}_{chain}.npy   (L x 1280 float32)
  p1_out/esm2_emb/meta.json           chain -> {len, file}
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import esm

OUT_DIR = Path(r'd:\lht\审计验证与跨领域')
P1_DIR = OUT_DIR / 'p1_out'
EMB_DIR = P1_DIR / 'esm2_emb'
EMB_DIR.mkdir(exist_ok=True)

res = pd.read_json(P1_DIR / 'sites_resolved.json')
pdb_fasta = json.load(open(OUT_DIR / 'pdb_fasta_cache.json'))

needed = sorted({(r.pdb_id, r.chain) for r in res.itertuples()})
print(f'Chains needing embeddings: {len(needed)}')

MODEL_NAME = 'esm2_t33_650M_UR50D'
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.eval().cuda().half()
batch_converter = alphabet.get_batch_converter()
embed_scale = float(model.embed_scale)
# amino-acid token embeddings (for mutant residue features)
aa_token_emb = model.embed_tokens.weight.detach().float().cpu().numpy()  # vocab x 1280

meta = {}
if (EMB_DIR / 'meta.json').exists():
    meta = json.load(open(EMB_DIR / 'meta.json'))

tok_to_idx = alphabet.tok_to_idx
MAXLEN = 1022

def embed_sequence(seq):
    """Return (L,1280) float32 final-layer embeddings, chunking if needed."""
    L = len(seq)
    if L <= MAXLEN:
        return _embed_chunk(seq)
    # overlapping chunks
    step = MAXLEN - 100
    starts = list(range(0, L, step))
    out = np.zeros((L, 1280), dtype=np.float32)
    cnt = np.zeros((L, 1), dtype=np.float32)
    for s in starts:
        e = min(L, s + MAXLEN)
        emb = _embed_chunk(seq[s:e])  # (e-s, 1280)
        out[s:e] += emb
        cnt[s:e] += 1
    return out / np.maximum(cnt, 1)

def _embed_chunk(seq):
    data = [('chain', seq)]
    _, _, toks = batch_converter(data)
    toks = toks.cuda()
    with torch.no_grad():
        out = model(toks, repr_layers=[33], return_contacts=False)
    rep = out['representations'][33]  # (1, T, 1280) fp16
    rep = rep[0, 1:-1].float().cpu().numpy()  # strip BOS/EOS
    return rep

t0 = time.time()
done = 0
for pdb, chain in needed:
    key = f'{pdb}_{chain}'
    emb_file = EB_DIR_FILE = EMB_DIR / f'{key}.npy'
    if key in meta and emb_file.exists():
        continue
    seq = pdb_fasta.get(pdb, {}).get(chain)
    if seq is None:
        print(f'  MISSING seq {key}')
        continue
    emb = embed_sequence(seq)
    np.save(emb_file, emb.astype(np.float32))
    meta[key] = {'len': len(seq), 'file': emb_file.name}
    done += 1
    if done % 25 == 0:
        print(f'  {done} embedded, {time.time()-t0:.0f}s elapsed')
        json.dump(meta, open(EMB_DIR / 'meta.json', 'w'))

json.dump(meta, open(EMB_DIR / 'meta.json', 'w'))
# also save alphabet info for mutant-token features
json.dump({'embed_scale': embed_scale,
           'aa_tok_idx': {aa: tok_to_idx.get(aa, tok_to_idx['<unk>'])
                          for aa in 'ACDEFGHIKLMNPQRSTVWYBXZUO'}},
          open(P1_DIR / 'esm2_alphabet_info.json', 'w'))
np.save(P1_DIR / 'esm2_aa_token_emb.npy', aa_token_emb)
print(f'Done: {len(meta)} chains embedded in {time.time()-t0:.0f}s')
