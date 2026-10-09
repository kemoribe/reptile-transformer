# -*- coding: utf-8 -*-
"""P1-3: PPI model zoo — standard vs audited (homology-free) evaluation.

Models (capacity gradient):
  Zero                : ddG = 0
  Mean                : ddG = train mean
  Linear              : 5 mutation scalar features
  RF                  : RandomForest on the same 5 scalars
  ESM2-mean+MLP       : chain mean-pool embedding -> MLP (protein-identity probe)
  ESM2-site+MLP       : site/window/mut embeddings + scalars -> MLP
  ESM2-site+TF        : window embedding + [MUT] token -> Transformer encoder

Evaluation sets:
  Standard    : all resolved test rows
  Audited-40% : exclude complexes containing any chain with same_cluster_40 = True
  Audited-30% : same with same_cluster_30

Protocol: TargetScaler (fit on train) for training; de-normalized predictions
for metrics; 3 seeds; group split (by complex) for early-stopping val set.
"""
import json
import copy
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.amp

OUT_DIR = Path(r'd:\lht\审计验证与跨领域')
P1_DIR = OUT_DIR / 'p1_out'
LOG_FILE = P1_DIR / 'p1_3_log.txt'

R = 1.987e-3
SEEDS = [42, 43, 44]
DEVICE = 'cuda'

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(str(msg) + '\n')

# ---------------- data ----------------
res = pd.DataFrame(json.load(open(P1_DIR / 'sites_resolved.json')))
res['chain_key'] = res['pdb_id'] + '_' + res['chain']

# ---- ESM2 assets ----
meta = json.load(open(P1_DIR / 'esm2_emb' / 'meta.json'))
alpha_info = json.load(open(P1_DIR / 'esm2_alphabet_info.json'))
embed_scale = alpha_info['embed_scale']
aa_tok_idx = alpha_info['aa_tok_idx']
aa_token_emb = np.load(P1_DIR / 'esm2_aa_token_emb.npy')  # vocab x 1280

emb_cache = {}
def chain_emb(key):
    if key not in emb_cache:
        emb_cache[key] = np.load(P1_DIR / 'esm2_emb' / f'{key}.npy')
    return emb_cache[key]

# ---- scalar mutation features ----
BLOSUM62 = {
    ('A','A'):4,('R','R'):5,('N','N'):6,('D','D'):6,('C','C'):9,('Q','Q'):5,('E','E'):5,('G','G'):6,('H','H'):8,('I','I'):4,('L','L'):4,('K','K'):5,('M','M'):5,('F','F'):6,('P','P'):7,('S','S'):4,('T','T'):5,('W','W'):11,('Y','Y'):7,('V','V'):4}
_B62 = {
 'A':-1,'R':-1,'N':-2,'D':-2,'C':0,'Q':0,'E':-1,'G':0,'H':-2,'I':-1,'L':-1,'K':-1,'M':1,'F':-2,'P':-1,'S':1,'T':0,'W':-3,'Y':-2,'V':0,}  # placeholder (unused)
def blosum(wt, mut):
    if wt == mut:
        return BLOSUM62.get((wt, wt), 0)
    pair = {wt: {mut: None}}
    # compact full table
    full = {
    'A':{'R':-1,'N':-2,'D':-2,'C':0,'Q':0,'E':-1,'G':0,'H':-2,'I':-1,'L':-1,'K':-1,'M':1,'F':-2,'P':-1,'S':1,'T':0,'W':-3,'Y':-2,'V':0},
    'R':{'A':-1,'N':0,'D':-2,'C':-3,'Q':1,'E':0,'G':-2,'H':0,'I':-3,'L':-2,'K':2,'M':-1,'F':-3,'P':-2,'S':-1,'T':-1,'W':-3,'Y':-2,'V':-3},
    'N':{'A':-2,'R':0,'D':1,'C':-3,'Q':0,'E':0,'G':0,'H':1,'I':-3,'L':-3,'K':0,'M':-2,'F':-3,'P':-2,'S':1,'T':0,'W':-4,'Y':-2,'V':-3},
    'D':{'A':-2,'R':-2,'N':1,'C':-3,'Q':0,'E':2,'G':-1,'H':-1,'I':-3,'L':-4,'K':-1,'M':-3,'F':-3,'P':-1,'S':0,'T':-1,'W':-4,'Y':-3,'V':-3},
    'C':{'A':0,'R':-3,'N':-3,'D':-3,'Q':-3,'E':-4,'G':-3,'H':-3,'I':-1,'L':-1,'K':-3,'M':-1,'F':-2,'P':-3,'S':-1,'T':-1,'W':-2,'Y':-2,'V':-1},
    'Q':{'A':0,'R':1,'N':0,'D':0,'C':-3,'E':2,'G':-2,'H':0,'I':-3,'L':-2,'K':1,'M':0,'F':-3,'P':-1,'S':0,'T':-1,'W':-2,'Y':-1,'V':-2},
    'E':{'A':-1,'R':0,'N':0,'D':2,'C':-4,'Q':2,'G':-2,'H':0,'I':-3,'L':-3,'K':1,'M':-2,'F':-3,'P':-1,'S':0,'T':-1,'W':-3,'Y':-2,'V':-2},
    'G':{'A':0,'R':-2,'N':0,'D':-1,'C':-3,'Q':-2,'E':-2,'H':-2,'I':-4,'L':-4,'K':-2,'M':-3,'F':-3,'P':-2,'S':0,'T':-2,'W':-2,'Y':-3,'V':-3},
    'H':{'A':-2,'R':0,'N':1,'D':-1,'C':-3,'Q':0,'E':0,'G':-2,'I':-3,'L':-3,'K':-1,'M':-2,'F':-1,'P':-2,'S':-1,'T':-2,'W':-2,'Y':2,'V':-3},
    'I':{'A':-1,'R':-3,'N':-3,'D':-3,'C':-1,'Q':-3,'E':-3,'G':-4,'H':-3,'L':2,'K':-3,'M':1,'F':0,'P':-3,'S':-2,'T':-1,'W':-3,'Y':-1,'V':3},
    'L':{'A':-1,'R':-2,'N':-3,'D':-4,'C':-1,'Q':-2,'E':-3,'G':-4,'H':-3,'I':2,'K':-2,'M':2,'F':0,'P':-3,'S':-2,'T':-1,'W':-2,'Y':-1,'V':1},
    'K':{'A':-1,'R':2,'N':0,'D':-1,'C':-3,'Q':1,'E':1,'G':-2,'H':-1,'I':-3,'L':-2,'M':-1,'F':-3,'P':-1,'S':0,'T':-1,'W':-3,'Y':-2,'V':-2},
    'M':{'A':1,'R':-1,'N':-2,'D':-3,'C':-1,'Q':0,'E':-2,'G':-3,'H':-2,'I':1,'L':2,'K':-1,'F':0,'P':-2,'S':1,'T':1,'W':-2,'Y':-1,'V':1},
    'F':{'A':-2,'R':-3,'N':-3,'D':-3,'C':-2,'Q':-3,'E':-3,'G':-3,'H':-1,'I':0,'L':0,'K':-3,'M':0,'P':-4,'S':-2,'T':-2,'W':1,'Y':3,'V':-1},
    'P':{'A':-1,'R':-2,'N':-2,'D':-1,'C':-3,'Q':-1,'E':-1,'G':-2,'H':-2,'I':-3,'L':-3,'K':-1,'M':-2,'F':-4,'S':-1,'T':-1,'W':-4,'Y':-3,'V':-2},
    'S':{'A':1,'R':-1,'N':1,'D':0,'C':-1,'Q':0,'E':0,'G':0,'H':-1,'I':-2,'L':-2,'K':0,'M':1,'F':-2,'P':-1,'T':1,'W':-3,'Y':-2,'V':-2},
    'T':{'A':0,'R':-1,'N':0,'D':-1,'C':-1,'Q':-1,'E':-1,'G':-2,'H':-2,'I':-1,'L':-1,'K':-1,'M':1,'F':-2,'P':-1,'S':1,'W':-2,'Y':-2,'V':0},
    'W':{'A':-3,'R':-3,'N':-4,'D':-4,'C':-2,'Q':-2,'E':-3,'G':-2,'H':-2,'I':-3,'L':-2,'K':-3,'M':-2,'F':1,'P':-4,'S':-3,'T':-2,'Y':2,'V':-3},
    'Y':{'A':-2,'R':-2,'N':-2,'D':-3,'C':-2,'Q':-1,'E':-2,'G':-3,'H':2,'I':-1,'L':-1,'K':-2,'M':-1,'F':3,'P':-3,'S':-2,'T':-2,'W':2,'V':-1},
    'V':{'A':0,'R':-3,'N':-3,'D':-3,'C':-1,'Q':-2,'E':-2,'G':-3,'H':-3,'I':3,'L':1,'K':-2,'M':1,'F':-1,'P':-2,'S':-2,'T':0,'W':-3,'Y':-1},
    }
    if wt in full and mut in full[wt]:
        return full[wt][mut]
    if mut in full and wt in full[mut]:
        return full[mut][wt]
    return 0

KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2,
      'B':-3.5,'Z':-3.5,'X':0.0,'U':2.5,'O':0.0}
VOL = {'A':88.6,'R':173.4,'N':114.1,'D':111.1,'C':108.5,'Q':143.8,'E':138.4,'G':60.1,'H':153.2,'I':166.7,'L':166.7,'K':168.6,'M':162.9,'F':189.9,'P':112.7,'S':89.0,'T':116.1,'W':227.8,'Y':193.6,'V':140.0,
       'B':111.1,'Z':138.4,'X':135.0,'U':108.5,'O':168.6}
CHG = {'A':0,'R':1,'N':0,'D':-1,'C':0,'Q':0,'E':-1,'G':0,'H':0.1,'I':0,'L':0,'K':1,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0,
       'B':0,'Z':0,'X':0,'U':0,'O':1}

def scalar_feats(row):
    wt, mut = row['wt'], row['mut']
    L = meta[row['chain_key']]['len']
    return np.array([blosum(wt, mut),
                     KD.get(mut, 0) - KD.get(wt, 0),
                     VOL.get(mut, 0) - VOL.get(wt, 0),
                     CHG.get(mut, 0) - CHG.get(wt, 0),
                     row['pos_seq0'] / max(L, 1)], dtype=np.float32)

def mut_emb(row):
    tok = aa_tok_idx.get(row['mut'], aa_tok_idx['X'])
    return (aa_token_emb[tok] * embed_scale).astype(np.float32)

def build_features(df, feat_type):
    """feat_type: 'scalar' | 'mean' | 'site'"""
    outs = []
    for row in df.itertuples():
        key = row.chain_key
        emb = chain_emb(key)
        pos = int(row.pos_seq0)
        pos = min(max(pos, 0), len(emb) - 1)
        if feat_type == 'scalar':
            outs.append(scalar_feats(row._asdict() | {}))
        elif feat_type == 'mean':
            outs.append(emb.mean(0))
        else:  # site
            w = emb[max(0, pos - 5):pos + 6].mean(0)
            sf = scalar_feats({'wt': row.wt, 'mut': row.mut, 'chain_key': key,
                               'pos_seq0': pos})
            me = mut_emb({'mut': row.mut})
            outs.append(np.concatenate([emb[pos], me, emb[pos] - me, w, sf * 3.0]))
    return np.array(outs, dtype=np.float32)

def scalar_feats(d):
    wt, mut = d['wt'], d['mut']
    L = meta[d['chain_key']]['len']
    return np.array([blosum(wt, mut), KD.get(mut, 0) - KD.get(wt, 0),
                     VOL.get(mut, 0) - VOL.get(wt, 0),
                     CHG.get(mut, 0) - CHG.get(wt, 0),
                     d['pos_seq0'] / max(L, 1)], dtype=np.float32)

# ---------------- audited exclusions ----------------
labels = pd.read_csv(OUT_DIR / 'skempi_prpi' / 'labels.csv')
audit = pd.read_csv(OUT_DIR / 'audit_results' / 'skempi_prpi_expanded' / 'per_target_audit.csv')

# chain_key -> sequence (train+test labels contain all)
chain_seqs = json.load(open(OUT_DIR / 'pdb_chain_sequences.json'))
key2seq = {k: chain_seqs[k] for k in labels['chain_key'] if k in chain_seqs}
seq2tid = {r.target_sequence: r.target_id for r in audit.itertuples()}
tid_flags = {r.target_id: {'c40': bool(r.same_cluster_40), 'c30': bool(r.same_cluster_30)}
             for r in audit.itertuples()}

def complex_leaks(complex_id):
    """True if any chain of the complex is same-cluster at 40% (and separately 30%)."""
    chs = complex_id.split('_')[1:]
    l40 = l30 = False
    for ch in chs:
        key = f'{complex_id.split("_")[0]}_{ch}'
        tid = seq2tid.get(key2seq.get(key, ''))
        if tid and tid in tid_flags:
            l40 |= tid_flags[tid]['c40']
            l30 |= tid_flags[tid]['c30']
    return l40, l30

test_complexes = res[res['split'] == 'test']['complex'].unique()
excl40, excl30 = set(), set()
for c in test_complexes:
    l40, l30 = complex_leaks(c)
    if l40:
        excl40.add(c)
    if l30:
        excl30.add(c)
log(f'Audited exclusions: 40% -> {len(excl40)} complexes {sorted(excl40)[:5]}..., '
    f'30% -> {len(excl30)} complexes')

test_df = res[res['split'] == 'test']
eval_sets = {
    'Standard': test_df,
    'Audited-40%': test_df[~test_df['complex'].isin(excl40)],
    'Audited-30%': test_df[~test_df['complex'].isin(excl30)],
}
for k, v in eval_sets.items():
    log(f'  {k}: {len(v)} rows, {v["complex"].nunique()} complexes')

train_df = res[res['split'] == 'train'].reset_index(drop=True)
y_train_raw = train_df['ddG'].values.astype(np.float32)

# TargetScaler (fit on train only)
ts_mean, ts_std = float(y_train_raw.mean()), float(y_train_raw.std())
log(f'TargetScaler: mean={ts_mean:.4f}, std={ts_std:.4f}')

# ---------------- models ----------------
def mlp(dims, dropout=0.2):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)

class MeanMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp([1280, 256, 64, 1], dropout=0.3)
    def forward(self, x):
        if not torch.is_tensor(x):
            x = x['mean']
        return self.net(x).squeeze(-1)

class SiteMLP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), mlp([d_in, 1024, 256, 64, 1], dropout=0.2))
    def forward(self, x):
        if not torch.is_tensor(x):
            x = x['site']
        return self.net(x).squeeze(-1)

class SiteTF(nn.Module):
    def __init__(self, d_emb=1280, d_model=256, nhead=8, layers=2, win=10):
        super().__init__()
        self.win = win
        self.proj = nn.Linear(d_emb, d_model)
        self.mut_proj = nn.Linear(d_emb, d_model)
        self.pos = nn.Parameter(torch.randn(2 * win + 2, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512,
                                         batch_first=True, dropout=0.1,
                                         activation='gelu')
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = mlp([d_model, 128, 1], dropout=0.1)
    def forward(self, x):
        # x['win']: (B, 2*win+1, 1280); x['mut']: (B, 1280)
        w = self.proj(x['win'])
        m = self.mut_proj(x['mut']).unsqueeze(1)  # (B,1,D) [MUT] token
        seq = torch.cat([m, w], dim=1) + self.pos.unsqueeze(0)
        h = self.encoder(seq)
        h = h.mean(1)
        return self.head(h).squeeze(-1)

def make_window_feats(df, win=10):
    outs = []
    for row in df.itertuples():
        emb = chain_emb(row.chain_key)
        pos = min(max(int(row.pos_seq0), 0), len(emb) - 1)
        idx = np.arange(pos - win, pos + win + 1)
        idx = np.clip(idx, 0, len(emb) - 1)
        outs.append(emb[idx])
    return np.array(outs, dtype=np.float32)

def mut_embs(df):
    return np.array([mut_emb({'mut': r.mut}) for r in df.itertuples()], dtype=np.float32)

# precompute feature matrices once
t0 = time.time()
log('Building features ...')
F = {
    'scalar_train': build_features(train_df, 'scalar'),
    'scalar_std': build_features(eval_sets['Standard'], 'scalar'),
    'scalar_a40': build_features(eval_sets['Audited-40%'], 'scalar'),
    'scalar_a30': build_features(eval_sets['Audited-30%'], 'scalar'),
    'mean_train': build_features(train_df, 'mean'),
    'mean_std': build_features(eval_sets['Standard'], 'mean'),
    'mean_a40': build_features(eval_sets['Audited-40%'], 'mean'),
    'mean_a30': build_features(eval_sets['Audited-30%'], 'mean'),
    'site_train': build_features(train_df, 'site'),
    'site_std': build_features(eval_sets['Standard'], 'site'),
    'site_a40': build_features(eval_sets['Audited-40%'], 'site'),
    'site_a30': build_features(eval_sets['Audited-30%'], 'site'),
    'win_train': make_window_feats(train_df),
    'win_std': make_window_feats(eval_sets['Standard']),
    'win_a40': make_window_feats(eval_sets['Audited-40%']),
    'win_a30': make_window_feats(eval_sets['Audited-30%']),
    'mut_train': mut_embs(train_df),
    'mut_std': mut_embs(eval_sets['Standard']),
    'mut_a40': mut_embs(eval_sets['Audited-40%']),
    'mut_a30': mut_embs(eval_sets['Audited-30%']),
}
log(f'Features built in {time.time()-t0:.0f}s')

def group_split(df, frac=0.12, seed=42):
    rng = np.random.RandomState(seed)
    comps = df['complex'].unique()
    rng.shuffle(comps)
    n_val = max(1, int(len(comps) * frac))
    val_c = set(comps[:n_val])
    is_val = df['complex'].isin(val_c).values
    return ~is_val, is_val

def train_torch(model, data, y, seed, max_epochs=300, patience=30,
                batch=256, lr=1e-3, warmup=5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(DEVICE)
    tr_idx, va_idx = group_split(data['df'], seed=seed)
    X = data['X']
    y_t = torch.tensor((y - ts_mean) / ts_std, dtype=torch.float32)
    scaler = torch.amp.GradScaler('cuda')
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = int(tr_idx.sum())
    steps_per_epoch = max(1, (n + batch - 1) // batch)
    total_steps = max_epochs * steps_per_epoch
    def lr_at(step):
        if step < warmup * steps_per_epoch:
            return (step + 1) / (warmup * steps_per_epoch)
        p = (step - warmup * steps_per_epoch) / max(1, total_steps - warmup * steps_per_epoch)
        return 0.5 * (1 + np.cos(np.pi * min(p, 1.0))) * (1 - 1e-5 / lr) + 1e-5 / lr
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    X_tr = {k: torch.tensor(v[tr_idx], dtype=torch.float32) for k, v in X.items()}
    X_va = {k: torch.tensor(v[va_idx], dtype=torch.float32) for k, v in X.items()}
    y_tr, y_va = y_t[tr_idx], y_t[va_idx]
    best_val, best_state, bad = float('inf'), None, 0
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            with torch.amp.autocast('cuda'):
                pred = model({k: v[idx].to(DEVICE) for k, v in X_tr.items()})
                loss = nn.functional.mse_loss(pred, y_tr[idx].to(DEVICE))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda'):
            pv = model({k: v.to(DEVICE) for k, v in X_va.items()}).float()
        vloss = nn.functional.mse_loss(pv.cpu(), y_va).item()
        if vloss < best_val - 1e-5:
            best_val, bad = vloss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def predict_torch(model, X):
    model.eval()
    preds = []
    with torch.no_grad(), torch.amp.autocast('cuda'):
        if isinstance(X, dict):
            Xt = {k: (v if torch.is_tensor(v) else torch.tensor(v, dtype=torch.float32)).to(DEVICE)
                  for k, v in X.items()}
            n = len(next(iter(Xt.values())))
        else:
            Xt = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            n = len(Xt)
        for s in range(0, n, 1024):
            if isinstance(Xt, dict):
                batch = {k: v[s:s + 1024] for k, v in Xt.items()}
            else:
                batch = Xt[s:s + 1024]
            preds.append(model(batch).float().cpu().numpy())
    p = np.concatenate(preds)
    return p * ts_std + ts_mean  # de-normalize

def metrics(y, p):
    out = {'R2': r2_score(y, p), 'RMSE': float(np.sqrt(mean_squared_error(y, p))),
           'MAE': float(mean_absolute_error(y, p))}
    if np.std(p) > 1e-8:
        out['Spearman'] = float(spearmanr(y, p).statistic)
    else:
        out['Spearman'] = 0.0
    return out

model_defs = ['Zero', 'Mean', 'Linear', 'RF', 'ESM2-mean+MLP', 'ESM2-site+MLP', 'ESM2-site+TF']
results = []
y_true_sets = {k: v['ddG'].values.astype(np.float32) for k, v in eval_sets.items()}

for seed in SEEDS:
    log(f'=== seed {seed} ===')
    # trivial
    preds = {'Zero': {k: np.zeros(len(v)) for k, v in eval_sets.items()},
             'Mean': {k: np.full(len(v), y_train_raw.mean()) for k, v in eval_sets.items()}}
    # Linear / RF (sklearn, seeded)
    lin = LinearRegression().fit(F['scalar_train'], (y_train_raw - ts_mean) / ts_std)
    preds['Linear'] = {k: lin.predict(F['scalar_' + suf]) * ts_std + ts_mean
                       for k, suf in [('Standard', 'std'), ('Audited-40%', 'a40'), ('Audited-30%', 'a30')]}
    rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=seed, n_jobs=16)
    rf.fit(F['scalar_train'], (y_train_raw - ts_mean) / ts_std)
    preds['RF'] = {k: rf.predict(F['scalar_' + suf]) * ts_std + ts_mean
                   for k, suf in [('Standard', 'std'), ('Audited-40%', 'a40'), ('Audited-30%', 'a30')]}

    # ESM2-mean+MLP
    torch.cuda.empty_cache()
    m = train_torch(MeanMLP(), {'df': train_df, 'X': {'mean': F['mean_train']}}, y_train_raw, seed)
    preds['ESM2-mean+MLP'] = {k: predict_torch(m, F['mean_' + suf])
                              for k, suf in [('Standard', 'std'), ('Audited-40%', 'a40'), ('Audited-30%', 'a30')]}

    # ESM2-site+MLP
    d_in = F['site_train'].shape[1]
    m = train_torch(SiteMLP(d_in), {'df': train_df, 'X': {'site': F['site_train']}}, y_train_raw, seed)
    preds['ESM2-site+MLP'] = {k: predict_torch(m, F['site_' + suf])
                              for k, suf in [('Standard', 'std'), ('Audited-40%', 'a40'), ('Audited-30%', 'a30')]}

    # ESM2-site+TF
    class TFData:
        pass
    tf_data = {'df': train_df, 'X': {'win': F['win_train'], 'mut': F['mut_train']}}
    m = train_torch(SiteTF(), tf_data, y_train_raw, seed)
    preds['ESM2-site+TF'] = {}
    for k, suf in [('Standard', 'std'), ('Audited-40%', 'a40'), ('Audited-30%', 'a30')]:
        Xt = {'win': torch.tensor(F['win_' + suf], dtype=torch.float32),
              'mut': torch.tensor(F['mut_' + suf], dtype=torch.float32)}
        preds['ESM2-site+TF'][k] = predict_torch(m, Xt)

    for mname in model_defs:
        for k in eval_sets:
            mt = metrics(y_true_sets[k], preds[mname][k])
            results.append({'model': mname, 'seed': seed, 'eval_set': k,
                            'n': len(y_true_sets[k]), **mt})
    pd.DataFrame(results).to_csv(P1_DIR / 'p1_model_results.csv', index=False)

# ---------------- summary ----------------
res_df = pd.DataFrame(results)
agg = res_df.groupby(['model', 'eval_set'])[['R2', 'RMSE', 'MAE', 'Spearman']].agg(['mean', 'std'])
log('\n=== Aggregate (mean over 3 seeds) ===')
for metric in ['Spearman', 'RMSE', 'R2', 'MAE']:
    log(f'\n--- {metric} (mean±std) ---')
    piv_m = res_df.pivot_table(index='model', columns='eval_set', values=metric, aggfunc='mean')
    piv_s = res_df.pivot_table(index='model', columns='eval_set', values=metric, aggfunc='std')
    for m in piv_m.index:
        row = '  '.join(f'{c}:{piv_m.loc[m,c]:.4f}±{(piv_s.loc[m,c] if not np.isnan(piv_s.loc[m,c]) else 0):.4f}'
                        for c in piv_m.columns)
        log(f'{m:16s} {row}')

# rank reversal + mis-selection cost
log('\n=== Rank reversal & mis-selection cost ===')
for metric, higher_better in [('Spearman', True), ('RMSE', False), ('R2', True)]:
    piv = res_df.pivot_table(index='model', columns='eval_set', values=metric, aggfunc='mean')
    rank_std = piv['Standard'].rank(ascending=not higher_better)
    rank_a40 = piv['Audited-40%'].rank(ascending=not higher_better)
    log(f'\n[{metric}] rank under Standard -> Audited-40%:')
    for m in piv.index:
        flag = ' *** REVERSAL' if rank_std[m] != rank_a40[m] else ''
        log(f'  {m:16s} {int(rank_std[m])} -> {int(rank_a40[m])}{flag}')
    # mis-selection cost
    if higher_better:
        best_std = piv['Standard'].idxmax()
        best_aud = piv['Audited-40%'].idxmax()
        cost = piv.loc[best_aud, 'Audited-40%'] - piv.loc[best_std, 'Audited-40%']
    else:
        best_std = piv['Standard'].idxmin()
        best_aud = piv['Audited-40%'].idxmin()
        cost = piv.loc[best_std, 'Audited-40%'] - piv.loc[best_aud, 'Audited-40%']
    log(f'  best@Standard={best_std}, best@Audited-40%={best_aud}, '
        f'mis-selection cost={cost:.4f}')

res_df.to_csv(P1_DIR / 'p1_model_results.csv', index=False)
log('\nSaved p1_out/p1_model_results.csv')
