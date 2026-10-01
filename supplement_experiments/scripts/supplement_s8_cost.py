# -*- coding: utf-8 -*-
"""
Table S8: 计算成本对比
  A) 实测块：参数量 / batch / 计划epochs / best_epoch / 每epoch耗时 / 达最佳耗时 / 峰值显存
     - GNN(4模型×4数据集): GrapthDTA/result_{model}_{ds}.json 的 history.epoch_time
     - Reptile+Transformer: 各数据集训练日志的 Time / GPU 字段
     - MLP / Transformer baseline: 权重参数量 + 训练配置（控制台日志当时未留存）
  B) 统一微基准块（需 --bench）: RTX 4090 合成 batch 前向+反向，ms/batch 与峰值显存
用法:
  python supplement_s8_cost.py            # A块，<1分钟，无需GPU
  python supplement_s8_cost.py --bench    # A+B块，约3-5分钟，需GPU（建议无其他训练任务时）
输出: supplement_output/paper_revision/TableS8_cost.csv
"""
import sys, os, re, json, time
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r'd:\lht')
OUT = ROOT / 'supplement_output' / 'paper_revision'
OUT.mkdir(parents=True, exist_ok=True)
import torch

GNN = ['GCNNet', 'GATNet', 'GAT_GCN', 'GINConvNet']
DSS = ['chembl', 'davis', 'kiba', 'bindingdb']
DS_CN = {'chembl': 'ChEMBL', 'davis': 'Davis', 'kiba': 'KIBA', 'bindingdb': 'BindingDB'}
HARDWARE = 'NVIDIA RTX 4090 (24GB), AMP/FP16'


def count_params(state):
    if hasattr(state, 'parameters'):        # nn.Module
        return sum(p.numel() for p in state.parameters())
    if isinstance(state, dict):
        for k in ('model_state_dict', 'state_dict', 'model', 'net'):
            if k in state and isinstance(state[k], dict):
                return sum(v.numel() for v in state[k].values() if torch.is_tensor(v))
        if all(torch.is_tensor(v) for v in state.values()):
            return sum(v.numel() for v in state.values())
    return None


# ================= A1) GNN：实例化取参数量 + result JSON 取耗时 =================
print('== A1) GNN 参数量（实例化模型类，不依赖权重文件）==')
sys.path.insert(0, str(ROOT / 'GrapthDTA'))
from models.gcn import GCNNet           # noqa: E402
from models.gat import GATNet           # noqa: E402
from models.gat_gcn import GAT_GCN      # noqa: E402
from models.ginconv import GINConvNet   # noqa: E402

GNN_CLS = {'GCNNet': GCNNet, 'GATNet': GATNet, 'GAT_GCN': GAT_GCN, 'GINConvNet': GINConvNet}
gnn_params = {m: count_params(GNN_CLS[m]()) for m in GNN}
for m, n in gnn_params.items():
    print(f'  {m}: {n/1e6:.2f}M')

rows = []
print('== A2) GNN 每 epoch 耗时（result_*.json history 中位值）==')
for m in GNN:
    for ds in DSS:
        fp = ROOT / 'GrapthDTA' / f'result_{m}_{ds}.json'
        n_train, n_batch = None, None
        csvp = ROOT / 'GrapthDTA' / 'data' / f'{ds}_train.csv'
        if csvp.exists():
            with open(csvp, 'rb') as f:
                n_train = sum(1 for _ in f) - 1
            n_batch = int(np.ceil(n_train / 1024))
        if not fp.exists():
            rows.append(['GNN', m, DS_CN[ds], gnn_params[m] / 1e6, 1024, 1000,
                         None, None, None, None, 'result JSON 缺失'])
            continue
        d = json.load(open(fp, encoding='utf-8'))
        ts = [h['epoch_time'] for h in d.get('history', []) if 'epoch_time' in h]
        best_ep = d.get('best_epoch')
        ep_s = float(np.median(ts)) if ts else None
        t_best = best_ep * ep_s / 60 if (ep_s and best_ep) else None
        note = f'每epoch为训练耗时(不含验证); 每epoch {n_batch} batches/{n_train}样本; JSON内{len(ts)}条记录'
        rows.append(['GNN', m, DS_CN[ds], gnn_params[m] / 1e6, 1024, 1000,
                     best_ep, round(ep_s, 2) if ep_s else None,
                     round(t_best, 1) if t_best else None, None, note])
        print(f'  {m:9s} {ds:10s} best_ep={best_ep}  median={ep_s:.2f}s  '
              f'达最佳={t_best:.1f}min' if ep_s else f'  {m} {ds}: 无epoch记录')

# ================= A3) Reptile 日志解析 =================
REP_LOGS = [
    ('ChEMBL', ROOT / 'reptile_output' / 'logs',
     ROOT / 'reptile_output' / 'best_model.pt'),
    ('Davis', ROOT / 'reptile_output_davis' / 'logs',
     ROOT / 'reptile_output_davis' / 'best_model.pt'),
    ('KIBA', ROOT / 'reptile_output_kiba' / 'logs',
     ROOT / 'reptile_output_kiba' / 'best_model.pt'),
    ('BindingDB', ROOT / 'batch_bindingdb_logs',
     ROOT / 'output_bdb_mse' / 'best_model.pt'),
]

def parse_reptile_log(logfile):
    """返回 epochs计划, batch, best_epoch, 每epoch耗时列表, 峰值allocated(GB)"""
    epochs_plan, batch, cur_ep, best_ep = None, None, None, None
    times, vram = [], []
    for line in open(logfile, encoding='utf-8', errors='ignore'):
        me = re.search(r'Epochs:\s*(\d+)', line)
        if me:
            epochs_plan = int(me.group(1))
        mb = re.search(r'Batch Size:\s*(\d+)', line)
        if mb:
            batch = int(mb.group(1))
        mh = re.search(r'\[Epoch (\d+)/(\d+)\]', line)
        if mh:
            cur_ep = int(mh.group(1))
            epochs_plan = int(mh.group(2))
            mt = re.search(r'Time:\s*([\d.]+)s', line)
            if mt:
                times.append(float(mt.group(1)))
            mg = re.search(r'GPU:\s*([\d.]+)GB/([\d.]+)GB', line)
            if mg:
                vram.append(float(mg.group(1)))
        if '新最佳模型' in line and cur_ep is not None:
            best_ep = cur_ep
    return epochs_plan, batch, best_ep, times, max(vram) if vram else None

print('== A3) Reptile+Transformer 训练日志 ==')
for ds_cn, logdir, ckpt in REP_LOGS:
    logs = sorted(logdir.glob('*.log')) if logdir.exists() else []
    if ds_cn == 'BindingDB':
        logs = [p for p in logs if 'reptile' in p.name]
    params = None
    if ckpt and ckpt.exists():
        try:
            params = count_params(torch.load(ckpt, map_location='cpu')) / 1e6
        except Exception:
            pass
    if not logs:
        rows.append(['Reptile+Transformer', 'Reptile+Transformer', ds_cn, params,
                     None, None, None, None, None, None, '未找到训练日志'])
        continue
    # 选 epoch 记录最多的日志（最完整的一次运行）
    best_log, best_stat, best_n = None, None, -1
    for lf in logs:
        try:
            stat = parse_reptile_log(lf)
        except Exception:
            continue
        if len(stat[3]) > best_n:
            best_log, best_stat, best_n = lf, stat, len(stat[3])
    epochs_plan, batch, best_ep, times, vram = best_stat
    batch = batch or (2048 if ds_cn == 'BindingDB' else 512)
    pnote = ''
    if ds_cn == 'BindingDB' and ckpt and 'output_bdb_mse' in str(ckpt):
        pnote = '; 参数量取自 output_bdb_mse 同构权重'
    if not times:
        rows.append(['Reptile+Transformer', 'Reptile+Transformer', ds_cn, params,
                     batch, epochs_plan, best_ep, None, None, vram,
                     f'日志无耗时字段: {best_log.name}{pnote}'])
        continue
    # epoch1 含 ESM/特征预计算（数百秒），稳态取 <=30s 的记录；BDB 全部~400s 则全取
    fast = [t for t in times if t <= 30]
    steady = fast if len(fast) >= 10 else times
    ep_s = float(np.median(steady))
    t_best = best_ep * ep_s / 60 if best_ep else None
    rows.append(['Reptile+Transformer', 'Reptile+Transformer', ds_cn, params,
                 batch, epochs_plan, best_ep, round(ep_s, 2),
                 round(t_best, 1) if t_best else None,
                 round(vram, 2) if vram else None,
                 f'稳态中位(共{len(steady)}条); 日志 {best_log.name}{pnote}'])
    print(f'  {ds_cn:9s}: batch={batch}, {ep_s:.2f}s/epoch (n={len(steady)}), '
          f'best_ep={best_ep}, VRAM={vram}, epochs={epochs_plan}')

# ================= A4) MLP / Transformer baseline：参数量 + 配置 =================
print('== A4) MLP / Transformer baseline 参数量与配置 ==')
BASELINE = [
    ('MLP', 'ChEMBL', ROOT / 'baseline_output' / 'best_model.pt', 2048, 220),
    ('MLP', 'KIBA', ROOT / 'baseline_output_kiba' / 'best_model.pt', 2048, 220),
    ('Transformer', 'ChEMBL', ROOT / 'transformer_baseline_output' / 'best_model.pt', 2048, 300),
    ('Transformer', 'KIBA', ROOT / 'transformer_baseline_output_kiba' / 'best_model.pt', 2048, 300),
]
for mname, ds_cn, ckpt, batch, epochs in BASELINE:
    params = None
    if ckpt.exists():
        try:
            params = count_params(torch.load(ckpt, map_location='cpu')) / 1e6
        except Exception as e:
            print(f'  {mname}|{ds_cn} 权重加载失败: {e}')
    rows.append(['非GNN', mname, ds_cn, params, batch, epochs, None, None, None, None,
                 '训练输出仅控制台、日志未留存; 耗时见B块微基准(若已运行--bench)'])
    print(f'  {mname} | {ds_cn}: {params}M, batch={batch}, epochs={epochs}')

# ================= B) 统一微基准（可选） =================
if '--bench' in sys.argv:
    print('== B) 统一微基准（RTX 4090, AMP, 合成 batch, 前向+反向 ×10）==')
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as GLoader
    import torch.nn as nn

    bench_rows = []
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    assert dev.type == 'cuda', '微基准需要 GPU'
    print(f'  device={torch.cuda.get_device_name(0)}')

    def bench(tag, build_batch, model, loss_target):
        try:
            model = model.to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=1e-4)
            batch = build_batch(dev)
            for _ in range(3):
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    out = model(*batch['x']) if 'x' in batch else model(batch['g'])
                    out = out if torch.is_tensor(out) else out[0]
                    loss = nn.MSELoss()(out.view(-1), loss_target(batch).view(-1))
                loss.backward(); opt.step()
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            for _ in range(10):
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    out = model(*batch['x']) if 'x' in batch else model(batch['g'])
                    out = out if torch.is_tensor(out) else out[0]
                    loss = nn.MSELoss()(out.view(-1), loss_target(batch).view(-1))
                loss.backward(); opt.step()
            torch.cuda.synchronize()
            ms = (time.time() - t0) / 10 * 1000
            vram = torch.cuda.max_memory_allocated() / 1024**3
            bench_rows.append([tag, round(ms, 1), round(vram, 2), 'AMP FP16 前向+反向(含优化器)'])
            print(f'  {tag}: {ms:.1f} ms/batch, 峰值 {vram:.2f} GiB')
        except Exception as e:
            bench_rows.append([tag, None, None, f'基准失败: {e}'])
            print(f'  {tag} 失败: {e}')
        finally:
            try:
                del model, opt, batch
            except Exception:
                pass
            torch.cuda.empty_cache()

    # B1) GNN: Davis 尺度合成图 (B=1024, 60原子/分子, 120有向边)
    B, N_AVG, E_AVG = 1024, 60, 120
    def make_gnn_batch(dev):
        g = Data(x=torch.rand(B * N_AVG, 78),
                 edge_index=torch.randint(0, B * N_AVG, (2, B * E_AVG)),
                 batch=torch.arange(B).repeat_interleave(N_AVG),
                 target=torch.randint(0, 25, (B, 1000)), y=torch.rand(B, 1)).to(dev)
        return {'g': g}
    for m in GNN:
        bench(f'GNN-{m} (B=1024, Davis尺度)', make_gnn_batch, GNN_CLS[m](),
              lambda b: b['g'].y.view(-1, 1).float())

    # B2) MLP: 与 BaselineMLP 同构（避免 import 触发其 main()）
    class BaselineMLP(nn.Module):
        def __init__(self, hidden_dim=512, dropout=0.1):
            super().__init__()
            H = hidden_dim
            self.morgan_proj = nn.Sequential(nn.Linear(2048, H), nn.LayerNorm(H), nn.GELU(), nn.Dropout(dropout))
            self.maccs_proj = nn.Sequential(nn.Linear(167, H // 2), nn.LayerNorm(H // 2), nn.GELU(), nn.Dropout(dropout))
            self.desc_proj = nn.Sequential(nn.Linear(10, H // 4), nn.LayerNorm(H // 4), nn.GELU(), nn.Dropout(dropout))
            self.protein_proj = nn.Sequential(nn.Linear(128, H), nn.LayerNorm(H), nn.GELU(), nn.Dropout(dropout))
            self.fusion = nn.Sequential(nn.Linear(H + H // 2 + H // 4 + H, H), nn.LayerNorm(H), nn.GELU(), nn.Dropout(dropout))
            self.predictor = nn.Sequential(nn.Linear(H, H // 2), nn.LayerNorm(H // 2), nn.GELU(),
                                            nn.Dropout(dropout), nn.Linear(H // 2, 1))
            self.output_scale = nn.Parameter(torch.ones(1))
            self.output_bias = nn.Parameter(torch.zeros(1))

        def forward(self, morgan_fp, maccs_fp, descriptors, protein):
            f = torch.cat([self.morgan_proj(morgan_fp), self.maccs_proj(maccs_fp),
                           self.desc_proj(descriptors), self.protein_proj(protein)], dim=-1)
            v = self.predictor(self.fusion(f))
            return v * self.output_scale + self.output_bias

    B2 = 2048
    def make_tab_batch(dev):
        x = (torch.rand(B2, 2048, device=dev), torch.rand(B2, 167, device=dev),
             torch.rand(B2, 10, device=dev), torch.rand(B2, 128, device=dev))
        return {'x': x, 'y': torch.rand(B2, 1, device=dev)}
    bench('MLP (B=2048)', make_tab_batch, BaselineMLP(), lambda b: b['y'])

    # B3) ReptileTransformer（蛋白侧输入预计算128维特征，不加载ESM2）
    from reptile_transformer_model import ReptileTransformer
    rt = ReptileTransformer()  # protein_encoder=None, 前向走预计算特征分支
    bench('ReptileTransformer (B=2048, 预计算蛋白特征)', make_tab_batch, rt, lambda b: b['y'])

    bdf = pd.DataFrame(bench_rows, columns=['基准项', 'ms/batch', '峰值显存(GiB)', '备注'])
    bdf.to_csv(OUT / 'TableS8_benchmark.csv', index=False, encoding='utf-8-sig')
    print('  B块已保存 → TableS8_benchmark.csv')

# ================= 汇总输出 =================
df = pd.DataFrame(rows, columns=[
    '类别', '模型', '数据集', '参数量(M)', 'Batch size', '计划Epochs', 'Best epoch',
    '每Epoch耗时(s)', '达Best累计(min)', '峰值显存(GiB,日志)', '备注'])
df.insert(0, '硬件', HARDWARE)
out = OUT / 'TableS8_cost.csv'
df.to_csv(out, index=False, encoding='utf-8-sig')
print('\n已保存 →', out)
print(df.drop(columns=['硬件', '备注']).to_string(index=False))
