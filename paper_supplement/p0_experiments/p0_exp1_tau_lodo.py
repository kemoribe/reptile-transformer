# -*- coding: utf-8 -*-
"""
P0 实验1：τ≈0.13 GNN 适用性阈值的留一数据集验证（LODO）
数据源（论文权威口径，V3 Table S3 / Figure 6 / exp6_conceptual_framework.txt）：
  ChEMBL    τ=0.122  ΔR²=-0.2041
  Davis     τ=0.137  ΔR²=+0.1412  （唯一 GNN 占优点）
  KIBA      τ=0.250  ΔR²=-0.0989
  BindingDB τ=0.360  ΔR²=-0.1340
规则族：GNN-favourable 带 [b_lo, b_hi]，边界=正类点与最近异类点的中点；
        另报告 1-近邻（1-NN）留一准确率与多数类基线。
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(r'd:\lht\修改supplement_output\p0_experiments')
OUT.mkdir(parents=True, exist_ok=True)

# x = training-panel mean ECFP4 Tanimoto (Table S3); y = 1 if GNN-best R2 (ΔR²>0)
DATA = [
    ('ChEMBL',    0.122, -0.2041, 0),
    ('Davis',     0.137,  0.1412, 1),
    ('KIBA',      0.250, -0.0989, 0),
    ('BindingDB', 0.360, -0.1340, 0),
]


def fit_band(train):
    """返回 (b_lo, b_hi)；None 表示该侧无界，'undef' 表示无正类、规则不可辨识。"""
    pos = [x for x, y in train if y == 1]
    neg = [x for x, y in train if y == 0]
    if not pos:
        return 'undef', 'undef'
    p = max(pos)  # 正类支撑点（本数据中恒为 1 个）
    below = [x for x in neg if x < p]
    above = [x for x in neg if x > p]
    b_lo = (p + max(below)) / 2 if below else None
    b_hi = (p + min(above)) / 2 if above else None
    return b_lo, b_hi


def in_band(x, band):
    b_lo, b_hi = band
    if b_lo == 'undef':
        return 0  # 训练折中无正类 -> 规则不可辨识，保守预测“非GNN优势”
    ok = True
    if b_lo is not None:
        ok &= x >= b_lo
    if b_hi is not None:
        ok &= x <= b_hi
    return int(ok)


def main():
    rows, bounds = [], []
    for i, (name, x, d, y) in enumerate(DATA):
        train = [(DATA[j][1], DATA[j][3]) for j in range(len(DATA)) if j != i]
        band = fit_band(train)
        pred = in_band(x, band)

        # 1-NN：按 |Δτ| 最近的训练数据集标签
        nn_j = min((j for j in range(len(DATA)) if j != i),
                   key=lambda j: abs(DATA[j][1] - x))
        nn_pred = DATA[nn_j][3]

        rows.append({
            'held_out': name, 'tau': x, 'delta_R2': d, 'true_GNN_adv': y,
            'b_lo': band[0] if band[0] != 'undef' else 'unidentifiable',
            'b_hi': band[1] if band[1] != 'undef' else 'unidentifiable',
            'band_pred': pred, 'band_correct': int(pred == y),
            'nn_pred': nn_pred, 'nn_correct': int(nn_pred == y),
        })
        for tag, b in (('b_lo', band[0]), ('b_hi', band[1])):
            if b not in (None, 'undef'):
                bounds.append({'held_out': name, 'boundary': tag, 'value': b})

    df = pd.DataFrame(rows)
    bd = pd.DataFrame(bounds)
    acc_band = df['band_correct'].mean()
    acc_nn = df['nn_correct'].mean()
    majority = 1 - np.mean([r[3] for r in DATA])  # 全为负类预测的准确率

    # 含两侧边界的完整模型（用全部 4 点拟合，仅作展示，非验证）
    full_band = fit_band([(r[1], r[3]) for r in DATA])

    summary_lines = []
    w = summary_lines.append
    w('P0 实验1：τ≈0.13 GNN 适用性阈值 —— 留一数据集(LODO)验证')
    w('=' * 64)
    w('数据（Table S3 / Figure 6 权威口径）:')
    for name, x, d, y in DATA:
        w(f'  {name:9s} τ={x:.3f}  ΔR²={d:+.4f}  GNN占优={y}')
    w('')
    w('逐折结果（带规则：GNN 优势带边界=正类点与最近异类 τ 的中点）:')
    for r in rows:
        w(f"  留出 {r['held_out']:9s} 训练带=[{r['b_lo']}, {r['b_hi']}]  "
          f"预测GNN={r['band_pred']} 实际={r['true_GNN_adv']} "
          f"{'✓' if r['band_correct'] else '✗'}")
    w('')
    g = bd.groupby('boundary')['value'].agg(['mean', 'min', 'max', 'std', 'count'])
    w('边界估计跨折稳定性:')
    for bn, rr in g.iterrows():
        w(f"  {bn}: mean={rr['mean']:.4f} range=[{rr['min']:.4f},{rr['max']:.4f}] "
          f"sd={rr['std']:.4f} n={int(rr['count'])}")
    w('')
    w(f'LODO 准确率（带规则）      = {acc_band:.3f} ({df["band_correct"].sum()}/4)')
    w(f'LODO 准确率（1-NN 参照）   = {acc_nn:.3f} ({df["nn_correct"].sum()}/4)')
    w(f'多数类基线（恒预测非GNN）  = {majority:.3f} (3/4)')
    w(f'全数据拟合带（仅供展示）   = [{full_band[0]}, {full_band[1]}]')
    w('')
    w('结论：4 个数据点中仅 1 个正类（Davis）。留出 Davis 时训练折无正类、规则不可')
    w('辨识；留出 ChEMBL 时下界不可辨识导致其被误判入 GNN 带。LODO 准确率 0.50，')
    w('低于多数类基线 0.75。τ≈0.13 不具备样本外预测效度，按预注册降级方案应表述为')
    w('descriptive summary（经验描述区域），不得作为部署阈值声称。')

    df.to_csv(OUT / 'p0_exp1_tau_lodo_folds.csv', index=False, encoding='utf-8-sig')
    bd.to_csv(OUT / 'p0_exp1_tau_lodo_boundaries.csv', index=False, encoding='utf-8-sig')
    summary = {
        'accuracy_band_rule': float(acc_band),
        'accuracy_1nn': float(acc_nn),
        'majority_baseline': float(majority),
        'full_data_band': [None if b == 'undef' else b for b in full_band],
        'boundary_stability': json.loads(g.reset_index().to_json(orient='records')),
    }
    with open(OUT / 'p0_exp1_tau_lodo_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    (OUT / 'p0_exp1_tau_lodo_report.txt').write_text(
        '\n'.join(summary_lines), encoding='utf-8')
    print('\n'.join(summary_lines))


if __name__ == '__main__':
    main()
