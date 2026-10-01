# -*- coding: utf-8 -*-
"""Generate the tiny synthetic example datasets shipped in examples/.

Design (sequences of exactly 160 residues, so bidirectional coverage -c 0.8
never filters a pair for length reasons):

dataset ``demo``
* family A — train A1/A2 (~85% identical); test A3 ~65% identical to A1
  -> same cluster with training at 40%, remote at 80%
* family B — train B1; test B2 ~75% identical
  -> same cluster at 40/60%, remote at 80%
* two independent random test sequences R1/R2
  -> cross-cluster (homology-free) at every threshold

dataset ``demo2`` (needed so the router LODO check has >=2 panels)
* family C — train C1/C2; test C3 ~70% identical -> same at 40/60%
* three independent random test sequences -> remote at every threshold

Predictions for 3 demo models are synthesized on ``demo`` so that the
*standard* full-panel winner differs from the *audited* (homology-free
subset) winner, demonstrating the mis-selection cost analysis.

Run:  python examples/make_example_data.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
SEED = 20240517
SEQ_LEN = 160
HERE = Path(__file__).resolve().parent


def random_seq(rng):
    return "".join(rng.choice(list(AA), size=SEQ_LEN))


def mutate(seq, rate, rng):
    """Replace ``rate`` fraction of positions with a *different* amino acid."""
    arr = list(seq)
    pos = rng.permutation(SEQ_LEN)[: int(round(rate * SEQ_LEN))]
    for i in pos:
        choices = [a for a in AA if a != arr[i]]
        arr[i] = rng.choice(choices)
    return "".join(arr)


def write_split(name, train_seqs, test_specs, rng):
    """test_specs = [(seq, true_base_affinity, n_compounds), ...]."""
    data_dir = HERE / "data"
    pd.DataFrame({"target_sequence": train_seqs}).to_csv(
        data_dir / f"{name}_train.csv", index=False)
    rows = []
    for j, (seq, base, n_comp) in enumerate(test_specs):
        for k in range(n_comp):
            rows.append({"compound_id": f"{name}_C{j:02d}_{k:02d}",
                         "target_sequence": seq,
                         "y": base + rng.normal(0, 0.3)})
    test_df = pd.DataFrame(rows)
    test_df.to_csv(data_dir / f"{name}_test.csv", index=False)
    return test_df


def main():
    data_dir = HERE / "data"
    data_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---- demo ---------------------------------------------------------------
    a1 = random_seq(rng)
    a2 = mutate(a1, 0.15, rng)
    a3 = mutate(a1, 0.35, rng)       # ~65% identity to A1 -> same @40%
    b1 = random_seq(rng)
    b2 = mutate(b1, 0.25, rng)       # ~75% identity to B1 -> same @40/60%
    r1, r2 = random_seq(rng), random_seq(rng)
    # 60 pseudo-compounds/target -> 120-pair audited (remote) panel, enough
    # for EF@1% to have at least one top-ranked position
    test_specs_demo = [(a3, 6.0, 60), (b2, 5.0, 60), (r1, 7.0, 60), (r2, 4.0, 60)]
    test_df = write_split("demo", [a1, a2, b1], test_specs_demo, rng)

    # ---- demo2 --------------------------------------------------------------
    c1 = random_seq(rng)
    c2 = mutate(c1, 0.10, rng)
    c3 = mutate(c1, 0.30, rng)       # ~70% identity -> same @40/60%
    r3, r4, r5 = random_seq(rng), random_seq(rng), random_seq(rng)
    write_split("demo2", [c1, c2],
                [(c3, 6.5, 60), (r3, 5.5, 60), (r4, 4.5, 60), (r5, 6.0, 60)], rng)

    # ---- predictions on demo ------------------------------------------------
    # GraphLikeModel: accurate on homologous targets, poor on remote targets
    # RemoteExpert : accurate on remote targets, mediocre on homologous ones
    # MiddleModel  : mediocre everywhere
    rows = []
    rng2 = np.random.default_rng(SEED + 1)
    for j, (seq, _, _) in enumerate(test_specs_demo):
        remote = j >= 2
        sub = test_df[test_df.target_sequence == seq]
        for _, r in sub.iterrows():
            y = r["y"]
            # Full panel (2 homologous + 2 remote targets): GraphLikeModel's
            # smaller cross-family penalty (0.45 vs 0.70) makes it win; on the
            # audited remote-only panel RemoteExpert wins -> winner reversal.
            preds = {
                "GraphLikeModel": y + rng2.normal(0, 0.25) + (0.45 if remote else 0.0),
                "RemoteExpert": y + rng2.normal(0, 0.25) + (0.0 if remote else 0.70),
                "MiddleModel": y + rng2.normal(0, 0.5),
            }
            for model, yhat in preds.items():
                rows.append({"dataset": "demo", "model": model,
                             "target_sequence": seq, "y_true": y, "y_pred": yhat})
    pred_dir = HERE / "predictions"
    pred_dir.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(pred_dir / "demo_predictions.csv", index=False)

    print("example data written under", data_dir, "and", pred_dir)


if __name__ == "__main__":
    main()
