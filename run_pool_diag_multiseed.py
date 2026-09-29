# -*- coding: utf-8 -*-
"""pool_diag_multiseed: 10-seed version of the F-critic overlap probe
(reviewer P0 item: "run the probe on 10 seeds, it costs hours").

For each seed 0-9: train the F-class gated WGAN-GP with the identical
protocol, then measure
  - overlap coefficient of (real source-class S scores) vs (F-pool scores)
  - the same for N / V
  - AUC of distinguishing real-S from F-synthetic scores
  - fraction of real-S beats above the 10%/50%/90% quantiles of pool scores
Outputs results/pool_diag_multiseed_<date>.json (mean +/- sd across seeds).
"""
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

import run_mitbih_aug as rmb  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CAP = 200
SEEDS = list(range(10))


def overlap_coef(a, b, nbins=50):
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    bins = np.linspace(lo, hi, nbins + 1)
    h1, _ = np.histogram(a, bins=bins, density=True)
    h2, _ = np.histogram(b, bins=bins, density=True)
    return float(np.minimum(h1, h2).sum() * (bins[1] - bins[0]))


def auc(pos, neg):
    """AUC that pos scores rank above neg scores (Mann-Whitney)."""
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv)) + 1
    rp = r[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


x_tr, y_tr, x_te, y_te = rmb.load_data()
rows = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    tr_i = []
    for c in range(5):
        idx = np.where(y_tr == c)[0]
        tr_i.append(idx[rng.permutation(len(idx))[:CAP]])
    xs, ys = x_tr[np.concatenate(tr_i)], y_tr[np.concatenate(tr_i)]
    xc = xs[ys == 3]
    w = rmb.WGAN1D(DEVICE)
    g, cc = w.train_on(xc, seed=seed * 1000 + 3)
    cand = w.generate(4000, seed=seed * 1000 + 3)
    sc = w.score(cand)
    kept = cand[sc >= np.quantile(sc, 0.10)]
    sc_kept = w.score(kept)
    row = {"seed": seed, "C": float(cc)}
    for cls, ci in [("S", 1), ("N", 0), ("V", 2)]:
        s = w.score(xs[ys == ci])
        row["overlap_" + cls] = overlap_coef(s, sc_kept)
        row["auc_" + cls] = auc(s, sc_kept)
        row["q90_" + cls] = float((s >= np.quantile(sc_kept, 0.90)).mean())
        row["q50_" + cls] = float((s >= np.quantile(sc_kept, 0.50)).mean())
    rows.append(row)
    print("seed %d: overlap S=%.3f AUC S=%.3f (C=%.2f)"
          % (seed, row["overlap_S"], row["auc_S"], row["C"]))

out = {"config": {"exp": "pool_diag_multiseed", "seeds": SEEDS, "cap": CAP,
                  "protocol": "identical F-class gated WGAN-GP (4000 cand, "
                              "keep top-90%)",
                  "gpu": torch.cuda.get_device_name(0),
                  "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")},
       "per_seed": rows}
for k in ["overlap_S", "overlap_N", "overlap_V", "auc_S", "auc_N", "auc_V",
          "q90_S", "q50_S"]:
    v = np.array([r[k] for r in rows], float)
    out.setdefault("summary", {})[k] = {"mean": float(v.mean()),
                                        "sd": float(v.std(ddof=1))}
    print("%-10s mean=%.4f sd=%.4f" % (k, v.mean(), v.std(ddof=1)))
date = time.strftime("%Y%m%d")
path = os.path.join(_ROOT, "results", "pool_diag_multiseed_%s.json" % date)
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("saved ->", path)
