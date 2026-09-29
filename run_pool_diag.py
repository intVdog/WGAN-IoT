# -*- coding: utf-8 -*-
"""run_pool_diag.py -- provenance for Sec. 4.3 (pool-inspection limits).

Computes, from REAL data and REAL gated WGAN-GP pools (seed 0, cap-200
protocol identical to run_mitbih_aug.py), the four quantities quoted in
Sec. 4.3 of the paper:

  A  within-class two-half mean deviation   (expected small)
  B  clean-pool vs class-mean deviation     (expected larger, overlapping A)
  C  N-class vs F-class mean deviation      (cross-class contrast)
  D  threshold-overlap probe: fraction of REAL source-class (S) beats whose
     F-critic score is >= the 10% quantile of the F-pool scores
     (paper quotes ~43% on seed 0)

Outputs results/pool_diag_seed0_<date>.json with an env fingerprint.
CPU+GPU, one-off diagnostic (~3-5 min on RTX 4060).
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, ".pkgs", "docx_pkg"))
sys.path.insert(0, os.path.join(_ROOT, ".pkgs", "lxml_pkg"))

import run_mitbih_aug as rmb  # noqa: E402  (load_data, prep, WGAN1D, CLASSES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0
CAP = 200


def sha256_of(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def mean_dev(a: np.ndarray, b: np.ndarray) -> float:
    """Paper's mean-deviation metric: mean over features of |mean_a - mean_b|."""
    return float(np.abs(a.mean(0) - b.mean(0)).mean())


def main() -> None:
    t0 = time.strftime("%Y-%m-%d %H:%M:%S")
    x_tr, y_tr, x_te, y_te = rmb.load_data()
    rng = np.random.default_rng(SEED)
    tr_i = []
    for c in range(5):
        idx = np.where(y_tr == c)[0]
        tr_i.append(idx[rng.permutation(len(idx))[:CAP]])
    tr_i = np.concatenate(tr_i)
    xs, ys = x_tr[tr_i], y_tr[tr_i]

    res = {"A_two_half": [], "B_pool_vs_real": {}, "C_cross_class": {},
           "D_overlap": {}}

    # ---- A: within-class two-half deviation (20 random splits per class) --
    for c in range(5):
        xc = xs[ys == c]
        vals = []
        for k in range(20):
            perm = np.random.default_rng(SEED * 100 + c * 20 + k)
            idx = perm.permutation(len(xc))
            h1, h2 = xc[idx[:len(idx) // 2]], xc[idx[len(idx) // 2:]]
            vals.append(mean_dev(h1, h2))
        res["A_two_half"].append({"class": rmb.CLASSES[c],
                                  "min": min(vals), "max": max(vals),
                                  "mean": float(np.mean(vals))})

    # ---- pools: gated WGAN-GP per class (same protocol as run_mitbih_aug) --
    pools, critics = {}, {}
    gate_info = {}
    for c in range(5):
        xc = xs[ys == c]
        w = rmb.WGAN1D(DEVICE)
        g, cc = w.train_on(xc, seed=SEED * 1000 + c)
        cand = w.generate(4000, seed=SEED * 1000 + c)
        scores = w.score(cand)
        kept = cand[scores >= np.quantile(scores, 0.10)]
        pools[c], critics[c] = kept, w
        gate_info[c] = {"C": float(cc), "dev": mean_dev(kept, xc),
                        "pool_len": int(len(kept))}
        # ---- B: clean-pool vs class-mean deviation ----
        res["B_pool_vs_real"][rmb.CLASSES[c]] = mean_dev(kept, xc)
        print(f"[pool] {rmb.CLASSES[c]}: C={cc:.2f} dev="
              f"{res['B_pool_vs_real'][rmb.CLASSES[c]]:.4f}")

    # ---- C: cross-class N vs F mean deviation ----
    res["C_cross_class"] = {
        "N_vs_F": mean_dev(xs[ys == 0], xs[ys == 3]),
        "N_vs_S": mean_dev(xs[ys == 0], xs[ys == 1]),
        "V_vs_F": mean_dev(xs[ys == 2], xs[ys == 3]),
    }

    # ---- D: threshold-overlap probe (source S beats vs F pool) ----
    f_pool_scores = critics[3].score(pools[3])
    thr = float(np.quantile(f_pool_scores, 0.10))
    s_real = xs[ys == 1]
    s_scores = critics[3].score(s_real)
    frac_s = float((s_scores >= thr).mean())
    n_real = xs[ys == 0]
    n_scores = critics[3].score(n_real)
    frac_n = float((n_scores >= thr).mean())
    v_real = xs[ys == 2]
    frac_v = float((critics[3].score(v_real) >= thr).mean())
    res["D_overlap"] = {"threshold_10pct_of_F_pool": thr,
                        "source_S_above_thr": frac_s,
                        "source_N_above_thr": frac_n,
                        "source_V_above_thr": frac_v}
    print(f"[probe] thr={thr:.4f}  S>=thr: {frac_s:.3f}  "
          f"N>=thr: {frac_n:.3f}  V>=thr: {frac_v:.3f}")

    out = {
        "config": {"exp": "pool_diag", "seed": SEED, "cap": CAP,
                   "protocol": "identical pools/gate to run_mitbih_aug "
                               "(4000 cand, keep top-90%)",
                   "gpu": torch.cuda.get_device_name(0)
                   if DEVICE.type == "cuda" else "cpu",
                   "torch": torch.__version__,
                   "started_at": t0,
                   "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "rmb_sha256": sha256_of(os.path.join(
                       _HERE, "run_mitbih_aug.py")),
                   "self_sha256": sha256_of(os.path.abspath(__file__))},
        "gates": {rmb.CLASSES[c]: gate_info[c] for c in gate_info},
        "results": res,
    }
    date = time.strftime("%Y%m%d")
    path = os.path.join(_ROOT, "results", f"pool_diag_seed{SEED}_{date}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("saved ->", path)


if __name__ == "__main__":
    main()
