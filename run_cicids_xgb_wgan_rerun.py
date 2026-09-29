"""CICIDS-2017: XGBoost (no graph) x per-class WGAN-GP augmentation.

Closes the last open cell of the augmentation matrix on CICIDS2017:
strong tabular backbone (XGB, macro-F1 91.66 plain) fed with per-class
WGAN-GP synthetic samples (the exact generation pipeline of
run_cicids_perclass_wgan.py: 7 minority classes, critic top-90% filter,
target 8000/class).  Same train/test split as every CICIDS run
(stratified 0.0/0.2, seed 42; GAT no_aug reference macro-F1 61.56,
XGB plain reference 91.66 from results/cicids_xgb_baseline.json).

Synthetic samples are cached to results/cicids_perclass_synthetic.npz
(+ meta json) so reruns / other backbones skip the ~40 min GAN training.

Variants (all XGBoost 400 trees hist, same hyper-parameters):
  xgb_plain          : real only, no weighting
  xgb_plain_aug      : real + synthetic, hard labels
  xgb_balanced       : real only, sklearn-style balanced sample weights
  xgb_balanced_aug   : balanced weights + synthetic

Usage:
    python scripts/run_cicids_xgb_wgan.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from xgboost import XGBClassifier

from gan_gat.config import Config, add_common_args, config_from_args
from gan_gat.evaluate import classification_metrics, save_json
from gan_gat.preprocessing import stratified_split
from gan_gat.utils import seed_everything
from gan_gat.wgan_gp import ConditionalWGANGP

CLASS_NAMES = ["BENIGN", "DoS_Hulk", "PortScan", "DDoS", "DoS_GoldenEye",
               "FTP_Patator", "SSH_Patator", "DoS_slowloris",
               "DoS_Slowhttptest", "Bot", "Web_Attack_Brute_Force",
               "Web_Attack_XSS", "Infiltration", "Web_Attack_Sql_Injection",
               "Heartbleed"]
AUG_CLASSES = [5, 6, 7, 8, 9, 10, 11]
MIN_REAL = 100
N_CLASSES = 15

CACHE_NPZ = "results/cicids_perclass_synthetic.npz"
CACHE_META = "results/cicids_perclass_synthetic_meta.json"


def load_or_generate_synthetic(cfg, x_tr, y_tr, device,
                               target: int, keep_q: float):
    """Return (x_syn, y_syn, meta).  Reuses the on-disk cache if present."""
    if os.path.exists(CACHE_NPZ) and os.path.exists(CACHE_META):
        d = np.load(CACHE_NPZ)
        meta = json.load(open(CACHE_META))
        print(f"[cache] loaded {len(d['x_syn']):,} synthetic samples "
              f"from {CACHE_NPZ}")
        return d["x_syn"], d["y_syn"], meta

    counts = np.bincount(y_tr, minlength=N_CLASSES)
    sx, sy = [], []
    meta = {}
    for c in AUG_CLASSES:
        n_real = int(counts[c])
        if n_real < MIN_REAL:
            continue
        n_need = max(0, target - n_real)
        if n_need <= 0:
            continue
        xc = x_tr[y_tr == c]
        yc = np.full(len(xc), c, dtype=np.int64)
        wgan = ConditionalWGANGP(cfg, x_tr.shape[1], N_CLASSES,
                                 device=device)
        print(f"[per-class WGAN] class {CLASS_NAMES[c]} "
              f"({n_real} real, need {n_need})")
        wgan.train_on(xc, yc, minority_classes=[c], log=True)
        n_cand = int(n_need / keep_q) + 200
        cand = wgan.generate(n_cand, c)
        scores = wgan.score(cand, c)
        thr = np.quantile(scores, 1.0 - keep_q)
        kept = cand[scores >= thr][:n_need]
        meta[CLASS_NAMES[c]] = {"real": n_real, "candidates": n_cand,
                                "kept": len(kept),
                                "keep_rate": len(kept) / n_cand}
        print(f"[per-class WGAN] {CLASS_NAMES[c]}: {n_cand:,} candidates -> "
              f"kept {len(kept):,} ({len(kept) / n_cand * 100:.0f}%)")
        sx.append(kept)
        sy.append(np.full(len(kept), c, dtype=np.int64))
    x_syn = np.concatenate(sx)
    y_syn = np.concatenate(sy)
    np.savez(CACHE_NPZ, x_syn=x_syn, y_syn=y_syn)
    save_json(CACHE_META, meta)
    print(f"[cache] saved {len(x_syn):,} synthetic samples -> {CACHE_NPZ}")
    return x_syn, y_syn, meta


def balanced_weights(y: np.ndarray, counts: np.ndarray) -> np.ndarray:
    w = np.zeros(len(y))
    for c in range(N_CLASSES):
        w[y == c] = len(y) / (N_CLASSES * max(1, counts[c]))
    return w


def rare_recall(y_te: np.ndarray, pred: np.ndarray) -> dict:
    out = {}
    for c in range(5, N_CLASSES):
        name = CLASS_NAMES[c]
        tp = int(((pred == c) & (y_te == c)).sum())
        fn = int(((pred != c) & (y_te == c)).sum())
        out[name] = tp / max(1, tp + fn)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p)
    p.add_argument("--target", type=int, default=8000)
    p.add_argument("--keep-q", type=float, default=0.90)
    p.add_argument("--trees", type=int, default=400)
    p.add_argument("--depth", type=int, default=7)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--force-regenerate", action="store_true",
                   help="ignore the synthetic cache and retrain the GANs")
    args = p.parse_args()
    cfg = config_from_args(args)
    cfg.gat_epochs = 0          # GAN-only config fields
    seed_everything(cfg.random_state)
    device = torch.device(cfg.device)

    d = np.load("results/cicids_prepared.npz")
    x, y = d["X"], d["y"]
    x_tr, y_tr, _, _, x_te, y_te = stratified_split(x, y, 0.0, 0.2, 42)
    print(f"train {len(x_tr):,}  test {len(x_te):,}")

    if args.force_regenerate:
        for f in (CACHE_NPZ, CACHE_META):
            if os.path.exists(f):
                os.remove(f)
    t0 = time.time()
    x_syn, y_syn, gan_meta = load_or_generate_synthetic(
        cfg, x_tr, y_tr, device, args.target, args.keep_q)
    print(f"[gan] {time.time() - t0:.0f}s")

    counts_tr = np.bincount(y_tr, minlength=N_CLASSES)
    counts_aug = np.bincount(np.concatenate([y_tr, y_syn]),
                             minlength=N_CLASSES)
    x_aug = np.concatenate([x_tr, x_syn])
    y_aug = np.concatenate([y_tr, y_syn])
    print(f"augmented train: {len(x_aug):,} "
          f"(+{len(x_syn):,} synthetic, {len(x_syn) / len(x_tr) * 100:.1f}%)")

    results = {"config": vars(args), "gan_meta": gan_meta,
               "variants": {}, "rare_recall": {}}
    jobs = [("xgb_plain", x_tr, y_tr, counts_tr, False),
            ("xgb_plain_aug", x_aug, y_aug, counts_aug, False),
            ("xgb_balanced", x_tr, y_tr, counts_tr, True),
            ("xgb_balanced_aug", x_aug, y_aug, counts_aug, True)]
    for tag, xf, yf, counts, use_w in jobs:
        t1 = time.time()
        print(f"[XGB] training {tag} ({args.trees} trees)...")
        model = XGBClassifier(
            n_estimators=args.trees, max_depth=args.depth,
            learning_rate=args.lr, subsample=0.9, colsample_bytree=0.9,
            tree_method="hist", n_jobs=8, random_state=42,
            eval_metric="mlogloss")
        if use_w:
            model.fit(xf, yf, sample_weight=balanced_weights(yf, counts))
        else:
            model.fit(xf, yf)
        pred = model.predict(x_te)
        m = classification_metrics(y_te, pred, n_classes=N_CLASSES)
        print(f"[XGB {tag}] acc={m['accuracy'] * 100:.2f}  "
              f"P/R/F1={m['precision'] * 100:.2f}/{m['recall'] * 100:.2f}/"
              f"{m['f1'] * 100:.2f}  ({time.time() - t1:.0f}s)")
        results["variants"][tag] = m
        results["rare_recall"][tag] = rare_recall(y_te, pred)

    print("\n=== rare-class recall ===")
    for name in CLASS_NAMES[5:]:
        row = "  ".join(f"{k}={results['rare_recall'][k][name]:.3f}"
                        for k in ("xgb_plain", "xgb_plain_aug",
                                  "xgb_balanced", "xgb_balanced_aug"))
        print(f"  {name:<24}: {row}")

    save_json("results/cicids_xgb_wgan_rerun_20260908.json", results)
    print("saved -> results/cicids_xgb_wgan_rerun_20260908.json")


if __name__ == "__main__":
    main()
