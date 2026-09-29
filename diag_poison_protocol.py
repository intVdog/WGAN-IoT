# -*- coding: utf-8 -*-
"""Diagnostic: compare legacy (4x) vs exact-rho poisoning protocols.

Legacy build() (run_mitbih_attack.py) injects 4*n_bad source-class beats,
so a "rho=0.5" arm actually carries a 500-beat F pool (100 synth + 400
injected = 80%) vs the clean arm's 200.  This diagnostic trains, per seed,
the same WGAN pools as the P0 run and evaluates BOTH protocols under
identical pools:

  no_aug / clean
  p50_legacy, p30_legacy   (current script behaviour; should reproduce P0)
  p50_exact,  p30_exact    (fixed: pool length == n_add, injected == n_bad)

Output: results/diag_poison_protocol_seeds<ids>_<date>.json (+ prints).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from gan_gat.evaluate import classification_metrics, save_json
from gan_gat.utils import seed_everything

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "rmb", os.path.join(_THIS_DIR, "run_mitbih_aug.py"))
rmb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rmb)

CLASSES = rmb.CLASSES
CAP = 200
TARGET = 3      # F
SOURCE = 1      # S


def sha256_of(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device)
    FC, SC = TARGET, SOURCE

    seed_ids: list[int] = []
    for tok in args.seeds.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            seed_ids.extend(range(int(a), int(b) + 1))
        else:
            seed_ids.append(int(tok))
    seed_ids = sorted(set(seed_ids))

    date = time.strftime("%Y%m%d")
    out_path = (f"results/diag_poison_protocol_seeds"
                f"{min(seed_ids)}-{max(seed_ids)}_{date}.json")
    if os.path.exists(out_path) and not args.force:
        sys.exit(f"[P0] refusing to overwrite {out_path} (use --force)")

    x_tr_all, y_tr_all, x_te, y_te = rmb.load_data()
    print("test counts:", np.bincount(y_te, minlength=5).tolist())

    arms = ["no_aug", "clean", "p50_legacy", "p30_legacy",
            "p50_exact", "p30_exact"]
    results = {"config": {"exp": "diag_poison_protocol", "cap": args.cap,
                          "seeds": seed_ids,
                          "source_class": CLASSES[SC],
                          "target_class": CLASSES[FC],
                          "arms": arms,
                          "note": "legacy=4*n_bad injection (current "
                                  "script); exact=pool n_add, inject n_bad",
                          "env": {
                              "torch": torch.__version__,
                              "numpy": np.__version__,
                              "python": platform.python_version(),
                              "gpu": (torch.cuda.get_device_name(0)
                                      if torch.cuda.is_available()
                                      else "cpu"),
                              "cudnn_deterministic":
                                  torch.backends.cudnn.deterministic,
                              "script_sha256":
                                  sha256_of(os.path.abspath(__file__)),
                              "started_at":
                                  time.strftime("%Y-%m-%d %H:%M:%S"),
                          }},
               "per_seed": {}, "summary": {}}

    for seed in seed_ids:
        print(f"\n===== seed {seed} =====")
        seed_everything(seed)
        rng = np.random.RandomState(1000 + seed)
        tr_i = []
        for c in range(5):
            idx = np.where(y_tr_all == c)[0]
            n = min(args.cap, len(idx))
            tr_i.append(idx[rng.permutation(len(idx))[:n]])
        tr_i = np.concatenate(tr_i)
        xs, ys = x_tr_all[tr_i], y_tr_all[tr_i]
        n_tr_cls = np.bincount(ys, minlength=5)
        print("scarce:", n_tr_cls.tolist())

        # ---- WGAN pools identical to P0 ----
        pools, critics = {}, {}
        for c in range(5):
            n_c = int(n_tr_cls[c])
            if n_c < 40:
                continue
            xc = xs[ys == c]
            w = rmb.WGAN1D(device)
            g, cc = w.train_on(xc, seed=seed * 1000 + c)
            cand = w.generate(4000, seed=seed * 1000 + c)
            scores = w.score(cand)
            kept = cand[scores >= np.quantile(scores, 0.10)]
            pools[c] = kept
            critics[c] = w
            print(f"  [wgan] {CLASSES[c]}: C={cc:.2f} pool={len(kept)}")

        def build_arm(mode: str, rho: float):
            """mode: 'clean' | 'legacy' | 'exact'."""
            sx, sy = [], []
            f_len = None
            for c in range(5):
                n_c = int(n_tr_cls[c])
                if c not in pools or n_c < 40:
                    continue
                n_add = int(1.0 * n_c)
                if c == FC and mode != "clean":
                    n_good = int(n_add * (1 - rho))
                    n_bad = n_add - n_good
                    pool_f = pools[c][:n_good]
                    s_idx = np.where(ys == SC)[0]
                    rng2 = np.random.RandomState(seed * 7000 + c)
                    if mode == "legacy":
                        pick = s_idx[rng2.randint(
                            len(s_idx), size=4 * n_bad)]
                    else:  # exact
                        pick = s_idx[rng2.randint(
                            len(s_idx), size=n_bad)]
                    tampered = np.concatenate([pool_f, xs[pick]])
                    f_len = len(tampered)
                    sx.append(tampered)
                    sy.append(np.full(len(tampered), c, dtype=np.int64))
                    continue
                sx.append(pools[c][:n_add])
                sy.append(np.full(n_add, c, dtype=np.int64))
            return (np.concatenate([xs] + sx),
                    np.concatenate([ys] + sy)), f_len

        specs = [("no_aug", (xs, ys), None),
                 ("clean", build_arm("clean", 0.0)[0], 200)]
        for tag, rho in [("p50", 0.5), ("p30", 0.3)]:
            (xf_leg, yf_leg), l_leg = build_arm("legacy", rho)
            (xf_ex, yf_ex), l_ex = build_arm("exact", rho)
            specs.append((f"{tag}_legacy", (xf_leg, yf_leg), l_leg))
            specs.append((f"{tag}_exact", (xf_ex, yf_ex), l_ex))
            print(f"  rho={rho}: legacy F-pool len={l_leg} "
                  f"(injected {l_leg - int(200*(1-rho))}), "
                  f"exact F-pool len={l_ex}")

        per = {}
        for tag, (xf, yf), f_len in specs:
            m = rmb.build_backbone("se", 5).to(device)
            m = rmb.train_cnn(m, xf, yf, device, seed)
            p = rmb.predict(m, x_te, device)
            mm = classification_metrics(y_te, p, n_classes=5)
            cm = np.zeros((5, 5), dtype=int)
            for a_, b_ in zip(y_te, p):
                cm[a_, b_] += 1
            with np.errstate(divide="ignore", invalid="ignore"):
                rec = cm.diagonal() / cm.sum(1)
            per[tag] = {"acc": mm["accuracy"], "macro_f1": mm["f1"],
                        "recall": {CLASSES[c]: float(rec[c])
                                   for c in range(5)},
                        "f_pool_len": f_len}
            print(f"[{tag}] acc={mm['accuracy']*100:.2f} "
                  f"S={rec[SC]:.3f} F={rec[FC]:.3f} "
                  f"N={rec[0]:.2f} mF1={mm['f1']*100:.1f}")
        results["per_seed"][f"seed{seed}"] = per

    for a in arms:
        accs = [results["per_seed"][f"seed{s}"][a]["acc"]
                for s in seed_ids]
        srecs = [results["per_seed"][f"seed{s}"][a]["recall"]["S"]
                 for s in seed_ids]
        results["summary"][a] = {
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs, ddof=1)),
            "S_recall_mean": float(np.mean(srecs)),
            "S_recall_min": float(np.min(srecs)),
            "S_recall_max": float(np.max(srecs)),
        }
        print(f"\n{a}: acc {np.mean(accs)*100:.2f}±"
              f"{np.std(accs, ddof=1)*100:.2f} | "
              f"S {np.mean(srecs):.3f} "
              f"({np.min(srecs):.2f}-{np.max(srecs):.2f})")
    save_json(out_path, results)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
