# -*- coding: utf-8 -*-
"""P4: paired statistics for the paper's key comparisons (reproducible).

Paired by seed on the same test set: t-test, Wilcoxon signed-rank,
bootstrap 95% CI of the mean difference (10k resamples, seed 42).

Usage:
    python scripts/p4_paired_stats.py <attack-json> [--metric S]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy import stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--metric", default="S", help="class letter")
    args = ap.parse_args()

    F = json.load(open(args.json_path, encoding="utf-8"))
    ps = F["per_seed"]
    seeds = sorted(ps.keys(), key=lambda s: int(s.replace("seed", "")))

    def vec(arm):
        return np.array([ps[s][arm]["recall"][args.metric] for s in seeds],
                        dtype=float)

    def vec_acc(arm):
        return np.array([ps[s][arm]["acc"] for s in seeds], dtype=float)

    def report(name, a, b, use_acc=False):
        x = vec_acc(a) if use_acc else vec(a)
        y = vec_acc(b) if use_acc else vec(b)
        d = y - x
        t, p_t = stats.ttest_rel(y, x)
        try:
            _, p_w = stats.wilcoxon(y, x)
        except ValueError:
            p_w = float("nan")
        rng = np.random.RandomState(42)
        boot = [d[rng.randint(0, len(d), size=len(d))].mean()
                for _ in range(10000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        unit = "pp" if use_acc else ""
        print(f"{name:<44} d {d.mean():+.4f}{unit}  t_p {p_t:.4f}  "
              f"w_p {p_w:.4f}  boot95 [{lo:+.4f}{unit}, "
              f"{hi:+.4f}{unit}]")

    print(f"file: {args.json_path}  seeds: {len(seeds)}  "
          f"metric: {args.metric}-recall\n")
    report("poisoned50 vs clean", "clean", "poisoned_sSr50")
    report("poisoned80 vs clean", "clean", "poisoned_sSr80")
    report("adapt80 vs clean", "clean", "adapt_sSr80")
    report("flood400 vs clean", "clean", "flood_sSn400")
    report("flood400 vs poisoned50", "poisoned_sSr50", "flood_sSn400")
    report("defence(r80) vs no_aug (recovery)", "no_aug", "defence_sSr80")
    report("deffl(fl400) vs no_aug (recovery)", "no_aug", "deffl_sSn400")
    print("\n=== acc drift (pp) ===")
    report("poisoned50 acc vs no_aug", "no_aug", "poisoned_sSr50",
           use_acc=True)
    report("flood400 acc vs no_aug", "no_aug", "flood_sSn400", use_acc=True)
    report("deffl acc vs no_aug", "no_aug", "deffl_sSn400", use_acc=True)


if __name__ == "__main__":
    main()
