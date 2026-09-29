# -*- coding: utf-8 -*-
"""P1a: stealth working-point analysis (paper reproducible script).

Reads a v2 attack-run JSON and reports, for every attack arm:
  * source-class recall damage  dS  = no_aug.S - arm.S
  * global accuracy drift       dAcc = arm.acc - no_aug.acc (pp, mean)
and the empirical stealth front.  A working point is "silent destruction"
if |dAcc| < 1pp while S recall < 0.2 (10-seed means).  The paper's claim
is that no such stable mean working point exists on MIT-BIH under the
fixed protocol; this script is the evidence generator.

Usage:
    python scripts/p1a_stealth_curve.py \
        results/mitbih_attack_v2_cap200_r50-80_sS_fl100-200-400_def80_\
deffl400_adapt_seeds0-9_20260908.json [--source S]
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--source", default="S", help="source class letter")
    args = ap.parse_args()

    F = json.load(open(args.json_path, encoding="utf-8"))
    ps = F["per_seed"]
    seeds = sorted(ps.keys(), key=lambda s: int(s.replace("seed", "")))
    arms = F["config"]["arms"]
    noaug = f"no_aug"
    print(f"file: {args.json_path}")
    print(f"seeds: {len(seeds)}  source class: {args.source}\n")

    def mean(arm, metric, cls=None):
        vals = [ps[s][arm][metric] for s in seeds]
        return float(np.mean([v[cls] if cls else v for v in vals]))

    no_s = mean(noaug, "recall", args.source)
    no_a = mean(noaug, "acc") * 100
    print(f"no_aug: acc {no_a:.2f}  {args.source}-recall {no_s:.3f}\n")
    print(f"{'arm':<24} {'acc':>7} {'dAcc(pp)':>9} {'S':>6} "
          f"{'dS':>6} {'S range':>11}  note")
    for a in arms:
        if a == noaug:
            continue
        acc = mean(a, "acc") * 100
        s = mean(a, "recall", args.source)
        s_lo = min(ps[s_][a]["recall"][args.source] for s_ in seeds)
        s_hi = max(ps[s_][a]["recall"][args.source] for s_ in seeds)
        dacc = acc - no_a
        ds = no_s - s
        if abs(dacc) < 1.0 and s < 0.2:
            note = "*** silent destruction ***"
        elif s < 0.2:
            note = "destroyed, drifts"
        elif abs(dacc) < 1.0:
            note = "silent, weak"
        else:
            note = ""
        print(f"{a:<24} {acc:7.2f} {dacc:+9.2f} {s:6.3f} {ds:6.3f} "
              f"[{s_lo:.2f}-{s_hi:.2f}]  {note}")

    print("\n--- stealth curve points (S_recall, |dAcc|pp) ---")
    for a in arms:
        if a == noaug:
            continue
        # mean() returns acc as a FRACTION; no_a is in pp already.
        # convert fraction to pp consistently: dAcc_pp = (acc_frac - no_a_frac)*100
        dacc = (mean(a, "acc") - no_a / 100.0) * 100.0
        s = mean(a, "recall", args.source)
        print(f"({s:.3f}, {abs(dacc):.2f})  <- {a}")


if __name__ == "__main__":
    main()
