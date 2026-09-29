# -*- coding: utf-8 -*-
"""Verify: ungated cap300 seed1 (N critic C=2.20) -> wgan100 acc collapse
of ~-28pp vs no_aug; and gated run recovery (5/5 seeds mF1 >= no_aug)."""
import re
import numpy as np

def parse(path):
    raw = open(path, 'rb').read()
    txt = raw.decode('utf-16', errors='replace')
    out = {}
    cur = None
    for line in txt.splitlines():
        m = re.match(r'===== seed (\d+) =====', line.strip())
        if m:
            cur = int(m.group(1))
            out[cur] = {'wgan_C': {}, 'acc': {}}
            continue
        if cur is None:
            continue
        m = re.match(r'\[wgan\] (\w+): kept \d+ \(G [-\d.]+ C ([-\d.]+)\)',
                     line.strip())
        if m:
            out[cur]['wgan_C'][m.group(1)] = float(m.group(2))
            continue
        m = re.match(r'\[(\w+)\] acc=([\d.]+) macroF1=([\d.]+)',
                     line.strip())
        if m and m.group(1) in ('no_aug', 'wgan100', 'wgan30', 'realcopy',
                                'smote'):
            out[cur]['acc'][m.group(1)] = (float(m.group(2)),
                                           float(m.group(3)))
    return out

for f in ['results/logs/mitbih_wide_cap300.log',
          'results/logs/mitbih_wide_cap300_gated.log']:
    d = parse(f)
    print(f'\n===== {f} =====')
    for s in sorted(d):
        row = d[s]
        cs = {k: f'{v:+.2f}' for k, v in row['wgan_C'].items()}
        na = row['acc'].get('no_aug')
        w1 = row['acc'].get('wgan100')
        drop = (w1[0] - na[0]) if (na and w1) else None
        print(f'seed {s}: N_C={cs.get("N","-"):>6} '
              f'no_aug={na[0] if na else float("nan"):7.2f} '
              f'wgan100={w1[0] if w1 else float("nan"):7.2f} '
              f'drop={drop:+.1f}pp' if drop is not None else
              f'seed {s}: N_C={cs.get("N","-"):>6} no_aug='
              f'{na[0] if na else float("nan"):.2f}')
