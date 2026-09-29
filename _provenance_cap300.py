# -*- coding: utf-8 -*-
"""Strict provenance: match table-1 cap300 rows (85.9/86.8/86.3/85.8 acc,
65.6/66.7/66.2/65.5 mF1) against candidate logs (wide cap300 gated/ungated)."""
import re
import glob

# table-1 cap300 target means (acc, macroF1) per arm
target = {'no_aug': (85.9, 65.6), 'wgan100': (86.8, 66.7),
          'smote': (86.3, 66.2), 'realcopy': (85.8, 65.5)}

print('=== candidate logs ===')
for f in ['results/logs/mitbih_wide_cap300.log',
          'results/logs/mitbih_wide_cap300_gated.log',
          'results/logs/mitbih_light_cap100_gated.log']:
    raw = open(f, 'rb').read()
    txt = raw.decode('utf-16', errors='replace')
    bb = [l.strip() for l in txt.splitlines() if 'backbone=' in l][:1]
    # parse per-seed arm acc/macroF1
    data = {}
    cur = None
    for line in txt.splitlines():
        m = re.match(r'===== seed (\d+) =====', line.strip())
        if m:
            cur = int(m.group(1))
            data[cur] = {}
            continue
        m = re.match(r'\[(\w+)\] acc=([\d.]+) macroF1=([\d.]+)',
                     line.strip())
        if m and cur is not None:
            data[cur][m.group(1)] = (float(m.group(2)),
                                     float(m.group(3)))
    import numpy as np
    print(f'\n{f}')
    print('  backbone:', bb[0] if bb else '?')
    print('  seeds present:', sorted(data.keys()))
    for arm, (ta, tm) in target.items():
        vals_a = [data[s][arm][0] for s in sorted(data) if arm in data[s]]
        vals_m = [data[s][arm][1] for s in sorted(data) if arm in data[s]]
        if vals_a:
            ma = np.mean(vals_a)
            mm = np.mean(vals_m)
            print(f'  {arm:9s} mean acc {ma:6.2f} (target {ta})  '
                  f'mF1 {mm:6.2f} (target {tm})  n={len(vals_a)}')
        else:
            print(f'  {arm:9s} (absent)')
