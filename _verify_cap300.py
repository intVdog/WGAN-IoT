# -*- coding: utf-8 -*-
"""Verify table-1 cap300 rows against the gated wide-cap300 log (acc AND
macroF1 per seed), and redo paired tests on macroF1."""
import re
import numpy as np
from scipy import stats

raw = open('results/logs/mitbih_wide_cap300_gated.log', 'rb').read()
txt = raw.decode('utf-16', errors='replace')

data = {}
cur = None
for line in txt.splitlines():
    m = re.match(r'===== seed (\d+) =====', line.strip())
    if m:
        cur = int(m.group(1))
        data[cur] = {}
        continue
    m = re.match(r'\[(\w+)\] acc=([\d.]+) macroF1=([\d.]+)', line.strip())
    if m and cur is not None and m.group(1) in (
            'no_aug', 'wgan30', 'wgan100', 'realcopy', 'smote'):
        data[cur][m.group(1)] = (float(m.group(2)), float(m.group(3)))

print('per-seed acc/macroF1 (gated wide cap300):')
for s in sorted(data):
    row = data[s]
    print(f'seed {s}: ' + '  '.join(
        f'{k}={v[0]:.2f}/{v[1]:.2f}' for k, v in row.items()))

print('\nmean±std per arm:')
accs = {k: [] for k in ('no_aug', 'wgan100', 'realcopy', 'smote')}
mf1s = {k: [] for k in accs}
for s in sorted(data):
    for k in accs:
        if k in data[s]:
            accs[k].append(data[s][k][0])
            mf1s[k].append(data[s][k][1])
for k in accs:
    a = np.array(accs[k])
    m = np.array(mf1s[k])
    print(f'{k:9s} acc {a.mean():.2f}±{a.std(ddof=1):.2f}  '
          f'mF1 {m.mean():.2f}±{m.std(ddof=1):.2f}  (n={len(a)})')

print('\npaired t vs no_aug (macroF1):')
na = np.array(mf1s['no_aug'])
for k in ('wgan100', 'smote', 'realcopy'):
    t, p = stats.ttest_rel(np.array(mf1s[k]), na)
    print(f'{k:9s} mF1 d {np.mean(np.array(mf1s[k])-na):+.3f}  p={p:.4f}')
na_a = np.array(accs['no_aug'])
for k in ('wgan100', 'smote', 'realcopy'):
    t, p = stats.ttest_rel(np.array(accs[k]), na_a)
    print(f'{k:9s} acc d {np.mean(np.array(accs[k])-na_a):+.3f}  p={p:.4f}')
