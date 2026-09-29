# -*- coding: utf-8 -*-
"""Deep check: se cap100 - is the augmentation effect significant on acc?
Also Wilcoxon + per-seed signs. Compare light vs se."""
import json
import numpy as np
from scipy import stats

for path, label in [
        ('results/mitbih_aug_cap100_se_seeds0-4_20260909.json', 'se cap100'),
        ('results/mitbih_scarce_aug.json', 'light cap100 (old json)')]:
    d = json.load(open(path, encoding='utf-8'))
    ps = d['per_seed']
    seeds = sorted(ps.keys(), key=lambda s: int(s.replace('seed', '')))
    acc = {k: np.array([ps[s]['acc'][k] for s in seeds]) for k in
           ['no_aug', 'wgan30', 'wgan100', 'realcopy', 'smote']}
    mf1 = {k: np.array([ps[s]['macro_f1'][k] for s in seeds]) for k in acc}
    print(f'\n===== {label} (n={len(seeds)}) =====')
    for met, dd in [('acc', acc), ('macroF1', mf1)]:
        for k in ['wgan30', 'wgan100', 'realcopy', 'smote']:
            t, pt = stats.ttest_rel(dd[k], dd['no_aug'])
            try:
                _, pw = stats.wilcoxon(dd[k], dd['no_aug'])
            except ValueError:
                pw = float('nan')
            diff = dd[k] - dd['no_aug']
            pos = int((diff > 0).sum())
            print(f'{met:7s} {k:9s} mean {dd["no_aug"].mean()*100:6.2f} -> '
                  f'{dd[k].mean()*100:6.2f}  d {diff.mean()*100:+5.2f}pp '
                  f't_p {pt:.4f} w_p {pw:.4f}  +sign {pos}/{len(seeds)}')
