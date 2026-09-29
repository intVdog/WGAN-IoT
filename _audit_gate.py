# -*- coding: utf-8 -*-
"""Audit: which backbone produced the gate evidence in section 4.6?
Claims: catastrophic seed C=+2.2 with -28pp acc; C in [0.81,1.50]; gated
5/5 seeds macro-F1 >= no_aug (cap300)."""
import re
import glob

# look in cap300 logs for C values and gated/no-gated comparisons
for f in ['results/logs/mitbih_wide_cap300.log',
          'results/logs/mitbih_wide_cap300_gated.log']:
    raw = open(f, 'rb').read()
    txt = raw.decode('utf-16', errors='replace')
    print(f'\n===== {f} =====')
    # find SKIPPED events and C values
    for line in txt.splitlines():
        if 'SKIPPED' in line or ('C=' in line and 'wgan' in line.lower()):
            print('   ', line.strip()[:130])
