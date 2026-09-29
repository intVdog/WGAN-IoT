# -*- coding: utf-8 -*-
"""Audit the section-4.6 claims: C=+2.2 catastrophe with -28pp acc drop."""
import re
import glob

print('=== 1. raw C/G lines in wide_cap300.log (ungated) ===')
raw = open('results/logs/mitbih_wide_cap300.log', 'rb').read()
txt = raw.decode('utf-16', errors='replace')
for line in txt.splitlines():
    if re.search(r'\[wgan\]|G=|C=', line):
        print('   ', line.strip()[:150])
print('\nfirst 30 lines:')
for line in txt.splitlines()[:30]:
    print('   ', line.strip()[:150])

print('\n=== 2. search EVERYTHING for "2.2" or "28" near wgan/collapse ===')
for f in sorted(glob.glob('results/logs/*.log')) + sorted(
        glob.glob('docs/*.md')):
    try:
        raw = open(f, 'rb').read()
    except Exception:
        continue
    txt = None
    for enc in ('utf-16', 'utf-8'):
        try:
            txt = raw.decode(enc)
            break
        except Exception:
            continue
    if txt is None:
        continue
    for i, line in enumerate(txt.splitlines()):
        if ('2.2' in line or '28' in line) and re.search(
                r'wgan|C=|collapse|catast|灾难|崩溃|28 个', line):
            print(f'{f}:{i}: {line.strip()[:150]}')
