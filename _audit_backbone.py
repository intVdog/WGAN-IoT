# -*- coding: utf-8 -*-
"""Compare light vs se vs wide cap100 results to see which one the paper
table-1 uses, and whether an 'se' run exists at cap100/300."""
import json
import glob

d = json.load(open('results/mitbih_scarce_aug.json', encoding='utf-8'))
print('scarce_aug.json (cap100) no_aug acc mean:',
      d['summary']['acc']['no_aug'])

# check log heads for backbone identity
for f in ['results/logs/mitbih_light_cap100_gated.log',
          'results/logs/mitbih_wide_cap300_gated.log']:
    raw = open(f, 'rb').read()
    txt = raw.decode('utf-16', errors='replace')
    first = [l for l in txt.splitlines() if 'backbone' in l or 'cap=' in l
             or 'scarce train' in l]
    print(f'\n{f}:')
    for l in first[:3]:
        print('   ', l.strip()[:110])

# does a cap100 'se' backbone log exist?
print('\nlogs with cap100:')
for f in sorted(glob.glob('results/logs/*.log')):
    raw = open(f, 'rb').read()
    txt = None
    for enc in ('utf-16', 'utf-8'):
        try:
            txt = raw.decode(enc)
            break
        except Exception:
            continue
    if txt and 'cap' in txt.lower() and '100' in txt:
        heads = [l.strip() for l in txt.splitlines()
                 if 'backbone' in l.lower()][:1]
        print(' ', f, heads[0][:80] if heads else '(no backbone line)')
