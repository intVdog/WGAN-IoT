# -*- coding: utf-8 -*-
"""Audit: does cap300 table-1 data (smote/realcopy rows) exist anywhere?"""
import re
import glob

# paper table 1 cap300 rows:
# no_aug 85.9±0.9/65.6±0.7 ; wgan100 86.8±1.2/66.7±1.1 ; smote 86.3±1.1/66.2±1.1
# realcopy 85.8±1.3/65.5±1.3 ; p: 0.063/0.26/0.76
print('=== logs mentioning smote or realcopy at cap300 ===')
for f in sorted(glob.glob('results/logs/*.log')):
    raw = open(f, 'rb').read()
    txt = None
    for enc in ('utf-16', 'utf-8'):
        try:
            txt = raw.decode(enc)
            break
        except Exception:
            continue
    if txt is None:
        continue
    if 'smote' in txt.lower() or 'realcopy' in txt.lower():
        lines = [l.strip() for l in txt.splitlines()
                 if re.search(r'(smote|realcopy|no_aug|wgan)', l, re.I)
                 and 'acc=' in l]
        if lines:
            print(f'--- {f}')
            for l in lines[:10]:
                print('   ', l[:130])

print('\n=== any cap300 json? ===')
for f in sorted(glob.glob('results/*.json')):
    if 'cap300' in f or 'wide' in f:
        print(f)

print('\n=== check existing scarce json configs ===')
import json
for f in sorted(glob.glob('results/*scarce*.json')):
    try:
        d = json.load(open(f, encoding='utf-8'))
        print(f, '-> cap', d.get('config', {}).get('cap'),
              'seeds', d.get('config', {}).get('seeds'))
    except Exception as e:
        print(f, 'ERR', e)
