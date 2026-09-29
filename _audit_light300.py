# -*- coding: utf-8 -*-
"""Check: does a light-backbone cap300 run exist? List every aug-run log's
backbone+cap, plus any json with backbone=light & cap=300."""
import json
import glob
import re

print('=== all aug-related logs: backbone & cap ===')
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
    bb = None
    cap = None
    for line in txt.splitlines():
        m = re.search(r'backbone=(\w+)', line)
        if m:
            bb = m.group(1)
        m = re.search(r'scarce train (\d+)', line)
        if m:
            # 500 -> cap100, 1000 -> cap200, 1500 -> cap300
            cap = int(m.group(1)) // 5
    if bb or cap:
        print(f'{f}: backbone={bb} cap={cap}')

print('\n=== all mitbih aug json configs ===')
for f in sorted(glob.glob('results/mitbih*.json')) + \
        sorted(glob.glob('results/*aug*.json')):
    try:
        d = json.load(open(f, encoding='utf-8'))
        c = d.get('config', {})
        print(f'{f}: cap={c.get("cap")} backbone={c.get("backbone")} '
              f'seeds={c.get("seeds")}')
    except Exception:
        pass
