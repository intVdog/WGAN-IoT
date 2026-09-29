# -*- coding: utf-8 -*-
"""Find which run produced mitbih_scarce_aug.json (cap100, 5 seeds)."""
import json
import glob

d = json.load(open('results/mitbih_scarce_aug.json', encoding='utf-8'))
print('config:', d['config'])
print('per_seed seed0 acc:', d['per_seed']['seed0']['acc'])

# search logs for matching per-seed acc pattern
target = d['per_seed']['seed0']['acc']['no_aug']
print(f'\nlooking for no_aug acc ~ {target:.4f} in logs...')
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
    if f'no_aug] acc={target*100:.2f}' in txt:
        print('MATCH:', f)
    elif f'no_aug] acc={target*100:.1f}' in txt:
        print('approx:', f)

# check backbones mentioned in candidate logs
for f in ['results/logs/mitbih_light_cap100_gated.log',
          'results/logs/mitbih_run.log',
          'results/logs/scarce_aug_run.log']:
    try:
        raw = open(f, 'rb').read()
        txt = raw.decode('utf-16', errors='replace')
    except Exception:
        try:
            txt = open(f, encoding='utf-8', errors='replace').read()
        except Exception:
            continue
    head = [l.strip() for l in txt.splitlines()[:6] if l.strip()]
    print(f'\n--- {f} head ---')
    for l in head[:4]:
        print('   ', l[:120])
