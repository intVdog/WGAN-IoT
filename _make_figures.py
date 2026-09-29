# -*- coding: utf-8 -*-
"""Make submission-grade figures for the Chinese master draft (v3.0):
  fig1_pipeline.png   - training-pipeline & threat-model schematic
  fig2_spectrum.png   - prior spectrum: measured drift vs -f*p_s prediction
  fig3_frontier.png   - measured stealth frontier (S->F, 10 seeds)

  NOTE on numbering: the numeric prefix matches the figure number used in the
  paper (图 2 = prior spectrum, 图 3 = measured stealth frontier).  Earlier
  revisions saved these two under swapped prefixes (fig2_frontier /
  fig3_spectrum), which did not match the manuscript's figure order.

All empirical numbers are read from the canonical JSONs (same source the
paper tables are audited against), never hardcoded.
Output: docs/figs/*.png  (300 dpi, grayscale-friendly, Chinese labels).
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
from matplotlib import font_manager  # noqa: E402
for _fp in [r'C:\Windows\Fonts\simhei.ttf',
            r'C:\Windows\Fonts\msyh.ttc']:
    if os.path.exists(_fp):
        font_manager.fontManager.addfont(_fp)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei',
                                   'sans-serif']

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'docs', 'figs')
os.makedirs(OUT, exist_ok=True)

SS = json.load(open(os.path.join(
    ROOT, 'results', 'mitbih_attack_v2_cap200_r50-80_sS_fl100-200-400_'
    'def80_deffl400_adapt_seeds0-9_20260908.json'), encoding='utf-8'))
E1 = json.load(open(os.path.join(
    ROOT, 'results', 'mitbih_attack_v2_cap200_r50-80_sN_fl400_def80_'
    'deffl400_seeds0-9_20260909.json'), encoding='utf-8'))


def vec(j, seedn, arm, met, cls=None):
    return np.array([j['per_seed']['seed%d' % s][arm][met][cls] if cls else
                     j['per_seed']['seed%d' % s][arm][met]
                     for s in range(seedn)], float)


# --------------------------------------------------------------------------
# Figure 1: pipeline & threat model (schematic boxes)
# --------------------------------------------------------------------------
def fig1():
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    def box(x, y, w, h, text, fc='#eaf2fb', ec='black', fs=11, bold=False):
        b = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.6',
                           fc=fc, ec=ec, lw=1.2)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fs, fontweight='bold' if bold else 'normal')

    def arrow(x1, y1, x2, y2, color='black', ls='-'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='-|>', lw=1.3, color=color,
                                    linestyle=ls))

    # main vertical pipeline (centered at x=30..52)
    cx = 34
    bw, bh = 26, 11
    box(cx - bw / 2, 84, bw, bh,
        'Real scarce data D\n(100–200 beats/class)', fs=10)
    box(cx - bw / 2, 66, bw, bh,
        'Per-class WGAN-GP {G$_c$, D$_c$}\nspectral-norm critic + gate',
        fc='#fdf3e7', fs=9.5)
    box(cx - bw / 2, 48, bw, bh,
        'Generation pool P$_t$\n(top-90% kept)', fs=10)
    box(cx - bw / 2, 30, bw, bh,
        'Training set D$\\prime$\n(real + pool)', fs=10)
    box(cx - bw / 2, 12, bw, bh, 'Classifier h\n(se backbone)', fs=10)
    arrow(cx, 84, cx, 77.5)
    arrow(cx, 66, cx, 59.5)
    arrow(cx, 48, cx, 41.5)
    arrow(cx, 30, cx, 23.5)
    # attack 1: interface / transit of the pool
    ax.annotate('Attack 1: pool interface / transit\n'
                'relabel · adaptive · flood',
                xy=(cx + bw / 2, 53), xytext=(78, 62), fontsize=9.5,
                ha='center', color='#b00000',
                arrowprops=dict(arrowstyle='-|>', color='#b00000', lw=1.4))
    # attack 2: generator training data
    ax.annotate('Attack 2: generator training data\nmix γ source beats',
                xy=(cx - bw / 2, 72), xytext=(6, 80), fontsize=9.5,
                ha='center', color='#b00000',
                arrowprops=dict(arrowstyle='-|>', color='#b00000', lw=1.4))
    # defence: full-pool rebuild (dashed, green) + calibration gate
    ax.annotate('Defence: full-pool rebuild\n'
                'drop P$\\prime$, regen. from clean G$_t$',
                xy=(cx + bw / 2, 36), xytext=(84, 34), fontsize=9.5,
                ha='center', color='#005000',
                arrowprops=dict(arrowstyle='-|>', color='#005000', lw=1.4,
                                linestyle='--'))
    ax.annotate('Defence: calibration gate\n(skip class if C > 0.5)',
                xy=(cx - bw / 2, 70), xytext=(2, 58), fontsize=9.5,
                ha='center', color='#005000',
                arrowprops=dict(arrowstyle='-|>', color='#005000', lw=1.4,
                                linestyle='--'))
    ax.text(50, 3,
            'Note: injection occurs after critic filtering; all defence '
            'inputs lie in the normal pipeline data flow.',
            ha='center', fontsize=8.5, color='dimgray')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig1_pipeline.png'), dpi=300,
                bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 2: prior spectrum (N/V/S): measured drift vs -f*p_s prediction
# --------------------------------------------------------------------------
def fig2():
    pN = 18118 / 21892.0
    pS = 556 / 21892.0
    pV = 1448 / 21892.0

    def arm_mean(j, seedn, arm, met, cls=None):
        if cls:
            return float(vec(j, seedn, arm, met, cls).mean())
        return float(vec(j, seedn, arm, met).mean())

    # N (n=10): r50/r80/flood400
    N0 = arm_mean(E1, 10, 'no_aug', 'recall', 'N')
    Npts = []
    for arm, lab in [('poisoned_sNr50', 'replacement 50%'),
                     ('poisoned_sNr80', 'replacement 80%'),
                     ('flood_sNn400', 'flood 400')]:
        f = N0 - arm_mean(E1, 10, arm, 'recall', 'N')
        meas = (arm_mean(E1, 10, arm, 'acc')
                - arm_mean(E1, 10, 'no_aug', 'acc')) * 100
        Npts.append((pN, f, meas, lab))
    # S (n=10, S->F)
    S0 = arm_mean(SS, 10, 'no_aug', 'recall', 'S')
    Spts = []
    for arm, lab in [('poisoned_sSr50', 'replacement 50%'),
                     ('flood_sSn400', 'flood 400')]:
        f = S0 - arm_mean(SS, 10, arm, 'recall', 'S')
        meas = (arm_mean(SS, 10, arm, 'acc')
                - arm_mean(SS, 10, 'no_aug', 'acc')) * 100
        Spts.append((pS, f, meas, lab))
    # V (n=3)
    VJ = json.load(open(os.path.join(
        ROOT, 'results', 'mitbih_attack_v2_cap200_r50-80_sV_seeds0-2_'
        '20260908.json'), encoding='utf-8'))

    def vmean(arm, met, cls=None):
        if cls:
            return float(np.mean([VJ['per_seed']['seed%d' % s][arm][met][cls]
                                  for s in range(3)]))
        return float(np.mean([VJ['per_seed']['seed%d' % s][arm][met]
                              for s in range(3)]))

    V0 = vmean('no_aug', 'recall', 'V')
    fV = V0 - vmean('poisoned_sVr50', 'recall', 'V')
    mV = (vmean('poisoned_sVr50', 'acc') - vmean('no_aug', 'acc')) * 100

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    series = [('N', pN, Npts, 'o'), ('V', pV, [(pV, fV, mV, 'replacement 50%')], 's'),
              ('S', pS, Spts, '^')]
    for name, p, pts, mk in series:
        xs = [p] * len(pts)
        preds = [-f * p * 100 for _, f, _, _ in pts]
        meass = [m for _, _, m, _ in pts]
        labs = [l for _, _, _, l in pts]
        ax.scatter(xs, meass, marker=mk, s=90, zorder=6,
                   edgecolor='black', linewidths=.9,
                   label='source %s (measured)' % name)
        for x, pr, lab in zip(xs, preds, labs):
            ax.scatter([x], [pr], marker='x', s=70, zorder=5,
                       color='black')
        for x, m, lab in zip(xs, meass, labs):
            ax.annotate(lab, (x, m), textcoords='offset points',
                        xytext=(10, 6), fontsize=8)
    # silent band
    ax.axhspan(-1, 1, color='gray', alpha=0.18)
    ax.text(0.14, 0.4, 'near-silent band', fontsize=9, color='dimgray',
            ha='center')
    ax.set_xscale('log')
    ax.set_xticks([pN, pV, pS])
    ax.set_xticklabels(['N\n0.828', 'V\n0.066', 'S\n0.025'])
    ax.set_xlabel('test prior of source class p$_s$ (log scale)')
    ax.set_ylabel('accuracy drift of 50% replacement (pp)')
    ax.set_ylim(-60, 15)
    ax.axhline(0, color='black', lw=.8)
    ax.grid(alpha=.25, ls=':')
    ax.legend(fontsize=9, loc='lower left')
    ax.set_title('drift = -f·p$_s$ + collateral: large-prior sources must '
                 'ring (x = prediction, markers = measured; flood 400 also '
                 'shown for N)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig2_spectrum.png'), dpi=300,
                bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 3: measured stealth frontier (S->F; per-seed points per arm)
# --------------------------------------------------------------------------
def fig3():
    arms = [('poisoned_sSr50', 'replacement 50%'), ('poisoned_sSr80', 'replacement 80%'),
            ('adapt_sSr80', 'adaptive 80%'), ('flood_sSn100', 'flood +100'),
            ('flood_sSn200', 'flood +200'), ('flood_sSn400', 'flood +400')]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for arm, lab in arms:
        ds = 1 - vec(SS, 10, arm, 'recall', 'S')
        da = np.abs(vec(SS, 10, arm, 'acc') - vec(SS, 10, 'no_aug', 'acc'))
        ax.scatter(ds * 100, da * 100, s=22, alpha=0.55, edgecolor='none',
                   label=lab)
        ax.scatter([ds.mean() * 100], [da.mean() * 100], s=70,
                   marker='D', zorder=5, edgecolor='black', linewidths=.8)
    # silent region: |dAcc| <= 1pp
    ax.axhspan(0, 1, color='gray', alpha=0.18)
    ax.text(62, 0.5, 'near-silent band\n(|ΔAcc| ≤ 1pp)', fontsize=9,
            va='bottom', ha='left', color='dimgray')
    ax.set_xlabel('source-class recall loss dS = 1 - R$_S$ (%)')
    ax.set_ylabel('|ΔAcc| (pp)')
    ax.set_xlim(-2, 105)
    ax.set_ylim(0, 70)
    ax.legend(fontsize=8.5, ncol=2, framealpha=.9, loc='upper left')
    ax.grid(alpha=.25, ls=':')
    ax.set_title('Measured operating points: no stable point is both silent and destructive', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig3_frontier.png'), dpi=300,
                bbox_inches='tight')
    plt.close(fig)


fig1()
fig2()
fig3()
print('figures ->', OUT)
for f in sorted(os.listdir(OUT)):
    print('  ', f, os.path.getsize(os.path.join(OUT, f)))
