# -*- coding: utf-8 -*-
"""Audit results/edge_deploy_*.json: internal consistency + derived math.

Checks (all recomputed from the JSON, nothing hard-coded except the known
backbone parameter counts and the assumed device-model constants):
  1) profile: MACs == sum(layer MACs), params match build_backbone counts
  2) latency: per_beat == total / batch; throughput == batch / total
  3) int8: delta == int8 - fp32 (per seed); bytes accounting closes
  4) device model: cycles == MACs / eta; latency == cycles / clock;
     duty == latency / interval; required clock == MACs / eta / budget;
     energy == active_mW * latency; battery days == cell J / daily J
  5) console log <-> JSON cross-check for the headline numbers

Usage: python scripts/_audit_edge_deploy.py [json] [log]
"""
import glob
import io
import os
import re
import sys

import numpy as np

PARAMS = {"light": 48517, "se": 70525, "wide": 1068405, "resnet": 1301669}
fails = []


def chk(name, got, want, tol):
    ok = abs(got - want) <= tol
    if not ok:
        fails.append("%s: got %.6f want %.6f" % (name, got, want))
    print("  [%s] %s: %.6f vs %.6f" % ("OK" if ok else "XX", name, got, want))


json_path = sys.argv[1] if len(sys.argv) > 1 else sorted(
    glob.glob("results/edge_deploy_*.json"))[-1]
print("== file:", json_path)
import json  # noqa: E402
J = json.load(open(json_path, encoding="utf-8"))

print("== P1 profile ==")
for bb, p in J["profile"].items():
    chk("profile %s MACs == sum(layers)" % bb,
        p["macs"], sum(r["macs"] for r in p["layers"]), 0)
    chk("profile %s params == known" % bb, p["params"], PARAMS[bb], 0)
    chk("profile %s fp32 bytes == params*4" % bb,
        p["fp32_weight_bytes"], p["params"] * 4, 0)
    chk("profile %s peak act <= total act" % bb,
        float(p["peak_act_bytes_int8"] <= p["act_bytes_total_int8"]), 1.0, 0)

print("== P2 latency ==")
for bb, lat in J["latency"].items():
    if bb == "matrix_optimized":
        continue
    for mode in ("eager", "optimized"):
        for k, v in lat[mode].items():
            if v <= 0:
                fails.append("%s %s %s non-positive" % (bb, mode, k))
print("  [OK] all latencies positive" if not any(
    "non-positive" in f for f in fails) else "  [XX] negative latency")
mx = J["latency"]["matrix_optimized"]
for k, cell in mx["cells"].items():
    b = int(re.search(r"batch(\d+)", k).group(1))
    chk("matrix %s per_beat == total/batch" % k,
        cell["per_beat_ms"], cell["total_ms"] / b, 1e-3)
    chk("matrix %s throughput == batch/total" % k,
        cell["throughput_beat_per_s"],
        b / (cell["total_ms"] / 1e3), 0.5)

print("== P3 int8 ==")
for sk, e in J["per_seed"].items():
    i8 = e["int8"]["both"]
    chk("%s delta_acc == int8 - fp32" % sk,
        i8["delta_acc_pp"],
        (i8["acc"] - e["fp32"]["acc"]) * 100, 0.002)
    chk("%s delta_mF1 == int8 - fp32" % sk,
        i8["delta_macro_f1_pp"],
        (i8["macro_f1"] - e["fp32"]["macro_f1"]) * 100, 0.002)
    by = i8["bytes"]
    chk("%s bytes total closes" % sk, by["total_bytes"],
        by["int8_weight_bytes"] + by["scale_bias_bytes"]
        + by["float_params_and_buffers_bytes"], 0)
    chk("%s int8 weight params < fp32 params" % sk,
        float(by["int8_weight_params"] < PARAMS[J["config"]["backbone"]]),
        1.0, 0)
sm = J["summary"]
n = sm["n_seeds"]
for key, src in (("int8_acc", "acc"), ("int8_macro_f1", "macro_f1"),
                 ("int8_acc_delta_pp", "delta_acc_pp"),
                 ("int8_macro_f1_delta_pp", "delta_macro_f1_pp")):
    if key in sm:
        vals = [J["per_seed"]["seed%d" % s]["int8"]["both"][src]
                for s in range(n)]
        chk("summary %s mean" % key, sm[key]["mean"],
            float(np.mean(vals)), 5e-5)
        if n > 1:
            chk("summary %s sd" % key, sm[key]["sd"],
                float(np.std(vals, ddof=1)), 5e-5)

print("== P5 device model ==")
for bb in ("se", "light"):
    dm = J["device_model"][bb]
    macs = J["profile"][bb]["macs"]
    chk("device %s macs source" % bb, dm["macs"], macs, 0)
    for t in dm["tiers"]:
        eta_lo = min(t["eta_macs_per_cycle"])
        eta_hi = max(t["eta_macs_per_cycle"])
        f_hz = t["clock_MHz"] * 1e6
        chk("device %s %s cycles_worst" % (bb, t["tier"][:12]),
            t["cycles_worst"], macs / eta_lo, 1.0)
        chk("device %s %s lat_worst" % (bb, t["tier"][:12]),
            t["latency_ms_worst"], macs / eta_lo / f_hz * 1e3, 1e-3)
        chk("device %s %s lat_best" % (bb, t["tier"][:12]),
            t["latency_ms_best"], macs / eta_hi / f_hz * 1e3, 1e-3)
        for iv in (1.0, 0.6):
            chk("device %s duty@%.1fs" % (t["tier"][:12], iv),
                t["duty_cycle_worst"]["%.1fs" % iv],
                t["latency_ms_worst"] / 1e3 / iv, 1e-6)
        chk("device %s f_needed_worst" % t["tier"][:12],
            t["clock_MHz_for_100ms_worst"], macs / eta_lo / 0.1 / 1e6, 0.01)
        chk("device %s energy_worst" % t["tier"][:12],
            t["energy_uJ_worst"], t["active_mW_assumed"]
            * t["latency_ms_worst"], 0.05)
        cell_j = 220e-3 * 3.0 * 3600.0
        day_j = t["energy_uJ_worst"] * 1e-6 * 86400.0
        chk("device %s battery days" % t["tier"][:12],
            t["coin_cell_days_worst"], cell_j / day_j, 0.5)

print("== P4 pipeline ==")
for dn, pc in J["pipeline_cost"].items():
    rb = pc["rebuild"]
    chk("pipeline %s rebuild total == gen+score+select" % dn,
        rb["total_ms"], rb["generate_ms"] + rb["score_ms"]
        + rb["select_ms"], 5e-3)
    chk("pipeline %s ms/cand total" % dn, rb["ms_per_candidate_total"],
        rb["total_ms"] / rb["candidates"], 1e-5)
    chk("pipeline %s ms/cand gen" % dn, rb["ms_per_candidate_generate"],
        rb["generate_ms"] / rb["candidates"], 1e-5)
    chk("pipeline %s ms/cand score" % dn, rb["ms_per_candidate_score"],
        rb["score_ms"] / rb["candidates"], 1e-5)
    g = pc["gate_scoring"]
    # fixed kernel-launch overhead -> per-candidate cost must fall as n grows
    for tag in ("generate", "score"):
        a = g["ms_per_candidate_" + tag]
        b_ = rb["ms_per_candidate_" + tag]
        chk("pipeline %s %s per-cand consistent within 2x" % (dn, tag),
            float(max(a, b_) <= 2 * min(a, b_)), 1.0, 0)
    # marginal cost implied by the two sizes (paper reports this)
    marg = ((rb["score_ms"] - g["score_ms"])
            / (rb["candidates"] - g["candidates"]))
    chk("pipeline %s marginal score ms/cand positive" % dn,
        float(marg > 0), 1.0, 0)
    chk("pipeline %s gate cheaper than rebuild" % dn,
        float(g["total_ms"] < rb["total_ms"]), 1.0, 0)
    print("    %s marginal score cost = %.5f ms/candidate, "
          "implied fixed cost = %.2f ms"
          % (dn, marg, g["score_ms"] - marg * g["candidates"]))

print()
if len(sys.argv) > 2:
    log_path = sys.argv[2]
    raw = open(log_path, "rb").read()
    txt = None
    for enc in ("utf-16", "utf-8", "gbk"):
        try:
            txt = raw.decode(enc)
            break
        except Exception:
            continue
    txt = txt.replace("\x00", "")
    print("== log <-> JSON cross-check:", os.path.basename(log_path))
    n_ok = 0
    for bb, p in J["profile"].items():
        m = re.search(r"%s\s+MACs=\s*([\d.]+)M" % bb, txt)
        if m:
            got = float(m.group(1))
            chk("log %s MACs" % bb, got, round(p["macs"] / 1e6, 3), 0.001)
            n_ok += 1
    accs = re.findall(r"fp32: acc=([\d.]+) mF1=([\d.]+)", txt)
    i8s = re.findall(r"int8\(both\): acc=([\d.]+) mF1=([\d.]+)\s+"
                     r"Δacc=([+-][\d.]+)pp ΔmF1=([+-][\d.]+)pp\s+bytes=(\d+)",
                     txt)
    for seed in range(J["summary"]["n_seeds"]):
        e = J["per_seed"]["seed%d" % seed]
        chk("log seed%d fp32 acc" % seed, float(accs[seed][0]),
            round(e["fp32"]["acc"] * 100, 2), 0.005)
        chk("log seed%d fp32 mF1" % seed, float(accs[seed][1]),
            round(e["fp32"]["macro_f1"] * 100, 2), 0.005)
        i8 = e["int8"]["both"]
        chk("log seed%d int8 acc" % seed, float(i8s[seed][0]),
            round(i8["acc"] * 100, 2), 0.005)
        chk("log seed%d int8 Δacc" % seed, float(i8s[seed][2]),
            i8["delta_acc_pp"], 0.005)
        chk("log seed%d int8 bytes" % seed, float(i8s[seed][4]),
            i8["bytes"]["total_bytes"], 0)
    chk("log cells cross-checked", float(n_ok > 0), 1.0, 0)

print()
if fails:
    print("===== %d 处不一致 =====" % len(fails))
    for f in fails:
        print("  XX", f)
    sys.exit(1)
print("===== 边缘部署 JSON 内部一致性与派生量全部通过 =====")
