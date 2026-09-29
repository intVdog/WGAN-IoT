# -*- coding: utf-8 -*-
"""Deepened edge-deployment evaluation for the MIT-BIH backbones.

Provenance is labelled per section (this matters for the paper):

  P1 profile      MEASURED   per-layer MACs, params, weight bytes, peak
                             activation bytes, analytic int8 weight bytes.
  P2 latency      MEASURED   this host, 1..4 CPU threads, batch 1/8/32,
                             eager vs jit-optimized (BN folded by the graph
                             optimizer).  PC proxy, NOT device numbers.
  P3 int8_sim     MEASURED   full-int8 simulation: per-channel symmetric
                             int8 weights + per-tensor asymmetric int8
                             activations calibrated on the training split.
                             Accuracy delta is a real test-set measurement;
                             the serialized int8 bytes are counted from the
                             quantized tensors, not estimated as params x 1.
                             NOTE: this host's torch exposes only the oneDNN
                             quantized engine and has no quantized conv1d
                             kernel, so int8 SIMULATION replaces int8
                             execution; no int8 latency is claimed.
  P4 pipeline     MEASURED   per-beat inference, gate scoring of 4000
                             candidates, full rebuild of 8000 candidates
                             (generate + score + top-90% + top-up).
                             Generator/critic weights are untrained: cost
                             depends on architecture and tensor shapes only.
  P5 device_model ANALYTIC   cycles = MACs / eta, latency = cycles / f.
                             eta (int8 MACs per cycle) and active power are
                             ASSUMPTIONS taken as ranges; results are
                             reported as ranges plus a corner analysis and
                             the minimum clock needed for a 100 ms budget.

P0 discipline: auto-named output, refuses to overwrite, env fingerprint.

Usage:
    python scripts/run_edge_deploy.py [--cap 200] [--backbone se]
        [--seed 0] [--device cuda] [--repeats 200] [--quick] [--force]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from gan_gat.evaluate import classification_metrics, save_json  # noqa: E402
from gan_gat.utils import seed_everything  # noqa: E402

_THIS = os.path.abspath(__file__)
_spec = importlib.util.spec_from_file_location(
    "rmb", os.path.join(os.path.dirname(_THIS), "run_mitbih_aug.py"))
rmb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rmb)

BACKBONES = ["light", "se", "wide", "resnet"]
PROFILE_BACKBONES = ["light", "se", "wide", "resnet"]
SIM_BACKBONES = ["light", "se", "wide"]     # LightCNN1D family (foldable BN)

# ---------------------------------------------------------------- helpers --


def sha256_of(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def bench(fn, reps: int, warmup: int = 5) -> float:
    """Seconds per call, best-of-3 medians of the mean over ``reps``."""
    for _ in range(warmup):
        fn()
    means = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        means.append((time.perf_counter() - t0) / reps)
    return float(np.median(means))


def time_cuda(fn, reps: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


# ------------------------------------------------------------- P1 profile --
def profile_model(model: nn.Module, x: torch.Tensor) -> dict:
    """MACs / params / weight bytes / peak activation bytes, per layer."""
    rows: list[dict] = []
    handles = []

    def hook(mod, inp, out):
        if isinstance(mod, nn.Conv1d):
            macs = (out.shape[1] * out.shape[2]
                    * (mod.in_channels // mod.groups) * mod.kernel_size[0])
            kind = "conv1d"
        elif isinstance(mod, nn.Linear):
            macs = mod.in_features * mod.out_features
            kind = "linear"
        else:
            return
        w_par = mod.weight.numel()
        b_par = mod.bias.numel() if mod.bias is not None else 0
        n_scale = mod.out_channels if kind == "conv1d" else 1
        rows.append({
            "kind": kind,
            "out_shape": list(out.shape),
            "macs": int(macs),
            "params": int(sum(p.numel() for p in mod.parameters())),
            "fp32_weight_bytes": int((w_par + b_par) * 4),
            # int8 weights + fp32 bias + fp32 per-channel scale
            "int8_weight_bytes": int(w_par + 4 * (b_par + n_scale)),
            "act_bytes_fp32": int(out.numel() * 4),
            "act_bytes_int8": int(out.numel()),
        })

    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            handles.append(m.register_forward_hook(hook))
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()

    total_params = sum(p.numel() for p in model.parameters())
    peak_fp32 = max(r["act_bytes_fp32"] for r in rows)
    peak_i8 = max(r["act_bytes_int8"] for r in rows)
    return {
        "params": int(total_params),
        "macs": int(sum(r["macs"] for r in rows)),
        "fp32_weight_bytes": int(total_params * 4),
        "int8_weight_bytes_analytic": int(sum(r["int8_weight_bytes"]
                                              for r in rows)),
        "peak_act_bytes_fp32": int(peak_fp32),
        "peak_act_bytes_int8": int(peak_i8),
        "act_bytes_total_int8": int(sum(r["act_bytes_int8"] for r in rows)),
        "layers": rows,
    }


# ------------------------------------------------------------ P3 int8 sim --
class QAct(nn.Module):
    """Per-tensor asymmetric int8 fake-quant for activations."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("lo", torch.zeros(1))
        self.register_buffer("hi", torch.zeros(1))
        self.inited = False
        self.ready = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.ready:                       # calibration pass
            lo, hi = float(x.min()), float(x.max())
            if not self.inited:
                self.lo.fill_(lo)
                self.hi.fill_(hi)
                self.inited = True
            else:
                self.lo.fill_(min(float(self.lo), lo))
                self.hi.fill_(max(float(self.hi), hi))
            return x
        lo, hi = float(self.lo), float(self.hi)
        if hi - lo < 1e-12:
            return x
        scale = (hi - lo) / 255.0
        zp = round(-lo / scale)
        q = torch.clamp(torch.round(x / scale) + zp, 0.0, 255.0)
        return (q - zp) * scale


class QConv1d(nn.Module):
    """int8 weights (per output channel, symmetric) + int8 in/out acts."""

    def __init__(self, conv: nn.Conv1d) -> None:
        super().__init__()
        w = conv.weight.data
        scale = (w.abs().amax(dim=(1, 2), keepdim=True)
                 / 127.0).clamp(min=1e-12)
        self.register_buffer("w_int8",
                             torch.round(w / scale).clamp(-127, 127)
                             .to(torch.int8))
        self.register_buffer("w_scale", scale)
        self.register_buffer("bias",
                             conv.bias.data.clone() if conv.bias is not None
                             else torch.zeros(conv.out_channels))
        self.stride, self.padding = conv.stride, conv.padding
        self.dilation, self.groups = conv.dilation, conv.groups
        self.in_q, self.out_q = QAct(), QAct()
        self.w_float = None          # set to bypass weight quantization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = (self.w_float if self.w_float is not None
             else self.w_int8.float() * self.w_scale)
        y = F.conv1d(self.in_q(x), w, self.bias, self.stride, self.padding,
                     self.dilation, self.groups)
        return self.out_q(y)


class QLinear(nn.Module):
    def __init__(self, lin: nn.Linear) -> None:
        super().__init__()
        w = lin.weight.data
        scale = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-12)
        self.register_buffer("w_int8",
                             torch.round(w / scale).clamp(-127, 127)
                             .to(torch.int8))
        self.register_buffer("w_scale", scale)
        self.register_buffer("bias",
                             lin.bias.data.clone() if lin.bias is not None
                             else torch.zeros(lin.out_features))
        self.in_q, self.out_q = QAct(), QAct()
        self.w_float = None          # set to bypass weight quantization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = (self.w_float if self.w_float is not None
             else self.w_int8.float() * self.w_scale)
        return self.out_q(F.linear(self.in_q(x), w, self.bias))


class SimQuant(nn.Module):
    """Wrapper that quantizes the input activation of the whole network."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net
        self.in_q = QAct()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.in_q(x))


def fold_bn_into_conv(model: nn.Module) -> int:
    """Fold Conv1d -> BatchNorm1d pairs in place; returns pairs folded."""
    folded = 0
    for seq in [m for m in model.modules() if isinstance(m, nn.Sequential)]:
        names = [n for n, _ in seq.named_children()]
        for i in range(len(names) - 1):
            a, b = getattr(seq, names[i]), getattr(seq, names[i + 1])
            if isinstance(a, nn.Conv1d) and isinstance(b, nn.BatchNorm1d):
                with torch.no_grad():
                    # conv(x) = W x + b_conv ; bn(y) = (y - mu)/sqrt(var+eps)
                    # * gamma + beta  ->  W' = W * s, b' = b_conv * s + beta
                    # - mu * s   with s = gamma / sqrt(var + eps)
                    s = b.weight / torch.sqrt(b.running_var + b.eps)
                    w_new = a.weight.data * s[:, None, None]
                    b_new = b.bias - b.running_mean * s
                    if a.bias is not None:
                        b_new = b_new + a.bias.data * s
                    a.weight.data = w_new
                    if a.bias is None:
                        a.bias = nn.Parameter(torch.zeros_like(s))
                    a.bias.data = b_new
                    b.weight.data.fill_(1.0)
                    b.bias.data.zero_()
                    b.running_mean.zero_()
                    b.running_var.fill_(1.0)
                setattr(seq, names[i + 1], nn.Identity())
                folded += 1
    return folded


def to_simquant(model: nn.Module) -> SimQuant:
    """Replace Conv1d/Linear by int8-simulating wrappers (in place)."""
    def repl(mod: nn.Module) -> None:
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.Conv1d):
                setattr(mod, name, QConv1d(child))
            elif isinstance(child, nn.Linear):
                setattr(mod, name, QLinear(child))
            else:
                repl(child)
    repl(model)
    return SimQuant(model)


def set_ready(model: nn.Module, ready: bool) -> None:
    for m in model.modules():
        if isinstance(m, QAct):
            m.ready = ready


def simquant_bytes(model: nn.Module) -> dict:
    """Serialized bytes of the simulated int8 model, counted per tensor."""
    w_i8 = w_f32 = other_f32 = 0
    n_w_int8 = 0
    for m in model.modules():
        if isinstance(m, (QConv1d, QLinear)):
            w_i8 += m.w_int8.numel() * m.w_int8.element_size()
            n_w_int8 += m.w_int8.numel()
            w_f32 += m.w_scale.numel() * 4 + m.bias.numel() * 4
    for name, p in model.named_parameters():
        other_f32 += p.numel() * 4
    for name, b in model.named_buffers():
        if b.dtype in (torch.float32, torch.float64):
            other_f32 += b.numel() * 4
    return {
        "int8_weight_bytes": int(w_i8),
        "int8_weight_params": int(n_w_int8),
        "scale_bias_bytes": int(w_f32),
        "float_params_and_buffers_bytes": int(other_f32),
        "total_bytes": int(w_i8 + w_f32 + other_f32),
    }


def quant_act_ranges(model: nn.Module) -> list[dict]:
    out = []
    for name, m in model.named_modules():
        if isinstance(m, QAct) and m.inited:
            out.append({"module": name, "lo": round(float(m.lo), 4),
                        "hi": round(float(m.hi), 4)})
    return out


def parse_seed_ids(s: str) -> list[int]:
    out: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return sorted(set(out))


def build_simquant(model: nn.Module, cal_x: np.ndarray, quant_w: bool = True,
                   quant_a: bool = True):
    """Fold BN, wrap in int8 simulators, calibrate, return (model, pairs)."""
    src = copy.deepcopy(model).to("cpu").eval()
    n_folded = fold_bn_into_conv(src)
    sq = to_simquant(src)
    sq.eval()
    if not quant_w:
        for m in sq.modules():
            if isinstance(m, (QConv1d, QLinear)):
                m.w_float = m.w_int8.float() * m.w_scale
    if not quant_a:
        for m in sq.modules():
            if isinstance(m, (QConv1d, QLinear)):
                m.in_q = nn.Identity()
                m.out_q = nn.Identity()
        sq.in_q = nn.Identity()
    cal = torch.as_tensor(cal_x, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        for s in range(0, len(cal), 256):
            sq(cal[s:s + 256])
    set_ready(sq, True)
    return sq, n_folded


def simquant_predict(sq: nn.Module, x: np.ndarray, bs: int = 1024):
    pred = []
    with torch.no_grad():
        xt = torch.as_tensor(x, dtype=torch.float32).unsqueeze(1)
        for s in range(0, len(xt), bs):
            pred.append(sq(xt[s:s + bs]).argmax(1).numpy())
    return np.concatenate(pred)


def metrics_of(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    m = classification_metrics(y_true, y_pred, n_classes=5)
    return {"acc": round(m["accuracy"], 4),
            "macro_f1": round(m["f1"], 4),
            "macro_recall": round(m["recall"], 4),
            "per_class_f1": {k: round(v, 4) for k, v in
                             rmb.per_class_f1(y_true, y_pred).items()}}


# --------------------------------------------------------- P5 device model --
DEVICE_TIERS = [
    {"name": "可穿戴低功耗 MCU (Cortex-M4F 级)", "clock_MHz": 48.0,
     "eta_macs_per_cycle": [0.5, 1.0, 2.0], "active_mW": 1.5},
    {"name": "主流 MCU (Cortex-M7 级)", "clock_MHz": 216.0,
     "eta_macs_per_cycle": [1.0, 2.0, 4.0], "active_mW": 20.0},
    {"name": "应用处理器核 (Cortex-A55 级)", "clock_MHz": 1800.0,
     "eta_macs_per_cycle": [4.0, 8.0, 16.0], "active_mW": 200.0},
]
BEAT_INTERVALS_S = [1.0, 0.6]              # 60 bpm and 100 bpm
BUDGET_MS = 100.0
COIN_CELL_J = 220e-3 * 3.0 * 3600.0        # CR2032 220 mAh @ 3.0 V


def device_model(macs: int) -> dict:
    tiers = []
    for t in DEVICE_TIERS:
        f_hz = t["clock_MHz"] * 1e6
        lo_eta, hi_eta = min(t["eta_macs_per_cycle"]), max(
            t["eta_macs_per_cycle"])
        cyc = {"worst": macs / lo_eta, "best": macs / hi_eta}
        lat = {k: v / f_hz * 1e3 for k, v in cyc.items()}
        duty = {k: {f"{iv}s": v / 1e3 / iv for iv in BEAT_INTERVALS_S}
                for k, v in lat.items()}
        f_need_MHz = {k: macs / e / (BUDGET_MS / 1e3) / 1e6
                      for k, e in (("worst", lo_eta), ("best", hi_eta))}
        energy_uJ = {k: t["active_mW"] * v for k, v in lat.items()}
        day_j = {k: v * 1e-6 * (86400.0 / BEAT_INTERVALS_S[0])
                 for k, v in energy_uJ.items()}
        tiers.append({
            "tier": t["name"],
            "clock_MHz": t["clock_MHz"],
            "eta_macs_per_cycle": t["eta_macs_per_cycle"],
            "active_mW_assumed": t["active_mW"],
            "cycles_worst": int(cyc["worst"]),
            "cycles_best": int(cyc["best"]),
            "latency_ms_worst": round(lat["worst"], 3),
            "latency_ms_best": round(lat["best"], 3),
            "duty_cycle_worst": {k: round(v, 6) for k, v in
                                 duty["worst"].items()},
            "duty_cycle_best": {k: round(v, 6) for k, v in
                                duty["best"].items()},
            "clock_MHz_for_100ms_worst": round(f_need_MHz["worst"], 2),
            "clock_MHz_for_100ms_best": round(f_need_MHz["best"], 2),
            "energy_uJ_worst": round(energy_uJ["worst"], 3),
            "energy_uJ_best": round(energy_uJ["best"], 3),
            "coin_cell_days_worst": round(COIN_CELL_J / day_j["worst"], 1),
            "coin_cell_days_best": round(COIN_CELL_J / day_j["best"], 1),
        })
    return {"macs": int(macs), "tiers": tiers,
            "assumptions": {
                "eta": "int8 MACs per cycle, taken as a range per tier",
                "active_mW": "assumed active power at the tier clock",
                "beat_interval_s": BEAT_INTERVALS_S,
                "budget_ms": BUDGET_MS,
                "battery": "CR2032 220 mAh at 3.0 V",
                "kind": "ANALYTIC MODEL - not a device measurement"}}


# ------------------------------------------------------------------- main --
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--backbone", default="se", choices=BACKBONES)
    ap.add_argument("--seeds", default="0-2",
                    help="seed ids: comma list or range, e.g. 0-2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--threads", default="1,2,4")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--quick", action="store_true",
                    help="skip WGAN training: train the no-aug model only")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    threads = [int(t) for t in args.threads.split(",")]
    batches = [int(b) for b in args.batches.split(",")]
    device = torch.device(args.device)
    date = time.strftime("%Y%m%d")
    _ids = parse_seed_ids(args.seeds)
    _tag = (f"{min(_ids)}-{max(_ids)}" if len(_ids) > 1 else str(_ids[0]))
    out_path = args.out or (f"results/edge_deploy_cap{args.cap}_"
                            f"{args.backbone}_seeds{_tag}_{date}.json")
    if os.path.exists(out_path) and not args.force:
        sys.exit(f"[P0] refusing to overwrite existing {out_path}")

    env_fp = {
        "torch": torch.__version__, "numpy": np.__version__,
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "gpu": (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "cpu"),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "script_sha256": sha256_of(_THIS),
        "rmb_sha256": sha256_of(os.path.join(os.path.dirname(_THIS),
                                             "run_mitbih_aug.py")),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "quantized_engines": list(
            torch.backends.quantized.supported_engines),
        "int8_path": "simulation (no quantized conv1d kernel on this host)",
    }
    res: dict = {"config": {
        "cap": args.cap, "backbone": args.backbone, "seeds": args.seeds,
        "threads": threads, "batches": batches, "repeats": args.repeats,
        "quick": args.quick, "classes": rmb.CLASSES,
        "gan_epochs": rmb.GAN_EPOCHS, "env": env_fp,
        "provenance": {
            "P1_profile": "MEASURED (analytic MACs from tensor shapes)",
            "P2_latency": "MEASURED on this host, PC proxy",
            "P3_int8_sim": "MEASURED accuracy under int8 simulation",
            "P4_pipeline": "MEASURED, untrained gen/critic weights",
            "P5_device_model": "ANALYTIC MODEL, assumption-based",
        }}, "profile": {}, "latency": {}, "int8_sim": {},
        "pipeline_cost": {}, "device_model": {}}

    # ---------------------------------------------------------- P1 profile
    print("== P1 profile ==")
    x1 = torch.randn(1, 1, rmb.LEN)
    for bb in PROFILE_BACKBONES:
        m = rmb.build_backbone(bb, 5).eval()
        p = profile_model(m, x1)
        top = sorted(p["layers"], key=lambda r: -r["macs"])[:3]
        p["top_layers_by_macs"] = [
            {"kind": r["kind"], "out_shape": r["out_shape"],
             "macs": r["macs"]} for r in top]
        res["profile"][bb] = p
        print(f"  {bb:<7} MACs={p['macs']/1e6:8.3f}M  params="
              f"{p['params']/1e3:6.1f}K  peak_act(int8)="
              f"{p['peak_act_bytes_int8']/1024:5.1f}KiB")

    # ---------------------------------------------------------- P2 latency
    print("== P2 latency ==")
    for bb in PROFILE_BACKBONES:
        m = rmb.build_backbone(bb, 5).eval()
        tr = torch.jit.trace(m, x1)
        opt = torch.jit.optimize_for_inference(tr)
        entry = {"eager": {}, "optimized": {}}
        for th in threads:
            torch.set_num_threads(th)
            with torch.inference_mode():
                entry["eager"][f"threads{th}_batch1_ms"] = round(
                    bench(lambda: m(x1), args.repeats) * 1e3, 3)
                entry["optimized"][f"threads{th}_batch1_ms"] = round(
                    bench(lambda: opt(x1), args.repeats) * 1e3, 3)
        res["latency"][bb] = entry

    bb = args.backbone
    m_bb = rmb.build_backbone(bb, 5).eval()
    x1 = torch.randn(1, 1, rmb.LEN)
    opt_bb = torch.jit.optimize_for_inference(torch.jit.trace(m_bb, x1))
    matrix = {}
    for th in threads:
        torch.set_num_threads(th)
        for b in batches:
            xb = torch.randn(b, 1, rmb.LEN)
            with torch.inference_mode():
                t = bench(lambda: opt_bb(xb), max(20, args.repeats // b))
            matrix[f"threads{th}_batch{b}"] = {
                "total_ms": round(t * 1e3, 3),
                "per_beat_ms": round(t * 1e3 / b, 4),
                "throughput_beat_per_s": round(b / t, 1)}
    res["latency"]["matrix_optimized"] = {"backbone": bb, "cells": matrix}
    torch.set_num_threads(threads[0])
    print(f"  {bb} optimized: " + "  ".join(
        f"t{th}/b1={res['latency'][bb]['optimized'][f'threads{th}_batch1_ms']}ms"
        for th in threads))

    # ------------------------------------------------------ data + models
    x_tr_all, y_tr_all, x_te, y_te = rmb.load_data()
    seed_ids = parse_seed_ids(args.seeds)
    per_seed: dict = {}
    for seed in seed_ids:
        print(f"-- seed {seed} --")
        seed_everything(seed)
        rng = np.random.RandomState(1000 + seed)
        tr_i = []
        for c in range(5):
            idx = np.where(y_tr_all == c)[0]
            tr_i.append(idx[rng.permutation(len(idx))[
                :min(args.cap, len(idx))]])
        tr_i = np.concatenate(tr_i)
        xs, ys = x_tr_all[tr_i], y_tr_all[tr_i]
        print("   scarce train per class:",
              dict(zip(rmb.CLASSES, np.bincount(ys, minlength=5).tolist())))

        pools, gate_stats = {}, {}
        t0 = time.perf_counter()
        if not args.quick:
            for c in range(5):
                xc = xs[ys == c]
                w = rmb.WGAN1D(device)
                _, cc = w.train_on(xc, seed=seed * 1000 + c)
                cand = w.generate(4000, seed=seed * 1000 + c)
                sc = w.score(cand)
                kept = cand[sc >= np.quantile(sc, 0.10)]
                pools[c] = kept
                gate_stats[rmb.CLASSES[c]] = {"C": round(float(cc), 3),
                                              "kept": int(len(kept))}
                print(f"   [wgan] {rmb.CLASSES[c]}: kept {len(kept)} "
                      f"(C {cc:.2f})")
        pool_s = round(time.perf_counter() - t0, 1)

        # mirror run_mitbih_aug's clean arm: scarce real + the SAME number of
        # gated synthetic candidates per class (n_add = 1.0 * n_real), i.e.
        # 200 + 200 per class, NOT the whole 3600-candidate gated pool.
        n_real = np.bincount(ys, minlength=5)
        sx, sy = [], []
        if not args.quick:
            for c in range(5):
                n_add = int(min(int(n_real[c]), len(pools[c])))
                sx.append(pools[c][:n_add])
                sy.append(np.full(n_add, c, dtype=np.int64))
        x_tr_model = np.concatenate([xs] + sx)
        y_tr_model = np.concatenate([ys] + sy)
        n_synth = int(len(x_tr_model) - len(xs))
        print(f"   training split: {len(x_tr_model)} = {len(xs)} real + "
              f"{n_synth} synthetic "
              f"({'no-aug' if args.quick else 'clean-aug, 200/class pool'})")

        seed_everything(seed * 1000)
        model = rmb.build_backbone(bb, 5).to(device)
        rmb.train_cnn(model, x_tr_model, y_tr_model, device, seed)
        pred_fp32 = rmb.predict(model, x_te, device)
        entry = {"train_size": int(len(x_tr_model)),
                 "n_real": int(len(xs)), "n_synth": n_synth,
                 "pool_seconds": pool_s, "gate": gate_stats,
                 "fp32": metrics_of(y_te, pred_fp32)}
        print(f"   fp32: acc={entry['fp32']['acc']*100:.2f} "
              f"mF1={entry['fp32']['macro_f1']*100:.2f}")

        # ------------------------------------------------------ P3 int8 sim
        if bb in SIM_BACKBONES:
            variants = {}
            for tag, qw, qa in (("both", True, True),
                                ("weights_only", True, False),
                                ("acts_only", False, True)):
                sq, n_folded = build_simquant(model, x_tr_model, qw, qa)
                met = metrics_of(y_te, simquant_predict(sq, x_te))
                variants[tag] = {
                    "acc": met["acc"], "macro_f1": met["macro_f1"],
                    "macro_recall": met["macro_recall"],
                    "per_class_f1": met["per_class_f1"],
                    "delta_acc_pp": round((met["acc"]
                                           - entry["fp32"]["acc"]) * 100, 3),
                    "delta_macro_f1_pp": round(
                        (met["macro_f1"]
                         - entry["fp32"]["macro_f1"]) * 100, 3)}
                if tag == "both":
                    variants[tag]["bytes"] = simquant_bytes(sq)
                    variants[tag]["bn_pairs_folded"] = n_folded
                    variants[tag]["peak_act_bytes_int8"] = \
                        res["profile"][bb]["peak_act_bytes_int8"]
                    variants[tag]["activation_ranges_sample"] = \
                        quant_act_ranges(sq)[:6]
            variants["scheme"] = (
                "weights int8 per-output-channel symmetric; activations "
                "int8 per-tensor asymmetric, calibrated on the training "
                "split; BN folded into conv")
            entry["int8"] = variants
            b = variants["both"]
            print(f"   int8(both): acc={b['acc']*100:.2f} "
                  f"mF1={b['macro_f1']*100:.2f}  "
                  f"Δacc={b['delta_acc_pp']:+.2f}pp "
                  f"ΔmF1={b['delta_macro_f1_pp']:+.2f}pp  "
                  f"bytes={b['bytes']['total_bytes']}")
            print(f"   int8(weights_only): "
                  f"Δacc={variants['weights_only']['delta_acc_pp']:+.2f}pp  "
                  f"int8(acts_only): "
                  f"Δacc={variants['acts_only']['delta_acc_pp']:+.2f}pp")
        per_seed[f"seed{seed}"] = entry

    def agg(key: str, sub: str):
        v = np.array([per_seed[f"seed{s}"][key][sub] for s in seed_ids])
        return {"mean": round(float(v.mean()), 4),
                "sd": round(float(v.std(ddof=1)), 4) if len(v) > 1 else None}

    def agg_i8(sub: str):
        v = np.array([per_seed[f"seed{s}"]["int8"]["both"][sub]
                      for s in seed_ids])
        return {"mean": round(float(v.mean()), 4),
                "sd": round(float(v.std(ddof=1)), 4) if len(v) > 1 else None}

    res["per_seed"] = per_seed
    res["summary"] = {
        "n_seeds": len(seed_ids),
        "fp32_acc": agg("fp32", "acc"),
        "fp32_macro_f1": agg("fp32", "macro_f1"),
    }
    if bb in SIM_BACKBONES:
        res["summary"].update({
            "int8_acc": agg_i8("acc"),
            "int8_macro_f1": agg_i8("macro_f1"),
            "int8_acc_delta_pp": agg_i8("delta_acc_pp"),
            "int8_macro_f1_delta_pp": agg_i8("delta_macro_f1_pp"),
        })
        print("== summary ==")
        print("   fp32 acc   %s" % res["summary"]["fp32_acc"])
        print("   int8 acc   %s" % res["summary"]["int8_acc"])
        print("   Δacc(pp)   %s" % res["summary"]["int8_acc_delta_pp"])

    # ------------------------------------------------------ P4 pipeline cost
    print("== P4 pipeline cost ==")
    n_gate, n_rebuild, keep, bs = 4000, 8000, 0.10, 512

    def measure_path(dev, n_cand, repeats=3):
        """Primitives measured separately, then composed.

        generate-only and score-only are benchmarked in tight loops (best of
        ``repeats`` after a discarded warm-up), selection is timed once.
        Composing avoids reading a per-candidate cost off a mixed path, which
        on a laptop GPU drifts by tens of percent between calls.
        """
        gen = rmb.Gen1D().to(dev).eval()
        cri = rmb.Critic1D().to(dev)          # train mode: spectral norm on
        sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" \
            else (lambda: None)

        def gen_pass(n):
            torch.manual_seed(12345)
            t0 = time.perf_counter()
            with torch.no_grad():
                for s in range(0, n, bs):
                    z = torch.randn(min(bs, n - s), 100, device=dev)
                    gen(z)
            sync()
            return time.perf_counter() - t0

        def score_pass(x_np):
            t0 = time.perf_counter()
            with torch.no_grad():
                for s in range(0, len(x_np), bs):
                    xb = torch.as_tensor(x_np[s:s + bs],
                                         device=dev).unsqueeze(1)
                    cri(xb)
            sync()
            return time.perf_counter() - t0

        gen_pass(1024)                       # discarded warm-up
        score_pass(np.zeros((1024, rmb.LEN), dtype=np.float32))
        gen_ms = min(gen_pass(n_cand) for _ in range(repeats)) * 1e3
        torch.manual_seed(12345)
        with torch.no_grad():
            cand = np.concatenate([
                gen(torch.randn(min(bs, n_cand - s), 100,
                                device=dev)).clamp(-1, 1).squeeze(1)
                .cpu().numpy() for s in range(0, n_cand, bs)])
        score_ms = min(score_pass(cand) for _ in range(repeats)) * 1e3
        sc = []
        with torch.no_grad():
            for s in range(0, len(cand), bs):
                sc.append(cri(torch.as_tensor(cand[s:s + bs],
                                              device=dev).unsqueeze(1))
                          .flatten().cpu().numpy())
        scores = np.concatenate(sc)
        t0 = time.perf_counter()
        thr = float(np.quantile(scores, keep))
        order = np.argsort(-scores)
        kept_all = int((scores >= thr).sum())
        top200 = cand[order[:200]]
        select_ms = (time.perf_counter() - t0) * 1e3
        total = gen_ms + score_ms + select_ms
        return {
            "candidates": int(n_cand),
            "generate_ms": round(gen_ms, 3),
            "score_ms": round(score_ms, 3),
            "select_ms": round(select_ms, 4),
            "total_ms": round(total, 3),
            "ms_per_candidate_generate": round(gen_ms / n_cand, 5),
            "ms_per_candidate_score": round(score_ms / n_cand, 5),
            "ms_per_candidate_total": round(total / n_cand, 5),
            "kept_above_10pct": kept_all,
            "topup_pool_shape": list(top200.shape),
            "repeats": repeats}

    for dev_name, dev in (("gpu", device), ("cpu", torch.device("cpu"))):
        reps = 3 if dev.type == "cuda" else 2
        gate = measure_path(dev, n_gate, reps)
        rebuild = measure_path(dev, n_rebuild, reps)
        res["pipeline_cost"][dev_name] = {
            "gate_scoring": gate, "rebuild": rebuild,
            "batch": bs, "keep_fraction": keep,
            "note": "generator/critic weights are untrained; cost depends "
                    "on architecture and tensor shapes only. Critic runs in "
                    "train mode so spectral-norm power iteration is included, "
                    "as in the pipeline.",
            "weights": "untrained"}
        print(f"  {dev_name}: gate4000 {gate['total_ms']:.0f} ms | "
              f"rebuild8000 {rebuild['total_ms']:.0f} ms "
              f"(gen {rebuild['generate_ms']:.0f} + score "
              f"{rebuild['score_ms']:.0f} + select "
              f"{rebuild['select_ms']:.2f})")

    # ------------------------------------------------------ P5 device model
    print("== P5 device model ==")
    res["device_model"] = {
        "se": device_model(res["profile"]["se"]["macs"]),
        "light": device_model(res["profile"]["light"]["macs"]),
    }
    for tier in res["device_model"]["se"]["tiers"]:
        print(f"  {tier['tier'][:22]:<22} cycles {tier['cycles_worst']:>10,} "
              f"lat {tier['latency_ms_worst']:8.2f}..{tier['latency_ms_best']:.2f} ms  "
              f"duty@1s {tier['duty_cycle_worst']['1.0s']*100:.3f}%  "
              f"f_needed {tier['clock_MHz_for_100ms_worst']:.1f} MHz")

    res["config"]["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    res["config"]["elapsed_s"] = round(time.perf_counter() - t_start, 1)
    save_json(out_path, res)
    print("saved ->", out_path)


t_start = time.perf_counter()
if __name__ == "__main__":
    main()
