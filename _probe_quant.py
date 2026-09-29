# -*- coding: utf-8 -*-
"""Probe: is the se backbone quantizable with torch static PTQ on this host?

Tries, in order:
  1) FX graph-mode PTQ (quantize_fx) with automatic fusion;
  2) eager PTQ after manual conv-bn-relu fusion;
  3) weight-only int8 emulation (fallback probe only).
Reports which path works and the serialized bytes of the converted model.
"""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "rmb", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "run_mitbih_aug.py"))
rmb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rmb)

import torch.ao.quantization as _tq  # noqa: E402

print("supported engines:", torch.backends.quantized.supported_engines)
for _e in list(torch.backends.quantized.supported_engines):
    try:
        torch.backends.quantized.engine = _e
        _qc = _tq.get_default_qconfig(_e)
        print("  engine", _e, "OK qconfig:", _qc)
    except Exception as _exc:
        print("  engine", _e, "unusable:", str(_exc)[:80])
ENGINE = next(e for e in ("x86", "fbgemm", "onednn", "qnnpack")
              if e in torch.backends.quantized.supported_engines)
torch.backends.quantized.engine = ENGINE
print("picked engine:", ENGINE)
x = torch.randn(16, 1, rmb.LEN)


def sd_bytes(m):
    total = 0
    for t in m.state_dict().values():
        total += t.numel() * t.element_size()
    return total


def try_fx():
    from torch.ao.quantization.quantize_fx import (prepare_fx, convert_fx)
    import torch.ao.quantization as tq
    m = rmb.build_backbone("se", 5).eval()
    qc = tq.get_default_qconfig(ENGINE)
    gm = torch.fx.symbolic_trace(m)
    pm = prepare_fx(gm, {"": qc}, example_inputs=(x,))
    with torch.no_grad():
        for _ in range(3):
            pm(x)
    qm = convert_fx(pm)
    with torch.no_grad():
        y = qm(x)
    return ("fx", sd_bytes(qm), float(y.float().sum()), type(qm).__name__)


def fuse_block(seq):
    """Fuse conv->bn->relu patterns inside a Sequential, in place."""
    names = [n for n, _ in seq.named_children()]
    kinds = [type(getattr(seq, n)).__name__ for n in names]
    for i in range(len(names) - 2):
        if (kinds[i] == "Conv1d" and kinds[i + 1] == "BatchNorm1d"
                and kinds[i + 2] == "ReLU"):
            try:
                torch.ao.quantization.fuse_modules(
                    seq, [names[i], names[i + 1], names[i + 2]],
                    inplace=True)
                return True
            except Exception as exc:  # pragma: no cover
                print("   fuse skip:", exc)
    return False


def try_eager():
    import torch.ao.quantization as tq
    m = rmb.build_backbone("se", 5).eval()
    n_fused = 0
    for mod in m.modules():
        if isinstance(mod, torch.nn.Sequential):
            n_fused += int(fuse_block(mod))
    print("   fused blocks:", n_fused)
    m.qconfig = tq.get_default_qconfig(ENGINE)
    tq.prepare(m, inplace=True)
    with torch.no_grad():
        for _ in range(3):
            m(x)
    tq.convert(m, inplace=True)
    with torch.no_grad():
        y = m(x)
    return ("eager", sd_bytes(m), float(y.float().sum()), type(m).__name__)


for fn in (try_fx, try_eager):
    print("==", fn.__name__)
    try:
        print("   OK ->", fn())
    except Exception as exc:
        print("   FAIL:", type(exc).__name__, str(exc)[:300])

# jit trace + optimize_for_inference sanity
print("== jit")
try:
    m = rmb.build_backbone("se", 5).eval()
    tr = torch.jit.trace(m, torch.randn(1, 1, rmb.LEN))
    opt = torch.jit.optimize_for_inference(tr)
    with torch.no_grad():
        y = opt(torch.randn(1, 1, rmb.LEN))
    print("   OK out", tuple(y.shape))
except Exception as exc:
    print("   FAIL:", type(exc).__name__, str(exc)[:300])
print("fp32 state_dict bytes:", sd_bytes(rmb.build_backbone("se", 5)))

