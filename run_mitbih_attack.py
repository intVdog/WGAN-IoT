"""Attack/defence experiments v2 (P0-P3 fixed protocol).

Protocol fixes vs v1 (2026-09-08):
  * exact-rho poisoning: the poisoned target pool has length n_add and
    contains EXACTLY rho*n_add injected source-class real beats labelled
    as target (v1 injected 4x, making "rho=0.5" actually 80% injection and
    the poisoned arm larger than the clean arm);
  * defence = full-pool rebuild: per-sample critic rejection is NOT used,
    because probes showed the critic cannot separate real cross-class beats
    from target-synthetic beats (threshold overlap ~43%; contrastive
    argmax across critics rejects 100% of good candidates since WGAN
    critics are realism discriminators, not morphology classifiers).  The
    defence therefore discards the received (possibly tampered) pool and
    rebuilds it from the clean generator + critic filter.  Attack 1 does
    not touch the generator, so this is in-threat-model.
  * arms are configurable: rho grid x source grid (P1a), optional
    adaptive injection (P2: only source beats with the highest target-
    critic score are injected), optional defence arm per rho.

Arms (per seed, cap=200, backbone 'se', 5 per-class WGAN pools shared):
  no_aug, clean
  poisoned_s{SC}r{rho}          exact-rho injection
  adapt_s{SC}r{rho}             adaptive (top-scored source beats) [P2]
  defence_s{SC}r{rho}           full rebuild after rho attack

P0 discipline: auto-named output results/<exp>_<setup>_seeds<ids>_<date>.json,
never silently overwritten (--force), full seeding + cudnn deterministic,
environment fingerprint in config.

Usage:
  python scripts/run_mitbih_attack.py --seeds 0-9 --rhos 0.5,0.8 \
      --sources 1 --defence-rho 0.8
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from gan_gat.evaluate import classification_metrics, save_json
from gan_gat.utils import seed_everything

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "rmb", os.path.join(_THIS_DIR, "run_mitbih_aug.py"))
rmb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rmb)

CLASSES = rmb.CLASSES          # N S V F Q
CAP = 200
TARGET = 3                      # F
EXP_NAME = "mitbih_attack_v2"
# Candidate counts.  The clean per-class pool draws CAND_DEFAULT candidates and
# gates the top 90%; the rebuild defence draws REBUILD_CAND_DEFAULT (= 2x) so
# that after the same 10% gate there is ample margin to top the pool back up to
# CAP synthetic candidates.  Both are exposed on the CLI so the candidate-size
# ablation (4000 / 8000 / 12000) can be run without touching this file.
CAND_DEFAULT = 4000
REBUILD_CAND_DEFAULT = 8000


def sha256_of(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def auto_out_path(seed_ids, rhos, sources, cap, defence_rho, adapt,
                  floods=None, defence_flood=None,
                  rebuild_cand=None) -> str:
    ids = (f"{min(seed_ids)}-{max(seed_ids)}" if len(seed_ids) > 1
           else str(seed_ids[0]))
    rho_s = "-".join(f"{int(r * 100):02d}" for r in rhos)
    src_s = "".join(CLASSES[s] for s in sorted(sources))
    date = time.strftime("%Y%m%d")
    tag = f"r{rho_s}_s{src_s}"
    if floods:
        tag += "_fl" + "-".join(str(f) for f in floods)
    if defence_rho is not None:
        tag += f"_def{int(defence_rho * 100):02d}"
    if defence_flood is not None:
        tag += f"_deffl{defence_flood}"
    if adapt:
        tag += "_adapt"
    # 候选数消融必须进入文件名，否则不同候选数会互相覆盖 (P0 guard)
    if rebuild_cand is not None and rebuild_cand != REBUILD_CAND_DEFAULT:
        tag += f"_rc{rebuild_cand}"
    return f"results/{EXP_NAME}_cap{cap}_{tag}_seeds{ids}_{date}.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--rhos", default="0.5,0.8",
                    help="comma list of attack ratios, e.g. 0.3,0.5,0.8")
    ap.add_argument("--sources", default="1",
                    help="comma list of source class ids (0=N,1=S,2=V,"
                         "3=F,4=Q), e.g. 0,1,2")
    ap.add_argument("--defence-rho", type=float, default=None,
                    help="if set, run the full-rebuild defence arm for "
                         "this rho (e.g. 0.8)")
    ap.add_argument("--flood", default=None,
                    help="comma list of ADDITIONAL injected source-beat "
                         "counts appended on top of the full clean pool "
                         "(flooding attack), e.g. 100,200,400")
    ap.add_argument("--defence-flood", type=int, default=None,
                    help="if set, run the full-rebuild defence arm for "
                         "this flood injection count")
    ap.add_argument("--adapt", action="store_true",
                    help="add adaptive-injection arms (P2)")
    ap.add_argument("--clean2", action="store_true",
                    help="append a clean-control arm that rebuilds the "
                         "target pool with the defence's deeper protocol "
                         "but WITHOUT any attack (attribution control)")
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--cand", type=int, default=CAND_DEFAULT,
                    help="number of candidates generated for the per-class "
                         "clean pool before critic gating (default %d)"
                         % CAND_DEFAULT)
    ap.add_argument("--rebuild-cand", type=int, default=REBUILD_CAND_DEFAULT,
                    help="number of candidates generated by the full-pool "
                         "rebuild defence before critic gating (default %d, "
                         "i.e. 2x the clean-pool candidate count); this is "
                         "the knob for the candidate-size ablation"
                         % REBUILD_CAND_DEFAULT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-cm", action="store_true",
                    help="also persist the full 5x5 confusion matrix per arm "
                         "(used for the confusion-structure appendix); the "
                         "matrix is computed anyway, this only stores it")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)
    FC = args.target

    def parse_ids(s: str):
        out = []
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

    seed_ids = parse_ids(args.seeds)
    rhos = sorted({float(x) for x in args.rhos.split(",") if x.strip()})
    sources = parse_ids(args.sources)
    if not seed_ids or not rhos or not sources:
        ap.error("--seeds/--rhos/--sources must not be empty")
    if any(not (0 < r < 1) for r in rhos):
        ap.error("rhos must be in (0, 1)")
    if args.defence_rho is not None and args.defence_rho not in rhos:
        rhos = sorted(rhos + [args.defence_rho])
    floods = []
    if args.flood:
        floods = sorted({int(x) for x in args.flood.split(",") if x.strip()})
    if args.defence_flood is not None and args.defence_flood not in floods:
        floods = sorted(floods + [args.defence_flood])

    out_path = args.out or auto_out_path(
        seed_ids, rhos, sources, args.cap, args.defence_rho, args.adapt,
        floods, args.defence_flood, args.rebuild_cand)
    if os.path.exists(out_path) and not args.force:
        sys.exit(f"[P0] refusing to overwrite existing {out_path} "
                 f"(use --force to replace, or a different --out)")

    x_tr_all, y_tr_all, x_te, y_te = rmb.load_data()
    print("test counts:", np.bincount(y_te, minlength=5).tolist())

    # ---- arm registry ----
    arm_tags = ["no_aug", "clean"]
    for sc in sources:
        for rho in rhos:
            arm_tags.append(f"poisoned_s{CLASSES[sc]}r{int(rho*100):02d}")
            if args.adapt:
                arm_tags.append(f"adapt_s{CLASSES[sc]}r{int(rho*100):02d}")
        for nfl in floods:
            arm_tags.append(f"flood_s{CLASSES[sc]}n{nfl}")
    if args.defence_rho is not None:
        for sc in sources:
            arm_tags.append(
                f"defence_s{CLASSES[sc]}r{int(args.defence_rho*100):02d}")
    if args.defence_flood is not None:
        for sc in sources:
            arm_tags.append(f"deffl_s{CLASSES[sc]}n{args.defence_flood}")
    if args.clean2:
        arm_tags.append("clean2")
    arm_tags = list(dict.fromkeys(arm_tags))   # unique, ordered

    env_fp = {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "gpu": (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "cpu"),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "script_sha256": sha256_of(os.path.abspath(__file__)),
        "rmb_sha256": sha256_of(os.path.join(_THIS_DIR,
                                             "run_mitbih_aug.py")),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    results = {"config": {"exp": EXP_NAME, "cap": args.cap,
                          "seeds": seed_ids,
                          "cand": args.cand,
                          "rebuild_cand": args.rebuild_cand,
                          "target_class": CLASSES[FC],
                          "rhos": rhos,
                          "sources": [CLASSES[s] for s in sources],
                          "defence_rho": args.defence_rho,
                          "floods": floods,
                          "defence_flood": args.defence_flood,
                          "adapt": args.adapt,
                          "protocol": "poisoned/adapt = exact-rho "
                                      "replacement within pool; flood = "
                                      "full clean pool + n extra injected "
                                      "source beats; defence = full-pool "
                                      "rebuild",
                          "arms": arm_tags,
                          "env": env_fp},
               "per_seed": {}, "summary": {}}
    stats = {a: {"acc": [], "mf1": [], "recall": {c: [] for c in range(5)}}
             for a in arm_tags}

    for seed in seed_ids:
        print(f"\n===== seed {seed} =====")
        seed_everything(seed)
        rng = np.random.RandomState(1000 + seed)
        tr_i = []
        for c in range(5):
            idx = np.where(y_tr_all == c)[0]
            n = min(args.cap, len(idx))
            tr_i.append(idx[rng.permutation(len(idx))[:n]])
        tr_i = np.concatenate(tr_i)
        xs, ys = x_tr_all[tr_i], y_tr_all[tr_i]
        n_tr_cls = np.bincount(ys, minlength=5)
        print("scarce:", n_tr_cls.tolist())

        # ---- per-class WGAN pools + critics (shared by all arms) ----
        pools, critics, gates = {}, {}, {}
        for c in range(5):
            n_c = int(n_tr_cls[c])
            if n_c < 40:
                continue
            xc = xs[ys == c]
            w = rmb.WGAN1D(device)
            g, cc = w.train_on(xc, seed=seed * 1000 + c)
            cand = w.generate(args.cand, seed=seed * 1000 + c)
            scores = w.score(cand)
            kept = cand[scores >= np.quantile(scores, 0.10)]
            gates[c] = {"C": float(cc),
                        "dev": float(np.abs(kept.mean(0) - xc.mean(0))
                                     .mean()),
                        "gate_ok": float(cc) <= 0.5}
            pools[c] = kept
            critics[c] = w
            print(f"  [wgan] {CLASSES[c]}: C={cc:.2f} pool={len(kept)}")

        def assemble(extra_pool_f):
            """xs + [pool per class]; extra_pool_f(c, n_add) -> np array."""
            sx, sy = [], []
            for c in range(5):
                n_c = int(n_tr_cls[c])
                if c not in pools or n_c < 40:
                    continue
                n_add = n_c          # frac=1.0
                if c == FC:
                    p = extra_pool_f(n_add)
                    sx.append(p)
                    sy.append(np.full(len(p), c, dtype=np.int64))
                    continue
                sx.append(pools[c][:n_add])
                sy.append(np.full(n_add, c, dtype=np.int64))
            return (np.concatenate([xs] + sx),
                    np.concatenate([ys] + sy))

        def clean_pool(n_add):
            return pools[FC][:n_add]

        def poisoned_pool(sc, rho, n_add, adaptive=False):
            """exact-rho: n_add total, rho*n_add injected source real."""
            n_bad = int(round(n_add * rho))
            n_good = n_add - n_bad
            good = pools[FC][:n_good]
            s_idx = np.where(ys == sc)[0]
            rng2 = np.random.RandomState(seed * 7000 + sc)
            if adaptive:
                # P2: inject only source beats with the HIGHEST target-
                # critic score (hardest / most confusable ones)
                sc_src = critics[FC].score(xs[s_idx])
                order = np.argsort(sc_src)[::-1]
                pick = s_idx[order[:n_bad]]
            else:
                pick = s_idx[rng2.randint(len(s_idx), size=n_bad)]
            return np.concatenate([good, xs[pick]])

        # End-to-end rebuild wall time, keyed by arm tag.  Rebuilds happen
        # during arm CONSTRUCTION (assemble() calls the pool lambda), which is
        # before the training loop, so the timings must be captured here and
        # carried into the loop rather than cleared per iteration.
        rebuild_ms_by_arm = {}
        _cur_rebuild_tag = [None]      # set by the lambdas below

        def defence_pool(sc, rho, n_add):
            """full rebuild: discard received pool, regenerate clean."""
            t0 = time.perf_counter()
            extra = critics[FC].generate(args.rebuild_cand,
                                         seed=seed * 9500 + FC)
            esc = critics[FC].score(extra)
            good = extra[esc >= np.quantile(esc, 0.10)]
            out = good[:n_add]
            if _cur_rebuild_tag[0] is not None:
                rebuild_ms_by_arm.setdefault(_cur_rebuild_tag[0], []).append(
                    (time.perf_counter() - t0) * 1e3)
            return out

        def flood_pool(sc, n_flood, n_add):
            """flooding: keep the FULL clean pool (n_add) and ADD n_flood
            source-class real beats on top (attacker controls the output
            interface and can inject arbitrary extra volume)."""
            good = pools[FC][:n_add]
            s_idx = np.where(ys == sc)[0]
            rng2 = np.random.RandomState(seed * 7200 + sc)
            pick = s_idx[rng2.randint(len(s_idx), size=n_flood)]
            return np.concatenate([good, xs[pick]])

        arms = [("no_aug", (xs, ys))]
        arms.append(("clean", assemble(clean_pool)))
        for sc in sources:
            for rho in rhos:
                tag = f"poisoned_s{CLASSES[sc]}r{int(rho*100):02d}"
                arms.append((tag, assemble(
                    lambda n, sc=sc, rho=rho: poisoned_pool(sc, rho, n))))
                if args.adapt:
                    tag = f"adapt_s{CLASSES[sc]}r{int(rho*100):02d}"
                    arms.append((tag, assemble(
                        lambda n, sc=sc, rho=rho:
                        poisoned_pool(sc, rho, n, adaptive=True))))
            for nfl in floods:
                tag = f"flood_s{CLASSES[sc]}n{nfl}"
                arms.append((tag, assemble(
                    lambda n, sc=sc, nfl=nfl: flood_pool(sc, nfl, n))))
        if args.defence_rho is not None:
            for sc in sources:
                tag = (f"defence_s{CLASSES[sc]}"
                       f"r{int(args.defence_rho*100):02d}")
                _cur_rebuild_tag[0] = tag
                arms.append((tag, assemble(
                    lambda n, sc=sc: defence_pool(
                        sc, args.defence_rho, n))))
        if args.defence_flood is not None:
            for sc in sources:
                tag = f"deffl_s{CLASSES[sc]}n{args.defence_flood}"
                _cur_rebuild_tag[0] = tag
                arms.append((tag, assemble(
                    lambda n, sc=sc: defence_pool(
                        sc, 0.5, n))))   # rebuild is rho-independent
        if args.clean2:
            # Clean-control for attribution (reviewer P0): NO attack, but the
            # target-class pool is rebuilt with the same deeper protocol as
            # the defence arms (candidates -> top-90% -> top-up).
            # Appended LAST so that every pre-existing arm keeps its index
            # (and hence its arm-index-derived initialization).
            _cur_rebuild_tag[0] = "clean2"
            arms.append(("clean2", assemble(
                lambda n: defence_pool(FC, 0.5, n))))
        _cur_rebuild_tag[0] = None

        for ai, (tag, (xf, yf)) in enumerate(arms):
            # deterministic, arm-set-INDEPENDENT init: re-seed before
            # build_backbone so the model init no longer depends on the
            # global RNG state left by arm construction (defence arms call
            # generate() -> torch.manual_seed, which used to corrupt the
            # init of every later-trained arm and made no_aug differ
            # between runs with different arm sets).
            seed_everything(seed * 1000 + ai)
            rb_this = rebuild_ms_by_arm.get(tag)
            m = rmb.build_backbone("se", 5).to(device)
            m = rmb.train_cnn(m, xf, yf, device, seed)
            p = rmb.predict(m, x_te, device)
            mm = classification_metrics(y_te, p, n_classes=5)
            cm = np.zeros((5, 5), dtype=int)
            for a_, b_ in zip(y_te, p):
                cm[a_, b_] += 1
            with np.errstate(divide="ignore", invalid="ignore"):
                rec = cm.diagonal() / cm.sum(1)
            stats[tag]["acc"].append(mm["accuracy"])
            stats[tag]["mf1"].append(mm["f1"])
            for c in range(5):
                stats[tag]["recall"][c].append(float(rec[c]))
            if rb_this:
                stats[tag]["rebuild_ms"] = float(np.mean(rb_this))
            if args.save_cm:
                stats[tag]["cm"] = cm.tolist()
            print(f"[{tag}] acc={mm['accuracy']*100:.2f} "
                  f"macroF1={mm['f1']*100:.2f}  "
                  f"S={rec[1]:.3f} F={rec[FC]:.3f} N={rec[0]:.2f} "
                  f"recalls={' '.join(f'{CLASSES[c]}={rec[c]:.2f}'
                                      for c in range(5))}"
                  + (f"  rebuild={np.mean(rb_this):.1f}ms" if rb_this else ""))

        results["per_seed"][f"seed{seed}"] = {
            a: dict({"acc": stats[a]["acc"][-1],
                     "macro_f1": stats[a]["mf1"][-1],
                     "recall": {CLASSES[c]: stats[a]["recall"][c][-1]
                                for c in range(5)}},
                    **({"rebuild_ms": stats[a]["rebuild_ms"]}
                       if "rebuild_ms" in stats[a] else {}),
                    **({"cm": stats[a]["cm"]} if "cm" in stats[a] else {}))
            for a in arm_tags}
        results["per_seed"][f"seed{seed}"]["gates"] = gates

    def mstd(v):
        return [round(float(np.mean(v)), 4),
                round(float(np.std(v, ddof=1)), 4)]
    results["summary"] = {
        a: {"acc": mstd(stats[a]["acc"]),
            "macro_f1": mstd(stats[a]["mf1"]),
            "F_recall": mstd(stats[a]["recall"][FC]),
            "all_recall": {CLASSES[c]: mstd(stats[a]["recall"][c])
                           for c in range(5)}} for a in arm_tags}
    print("\n===== summary (mean±std) =====")
    for a in arm_tags:
        s = results["summary"][a]
        print(f"  {a:<22} acc {s['acc'][0]*100:.2f}±{s['acc'][1]*100:.2f} "
              f"| mF1 {s['macro_f1'][0]*100:.2f} "
              f"| S {s['all_recall']['S'][0]:.3f} "
              f"| F {s['F_recall'][0]:.3f}")
    results["config"]["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if os.path.exists(out_path) and not args.force:
        sys.exit(f"[P0] output {out_path} appeared during the run; "
                 f"not overwriting (use --force to replace)")
    save_json(out_path, results)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
