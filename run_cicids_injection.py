# -*- coding: utf-8 -*-
"""E5: CIC-IDS-2017 目标类注入（池投毒）+ 重建式防御
   回应导师意见 7.1 的 (A) 路线：
     (a) 全局准确率静默  (b) 逐类召回崩塌  (c) 防御重建是否有效

   与既有脚本 run_cicids_xgb_wgan_rerun.py 的区别（该脚本只做逐类增强追加，
   并非池投毒）：
     * 目标类池被【改写】：以源类真实样本改标后替换目标类合成候选（exact-rho）；
     * 新增【重建式防御】臂：丢弃可疑池，改用缓存的干净合成候选重建；
     * 多种子：划分与 XGBoost 的随机种子均由 --seed 控制（原脚本硬编码 42）。

   无需 GPU：合成样本直接读 results/cicids_perclass_synthetic.npz 缓存。
   用法:
     python scripts/run_cicids_injection.py --seeds 0-2
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gan_gat.evaluate import classification_metrics, save_json  # noqa: E402
from gan_gat.utils import seed_everything  # noqa: E402
from gan_gat.preprocessing import stratified_split  # noqa: E402

CLASS_NAMES = ["BENIGN", "DoS_Hulk", "PortScan", "DDoS", "DoS_GoldenEye",
               "FTP_Patator", "SSH_Patator", "DoS_slowloris",
               "DoS_Slowhttptest", "Bot", "Web_Attack_Brute_Force",
               "Web_Attack_XSS", "Infiltration", "Web_Attack_Sql_Injection",
               "Heartbleed"]
AUG_CLASSES = [5, 6, 7, 8, 9, 10, 11]
N_CLASSES = 15
CACHE_NPZ = "results/cicids_perclass_synthetic.npz"
PREPARED = "results/cicids_prepared.npz"
EXP = "cicids_injection"


def sha256_of(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def parse_ids(s):
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


def rare_recall(y_te, pred):
    """全部 15 类的逐类召回（算 C_max 与 PMSB 预测需要前 5 个大类）。"""
    out = {}
    for c in range(N_CLASSES):
        tp = int(((pred == c) & (y_te == c)).sum())
        fn = int(((pred != c) & (y_te == c)).sum())
        out[CLASS_NAMES[c]] = tp / max(1, tp + fn)
    return out


def balanced_weights(y, counts):
    w = np.zeros(len(y))
    for c in range(N_CLASSES):
        w[y == c] = len(y) / (N_CLASSES * max(1, counts[c]))
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--target-class", type=int, default=10,
                    help="class id whose pool is TAMPERED (default 10 = "
                         "Web_Attack_Brute_Force)")
    ap.add_argument("--source-class", type=int, default=11,
                    help="class id whose REAL samples are relabelled and "
                         "injected (default 11 = Web_Attack_XSS, the "
                         "smallest-prior augmented class)")
    ap.add_argument("--rhos", default="0.5,0.8",
                    help="injection ratios for the replacement attack")
    ap.add_argument("--flood-n", type=str, default="0",
                    help="if >0, additionally run FLOODING arms: keep the full "
                         "clean pool and append N REAL source-class rows "
                         "relabelled as the target class.  Mirrors the "
                         "flooding attack of the MIT-BIH experiments; needed "
                         "to actually DESTROY a large-prior source, whose "
                         "training mass cannot be overwhelmed by pool "
                         "replacement alone (replacement is capped by pool "
                         "size).  Comma list allowed, e.g. 16000,32000.")
    ap.add_argument("--trees", type=int, default=400)
    ap.add_argument("--depth", type=int, default=7)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    seed_ids = parse_ids(args.seeds)
    rhos = sorted({float(x) for x in args.rhos.split(",") if x.strip()})
    TC, SC = args.target_class, args.source_class
    os.makedirs("results", exist_ok=True)
    out_path = args.out or (
        f"results/{EXP}_t{CLASS_NAMES[TC]}_s{CLASS_NAMES[SC]}_"
        f"seeds{seed_ids[0]}-{seed_ids[-1]}_{time.strftime('%Y%m%d')}.json")
    if os.path.exists(out_path) and not args.force:
        sys.exit(f"[P0] refusing to overwrite {out_path} (use --force)")

    d = np.load(PREPARED)
    x, y = d["X"], d["y"]
    syn = np.load(CACHE_NPZ)
    x_syn_all, y_syn_all = syn["x_syn"], syn["y_syn"]
    print(f"prepared {x.shape}  cached synthetic {x_syn_all.shape}")

    results = {"config": {
        "exp": EXP, "seeds": seed_ids, "target_class": CLASS_NAMES[TC],
        "source_class": CLASS_NAMES[SC], "rhos": rhos,
        "trees": args.trees, "depth": args.depth, "lr": args.lr,
        "protocol": "poisoned = exact-rho replacement INSIDE the target "
                    "class synthetic pool, injected samples are REAL source-"
                    "class rows relabelled as target; defence = discard the "
                    "received pool and rebuild it from the cached clean "
                    "synthetic candidates; no GAN retraining is performed in "
                    "either arm",
        "env": {"numpy": np.__version__, "python": platform.python_version(),
                "script_sha256": sha256_of(os.path.abspath(__file__)),
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}},
        "per_seed": {}, "summary": {}}

    flood_ns = sorted({int(x) for x in str(args.flood_n).split(",")
                       if x.strip() and int(x) > 0})
    arms = (["plain", "clean_aug"]
            + [f"poisoned_r{int(r*100):02d}" for r in rhos]
            + [f"flood_n{n}" for n in flood_ns]
            + [f"defence_r{int(r*100):02d}" for r in rhos]
            + [f"deffl_n{n}" for n in flood_ns])
    stats = {a: {"acc": [], "macro_f1": [], "rare": {}} for a in arms}
    for a in arms:
        stats[a]["rare"] = {n: [] for n in CLASS_NAMES}

    for seed in seed_ids:
        seed_everything(seed)
        rng = np.random.RandomState(seed)
        x_tr, y_tr, _, _, x_te, y_te = stratified_split(x, y, 0.0, 0.2, seed)
        counts_tr = np.bincount(y_tr, minlength=N_CLASSES)
        print(f"\n===== seed {seed} =====  train {len(x_tr):,}  "
              f"test {len(x_te):,}")

        # 该目标类的干净合成池（缓存里只含增强类）
        m_tc = y_syn_all == TC
        pool_clean = x_syn_all[m_tc]
        syn_other = x_syn_all[~m_tc]
        y_other = y_syn_all[~m_tc]
        # 源类真实训练样本（用于改标注入）
        sc_real = x_tr[y_tr == SC]
        print(f"  target {CLASS_NAMES[TC]}: clean synthetic pool "
              f"{len(pool_clean):,}; source {CLASS_NAMES[SC]}: "
              f"{len(sc_real):,} real train rows")

        def build(arm):
            """返回 (x, y) 训练集。"""
            if arm == "plain":
                return x_tr, y_tr
            # 合成部分：其它增强类照常 + 目标类池（视臂而定）
            if arm == "clean_aug":
                tgt_x, tgt_y = pool_clean, np.full(len(pool_clean), TC)
            elif arm.startswith("poisoned"):
                rho = int(arm.split("r")[1]) / 100.0
                n_add = len(pool_clean)
                n_bad = int(round(n_add * rho))
                n_good = n_add - n_bad
                good = pool_clean[:n_good]
                pick = rng.randint(len(sc_real), size=n_bad)
                bad = sc_real[pick]           # 真实源类样本，改标为目标类
                tgt_x = np.concatenate([good, bad])
                tgt_y = np.full(len(tgt_x), TC)
            elif arm.startswith("flood_n"):
                # 淹没型：保留完整干净池，额外追加 n 条改标源类真实样本。
                # 替换型受池容量（n_add）上限约束，无法压垮大先验源
                # （其训练质量远超池容量）；淹没型不受此约束。
                n = int(arm.split("n")[1])
                pick = rng.randint(len(sc_real), size=n)
                bad = sc_real[pick]
                tgt_x = np.concatenate([pool_clean, bad])
                tgt_y = np.full(len(tgt_x), TC)
            else:                              # defence_* : 重建干净池
                tgt_x, tgt_y = pool_clean, np.full(len(pool_clean), TC)
            xs = np.concatenate([x_tr, syn_other, tgt_x])
            ys = np.concatenate([y_tr, y_other, tgt_y])
            return xs, ys

        for arm in arms:
            xf, yf = build(arm)
            counts = np.bincount(yf, minlength=N_CLASSES)
            model = XGBClassifier(
                n_estimators=args.trees, max_depth=args.depth,
                learning_rate=args.lr, subsample=0.9, colsample_bytree=0.9,
                tree_method="hist", n_jobs=args.n_jobs,
                random_state=seed, eval_metric="mlogloss")
            model.fit(xf, yf, sample_weight=balanced_weights(yf, counts))
            pred = model.predict(x_te)
            m = classification_metrics(y_te, pred, n_classes=N_CLASSES)
            rr = rare_recall(y_te, pred)
            stats[arm]["acc"].append(m["accuracy"])
            stats[arm]["macro_f1"].append(m["f1"])
            for n in CLASS_NAMES:
                stats[arm]["rare"][n].append(rr[n])
            print(f"  [{arm:<14}] acc={m['accuracy']*100:.2f}  "
                  f"mF1={m['f1']*100:.2f}  "
                  f"{CLASS_NAMES[SC]}召回={rr[CLASS_NAMES[SC]]:.3f}  "
                  f"{CLASS_NAMES[TC]}召回={rr[CLASS_NAMES[TC]]:.3f}")

        results["per_seed"][f"seed{seed}"] = {
            a: {"acc": stats[a]["acc"][-1], "macro_f1": stats[a]["macro_f1"][-1],
                "rare_recall": {n: stats[a]["rare"][n][-1]
                                for n in CLASS_NAMES}}
            for a in arms}

    def mstd(v):
        return [round(float(np.mean(v)), 4), round(float(np.std(v, ddof=1)), 4)
                if len(v) > 1 else 0.0]

    for a in arms:
        results["summary"][a] = {
            "acc": mstd(stats[a]["acc"]), "macro_f1": mstd(stats[a]["macro_f1"]),
            "rare_recall": {n: mstd(stats[a]["rare"][n])
                            for n in CLASS_NAMES}}
    results["config"]["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(out_path, results)
    print(f"\nsaved -> {out_path}")

    print("\n=== E5 summary ===")
    print("%-16s %-10s %-10s %-14s %-14s" % (
        "arm", "acc", "macroF1", f"{CLASS_NAMES[SC]}(源)",
        f"{CLASS_NAMES[TC]}(目标)"))
    for a in arms:
        s = results["summary"][a]
        print("%-16s %-10.2f %-10.2f %-14.3f %-14.3f" % (
            a, s["acc"][0], s["macro_f1"][0],
            s["rare_recall"][CLASS_NAMES[SC]][0],
            s["rare_recall"][CLASS_NAMES[TC]][0]))


if __name__ == "__main__":
    main()
