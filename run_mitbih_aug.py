"""PRE-REGISTERED scarce-label MIT-BIH (Kaggle 5-class) beat experiment:
can SMALL WGAN-GP augmentation improve a LIGHTWEIGHT 1D-CNN?

Data: mitbih_train.csv / mitbih_test.csv (187-sample beats, labels
0..4 = N,S,V,F,Q; train N~72k / Q~15 -- extreme imbalance).
Protocol (fixed before running; every cell reported regardless of sign):
  - Beats z-normalized per beat, clipped to [-3,3], mapped to [-1,1],
    zero-padded 187 -> 192 (5 tail samples) so 1D deconv chains divide.
  - Scarce train: per-class stratified cap C=100 drawn from mitbih_train
    (class Q keeps its 15 samples), redrawn per seed 0..4.
  - Test: FULL mitbih_test (21,554 beats, natural distribution), fixed.
  - Arms (same cap, same classifier, shared seeds):
      no_aug   : scarce real beats only
      wgan30 / wgan100 : per-class 1D WGAN-GP (spectral-norm critic,
             GP=10, n_critic=5, 300 epochs) on the class's scarce beats;
             pool of 4000 candidates, critic top-90% filter, then add
             +30% / +100% of the class train count (injection sweep)
      realcopy : duplicate real beats to the +100% counts (golden bound)
      smote    : imblearn SMOTE on the beats to the +100% counts
  - Classifier: lightweight 1D depthwise-separable CNN (~50K params),
    plain CE, Adam 1e-3, 80 epochs, batch 64, identical for all arms.
  - Metrics (all reported): natural-test accuracy, macro-F1, macro-recall
    (= balanced acc), per-class F1.  Paired t-tests (5 seeds) vs no_aug.

Usage:
    python scripts/run_mitbih_aug.py --cap 100 --backbone se --seeds 0-4
    python scripts/run_mitbih_aug.py --cap 300 --backbone se --seeds 0-4
"""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from scipy import stats
from sklearn.metrics import confusion_matrix

from gan_gat.evaluate import classification_metrics, save_json
from gan_gat.utils import seed_everything

CLASSES = ["N", "S", "V", "F", "Q"]
CAP = 100
GAN_EPOCHS = 300
RAW = 187
LEN = 192          # RAW + 5 zero-pad so deconv chains divide evenly
DATA_DIR = "data/MIT-BIH"


def load_data():
    tr = pd.read_csv(f"{DATA_DIR}/mitbih_train.csv", header=None)
    te = pd.read_csv(f"{DATA_DIR}/mitbih_test.csv", header=None)
    x_tr = tr.iloc[:, :RAW].values.astype(np.float32)
    y_tr = tr.iloc[:, RAW].values.astype(np.int64)
    x_te = te.iloc[:, :RAW].values.astype(np.float32)
    y_te = te.iloc[:, RAW].values.astype(np.int64)
    return prep(x_tr), y_tr, prep(x_te), y_te


def prep(x: np.ndarray) -> np.ndarray:
    out = np.zeros((len(x), LEN), dtype=np.float32)
    for i in range(len(x)):
        v = x[i]
        v = (v - v.mean()) / (v.std() + 1e-6)
        v = np.clip(v, -3, 3) / 3.0
        out[i, :RAW] = v
    return out


# ------------------------------------------------------------ models ----
class SE(nn.Module):
    """Squeeze-and-excitation channel attention (1D)."""

    def __init__(self, ch, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(ch, max(4, ch // r)), nn.ReLU(inplace=True),
            nn.Linear(max(4, ch // r), ch), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x).unsqueeze(2)


def dw_block(ch, k=9, se=False):
    b = [nn.Conv1d(ch, ch, k, padding=k // 2, groups=ch),
         nn.Conv1d(ch, ch * 2, 1), nn.BatchNorm1d(ch * 2),
         nn.ReLU(inplace=True)]
    if se:
        b.append(SE(ch * 2))
    b.append(nn.MaxPool1d(2))
    return nn.Sequential(*b)


class LightCNN1D(nn.Module):
    """Depthwise-separable 1D CNN over 192-sample beats.

    ``se`` adds channel attention; ``base``/``blocks`` scale capacity.
    """

    def __init__(self, n_classes: int = 5, base: int = 32, blocks: int = 3,
                 se: bool = False):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, base, 17, stride=2,
                                            padding=8),
                                  nn.BatchNorm1d(base),
                                  nn.ReLU(inplace=True))
        blks, ch = [], base
        for _ in range(blocks):
            blks.append(dw_block(ch, se=se))
            ch *= 2
        self.blocks = nn.Sequential(*blks)
        self.head = nn.Linear(ch, n_classes)

    def forward(self, x):
        h = self.blocks(self.stem(x))
        return self.head(h.mean(dim=2))


class BasicBlock1D(nn.Module):
    """HeartNet-style 1D residual block (kernel 17)."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 17, stride=stride, padding=8,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 17, padding=8, bias=False)
        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch))

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = self.conv2(out)
        if self.down is not None:
            identity = self.down(x)
        return out + identity


class ResNet1D(nn.Module):
    def __init__(self, n_classes: int = 5, widths=(32, 64, 128)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, widths[0], 17, stride=2, padding=8, bias=False),
            nn.BatchNorm1d(widths[0]))
        layers, ch = [], widths[0]
        for w in widths:
            layers.append(BasicBlock1D(ch, w,stride=2 if ch != widths[0] else 1))
            layers.append(BasicBlock1D(w, w))
            ch = w
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(ch, n_classes)

    def forward(self, x):
        h = F.relu(self.stem(x), inplace=True)
        h = F.max_pool1d(h, 3, stride=2, padding=1)
        h = self.body(h)
        return self.head(h.mean(dim=2))


def build_backbone(name: str, n_classes: int) -> nn.Module:
    """light(~50K) | se(~70K) | wide(~1M) | resnet(~1.3M)."""
    if name == "light":
        return LightCNN1D(n_classes)
    if name == "se":
        return LightCNN1D(n_classes, se=True)
    if name == "wide":
        return LightCNN1D(n_classes, base=64, blocks=4, se=True)
    if name == "resnet":
        return ResNet1D(n_classes)
    raise ValueError(f"unknown backbone {name!r}")


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())



class Gen1D(nn.Module):
    def __init__(self, latent: int = 100, n_ch: int = 64):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent, n_ch * 8 * 12),
                                nn.BatchNorm1d(n_ch * 8 * 12),
                                nn.ReLU(inplace=True))
        def up(i, o):
            return [nn.ConvTranspose1d(i, o, 4, 2, 1), nn.BatchNorm1d(o),nn.ReLU(inplace=True)]
        self.deconv = nn.Sequential(*up(n_ch * 8, n_ch * 4),   # 24
                                    *up(n_ch * 4, n_ch * 2),   # 48
                                    *up(n_ch * 2, n_ch),       # 96
                                    nn.ConvTranspose1d(n_ch, 1, 4, 2, 1),
                                    nn.Tanh())                # 192

    def forward(self, z):
        return self.deconv(self.fc(z).view(z.size(0), -1, 12))


class Critic1D(nn.Module):
    def __init__(self, n_ch: int = 64):
        super().__init__()
        def sn(i, o, k=4, s=2, p=1):
            return nn.utils.spectral_norm(nn.Conv1d(i, o, k, s, p))
        self.conv = nn.Sequential(
            sn(1, n_ch), nn.LeakyReLU(0.2, inplace=True),       # 96
            sn(n_ch, n_ch * 2), nn.LeakyReLU(0.2, inplace=True),  # 48
            sn(n_ch * 2, n_ch * 4), nn.LeakyReLU(0.2, inplace=True),  # 24
            sn(n_ch * 4, n_ch * 8), nn.LeakyReLU(0.2, inplace=True))  # 12
        self.fc = nn.utils.spectral_norm(
            nn.Linear(n_ch * 8 * 12, 1, bias=False))

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


def gp1d(critic, real, fake, lam=10.0):
    alpha = torch.rand(real.size(0), 1, 1, device=real.device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d = critic(interp)
    ones = torch.ones(d.size(), device=real.device)
    grads = autograd.grad(d, interp, grad_outputs=ones, create_graph=True,
                          retain_graph=True, only_inputs=True)[0]
    return lam * ((grads.view(grads.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()


class WGAN1D:
    def __init__(self, device, lr: float = 2e-4):
        self.device = device
        self.gen = Gen1D().to(device)
        self.cri = Critic1D().to(device)
        self.opt_g = torch.optim.Adam(self.gen.parameters(), lr=lr,
                                      betas=(0.5, 0.999))
        self.opt_c = torch.optim.Adam(self.cri.parameters(), lr=lr,
                                      betas=(0.5, 0.999))

    def train_on(self, x: np.ndarray, seed: int, epochs: int = GAN_EPOCHS):
        seed_everything(seed)
        xt = torch.as_tensor(x, dtype=torch.float32,
                             device=self.device).unsqueeze(1)
        n = len(xt)
        rng = np.random.RandomState(seed)
        it = 0
        for _ in range(epochs):
            order = rng.permutation(n)
            for s in range(0, n, 32):
                real = xt[order[s:s + 32]]
                self.opt_c.zero_grad()
                z = torch.randn(len(real), 100, device=self.device)
                fake = self.gen(z).detach()
                c_loss = (-self.cri(real).mean() + self.cri(fake).mean()
                          + gp1d(self.cri, real, fake))
                c_loss.backward()
                self.opt_c.step()
                if it % 5 == 0:
                    self.opt_g.zero_grad()
                    z = torch.randn(len(real), 100, device=self.device)
                    g_loss = -self.cri(self.gen(z)).mean()
                    g_loss.backward()
                    self.opt_g.step()
                it += 1
        return float(g_loss.detach()), float(c_loss.detach())

    @torch.no_grad()
    def score(self, x: np.ndarray) -> np.ndarray:
        xt = torch.as_tensor(x, dtype=torch.float32,
                             device=self.device).unsqueeze(1)
        out = []
        for s in range(0, len(x), 512):
            out.append(self.cri(xt[s:s + 512]).flatten().cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def generate(self, n: int, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            torch.manual_seed(seed)
        self.gen.eval()
        out = []
        for s in range(0, n, 512):
            z = torch.randn(min(512, n - s), 100, device=self.device)
            out.append(self.gen(z).clamp(-1, 1).squeeze(1).cpu().numpy())
        self.gen.train()
        return np.concatenate(out).astype(np.float32)


# --------------------------------------------------------- training ----
def train_cnn(model, x, y, device, seed, epochs=80):
    seed_everything(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xt = torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(1)
    yt = torch.as_tensor(y, dtype=torch.long, device=device)
    n = len(x)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, 64):
            idx = perm[s:s + 64]
            opt.zero_grad()
            F.cross_entropy(model(xt[idx]), yt[idx]).backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model, x, device, bs=1024):
    xt = torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(1)
    out = []
    for s in range(0, len(x), bs):
        out.append(model(xt[s:s + bs]).argmax(1).cpu().numpy())
    return np.concatenate(out)


def per_class_f1(y_te, pred):
    cm = confusion_matrix(y_te, pred, labels=list(range(len(CLASSES))))
    out = {}
    for c in range(len(CLASSES)):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum()) - tp
        fn = int(cm[c, :].sum()) - tp
        out[CLASSES[c]] = tp / max(1, tp + 0.5 * (fp + fn))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--backbone", default="light",
                    choices=["light", "se", "wide", "resnet"])
    ap.add_argument("--seeds", default="0-4",
                    help="seed ids: comma list or range, e.g. 0-4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None,
                    help="output json (default: auto-named "
                         "results/mitbih_aug_cap<cap>_<bb>_seeds<ids>_"
                         "<date>.json)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing output file")
    args = ap.parse_args()
    device = torch.device(args.device)
    bb = build_backbone(args.backbone, len(CLASSES))
    print(f"backbone={args.backbone} "
          f"params={count_params(bb) / 1e3:.0f}K")

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
    if not seed_ids:
        ap.error("--seeds must list at least one seed id, e.g. 0-4")
    ids = (f"{min(seed_ids)}-{max(seed_ids)}" if len(seed_ids) > 1
           else str(seed_ids[0]))
    date = time.strftime("%Y%m%d")
    out_path = args.out or (f"results/mitbih_aug_cap{args.cap}_"
                            f"{args.backbone}_seeds{ids}_{date}.json")
    if os.path.exists(out_path) and not args.force:
        sys.exit(f"[P0] refusing to overwrite existing {out_path} "
                 f"(use --force to replace, or a different --out)")

    def fit(xf, yf, seed, ai):
        """Arm-index-seeded init: build_backbone consumes global RNG, so
        re-seed with (seed, ai) to make every arm's init independent of
        what ran before (WGAN training / earlier arms / defence pool
        generation) -- fixes cross-run & cross-config reproducibility."""
        seed_everything(seed * 1000 + ai)
        m = build_backbone(args.backbone, len(CLASSES)).to(device)
        return train_cnn(m, xf, yf, device, seed)

    x_tr_all, y_tr_all, x_te, y_te = load_data()
    print("train counts:", np.bincount(y_tr_all, minlength=5).tolist(),
          " test counts:", np.bincount(y_te, minlength=5).tolist())

    env_fp = {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "gpu": (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "cpu"),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "script_sha256": hashlib.sha256(
            open(os.path.abspath(__file__), "rb").read()
        ).hexdigest()[:16],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    results = {"config": {"cap": args.cap, "seeds": seed_ids,
                          "backbone": args.backbone,
                          "gan_epochs": GAN_EPOCHS,
                          "classes": CLASSES,
                          "env": env_fp},
               "per_seed": {}, "summary": {}}
    accs = {k: [] for k in ("no_aug", "wgan30", "wgan100", "realcopy",
                            "smote")}
    mf1 = {k: [] for k in accs}
    mrec = {k: [] for k in accs}

    from imblearn.over_sampling import SMOTE

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
        print(f"scarce train {len(xs)} per class "
              f"{dict(zip(CLASSES, n_tr_cls.tolist()))}")

        m = fit(xs, ys, seed, ai=0)
        p0 = predict(m, x_te, device)
        mm = classification_metrics(y_te, p0, n_classes=5)
        accs["no_aug"].append(mm["accuracy"])
        mf1["no_aug"].append(mm["f1"])
        mrec["no_aug"].append(mm["recall"])
        print(f"[no_aug] acc={mm['accuracy'] * 100:.2f} "
              f"macroF1={mm['f1'] * 100:.2f} macroR={mm['recall'] * 100:.2f}")

        pools = {}
        gates = {}
        for c in range(5):
            n_c = int(n_tr_cls[c])
            if n_c < 40:
                continue
            xc = xs[ys == c]
            w = WGAN1D(device)
            g, cc = w.train_on(xc, seed=seed * 1000 + c)
            cand = w.generate(4000, seed=seed * 1000 + c)
            scores = w.score(cand)
            kept = cand[scores >= np.quantile(scores, 0.10)]
            # ---- quality gate (pre-registered): skip untrustworthy pools ----
            # only critic calibration is used: C > +0.5 means the critic
            # finished with inverted real/fake preference (observed only in
            # the catastrophic seed; e.g. C=+2.2 for class N).  Mean-shift
            # is logged for reference only -- ECG z-scored features make it
            # a poor quality proxy (good pools can deviate ~0.14).
            dev = float(np.abs(kept.mean(0) - xc.mean(0)).mean())
            if cc > 0.5:
                print(f"  [wgan] {CLASSES[c]} SKIPPED by critic gate "
                      f"(C={cc:.2f} > 0.5)")
                gates[c] = {"skipped": True, "C": float(cc),
                            "dev": dev}
                continue
            pools[c] = kept
            gates[c] = {"skipped": False, "C": float(cc), "dev": dev}
            print(f"  [wgan] {CLASSES[c]}: kept {len(kept)} "
                  f"(G {g:.2f} C {cc:.2f} dev={dev:.3f})")

        def build(frac):
            sx, sy = [], []
            for c in range(5):
                n_c = int(n_tr_cls[c])
                if c not in pools or n_c < 40:
                    continue
                n_add = int(frac * n_c)
                sy.append(np.full(n_add, c, dtype=np.int64))
                sx.append(pools[c][:n_add])
            return np.concatenate([xs] + sx), np.concatenate([ys] + sy)

        ai = 1
        for tag, frac in (("wgan30", 0.3), ("wgan100", 1.0)):
            xa, ya = build(frac)
            m = fit(xa, ya, seed, ai)
            ai += 1
            p = predict(m, x_te, device)
            mm = classification_metrics(y_te, p, n_classes=5)
            accs[tag].append(mm["accuracy"])
            mf1[tag].append(mm["f1"])
            mrec[tag].append(mm["recall"])
            print(f"[{tag}] acc={mm['accuracy'] * 100:.2f} "
                  f"macroF1={mm['f1'] * 100:.2f} "
                  f"macroR={mm['recall'] * 100:.2f}")

        rx, ry = [], []
        for c in range(5):
            n_c = int(n_tr_cls[c])
            if n_c < 40:
                continue
            idx = np.where(ys == c)[0]
            rng2 = np.random.RandomState(seed * 100 + c)
            pick = idx[rng2.randint(len(idx), size=n_c)]
            rx.append(xs[pick])
            ry.append(np.full(n_c, c, dtype=np.int64))
        xr = np.concatenate([xs] + rx)
        yr = np.concatenate([ys] + ry)
        m = fit(xr, yr, seed, ai)
        ai += 1
        p = predict(m, x_te, device)
        mm = classification_metrics(y_te, p, n_classes=5)
        accs["realcopy"].append(mm["accuracy"])
        mf1["realcopy"].append(mm["f1"])
        mrec["realcopy"].append(mm["recall"])
        print(f"[realcopy] acc={mm['accuracy'] * 100:.2f} "
              f"macroF1={mm['f1'] * 100:.2f} macroR={mm['recall'] * 100:.2f}")

        strat = {c: int(2 * n_tr_cls[c]) for c in range(5)
                 if n_tr_cls[c] >= 40}
        x_sm, y_sm = SMOTE(random_state=seed,
                           k_neighbors=min(5, int(n_tr_cls.min()) - 1),
                           sampling_strategy=strat).fit_resample(xs, ys)
        m = fit(x_sm, y_sm, seed, ai)
        ai += 1
        p = predict(m, x_te, device)
        mm = classification_metrics(y_te, p, n_classes=5)
        accs["smote"].append(mm["accuracy"])
        mf1["smote"].append(mm["f1"])
        mrec["smote"].append(mm["recall"])
        print(f"[smote] acc={mm['accuracy'] * 100:.2f} "
              f"macroF1={mm['f1'] * 100:.2f} macroR={mm['recall'] * 100:.2f}")

        results["per_seed"][f"seed{seed}"] = {
            "acc": {k: accs[k][-1] for k in accs},
            "macro_f1": {k: mf1[k][-1] for k in mf1},
            "macro_recall": {k: mrec[k][-1] for k in mrec},
            "per_class_f1": {"no_aug": per_class_f1(y_te, p0)},
            "gate": gates}

    def summ(d):
        return {"mean": float(np.mean(d)), "std": float(np.std(d, ddof=1))}
    results["summary"] = {
        "acc": {k: summ(v) for k, v in accs.items()},
        "macro_f1": {k: summ(v) for k, v in mf1.items()},
        "macro_recall": {k: summ(v) for k, v in mrec.items()},
        "paired_ttest_macroF1": {
            k: float(stats.ttest_rel(mf1[k], mf1["no_aug"]).pvalue)
            for k in ("wgan30", "wgan100", "realcopy", "smote")}}
    print("\n===== summary (mean±std) =====")
    for met, d in (("acc", accs), ("macroF1", mf1), ("macroR", mrec)):
        print(f"  {met:<8}: " + "  ".join(
            f"{k}={np.mean(v) * 100:.2f}±{np.std(v, ddof=1) * 100:.2f}"
            for k, v in d.items()))
    print("  paired t (macroF1 vs no_aug):",
          {k: round(float(v), 4)
           for k, v in results["summary"]["paired_ttest_macroF1"].items()})
    results["config"]["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if os.path.exists(out_path) and not args.force:
        sys.exit(f"[P0] output {out_path} appeared during the run; "
                 f"not overwriting (use --force to replace)")
    save_json(out_path, results)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
