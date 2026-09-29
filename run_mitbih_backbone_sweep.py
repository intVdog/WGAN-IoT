"""Exploratory backbone sweep on MIT-BIH scarce augmentation.

Same scarce protocol as run_mitbih_aug.py (cap=100/class, WGAN pools
shared across backbones) but scans four 1D backbones to answer:
does a STRONGER backbone raise accuracy, and does WGAN augmentation
(+100%) still help (or help more) with it?

Backbones (all plain CE, 80 epochs, Adam 1e-3, batch 64):
  light : current depthwise-separable CNN (~50K params)      [baseline]
  se    : light + squeeze-excitation after each block (~60K)
  wide  : base channels 64, 4 blocks (~250K)
  resnet: 1D ResNet (HeartNet-style, kernel 17, 32/64/128,
          ~1.1M params)
Arms per backbone: no_aug | wgan100 (critic-filtered, +100%) | realcopy.
Seeds: 3 (exploratory; full protocol later for the chosen config).

Usage:
    python scripts/run_mitbih_backbone_sweep.py [--seeds 3] [--device cuda]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from gan_gat.evaluate import classification_metrics, save_json
from gan_gat.utils import seed_everything

CLASSES = ["N", "S", "V", "F", "Q"]
CAP = 100
GAN_EPOCHS = 300
RAW = 187
LEN = 192
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


class LightCNN(nn.Module):
    def __init__(self, n_classes, base=32, blocks=3, se=False):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base, 17, stride=2, padding=8),
            nn.BatchNorm1d(base), nn.ReLU(inplace=True))
        blks = []
        ch = base
        for _ in range(blocks):
            blks.append(dw_block(ch, se=se))
            ch *= 2
        self.blocks = nn.Sequential(*blks)
        self.head = nn.Linear(ch, n_classes)

    def forward(self, x):
        h = self.blocks(self.stem(x))
        return self.head(h.mean(dim=2))


class BasicBlock1D(nn.Module):
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
    def __init__(self, n_classes, widths=(32, 64, 128)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, widths[0], 17, stride=2, padding=8, bias=False),
            nn.BatchNorm1d(widths[0]))
        layers, ch = [], widths[0]
        for w in widths:
            layers.append(BasicBlock1D(ch, w, stride=2 if ch != widths[0]
                                       else 1))
            layers.append(BasicBlock1D(w, w))
            ch = w
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(ch, n_classes)

    def forward(self, x):
        h = F.relu(self.stem(x), inplace=True)
        h = F.max_pool1d(h, 3, stride=2, padding=1)
        h = self.body(h)
        return self.head(h.mean(dim=2))


def build_backbone(name, n_classes):
    if name == "light":
        return LightCNN(n_classes)
    if name == "se":
        return LightCNN(n_classes, se=True)
    if name == "wide":
        return LightCNN(n_classes, base=64, blocks=4, se=True)
    if name == "resnet":
        return ResNet1D(n_classes)
    raise ValueError(name)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


# --------------------------------------------------------------- GAN ----
class Gen1D(nn.Module):
    def __init__(self, latent=100, n_ch=64):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent, n_ch * 8 * 12),
                                nn.BatchNorm1d(n_ch * 8 * 12),
                                nn.ReLU(inplace=True))
        def up(i, o):
            return [nn.ConvTranspose1d(i, o, 4, 2, 1), nn.BatchNorm1d(o),
                    nn.ReLU(inplace=True)]
        self.deconv = nn.Sequential(*up(n_ch * 8, n_ch * 4),
                                    *up(n_ch * 4, n_ch * 2),
                                    *up(n_ch * 2, n_ch),
                                    nn.ConvTranspose1d(n_ch, 1, 4, 2, 1),
                                    nn.Tanh())

    def forward(self, z):
        return self.deconv(self.fc(z).view(z.size(0), -1, 12))


class Critic1D(nn.Module):
    def __init__(self, n_ch=64):
        super().__init__()
        def sn(i, o, k=4, s=2, p=1):
            return nn.utils.spectral_norm(nn.Conv1d(i, o, k, s, p))
        self.conv = nn.Sequential(
            sn(1, n_ch), nn.LeakyReLU(0.2, inplace=True),
            sn(n_ch, n_ch * 2), nn.LeakyReLU(0.2, inplace=True),
            sn(n_ch * 2, n_ch * 4), nn.LeakyReLU(0.2, inplace=True),
            sn(n_ch * 4, n_ch * 8), nn.LeakyReLU(0.2, inplace=True))
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
    def __init__(self, device, lr=2e-4):
        self.device = device
        self.gen = Gen1D().to(device)
        self.cri = Critic1D().to(device)
        self.opt_g = torch.optim.Adam(self.gen.parameters(), lr=lr,
                                      betas=(0.5, 0.999))
        self.opt_c = torch.optim.Adam(self.cri.parameters(), lr=lr,
                                      betas=(0.5, 0.999))

    def train_on(self, x, seed, epochs=GAN_EPOCHS):
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
    def score(self, x):
        xt = torch.as_tensor(x, dtype=torch.float32,
                             device=self.device).unsqueeze(1)
        out = []
        for s in range(0, len(x), 256):
            out.append(self.cri(xt[s:s + 256]).flatten().cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def generate(self, n):
        self.gen.eval()
        out = []
        for s in range(0, n, 256):
            z = torch.randn(min(256, n - s), 100, device=self.device)
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
def predict(model, x, device, bs=512):
    xt = torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(1)
    out = []
    for s in range(0, len(x), bs):
        out.append(model(xt[s:s + bs]).argmax(1).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)

    x_tr_all, y_tr_all, x_te, y_te = load_data()
    backbones = ["light", "se", "wide", "resnet"]
    params = {b: count_params(build_backbone(b, 5)) for b in backbones}
    print("params:", {b: f"{p/1e3:.0f}K" for b, p in params.items()})

    results = {"config": {"cap": args.cap, "seeds": args.seeds,
                          "backbones": params}, "per_seed": {},
               "summary": {}}
    stats = {b: {a: {"acc": [], "f1": [], "rec": []}
                 for a in ("no_aug", "wgan100", "realcopy")}
             for b in backbones}

    for seed in range(args.seeds):
        print(f"\n===== seed {seed} =====")
        rng = np.random.RandomState(1000 + seed)
        tr_i = []
        for c in range(5):
            idx = np.where(y_tr_all == c)[0]
            n = min(args.cap, len(idx))
            tr_i.append(idx[rng.permutation(len(idx))[:n]])
        tr_i = np.concatenate(tr_i)
        xs, ys = x_tr_all[tr_i], y_tr_all[tr_i]
        n_tr_cls = np.bincount(ys, minlength=5)

        # WGAN pools (shared by all backbones)
        pools = {}
        for c in range(5):
            n_c = int(n_tr_cls[c])
            if n_c < 40:
                continue
            xc = xs[ys == c]
            w = WGAN1D(device)
            g, cc = w.train_on(xc, seed=seed * 1000 + c)
            cand = w.generate(4000)
            scores = w.score(cand)
            kept = cand[scores >= np.quantile(scores, 0.10)]
            pools[c] = kept
            print(f"  [wgan] {CLASSES[c]}: kept {len(kept)} "
                  f"(G {g:.2f} C {cc:.2f})")

        sx, sy = [], []
        for c in range(5):
            n_c = int(n_tr_cls[c])
            if c not in pools or n_c < 40:
                continue
            sy.append(np.full(n_c, c, dtype=np.int64))
            sx.append(pools[c][:n_c])
        xa = np.concatenate([xs] + sx)
        ya = np.concatenate([ys] + sy)

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

        for b in backbones:
            line = f"  [{b}] "
            for tag, (xf, yf) in (("no_aug", (xs, ys)),
                                  ("wgan100", (xa, ya)),
                                  ("realcopy", (xr, yr))):
                m = build_backbone(b, 5).to(device)
                m = train_cnn(m, xf, yf, device, seed)
                p = predict(m, x_te, device)
                mm = classification_metrics(y_te, p, n_classes=5)
                stats[b][tag]["acc"].append(mm["accuracy"])
                stats[b][tag]["f1"].append(mm["f1"])
                stats[b][tag]["rec"].append(mm["recall"])
                line += (f"{tag} {mm['accuracy'] * 100:.2f}/"
                         f"{mm['f1'] * 100:.2f}  ")
            print(line)

        results["per_seed"][f"seed{seed}"] = {
            b: {a: {"acc": stats[b][a]["acc"][-1],
                    "macro_f1": stats[b][a]["f1"][-1]}
                for a in stats[b]} for b in backbones}

    for b in backbones:
        results["summary"][b] = {}
        for a in stats[b]:
            results["summary"][b][a] = {
                "acc_mean_std": [round(float(np.mean(stats[b][a]["acc"])),
                                       2),
                                 round(float(np.std(stats[b][a]["acc"],
                                                    ddof=1)), 2)],
                "macroF1_mean_std": [round(float(np.mean(stats[b][a]["f1"])),
                                           2),
                                     round(float(np.std(stats[b][a]["f1"],
                                                        ddof=1)), 2)]}
        d_acc = (np.array(stats[b]["wgan100"]["acc"])
                 - np.array(stats[b]["no_aug"]["acc"]))
        d_f1 = (np.array(stats[b]["wgan100"]["f1"])
                - np.array(stats[b]["no_aug"]["f1"]))
        results["summary"][b]["wgan_delta"] = {
            "acc": [round(float(np.mean(d_acc)), 2),
                    round(float(np.std(d_acc, ddof=1)), 2)],
            "macroF1": [round(float(np.mean(d_f1)), 2),
                        round(float(np.std(d_f1, ddof=1)), 2)]}

    print("\n===== summary (acc mean±std | macroF1) =====")
    for b in backbones:
        na = results["summary"][b]["no_aug"]
        wg = results["summary"][b]["wgan100"]
        dl = results["summary"][b]["wgan_delta"]
        print(f"  {b:<7} no_aug {na['acc_mean_std'][0]:.2f}±"
              f"{na['acc_mean_std'][1]:.2f} ({na['macroF1_mean_std'][0]:.2f})"
              f" | wgan100 {wg['acc_mean_std'][0]:.2f}±"
              f"{wg['acc_mean_std'][1]:.2f} ({wg['macroF1_mean_std'][0]:.2f})"
              f" | d_acc {dl['acc'][0]:+.2f}  d_F1 {dl['macroF1'][0]:+.2f}")
    save_json("results/mitbih_backbone_sweep.json", results)
    print("saved -> results/mitbih_backbone_sweep.json")


if __name__ == "__main__":
    main()
