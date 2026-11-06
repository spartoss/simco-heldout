"""Deep feed-forward networks (torch) on the measured tables, same splits / protocol as run_cv.py.

    python ml_dos/run_dnn.py --arm qtl --model dnn --n-jobs 16      # venv-cellot (torch 1.11)

Per outer split, on the training women only:
  impute (median) -> drop constant -> StandardScaler -> TopK(k) ; y standardised.
  Grid (fixed a priori): k in {500, all} x architecture {[256, 64], [512, 256, 128, 64]}
  x dropout {0.2, 0.5} x weight decay {1e-2, 1e-1}.
  Inner GroupKFold(5) over women: each config is trained on 4/5 and its validation MSE recorded every
  10 epochs; the config and epoch count with the lowest mean validation MSE win. Then 5 seeds are
  refitted on all training women for that many epochs and their predictions averaged.
Training: AdamW lr 1e-3, full batch, Gaussian input noise 0.1, input-feature dropout 0.1, GELU,
LayerNorm, max 400 epochs.

``--model dnn_wide`` uses one wider, shallower family ([1024], [1024, 256]) instead -- the
"is depth or width the issue" check.
"""
import argparse
import json
import os
import time
import warnings
from itertools import product

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import common as C
from models import TopK

MAXEP, EVERY, SEEDS = 400, 10, 5
FAMILIES = {
    "dnn": {"k": [500, "all"], "arch": [(256, 64), (512, 256, 128, 64)], "drop": [0.2, 0.5], "wd": [1e-2, 1e-1]},
    "dnn_wide": {"k": [500, "all"], "arch": [(1024,), (1024, 256)], "drop": [0.2, 0.5], "wd": [1e-2, 1e-1]},
}


class Net(torch.nn.Module):
    def __init__(self, p, arch, drop):
        super().__init__()
        L, d = [torch.nn.Dropout(0.1)], p
        for h in arch:
            L += [torch.nn.Linear(d, h), torch.nn.LayerNorm(h), torch.nn.GELU(), torch.nn.Dropout(drop)]
            d = h
        L.append(torch.nn.Linear(d, 1))
        self.f = torch.nn.Sequential(*L)

    def forward(self, x):
        return self.f(x).squeeze(-1)


def prep(Xtr, Xte, ytr, k):
    steps = [SimpleImputer(strategy="median"), VarianceThreshold(1e-10), StandardScaler()]
    for s in steps:
        Xtr = s.fit_transform(Xtr); Xte = s.transform(Xte)
    if k != "all":
        t = TopK(k).fit(Xtr, ytr); Xtr, Xte = t.transform(Xtr), t.transform(Xte)
    mu, sd = ytr.mean(), ytr.std()
    return (torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xte, dtype=torch.float32),
            torch.tensor((ytr - mu) / sd, dtype=torch.float32), mu, sd)


def train(Xtr, ytr, cfg, seed, epochs, Xval=None, yval=None):
    torch.manual_seed(seed)
    net = Net(Xtr.shape[1], cfg["arch"], cfg["drop"])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=cfg["wd"])
    curve = []
    for ep in range(1, epochs + 1):
        net.train()
        opt.zero_grad()
        loss = torch.mean((net(Xtr + 0.1 * torch.randn_like(Xtr)) - ytr) ** 2)
        loss.backward(); opt.step()
        if Xval is not None and ep % EVERY == 0:
            net.eval()
            with torch.no_grad():
                curve.append(float(torch.mean((net(Xval) - yval) ** 2)))
    return net, curve


def one_split(k, tr, te, X, y, g, model, outdir):
    f = outdir / f"split_{k:03d}.csv"
    if f.exists():
        return
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    t0 = time.time()
    Xtr, Xte, ytr = X.values[tr], X.values[te], y.values[tr]
    fam = FAMILIES[model]
    cfgs = [dict(zip(fam, v)) for v in product(*fam.values())]
    inner = list(GroupKFold(5).split(Xtr, ytr, g[tr]))
    curves = np.zeros((len(cfgs), MAXEP // EVERY))
    for i, cfg in enumerate(cfgs):
        for a, b in inner:
            A, B, ya, mu, sd = prep(Xtr[a], Xtr[b], ytr[a], cfg["k"])
            yb = torch.tensor((ytr[b] - mu) / sd, dtype=torch.float32)
            _, cv = train(A, ya, cfg, 0, MAXEP, B, yb)
            curves[i] += np.array(cv) / len(inner)
    i, e = np.unravel_index(np.argmin(curves), curves.shape)
    cfg, epochs = cfgs[i], (e + 1) * EVERY
    A, B, ya, mu, sd = prep(Xtr, Xte, ytr, cfg["k"])
    preds = []
    for s in range(SEEDS):
        net, _ = train(A, ya, cfg, s, epochs)
        net.eval()
        with torch.no_grad():
            preds.append(net(B).numpy() * sd + mu)
    pred = np.mean(preds, 0)
    info = {"best_params": {**{kk: str(vv) for kk, vv in cfg.items()}, "epochs": int(epochs)},
            "inner_rmse": float(np.sqrt(curves[i, e])) * float(sd), "seconds": time.time() - t0}
    json.dump(info, open(outdir / f"split_{k:03d}.json", "w"))
    pd.DataFrame({"sample": X.index[te], "y": y.values[te], "pred": pred, "split": k}
                 ).to_csv(f.with_suffix(".tmp"), index=False)
    os.replace(f.with_suffix(".tmp"), f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--model", default="dnn", choices=list(FAMILIES))
    ap.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--splits", default="0-99")
    a = ap.parse_args()
    lo, hi = map(int, a.splits.split("-"))
    X, y, g = C.load_arm(a.arm)
    sp = C.splits(C.rows())
    outdir = C.OUT / "preds" / a.arm / a.model
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO ] {a.arm}/{a.model}: X {X.shape}, splits {lo}-{hi}", flush=True)
    Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(one_split)(k, *sp[k], X, y, g, a.model, outdir)
                                         for k in range(lo, hi + 1))
    pl = pd.concat([pd.read_csv(f) for f in sorted(outdir.glob("split_*.csv"))])
    sc, _ = C.pooled(pl)
    print(f"[DONE ] {a.arm}/{a.model}: {pl.split.nunique()} splits, pooled R2 {sc['R2']:.3f} "
          f"median split R2 {C.per_split(pl).median():.3f}", flush=True)


if __name__ == "__main__":
    main()
