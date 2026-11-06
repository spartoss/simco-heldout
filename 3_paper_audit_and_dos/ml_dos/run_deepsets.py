"""DeepSets on the raw Unstim cells: the "no hand-made summary" deep model.

    python ml_dos/run_deepsets.py prepare --n-jobs 6            # once: cells -> cells.npz
    python ml_dos/run_deepsets.py fit --n-jobs 16 [--splits 0-99]

Input per sample: up to NC=1024 random gated Unstim cells per celltype (31 markers, arcsinh as stored),
a mask for celltypes with fewer cells, and the 28 log frequencies. The model:
  phi(cell) = MLP(31 -> 64 -> 64) + celltype embedding (shared weights across celltypes)
  per celltype: [mean, sd] of phi over its cells  -> 28 x 128
  rho = Dropout(0.3) -> Linear(28*128 + 28 -> 128) -> GELU -> Dropout(0.3) -> Linear(-> 1)
Each epoch draws a fresh 128 cells per celltype (augmentation); prediction uses all stored cells.
Markers are standardised with training-women statistics; y standardised. AdamW lr 1e-3, wd 1e-2,
batch 16, max 150 epochs. Hyperparameters are fixed a priori (no grid): the epoch count is chosen on
an inner GroupKFold(3) over training women (mean validation MSE every 5 epochs), then 3 seeds are
refitted on all training women and averaged. Same 100 splits / exclusions as everything else.
"""
import argparse
import json
import os
import time
import warnings

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

import common as C
from build_features import CT, MARKERS, load, read_list

NC, DRAW, MAXEP, EVERY, SEEDS = 1024, 128, 150, 5, 3
CELLS = C.OUT / "cells.npz"


def prep_one(patient, markers, seed):
    X, ct = load(patient, markers)
    rng = np.random.default_rng(seed)
    out = np.zeros((len(CT), NC, len(markers)), np.float16)
    n = np.zeros(len(CT), np.int32)
    freq = np.zeros(len(CT), np.float32)
    for i, lab in enumerate(CT.values()):
        idx = np.flatnonzero(ct == lab)
        freq[i] = np.log10(len(idx) / len(ct) + 1e-4)
        if len(idx) > NC:
            idx = rng.choice(idx, NC, replace=False)
        out[i, :len(idx)] = X[idx]
        n[i] = len(idx)
    return out, n, freq


def prepare(n_jobs):
    idx = C.rows()
    markers = read_list(MARKERS)
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(prep_one)(p, markers, s) for s, p in enumerate(idx))
    np.savez(CELLS, cells=np.stack([r[0] for r in res]), n=np.stack([r[1] for r in res]),
             freq=np.stack([r[2] for r in res]), samples=np.array(idx))
    print(f"[DONE ] {CELLS}: {np.stack([r[0] for r in res]).shape}")


def one_split(k, tr, te, D, y, g, outdir):
    import torch
    f = outdir / f"split_{k:03d}.csv"
    if f.exists():
        return
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    t0 = time.time()
    cells, n, freq = D["cells"], D["n"], D["freq"]
    nct, _, nm = cells.shape[1:]

    class DS(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.phi = torch.nn.Sequential(torch.nn.Linear(nm, 64), torch.nn.GELU(), torch.nn.Linear(64, 64),
                                           torch.nn.GELU())
            self.emb = torch.nn.Parameter(torch.zeros(nct, 64))
            self.rho = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(nct * 128 + nct, 128),
                                           torch.nn.GELU(), torch.nn.Dropout(0.3), torch.nn.Linear(128, 1))

        def forward(self, x, m, fr):            # x (B, T, N, M), m (B, T, N) float mask, fr (B, T)
            h = self.phi(x) + self.emb[None, :, None, :]
            w = m.unsqueeze(-1)
            cnt = w.sum(2).clamp(min=1)
            mu = (h * w).sum(2) / cnt
            sd = (((h - mu.unsqueeze(2)) ** 2 * w).sum(2) / cnt).clamp(min=1e-6).sqrt()
            z = torch.cat([mu, sd], -1).flatten(1)
            return self.rho(torch.cat([z, fr], 1)).squeeze(-1)

    def stats(rows):
        v = []
        for r in rows:
            for t in range(nct):
                if n[r, t]:
                    v.append(cells[r, t, :n[r, t]].astype(np.float32))
        v = np.concatenate(v)
        return v.mean(0), v.std(0) + 1e-3

    def batch(rows, mu, sd, rng, draw):
        B = len(rows)
        N = draw or NC
        x = np.zeros((B, nct, N, nm), np.float32)
        m = np.zeros((B, nct, N), np.float32)
        for b, r in enumerate(rows):
            for t in range(nct):
                c = n[r, t]
                if not c:
                    continue
                sel = rng.choice(c, min(c, N), replace=False) if draw else np.arange(c)
                x[b, t, :len(sel)] = (cells[r, t, sel].astype(np.float32) - mu) / sd
                m[b, t, :len(sel)] = 1
        fr = (freq[rows] - fmu) / fsd
        return torch.tensor(x), torch.tensor(m), torch.tensor(fr, dtype=torch.float32)

    def fit(rows, yy, seed, epochs, val=None):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = DS()
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        curve = []
        for ep in range(1, epochs + 1):
            net.train()
            for b in np.array_split(rng.permutation(len(rows)), max(1, len(rows) // 16)):
                x, m, fr = batch(rows[b], mu, sd, rng, DRAW)
                opt.zero_grad()
                loss = torch.mean((net(x, m, fr) - torch.tensor(yy[b], dtype=torch.float32)) ** 2)
                loss.backward(); opt.step()
            if val is not None and ep % EVERY == 0:
                curve.append(float(np.mean((predict(net, val[0]) - val[1]) ** 2)))
        return net, curve

    def predict(net, rows):
        net.eval()
        out = []
        with torch.no_grad():
            for b in np.array_split(np.arange(len(rows)), max(1, len(rows) // 8)):
                x, m, fr = batch(rows[b], mu, sd, None, None)
                out.append(net(x, m, fr).numpy())
        return np.concatenate(out)

    ytr = y.values[tr]
    ymu, ysd = ytr.mean(), ytr.std()
    curves = []
    for a, b in GroupKFold(3).split(tr, ytr, g[tr]):
        mu, sd = stats(tr[a])
        fmu, fsd = freq[tr[a]].mean(0), freq[tr[a]].std(0) + 1e-3
        _, cv = fit(tr[a], (ytr[a] - ymu) / ysd, 0, MAXEP, (tr[b], (ytr[b] - ymu) / ysd))
        curves.append(cv)
    curve = np.mean(curves, 0)
    epochs = (int(np.argmin(curve)) + 1) * EVERY
    mu, sd = stats(tr)
    fmu, fsd = freq[tr].mean(0), freq[tr].std(0) + 1e-3
    pred = np.mean([predict(fit(tr, (ytr - ymu) / ysd, s, epochs)[0], te) for s in range(SEEDS)], 0) * ysd + ymu
    json.dump({"best_params": {"epochs": epochs}, "inner_rmse": float(np.sqrt(curve.min())) * float(ysd),
               "seconds": time.time() - t0}, open(outdir / f"split_{k:03d}.json", "w"))
    pd.DataFrame({"sample": y.index[te], "y": y.values[te], "pred": pred, "split": k}
                 ).to_csv(f.with_suffix(".tmp"), index=False)
    os.replace(f.with_suffix(".tmp"), f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "fit"])
    ap.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--splits", default="0-99")
    a = ap.parse_args()
    if a.cmd == "prepare":
        return prepare(a.n_jobs)
    lo, hi = map(int, a.splits.split("-"))
    idx = C.rows()
    D = dict(np.load(CELLS, allow_pickle=True))
    assert (D["samples"] == np.array(idx)).all()
    y, g = C.outcome(idx), C.groups(idx)
    outdir = C.OUT / "preds" / "cells" / "deepsets"
    outdir.mkdir(parents=True, exist_ok=True)
    Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(one_split)(k, *C.splits(idx)[k], D, y, g, outdir)
                                         for k in range(lo, hi + 1))
    pl = pd.concat([pd.read_csv(f) for f in sorted(outdir.glob("split_*.csv"))])
    sc, _ = C.pooled(pl)
    print(f"[DONE ] cells/deepsets: {pl.split.nunique()} splits, pooled R2 {sc['R2']:.3f} "
          f"median split R2 {C.per_split(pl).median():.3f}", flush=True)


if __name__ == "__main__":
    main()
