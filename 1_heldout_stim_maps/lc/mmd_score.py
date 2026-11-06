"""Cell-level MMD of held-out-donor predictions (2026-10-12): the CellOT paper's metric, on the learning-curve design.

For each repeat, k = 15 map (stim, celltype) and test donor: MMD between the donor's TRUE stimulated cells and
    identity : her Unstim cells
    shift    : her Unstim cells + the training donors' mean (median stim - median Unstim), per marker
    <arm>    : her Unstim cells transported by the arm's saved network (final_maps.pt: g_best = val-best checkpoint,
               g_last = last iterate). ":dnorm" arms: cells minus her Unstim median offset, transported, offset added back.
Checkpoint averages exist only as quantiles (no intermediate weights saved), so they have no MMD here.
MMD = cellot/losses/mmd.py compute_scalar_mmd (RBF kernels, gammas 2, 1, 0.5, 0.1, 0.01, 0.005, averaged; biased
estimator), on the first --n-cells cells of each block (the cache is shuffled: a random subset).
Also per case: |median error| averaged over markers, for reference.
    python -m lc.mmd_score --arms unpaired=train/repeat_{r} none=cond_train/repeat_{r}/none \
        dnorm=cond_train/repeat_{r}/dnorm:dnorm --exclude-stims PI IL33 --out WORK/mmd
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lc.batched_icnn import BatchedICNN
from lc.paths import CACHE, PAIRS, WORK

GAMMAS = (2, 1, 0.5, 0.1, 0.01, 0.005)


def kmean(D2):
    return torch.stack([torch.exp(-g * D2).mean() for g in GAMMAS])


def single(sd, key, i):
    net = BatchedICNN(1, 13, fnorm_penalty=1)
    net.load_state_dict({k: v[i:i + 1] for k, v in sd[key].items() if k.split(".")[0] in ("W", "A", "Ab")})
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--repeats", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--exclude-stims", nargs="*", default=[])
    ap.add_argument("--n-cells", type=int, default=1000)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    arms = {}
    for s in a.arms:
        n, d = s.split("=", 1); arms[n] = (d.removesuffix(":dnorm"), d.endswith(":dnorm"))
    off = np.load(WORK / "cache_dnorm/offsets.npz")
    design = json.load(open(WORK / "design.json"))
    N = a.n_cells; rows = []; t0 = time.time()
    for dz in design:
        r = dz["repeat"]
        if r not in a.repeats: continue
        tests, trd_all = dz["test"], dz["order"][:a.k]
        nets = {}
        for n, (d, dn) in arms.items():
            dd = WORK / d.format(r=r); cm = pd.read_csv(dd / "maps.csv")
            sel = (cm.k == a.k) & ((cm.cap == 0) if "cap" in cm else True)
            nets[n] = (torch.load(dd / "final_maps.pt", map_location="cpu"),
                       {(s, c): i for i, s, c in zip(cm.index[sel], cm.stim[sel], cm.celltype[sel])}, dn)
        for c in sorted({c for s, c in PAIRS if s not in a.exclude_stims}):
            z = np.load(CACHE / f"{c}.npz")
            for s in sorted({s for s, cc in PAIRS if cc == c and s not in a.exclude_stims}):
                trd = [d for d in trd_all if f"Unstim|{d}" in z.files and f"{s}|{d}" in z.files]
                if len(trd) < 2: continue
                shift = np.mean([np.median(z[f"{s}|{d}"], 0) - np.median(z[f"Unstim|{d}"], 0) for d in trd], 0)
                for d in tests:
                    if f"Unstim|{d}" not in z.files or f"{s}|{d}" not in z.files: continue
                    U, S = z[f"Unstim|{d}"], z[f"{s}|{d}"]
                    if len(U) < 20 or len(S) < 20: continue
                    x = torch.from_numpy(U[:N]).float(); y = torch.from_numpy(S[:N]).float()
                    preds = {"identity": x, "shift": x + torch.from_numpy(shift).float()}
                    for n, (sd, pos, dn) in nets.items():
                        if (s, c) not in pos: continue
                        o = torch.from_numpy(off[f"{c}|{d}"]).float() if dn else 0
                        for key, lab in (("g_best", "vb"), ("g_last", "last")):
                            g = single(sd, key, pos[(s, c)])
                            xi = (x - o).unsqueeze(0).requires_grad_(True)
                            preds[f"{n} ({lab})"] = g.transport(xi).detach()[0] + o
                    kyy = kmean(torch.cdist(y, y) ** 2)
                    for n, p in preds.items():
                        mmd = (kmean(torch.cdist(p, p) ** 2) + kyy - 2 * kmean(torch.cdist(p, y) ** 2)).mean().item()
                        rows.append({"repeat": r, "stim": s, "celltype": c, "donor": d, "method": n, "mmd": mmd,
                                     "med_err": (p.median(0).values - torch.from_numpy(np.median(S, 0)).float()).abs().mean().item(),
                                     "effect": float(np.abs(np.median(S, 0) - np.median(U, 0)).mean())})
            print(f"repeat {r} {c} done, {(time.time() - t0) / 60:.1f} min", flush=True)
    D = pd.DataFrame(rows); D.to_csv(out / "cases.csv", index=False)
    print(D.groupby("method")[["mmd", "med_err"]].agg(["mean", "median"]).round(5).to_string())


if __name__ == "__main__":
    main()
