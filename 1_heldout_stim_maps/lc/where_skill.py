"""Per-case table for "where is CellOT good on new donors?" (stim maps, learning-curve design, held-out donors only).

For each repeat, map (stim, celltype, k), test donor, quantile q, marker:
    u      = test donor's Unstim quantile            t     = her stimulated quantile (truth)
    mu_u   = mean Unstim quantile of the k training donors, mu_s the same for the stimulated quantile
    shift  = mu_s + (u - mu_u): every donor gets the training-mean response on top of her own Unstim value
    noise  = |q of first half - q of second half| of her stimulated cells (split-half; var(full) ~ noise^2 / 4);
             noise_u the same for her Unstim cells
    sd_u   = SD of the training donors' Unstim quantile (to place the test donor: z_u = (u - mu_u) / sd_u)
    sd_r   = SD of the training donors' response (s - u) (how much donors differ in response in training)
    n_u, n_s = cells of the test donor (Unstim, stim)
Predictions: CellOT (the learning-curve run, work/lc/train, paper architecture), CellOT_none (cond_train, paired
context-free, repeats with it), gauss_ot, quantile_map (work/lc/baselines), shift.
Output WORK/where_skill/cases.pkl (all cases, no effect filter; the notebook filters).
--extra name=dir[:dnorm] adds more runs (dir relative to WORK, {r} = repeat). ":dnorm" = trained on the donor-normalised
cache (lc/dnorm_cache.py): the test donor's Unstim median offset is added back to the predicted quantiles.
    python -m lc.where_skill [--ks 15] [--extra dnorm=cond_train/repeat_{r}/dnorm:dnorm ...] [--out cases_v2.pkl]
"""
import argparse
import json

import numpy as np
import pandas as pd

from lc.cond_train_score import load_npz, load_train
from lc.paths import CACHE, DONORS, MARKERS, PAIRS, QS, WORK
from lc.score import truth_tables

KS = [3, 5, 8, 15]
OUT = WORK / "where_skill"


def counts():
    n = {}
    for c in sorted({c for _, c in PAIRS}):
        z = np.load(CACHE / f"{c}.npz")
        for key in z.files:
            cond, d = key.split("|"); n[(c, cond, d)] = len(z[key])
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", nargs="*", type=int, default=KS)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--out", default="cases.pkl")
    ap.add_argument("--npz", nargs="*", default=[], help="name=dir:key, closed-form predictions in <dir>/repeat_<r>.npz")
    a = ap.parse_args()
    extra = {}
    for e in a.extra:
        n, rest = e.split("=", 1); d, _, flag = rest.partition(":"); extra[n] = (d, flag == "dnorm")
    off = np.load(WORK / "cache_dnorm/offsets.npz") if any(v[1] for v in extra.values()) else None
    OUT.mkdir(parents=True, exist_ok=True)
    T = truth_tables(sorted({c for _, c in PAIRS}), set(DONORS)); N = counts()
    rows = []
    for dz in json.load(open(WORK / "design.json")):
        r, tests = dz["repeat"], dz["test"]
        maps = [(s, c, k) for s, c in PAIRS for k in a.ks]
        P = {"CellOT": load_train(WORK / f"train/repeat_{r}", maps)}
        nd = WORK / f"cond_train/repeat_{r}/none"
        if (nd / "pred_q.npy").exists(): P["CellOT_none"] = load_train(nd, maps)
        P.update(load_npz(WORK / f"baselines/repeat_{r}.npz", ["gauss_ot", "quantile_map"], maps))
        for e in a.npz:
            n, rest = e.split("=", 1); d, key = rest.rsplit(":", 1)
            P[n] = load_npz(WORK / d / f"repeat_{r}.npz", [key], maps)[key]
        DN = set()
        for n, (d, dn) in extra.items():
            ed = WORK / d.format(r=r)
            if (ed / "pred_q.npy").exists():
                P[n] = load_train(ed, maps)
                if dn: DN.add(n)
            else: print(f"[WARN ] {n}: no pred_q.npy in {ed}", flush=True)
        for (s, c, k) in maps:
            trd = [d for d in dz["order"][:k] if (c, s, d) in T and (c, "Unstim", d) in T]
            if len(trd) < 2: continue
            U = np.stack([T[(c, "Unstim", d)][0] for d in trd]); S = np.stack([T[(c, s, d)][0] for d in trd])
            mu_u, mu_s = U.mean(0), S.mean(0); sd_u = U.std(0, ddof=1); sd_r = (S - U).std(0, ddof=1)
            for j, d in enumerate(tests):
                if (c, s, d) not in T or (c, "Unstim", d) not in T: continue
                (u, nzu), (t, nz) = T[(c, "Unstim", d)], T[(c, s, d)]
                row = {"repeat": r, "stim": s, "celltype": c, "k": k, "donor": d,
                       "q": np.repeat(QS, len(MARKERS)), "marker": np.tile(MARKERS, len(QS)),
                       "u": u.ravel(), "t": t.ravel(), "noise": nz.ravel(), "noise_u": nzu.ravel(), "mu_u": mu_u.ravel(), "mu_s": mu_s.ravel(),
                       "sd_u": sd_u.ravel(), "sd_r": sd_r.ravel(), "n_u": N[(c, "Unstim", d)], "n_s": N[(c, s, d)],
                       "shift": (mu_s + u - mu_u).ravel()}
                for n, D in P.items():
                    p = D.get((s, c, k))
                    if p is None: row[n] = np.full(u.size, np.nan); continue
                    p = np.asarray(p[j], float)
                    if n in DN: p = p + off[f"{c}|{d}"][None, :]          # back from the donor-normalised space
                    row[n] = p.ravel()
                rows.append(pd.DataFrame(row))
        print(f"repeat {r} done", flush=True)
    D = pd.concat(rows, ignore_index=True)
    for col in D.select_dtypes("float64"): D[col] = D[col].astype("float32")
    for n in list(extra) + [e.split("=", 1)[0] for e in a.npz]:
        if n not in D: D[n] = np.nan
    D.to_pickle(OUT / a.out)
    print(D.shape, "->", OUT / a.out)


if __name__ == "__main__":
    main()
