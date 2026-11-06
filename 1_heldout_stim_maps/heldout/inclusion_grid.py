"""How many stim features pass under different inclusion criteria (2026-10-12, user question).

Feature = (stim, celltype, marker); stims without PI / IL33 by default. Per donor and quantile q in (50, 75, 90):
effect = |q(stim cells) - q(Unstim cells)| (cache_stim_all: DMSO wells pooled, uncorrected, <= 10k cells, as
heldout/inclusion.py). Criteria = statistic x donor aggregation x threshold:
    statistic : "q50" (median only) or "any q" (any of q50 / q75 / q90 passes)
    aggregation: mean over donors > thr (the current rule, heldout/inclusion.py), median over donors > thr, or at least
                 n donors with effect > thr (n = 1, 3, 5, 10, all available donors)
Null (no real effect): the same donor's Unstim cells split in two random halves, |q(half A) - q(half B)| / sqrt(2)
(var of a half-sample quantile difference = 4 var(q), of a stim - Unstim difference ~ 2 var(q)); the null of
(celltype, marker) is used for every stim, so "null features" = how many features the criterion passes by noise alone.
Outputs WORK/features/inclusion_grid/{counts.csv, by_stim.csv, features.csv (per-feature statistics)}.
    python -m heldout.inclusion_grid [--exclude-stims PI IL33]
"""
import argparse

import numpy as np
import pandas as pd

from heldout.common import ALL_PAIRS, CACHE_STIM_ALL, WORK
from lc.paths import MARKERS, PAIRS

QS = (50, 75, 90)
NS = (1, 3, 5, 10)
THRS = (0.05, 0.1, 0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude-stims", nargs="*", default=["PI", "IL33"])
    a = ap.parse_args()
    out = WORK / "features" / "inclusion_grid"; out.mkdir(parents=True, exist_ok=True)
    pairs = [(s, c) for s, c in ALL_PAIRS if s not in a.exclude_stims]
    feats, eff, nul = [], [], []                        # eff / nul: list of (n_donors, 3, 13) arrays per pair
    for ct in sorted({c for _, c in pairs}):
        z = np.load(CACHE_STIM_ALL / f"{ct}.npz")
        donors = sorted({k.split("|")[1] for k in z.files if k.startswith("Unstim|")})
        U = {d: np.percentile(z[f"Unstim|{d}"], QS, 0) for d in donors}
        N = {}
        for d in donors:
            x = z[f"Unstim|{d}"]; h = len(x) // 2
            N[d] = np.abs(np.percentile(x[:h], QS, 0) - np.percentile(x[h:], QS, 0)) / np.sqrt(2)
        for st in sorted({s for s, c in pairs if c == ct}):
            ds = [d for d in donors if f"{st}|{d}" in z.files]
            eff.append(np.stack([np.abs(np.percentile(z[f"{st}|{d}"], QS, 0) - U[d]) for d in ds]))
            nul.append(np.stack([N[d] for d in ds]))
            feats.append((st, ct, len(ds)))
        print(ct, flush=True)
    grid = {(s, c) for s, c in PAIRS}
    rows, frow = [], []
    for (st, ct, nd), E, Nl in zip(feats, eff, nul):
        for j, m in enumerate(MARKERS):
            r = {"stim": st, "celltype": ct, "marker": m, "n_donors": nd, "in_map_grid": (st, ct) in grid}
            for qi, q in enumerate(QS):
                r[f"mean_q{q}"] = E[:, qi, j].mean(); r[f"median_q{q}"] = np.median(E[:, qi, j])
                r[f"max_q{q}"] = E[:, qi, j].max()
            frow.append(r)
            for src, A in (("real", E), ("null", Nl)):
                x = A[:, :, j]                          # (donors, 3)
                for thr in THRS:
                    for stat, cols in (("q50", [0]), ("any q", [0, 1, 2])):
                        v = x[:, cols]
                        res = {"mean > thr": (v.mean(0) > thr).any(), "median > thr": (np.median(v, 0) > thr).any(),
                               **{f">= {n} donors": ((v > thr).sum(0) >= n).any() for n in NS},
                               "all donors": ((v > thr).sum(0) == len(v)).any()}
                        for agg, ok in res.items():
                            rows.append((src, thr, stat, agg, st, ct, m, (st, ct) in grid, bool(ok)))
    R = pd.DataFrame(rows, columns=["src", "thr", "stat", "agg", "stim", "celltype", "marker", "in_map_grid", "ok"])
    pd.DataFrame(frow).to_csv(out / "features.csv", index=False)
    aggs = ["mean > thr", "median > thr"] + [f">= {n} donors" for n in NS] + ["all donors"]
    C = R.groupby(["thr", "stat", "agg", "src"]).ok.sum().unstack("src").reset_index()
    C["null_share_%"] = (100 * C.null / C.real.clip(lower=1)).round(1)
    C["agg"] = pd.Categorical(C["agg"], aggs); C = C.sort_values(["thr", "stat", "agg"])
    C.to_csv(out / "counts.csv", index=False)
    B = R[R.src == "real"].groupby(["thr", "stat", "agg", "stim"]).ok.sum().unstack("stim").reset_index()
    B["agg"] = pd.Categorical(B["agg"], aggs); B = B.sort_values(["thr", "stat", "agg"]); B.to_csv(out / "by_stim.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"features: {len(frow)} ((stim, celltype, marker), stims {sorted({s for s, _ in pairs})}); in the map grid: "
          f"{sum(r['in_map_grid'] for r in frow)}")
    print(C.to_string(index=False))
    print(B[B.thr == 0.1].to_string(index=False))


if __name__ == "__main__":
    main()
