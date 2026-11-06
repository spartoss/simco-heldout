"""Calibrate the paired CellOT map's between-donor slope (2026-10-13, k = 15, closed form, no test truth used).

Per repeat and feature f = (stim, celltype, q, marker):
  a_f   = the map's slope across the repeat's test donors: OLS of its prediction on the donor's Unstim quantile u
          (predictions and inputs only)
  gam_f = within-donor widening on the TRAINING donors: (stim q+5 - q-5) / (Unstim q+5 - q-5), median over donors
  b_f   = OLS slope, across the 15 training donors, of their stim quantile on their Unstim quantile (noisy per feature)
  g     = E[b | gam] fitted across all features of the repeat (robust_g: binned medians, non-decreasing) = the slope a feature with this
          widening has between training donors, smoothed across features
  calib     pred = paired - (a_f - g(gam_f)) (u - ubar)   only the linear-in-u part changes; mean over test donors kept
  force1    pred = paired - (a_f - 1) (u - ubar)
  + 0.5 / 0.5 mixes with the shift.
Writes WORK/slope_calib/cases.pkl, reuses WORK/dnorm_slope/cases.pkl (paired, shift, truth).
    python -m lc.slope_calib
"""
import json

import numpy as np
import pandas as pd

from lc.dnorm_slope import OUT as DSO
from lc.paths import CACHE, MARKERS, PAIRS, QS, WORK

OUT = WORK / "slope_calib"
F = ["stim", "celltype", "q", "marker"]


def gammas(order, k=15):
    rows = []
    for c in sorted({c for _, c in PAIRS}):
        z = np.load(CACHE / f"{c}.npz")
        for s in sorted({s for s, cc in PAIRS if cc == c}):
            ds = [d for d in order[:k] if f"Unstim|{d}" in z.files and f"{s}|{d}" in z.files]
            for q in QS:
                lo, hi = max(q - 5, 1), min(q + 5, 99)
                g = []
                for d in ds:
                    du = np.percentile(z[f"Unstim|{d}"], hi, 0) - np.percentile(z[f"Unstim|{d}"], lo, 0)
                    dv = np.percentile(z[f"{s}|{d}"], hi, 0) - np.percentile(z[f"{s}|{d}"], lo, 0)
                    g.append(np.where(du > 1e-6, dv / np.maximum(du, 1e-6), np.nan))
                for j, m in enumerate(MARKERS):
                    rows.append((s, c, q, m, np.nanmedian([x[j] for x in g]) if g else np.nan))
    return pd.DataFrame(rows, columns=F + ["gam"])


def train_slopes(order, k=15):
    from lc.score import truth_tables
    from lc.paths import DONORS
    T = truth_tables(sorted({c for _, c in PAIRS}), set(DONORS))
    rows = []
    for s, c in PAIRS:
        ds = [d for d in order[:k] if (c, s, d) in T and (c, "Unstim", d) in T]
        if len(ds) < 4: continue
        U = np.stack([T[(c, "Unstim", d)][0] for d in ds]); S = np.stack([T[(c, s, d)][0] for d in ds])
        Uc = U - U.mean(0); b = (Uc * (S - S.mean(0))).sum(0) / np.maximum((Uc ** 2).sum(0), 1e-12)
        for qi, q in enumerate(QS):
            for j, m in enumerate(MARKERS):
                rows.append((s, c, q, m, b[qi, j], (Uc[:, qi, j] ** 2).sum()))
    return pd.DataFrame(rows, columns=F + ["b", "varu"])


def robust_g(gam, b, nbin=20):
    """E[b | gam] robustly: median b in nbin quantile bins of gam, made non-decreasing, linearly interpolated."""
    edges = np.unique(np.quantile(gam, np.linspace(0, 1, nbin + 1)))
    idx = np.clip(np.searchsorted(edges, gam, side="right") - 1, 0, len(edges) - 2)
    xs = np.array([np.median(gam[idx == i]) for i in range(len(edges) - 1) if (idx == i).any()])
    ys = np.maximum.accumulate([np.median(b[idx == i]) for i in range(len(edges) - 1) if (idx == i).any()])
    return np.interp(gam, xs, ys)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D = pd.read_pickle(DSO / "cases.pkl")[["repeat", "stim", "celltype", "donor", "q", "marker", "u", "t", "mu_s", "shift", "paired", "ens"]]
    out = []
    for dz in json.load(open(WORK / "design.json")):
        r = dz["repeat"]; X = D[D.repeat == r].copy()
        Gm = gammas(dz["order"]).merge(train_slopes(dz["order"]), on=F, how="inner").dropna()
        Gm = Gm[Gm.varu > 1e-8]
        Gm["g"] = robust_g(Gm.gam.values, Gm.b.values)
        g = [X[c] for c in F]
        X["uc"] = X.u - X.u.groupby(g).transform("mean"); X["pc"] = X.paired - X.paired.groupby(g).transform("mean")
        A = X.groupby(F).apply(lambda d: pd.Series({"a": (d.pc * d.uc).sum() / (d.uc ** 2).sum() if (d.uc ** 2).sum() > 1e-8 else np.nan}))
        X = X.join(A, on=F).merge(Gm[F + ["gam", "g", "b"]], on=F, how="left")
        X["a"] = X.a.fillna(1.0); X["g"] = X.g.fillna(X.a)
        X["calib"] = X.paired - (X.a - X.g) * X.uc
        X["force1"] = X.paired - (X.a - 1) * X.uc
        X["calib_ens"] = 0.5 * X.calib + 0.5 * X["shift"]
        X["force1_ens"] = 0.5 * X.force1 + 0.5 * X["shift"]
        bins = pd.cut(Gm.gam, [0, 0.8, 1.2, 1.5, 2, 3, np.inf])
        print(f"repeat {r}: g(gam) by gam bin", Gm.groupby(bins, observed=True).agg(gam=("gam", "median"), b=("b", "median"), g=("g", "median")).round(2).to_dict("index"), flush=True)
        out.append(X)
    D = pd.concat(out, ignore_index=True); D.to_pickle(OUT / "cases.pkl"); print(D.shape, "->", OUT)


if __name__ == "__main__":
    main()
