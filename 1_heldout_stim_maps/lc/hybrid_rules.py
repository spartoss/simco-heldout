"""Hybrid stim prediction: CellOT where it is trustworthy, the shift elsewhere, decided WITHOUT the test donors' truth
(2026-10-07). For Stabl the shift is the donor's own Unstim feature (+ a constant), so "use the shift" means "keep the
measured feature instead of a simulated one".

Per repeat (5 x 4 held-out donors, maps trained on k = 15 donors), feature f = (stim, celltype, marker, quantile):
  noise rule     real share of the donor-to-donor response differences, estimated on the 15 TRAINING donors from their
                 three plate wells (two-way variance components, as lc/plate_variance.py), for the 3-plate mean:
                 share = SD_donor^2 / (SD_donor^2 + SD_dxw^2 / 3). Use the shift if share < 0.5.
  stretch rule   true slope of the stimulated value on the Unstim value across the TRAINING donors (OLS), vs the
                 map's slope across the test donors (its predictions vs their Unstim values: inputs and outputs only,
                 no test truth). Use the shift if map slope - true slope > 0.5.
  range rule     per case: |test donor Unstim - training mean| > 2 training SDs -> shift.
Scored on the learning-curve scale (all cases above the floor, median skill per repeat, mean over repeats), paired
per donor-repeat. Base map: CellOT (paper training) and paired CellOT.
    python -m lc.hybrid_rules -> work/lc/where_skill/hybrid_rules.csv, hybrid_rules_paired.csv
"""
import json

import numpy as np
import pandas as pd

from lc.paths import MARKERS, PAIRS, QS, WORK

D_ = WORK / "dmso"; P = ["DMSO1", "DMSO2", "DMSO3"]; FLOOR = 0.0273
OUT = WORK / "where_skill"


def comps(A):
    """A: (n donors, 3 plates, F) -> per-feature variances (donor, well, donor x well), as lc/plate_variance.comps."""
    n, k, _ = A.shape
    gm = A.mean((0, 1)); dm = A.mean(1); pm = A.mean(0)
    s2_e = ((A - dm[:, None] - pm[None] + gm) ** 2).sum((0, 1)) / ((n - 1) * (k - 1))
    s2_w = np.clip(((pm - gm) ** 2).sum(0) / (k - 1) - s2_e / n, 0, None)
    s2_a = np.clip(((dm - gm) ** 2).sum(0) / (n - 1) - s2_e / k, 0, None)
    return s2_a, s2_w, s2_e, gm


def plate_quantiles():
    """{(celltype, cond, donor): (3 plates, 65)} for donors with >= 50 cells on every plate."""
    Q = {}
    for ct in sorted({c for _, c in PAIRS}):
        Z = [np.load(D_ / f"cache_{p}/{ct}.npz") for p in P]
        for key in Z[0].files:
            if all(key in z.files and len(z[key]) >= 50 for z in Z):
                cond, d = key.split("|")
                Q[(ct, cond, d)] = np.stack([np.percentile(z[key], QS, axis=0).ravel() for z in Z])
    return Q


def main():
    design = json.load(open(WORK / "design.json"))
    Q = plate_quantiles()
    D = pd.read_pickle(OUT / "cases.pkl")
    D = D[(D.k == 15)].copy()
    feats = [(q, m) for q in QS for m in MARKERS]          # same order as the 65-vector (q-major)
    rows = []
    for dz in design:
        r, train = dz["repeat"], dz["order"][:15]
        for s, c in PAIRS:
            ds = [d for d in train if (c, s, d) in Q and (c, "Unstim", d) in Q]
            if len(ds) < 5: continue
            A = np.stack([Q[(c, s, d)] - Q[(c, "Unstim", d)] for d in ds])
            a, w, e, _ = comps(A)
            share = a / np.maximum(a + e / 3, 1e-12)
            U = np.stack([Q[(c, "Unstim", d)].mean(0) for d in ds]); S = np.stack([Q[(c, s, d)].mean(0) for d in ds])
            Uc = U - U.mean(0); slope = (Uc * (S - S.mean(0))).sum(0) / np.maximum((Uc ** 2).sum(0), 1e-12)
            for i, (q, m) in enumerate(feats):
                rows.append({"repeat": r, "stim": s, "celltype": c, "q": q, "marker": m,
                             "real_share": share[i], "true_slope_train": slope[i]})
    R = pd.DataFrame(rows)
    D = D.merge(R, on=["repeat", "stim", "celltype", "q", "marker"], how="left")
    G = ["repeat", "stim", "celltype", "q", "marker"]
    for base in ["CellOT", "CellOT_none"]:
        uc = D.u - D.groupby(G).u.transform("mean"); pc = D[base] - D.groupby(G)[base].transform("mean")
        D[f"slope_{base}"] = (uc * pc).groupby([D[g] for g in G]).transform("sum") / \
            (uc ** 2).groupby([D[g] for g in G]).transform("sum").replace(0, np.nan)
    D["z"] = ((D.u - D.mu_u) / D.sd_u.replace(0, np.nan)).abs()
    D = D[(D.t - D.u).abs() > FLOOR].copy()
    sk = lambda p, g: (1 - (p - g.t).abs() / (g.t - g.u).abs()).groupby(g.repeat).median().mean()
    res, pr = [], []
    subsets = {"all 7 stims": D.index, "without IL33 and PI": D.index[~D.stim.isin(["IL33", "PI"])]}
    for base in ["CellOT", "CellOT_none"]:
        noise = D.real_share < 0.5
        stretch = (D[f"slope_{base}"] - D.true_slope_train) > 0.5
        rng = D.z > 2
        rules = {"base map only": pd.Series(False, D.index), "noise": noise, "stretch": stretch, "range": rng,
                 "noise + stretch": noise | stretch, "noise + stretch + range": noise | stretch | rng}
        for name, sw in rules.items():
            pred = np.where(sw.fillna(False), D["shift"], D[base])
            D["_p"] = pred
            for sub, idx in subsets.items():
                g = D.loc[idx]
                err = (g._p - g.t).abs(); eb = (g[base] - g.t).abs(); es = (g["shift"] - g.t).abs()
                per = lambda e: e.groupby([g.repeat, g.donor]).median()
                res.append({"base": base, "rule": name, "stims": sub, "skill": sk(g._p, g),
                            "share switched to shift": float(sw.loc[idx].fillna(False).mean()),
                            "better than base map (x/20)": int((per(err) < per(eb)).sum()) if name != "base map only" else np.nan,
                            "better than shift (x/20)": int((per(err) < per(es)).sum())})
    T = pd.DataFrame(res); T.to_csv(OUT / "hybrid_rules.csv", index=False)
    pd.set_option("display.width", 200)
    print(T.round(3).to_string(index=False))
    sw = (D.real_share < 0.5)
    print("\nshare of features switched by the noise rule, per stim:")
    print(sw.groupby(D.stim).mean().round(3).to_string())


if __name__ == "__main__":
    main()
