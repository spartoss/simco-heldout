"""Ceiling ladder with the well noise split into its shared and donor-specific parts (2026-10-07, meeting figure).

Every atlas well holds all donors (barcoded), one (stim, plate) per well. For a case (celltype, stim, donor, q, marker)
the difference between two of her wells, d_AB = w_A - w_B, splits into
    shared   s_AB = mean over donors of d_AB   (the same for every donor in those wells: learnable from training donors)
    donor x well   d_AB - s_AB                  (new for every donor: not learnable)
Ideal-model skill on the pooled (3-well) target, as wc/compare.py (independent equal-variance well noise):
    sampling only                    1 - noise / 2 / effect                         (wc/compare.py)
    sampling + donor x well          1 - rms_AB(d_AB - s_AB) / sqrt(6) / effect     (new)
    sampling + all well noise        1 - rms_AB(d_AB) / sqrt(6) / effect            (wc/compare.py, = 0.733)
Same case set and aggregation as wc/compare.py (k = 15, effect > floor, median per repeat, mean over repeats); also per
stim and without PI and IL33.
    python -m wc.ceiling_components -> work/summary/ceiling_components.csv
"""
import numpy as np
import pandas as pd

from wc.compare import CASES, FLOOR, QUANT, SUMMARY, WELLS, well_table


def main():
    C = pd.read_pickle(CASES)
    C = C[C.k == 15].copy()
    C["eff"] = (C.t - C.u).abs()
    C = C[C.eff > FLOOR]
    C = C[[(QUANT / f"{c}.npz").exists() for c in C.celltype]]
    key = ["celltype", "stim", "donor", "q", "marker"]
    W = well_table(sorted(C.celltype.unique()), drop_pi_dmso1=False).pivot_table(index=key, columns="well",
                                                                                values="value").reset_index()
    grp = ["celltype", "stim", "q", "marker"]
    pairs = [(a, b) for a in WELLS for b in WELLS if a < b]
    for a, b in pairs:
        d = W[a] - W[b]
        W[f"d_{a}{b}"] = d
        W[f"x_{a}{b}"] = d - d.groupby([W[g] for g in grp]).transform("mean")   # donor x well part
    J = C.merge(W, on=key, how="left")
    dcols = [f"d_{a}{b}" for a, b in pairs]; xcols = [f"x_{a}{b}" for a, b in pairs]
    rms = lambda cols: np.sqrt((J[cols] ** 2).mean(1, skipna=True))
    S = {"ideal: sampling only": 1 - J.noise / 2 / J.eff,
         "ideal: + donor x well noise": 1 - rms(xcols) / np.sqrt(6) / J.eff,
         "ideal: + shared plate/well effect": 1 - rms(dcols) / np.sqrt(6) / J.eff,
         "split-half replicate (previous ceiling)": 1 - J.noise / J.eff,
         "CellOT (paper training)": 1 - (J.CellOT - J.t).abs() / J.eff}
    if "CellOT_none" in J:
        S["CellOT paired"] = 1 - (J.CellOT_none - J.t).abs() / J.eff
    rows = []
    subsets = {"all 7 stims": J.index, "without PI and IL33": J.index[~J.stim.isin(["PI", "IL33"])]}
    for sub, idx in subsets.items():
        for lab, s in S.items():
            v = s.loc[idx].dropna()
            rows.append({"stims": sub, "quantity": lab, "skill": v.groupby(J.loc[v.index, "repeat"]).median().mean()})
    for stim, g in J.groupby("stim"):
        for lab, s in S.items():
            v = s.loc[g.index].dropna()
            rows.append({"stims": stim, "quantity": lab, "skill": v.groupby(J.loc[v.index, "repeat"]).median().mean()})
    # one-well target (a single plate): ideal error |w_A - w_B| / sqrt(2), with or without the shared part
    for sub, idx in subsets.items():
        g = J.loc[idx]
        for lab, pre in [("one well: ideal, + donor x well noise", "x_"), ("one well: ideal, + shared plate/well effect", "d_")]:
            vals = []
            for B in WELLS:
                effB = (g[B] - g.u).abs()
                for A in WELLS:
                    if A == B: continue
                    col = f"{pre}{min(A, B)}{max(A, B)}"
                    v = (1 - g[col].abs() / np.sqrt(2) / effB).replace([np.inf, -np.inf], np.nan)
                    vals.append(pd.DataFrame({"repeat": g.repeat, "v": v}).dropna())
            V = pd.concat(vals)
            rows.append({"stims": sub, "quantity": lab, "skill": V.groupby("repeat").v.median().mean()})
    R = pd.DataFrame(rows); R.to_csv(SUMMARY / "ceiling_components.csv", index=False)
    pd.set_option("display.width", 200)
    print(R.pivot(index="stims", columns="quantity", values="skill").round(3).to_string())


if __name__ == "__main__":
    main()
