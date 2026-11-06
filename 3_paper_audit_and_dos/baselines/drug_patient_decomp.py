"""Per-patient drug effects: split each true effect into a shared part and a patient-specific part, and score each
drug map on both, in-sample on the atlas (drug_patient_effects.py) and on held-out donors (atlas_heldout).

For one "group" = (drug, stim, celltype, marker, quantile) and its patients p (19 in-sample; the 4 test donors of a
learning-curve repeat held-out):
    shared true   = mean_p true_p                   shared pred   = mean_p pred_p
    specific true = true_p - shared true            specific pred = pred_p - shared pred
Per group and map we store the sums needed for every statistic in the notebook:
    bias      = shared pred - shared true            (error on the part common to all patients)
    ss_true   = sum_p specific true^2                (between-patient spread of the true effect, noise included)
    ss_pred   = sum_p specific pred^2                (between-patient spread the map predicts)
    cross     = sum_p specific true * specific pred
    ss_spec   = sum_p (specific true - specific pred)^2
    noise_var = mean_p noise_p^2 / 4                 (sampling variance of one true_p; noise = |half A - half B|)
so that a map's per-patient mean squared error is bias^2 + ss_spec / n, and a map that gives every patient the same
effect (median_shift) has ss_spec = ss_true exactly.
Writes simco_work/drug_choice/per_patient/{insample,heldout}_groups.csv.gz and heldout_cases.csv.gz (the held-out
cases with the columns the notebook needs).
    python -m baselines.drug_patient_decomp
"""
from pathlib import Path

import numpy as np
import pandas as pd

W = Path("/scratch/users/bmerwan/simco_work"); DC = W / "drug_choice"; OUT = DC / "per_patient"
HO = Path("/scratch/users/bmerwan/atlas_heldout/work/heldout/drugs/scores")
IN_METHODS = ["cellot", "quantile_map", "median_shift", "gauss_ot"]
HO_METHODS = ["step:CellOT", "step:quantile_map", "step:median_shift", "step:gauss_ot",
              "chain:CellOT>CellOT", "feature:CellOT>quantile_map", "feature:CellOT>median_shift",
              "chain:gauss_ot>gauss_ot", "feature:gauss_ot>quantile_map"]


def decomp(D, keys, methods):
    D = D.copy()
    g = D.groupby(keys, sort=False)
    D["dt"] = D["true"] - g["true"].transform("mean")
    cols = {"n": D["true"].notna().astype(float), "mean_true": D["true"], "noise_var": D["noise"] ** 2 / 4,
            "ss_true": D.dt ** 2}
    for m in methods:
        dp = D[m] - g[m].transform("mean")
        cols[f"{m}|mean_pred"] = D[m]; cols[f"{m}|ss_pred"] = dp ** 2
        cols[f"{m}|cross"] = D.dt * dp; cols[f"{m}|ss_spec"] = (D.dt - dp) ** 2
        cols[f"{m}|missing"] = D[m].isna().astype(float)
    X = pd.DataFrame(cols); X[keys] = D[keys]
    gx = X.groupby(keys, sort=False)
    mean_cols = ["mean_true", "noise_var"] + [f"{m}|mean_pred" for m in methods]
    G = gx[[c for c in cols if c not in mean_cols]].sum().join(gx[mean_cols].mean())
    for m in methods:
        bad = G.pop(f"{m}|missing") > 0          # any patient without a prediction: drop the map for this group
        G.loc[bad, [f"{m}|{k}" for k in ("mean_pred", "ss_pred", "cross", "ss_spec")]] = np.nan
        G[f"{m}|bias"] = G[f"{m}|mean_pred"] - G.mean_true
    return G.reset_index()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PE = pd.concat(pd.read_csv(f) for f in sorted((DC / "patient_effects").glob("*.csv")))
    G = decomp(PE, ["drug", "stim", "celltype", "marker", "q"], IN_METHODS)
    G.to_csv(OUT / "insample_groups.csv.gz", index=False); print("insample groups", G.shape)

    keep = ["repeat", "donor", "celltype", "stim", "drug", "q", "marker", "true", "noise"] + HO_METHODS
    H = pd.concat(pd.read_csv(f, usecols=keep) for f in sorted(HO.glob("r*.csv.gz")))
    GH = decomp(H, ["repeat", "drug", "stim", "celltype", "marker", "q"], HO_METHODS)
    GH.to_csv(OUT / "heldout_groups.csv.gz", index=False); print("heldout groups", GH.shape)
    H.to_csv(OUT / "heldout_cases.csv.gz", index=False, float_format="%.5g"); print("heldout cases", H.shape)


if __name__ == "__main__":
    main()
