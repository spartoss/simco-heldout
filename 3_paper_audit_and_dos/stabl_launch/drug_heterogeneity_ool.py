"""How woman-specific is each drug map on the OOL cohort? (CellOT drug maps vs quantile_map vs median_shift)

For every feature the Fig 5b model uses (selected in >= 5 % of the 100 folds) and every drug, with 36_G1 / 52_G1
excluded, let U = the untreated feature (CellOT-simulated stim, or measured Unstim) and D = treated - U, per woman:
- het     = SD(D across women) / SD(U across women): how much the drug's effect differs between women, in units of
            how much the women differ from each other. 0 means the same shift for everyone (median_shift, by
            construction).
- spread  = SD(D) / |mean D|: whether the between-women differences are small next to the common shift.
- rank_rho = Spearman(U, U + D): whether the treated women keep the untreated order on this feature. It is 1 for any
            monotone per-feature map (median_shift, quantile_map).
- own_R2  = the share of var(D) explained by a smooth function of the woman's own U (running mean over the 15 nearest women), i.e. the part
            that a per-feature 1-D map (quantile_map) could reproduce. 1 - own_R2 is the only part that needs
            the woman's other markers, i.e. the "single-cell" patient-specificity a CellOT map could add.
- r_vs_qmap = correlation across women of D_CellOT and D_quantile_map.
Writes simco_work/drug_choice/heterogeneity_ool.csv (one row per feature x drug x method).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

O = Path("/scratch/users/bmerwan/simco_work/onset_test"); M = O / "drug_models"
OUT = Path("/scratch/users/bmerwan/simco_work/drug_choice")
DRUGS = ["SA", "RIF", "SALPZ", "CHT", "THF", "LPZ", "MAP", "PRA", "MF"]
EXCL = ["36_G1", "52_G1"]
TABLES = {"CellOT drug map": "/scratch/users/bmerwan/simco_work/cellot_results/ina_13OG_{d}_pred_unstim.csv",
          "quantile_map": str(M / "tables_qmap/{d}_qmap_cellot.csv"),
          "median_shift": str(M / "tables_qmap/{d}_mshift_cellot.csv")}

U = pd.read_csv(O / "ina_13OG_final_long_allstims_filtered.csv", index_col=0).drop(index=EXCL, errors="ignore")
F = pd.read_csv(OUT / "selected_features_fig5b.csv")
F = F[(F.sel_freq >= 0.05) & F.feature.isin(U.columns)]
rows = []
for d in DRUGS:
    for meth, pat in TABLES.items():
        T = pd.read_csv(pat.format(d=d), index_col=0).reindex(index=U.index, columns=U.columns)
        for _, f in F.iterrows():
            u, t = U[f.feature].to_numpy(float), T[f.feature].to_numpy(float)
            ok = np.isfinite(u) & np.isfinite(t)
            u, t = u[ok], t[ok]; D = t - u
            sdU = u.std(); sdD = D.std()
            o = np.argsort(u)                       # running mean of D over the 15 women nearest in U
            fit = np.empty_like(D); fit[o] = pd.Series(D[o]).rolling(15, center=True, min_periods=1).mean().to_numpy()
            own = 1 - ((D - fit) ** 2).sum() / max(((D - D.mean()) ** 2).sum(), 1e-24) if sdD > 1e-12 else np.nan
            rows.append({"drug": d, "method": meth, "feature": f.feature, "stim": f.stim, "celltype": f.celltype,
                         "marker": f.marker, "sel_freq": f.sel_freq, "mean_D": D.mean(), "sd_D": sdD, "sd_U": sdU,
                         "het": sdD / sdU if sdU > 0 else np.nan,
                         "spread": sdD / abs(D.mean()) if abs(D.mean()) > 1e-12 else np.nan,
                         "rank_rho": spearmanr(u, t)[0], "own_R2": own, "D": D})
R = pd.DataFrame(rows)
q = R[R.method == "quantile_map"].set_index(["drug", "feature"]).D
R["r_vs_qmap"] = [np.corrcoef(r.D, q[(r.drug, r.feature)])[0, 1] if r.sd_D > 1e-12 and q[(r.drug, r.feature)].std() > 1e-12 else np.nan
                  for r in R.itertuples()]
R.drop(columns="D").to_csv(OUT / "heterogeneity_ool.csv", index=False)
w = lambda g, c: np.average(g[c].dropna(), weights=g.loc[g[c].notna(), "sel_freq"]) if g[c].notna().any() else np.nan
S = R.groupby("method").apply(lambda g: pd.Series({c: w(g, c) for c in ["het", "rank_rho", "own_R2", "r_vs_qmap"]}
                                                   | {"median spread": g.spread.median()}))
print("weighted by selection frequency, over", F.shape[0], "features x 9 drugs:")
print(S.round(3).to_string())
