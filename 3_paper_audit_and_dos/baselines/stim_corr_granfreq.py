"""Granulocyte share of the well vs every other celltype's response, per feature, with the pregnancy-group control.

Complements baselines/stim_cross_correlations.py (same donor-level deltas, same PC1). Per stim and per
non-granulocyte functional feature: Spearman r between the donor's Unstim granulocyte log-share and the response,
raw / after PC1 / after PC1 + HV-vs-PV indicator. Null: donors permuted on the frequency side (200x), statistic =
mean |r| over the 336 targets.  -> simco_work/stim_correlations/granfreq_features.csv, granfreq_group.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from stim_cross_correlations import CT, FUNC, GR, PLATES, STIMS, W, pc1, ranks, resid, usable, zcols

rng = np.random.default_rng(1)
q = pd.read_csv(W / "wells_q50.csv.gz"); cnt = pd.read_csv(W / "wells_counts.csv.gz")
wide = q.pivot_table(index=["stim", "plate", "donor"], columns=["celltype", "marker"], values="q50")
donors = sorted(q.donor.unique()); preg = np.array([d.startswith("PV") for d in donors], float)
n = cnt.pivot_table(index=["stim", "plate", "donor"], columns="celltype", values="n_cells").fillna(0)
lf = np.log(n.div(n.sum(1), axis=0).clip(lower=1e-5))
dmean = lambda df, s, pl: df.loc[s].loc[pl].groupby(level="donor").mean().reindex(donors)
g = dmean(lf, "Unstim", ["DMSO1", "DMSO2", "DMSO3"])[GR].values
share = np.exp(g)
grp = [("HV", share[preg == 0]), ("PV", share[preg == 1])]
rows_g = [(k, len(v), np.median(v), v.min(), v.max()) for k, v in grp]
from scipy.stats import mannwhitneyu
p_mw = mannwhitneyu(grp[0][1], grp[1][1]).pvalue
pd.DataFrame(rows_g, columns=["group", "n", "median_share", "min", "max"]).assign(mannwhitney_p=p_mw).to_csv(W / "granfreq_group.csv", index=False)

fcols = pd.MultiIndex.from_tuples([(c, m) for c in CT for m in FUNC])
rows, summ, pcg = [], [], []
for s in STIMS:
    pl = PLATES[s]
    D = dmean(wide, s, pl) - dmean(wide, "Unstim", pl)
    X = D[fcols].values; ok = usable(X)
    Z = ranks(X[:, ok]); cf = np.array([c for c, _ in fcols])[ok]; mf = np.array([m for _, m in fcols])[ok]
    g1, _ = pc1(Z)
    tgt = cf != GR
    pcg.append((s, np.corrcoef(rankdata(g1[:, 0]), preg)[0, 1], np.corrcoef(rankdata(g1[:, 0]), rankdata(g))[0, 1]))
    for var, C in (("raw", None), ("pc1", g1), ("pc1_group", np.column_stack([g1, preg])), ("group", preg[:, None])):
        Rz = zcols(resid(Z, C)); fz = zcols(resid(rankdata(g)[:, None], C))[:, 0]
        r = (Rz * fz[:, None]).mean(0)
        obs = np.nanmean(np.abs(r[tgt]))
        null = np.array([np.nanmean(np.abs((Rz[:, tgt] * fz[rng.permutation(len(fz))][:, None]).mean(0))) for _ in range(200)])
        summ.append((s, var, obs, null.mean(), (null >= obs).mean(), np.nanmean(r[tgt]), np.mean(r[tgt] > 0)))
        rows.append(pd.DataFrame({"stim": s, "variant": var, "celltype": cf[tgt], "marker": mf[tgt], "r": r[tgt]}))
pd.concat(rows).to_csv(W / "granfreq_features.csv", index=False)
pd.DataFrame(summ, columns=["stim", "variant", "mean_abs_r", "null_mean", "p_perm", "mean_signed_r", "frac_positive"]).to_csv(
    W / "granfreq_summary.csv", index=False)
pd.DataFrame(pcg, columns=["stim", "r_pc1_vs_pregnant", "r_pc1_vs_gran_share"]).to_csv(W / "pc1_vs_group.csv", index=False)
print(pd.read_csv(W / "pc1_vs_group.csv").round(2)); print(pd.read_csv(W / "granfreq_group.csv")); print(pd.read_csv(W / "granfreq_summary.csv").round(3).to_string())
