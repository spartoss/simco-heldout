"""Paired comparison of the null-map arms against CellOT and gauss_ot on identical splits.

Reads every finished ``Summary/`` under ``null_maps/stabl_n20/<arm>/<table>/`` plus the
existing references, and writes two tables next to them:

  null_maps_scores.csv   pooled-OOF R2 / RMSE / N features per (arm, table, model)
  null_maps_paired.csv   per-split R2 difference CellOT - arm: mean, 95% t-CI, Wilcoxon p,
                         CellOT wins -- the same statistic as stabl_results.ipynb Sec. 5

Partial results are fine: arms without a Summary are listed as missing.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ONSET = Path("/scratch/users/bmerwan/simco_work/onset_test")
W = ONSET / "null_maps"
REFS = {  # arm -> {table: dir}
    "cellot": {"no_Unstim": ONSET / "cmp_unstim_vs_simstim_n20/no_Unstim",
               "both": ONSET / "cellot_both_n20/both"},
    "gauss_ot": {"no_Unstim": ONSET / "gaussot_stabl_n20/no_Unstim",
                 "both": ONSET / "gaussot_stabl_n20/both"},
    "unstim_only": {"unstim_only": ONSET / "cmp_unstim_vs_simstim_n20/unstim_only"},
}
arms = dict(REFS)
for d in sorted((W / "stabl_n20").glob("*/*")):
    arms.setdefault(d.parent.name, {})[d.name] = d

scores, folds, missing = [], {}, []
for arm, tabs in arms.items():
    for tab, d in tabs.items():
        s, f = d / "Summary/Scores training CV.csv", d / "Summary/fold_metrics.csv"
        if not (s.exists() and f.exists()):
            missing.append(f"{arm}/{tab}")
            continue
        sc = pd.read_csv(s, index_col=0)
        for model, r in sc.iterrows():
            scores.append(dict(arm=arm, table=tab, model=model, **r.to_dict()))
        fm = pd.read_csv(f)
        for model, g in fm.groupby("model"):
            folds[(arm, tab, model)] = g.set_index("fold")["r2"]

pd.DataFrame(scores).to_csv(W / "null_maps_scores.csv", index=False)
paired = []
for (arm, tab, model), r2 in folds.items():
    if arm == "cellot" or ("cellot", tab, model) not in folds:
        continue
    ref = folds[("cellot", tab, model)]
    common = ref.index.intersection(r2.index)
    d = (ref.loc[common] - r2.loc[common]).values
    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d)) if len(d) > 1 else (np.nan, np.nan)
    paired.append(dict(arm=arm, table=tab, model=model, n_splits=len(d),
                       mean_diff_cellot_minus_arm=d.mean(), ci_lo=ci[0], ci_hi=ci[1],
                       wilcoxon_p=stats.wilcoxon(d).pvalue if len(d) > 5 and np.any(d) else np.nan,
                       cellot_wins=int((d > 0).sum())))
pp = pd.DataFrame(paired)
pp.to_csv(W / "null_maps_paired.csv", index=False)
with pd.option_context("display.width", 200):
    print(pd.DataFrame(scores)[["arm", "table", "model", "R2", "RMSE", "N features"]].to_string(index=False))
    print(pp.round(3).to_string(index=False))
print("missing:", missing or "none")
