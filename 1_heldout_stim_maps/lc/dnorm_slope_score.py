"""Score WORK/dnorm_slope/cases.pkl with the ctx_score readouts (effect > FLOOR; skill, dev_R2, between_R2, MSE,
% closer and Wilcoxon over (repeat, stim, celltype, donor) median errors) on: all features, the rule set
(q50/75/90 with inclusion effect > 0.1), the rule set without PI / IL33. Also the between-donor slope vs truth.
    python -m lc.dnorm_slope_score
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from heldout.common import WORK as HW
from lc.dnorm_slope import OUT
from lc.score import FLOOR

ARMS = ["shift", "paired", "dnorm", "ens", "dslope_loo", "dslope_loo_stim", "slope_shift", "dslope_w0.25", "dslope_w0.5",
        "dslope_w0.75", "dslope_w1", "ens_dslope_paired", "ens_dslope_shift_paired"]
G = ["repeat", "stim", "celltype", "q", "marker"]


def readout(D, refs=("paired", "ens", "shift")):
    D = D.copy(); D["ntest"] = D.groupby(G).t.transform("size")
    gb = D[D.ntest >= 3]; tc = gb.t - gb.groupby(G).t.transform("mean")
    res = {}
    for m in ARMS:
        e = (D[m] - D.t).abs(); pc = gb[m] - gb.groupby(G)[m].transform("mean")
        row = {"skill": (1 - e / D.eff).groupby(D.repeat).median().mean(), "MSE": (e ** 2).mean(),
               "dev_R2": 1 - (e ** 2).sum() / ((D.t - D.mu_s) ** 2).sum(),
               "between_R2": 1 - ((pc - tc) ** 2).sum() / (tc ** 2).sum()}
        for b in refs:
            if b == m: continue
            eb = (D[b] - D.t).abs(); k = [D.repeat, D.stim, D.celltype, D.donor]
            row[f"% closer vs {b}"] = 100 * (e < eb).mean()
            row[f"p vs {b}"] = wilcoxon(e.groupby(k).median(), eb.groupby(k).median()).pvalue
        res[m] = row
    return pd.DataFrame(res).T


def main():
    D = pd.read_pickle(OUT / "cases.pkl"); D["eff"] = (D.t - D.u).abs(); D = D[D.eff > FLOOR].copy()
    inc = pd.read_csv(HW / "features/inclusion.csv")
    L = inc.melt(id_vars=["stim", "celltype", "marker"], value_vars=[f"shift_q{q}" for q in (10, 25, 50, 75, 90)],
                 var_name="q", value_name="rule_eff"); L["q"] = L.q.str[7:].astype(int)
    D = D.merge(L, on=["stim", "celltype", "marker", "q"], how="left")
    rule = D.q.isin([50, 75, 90]) & (D.rule_eff > 0.1)
    pd.set_option("display.width", 250)
    for name, X in [("all features", D), ("rule set (q50-90, effect > 0.1)", D[rule]),
                    ("rule set without PI / IL33", D[rule & ~D.stim.isin(["PI", "IL33"])])]:
        R = readout(X); R.round(4).to_csv(OUT / f"score_{name.split(' (')[0].replace(' ', '_').replace('/', '')}.csv")
        print(f"\n=== {name}: {len(X)} cases"); print(R.round(4).to_string())
    # between-donor slope vs truth, rule set, by true-slope bin
    X = D[rule].copy(); X["n"] = X.groupby(G).t.transform("size"); X = X[X.n >= 3]; g = [X[c] for c in G]
    X["uc"] = X.u - X.u.groupby(g).transform("mean")
    A = ["t", "paired", "dnorm", "dslope_loo", "shift"]
    for c in A: X["c_" + c] = X[c] - X[c].groupby(g).transform("mean")
    S = X.groupby(["stim", "celltype", "q", "marker"]).apply(
        lambda d: pd.Series({c: (d["c_" + c] * d.uc).sum() / (d.uc ** 2).sum() for c in A})).dropna()
    S["bin"] = pd.cut(S.t, [-np.inf, 0.5, 0.8, 0.95, 1.05, 1.2, 1.5, np.inf])
    print("\n=== rule set: median between-donor slope per true-slope bin")
    print(S.groupby("bin", observed=True).agg(n=("t", "size"), **{c: (c, "median") for c in A}).round(2).to_string())
    print("Spearman with true slope:", {c: round(S[c].corr(S.t, method="spearman"), 3) for c in A[1:]})
    print("median |slope - true|:", {c: round((S[c] - S.t).abs().median(), 3) for c in A[1:]})


if __name__ == "__main__":
    main()
