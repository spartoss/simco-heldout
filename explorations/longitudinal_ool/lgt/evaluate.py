"""Score the leave-one-woman-out predictions of lgt.m012 per woman, as visits accumulate.

For each woman, the estimate of her gestational age at delivery after seeing a set of visits:
  single visit   G1 only, G2 only, G3 only (which visit is most informative)
  updating       G1 -> G1+G2 -> G1+G2+G3, equal mean or Bayes precision weights (lgt.m012)
compared on the SAME women (those with all three visits, and separately those with G1 and G2).
Leak-free landmarks: at t weeks, use only visits with EGA < t, and score only women still pregnant at t.
Metrics per woman: error in days; RMSE, MAE, Spearman with the truth, and preterm AUROC (descriptive: 4-5 positives).
Paired Wilcoxon of |error| against the constant baseline. The constant's Spearman / AUROC are left blank: leave-one-out
means rank women backwards (removing a late-delivering woman lowers the mean), an artefact, not a result.
    python -m lgt.evaluate --features freqmed
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

from lgt.data import WORK


def combine(P, col, visits, weights=None):
    """Per woman: estimate from the given visits (equal mean, or Bayes with the fold's prior and error variances)."""
    rows = {}
    for w, g in P.groupby("woman"):
        g = g[g.visit.isin(visits)]
        if set(g.visit) != set(visits): continue
        if weights is None:
            rows[w] = g[col].mean()
        else:
            W = weights[weights.woman == w].set_index("visit").err_var
            pv = weights[weights.woman == w].prior_var.iloc[0]; mu = g.constant.iloc[0]
            prec = 1 / pv + sum(1 / W[v] for v in g.visit)
            rows[w] = (mu / pv + sum(x / W[v] for x, v in zip(g[col], g.visit))) / prec
    return pd.Series(rows)


def scores(est, truth, pre, base=None):
    e = est - truth.reindex(est.index); r = {"women": len(e), "RMSE": np.sqrt((e ** 2).mean()), "MAE": e.abs().mean(),
                                            "Spearman": spearmanr(est, truth.reindex(est.index))[0]}
    p = pre.reindex(est.index)
    r["preterm women"] = int(p.sum()); r["AUROC preterm"] = roc_auc_score(p, -est) if 0 < p.sum() < len(p) else np.nan
    if base is not None:
        d = (est - truth.reindex(est.index)).abs() - (base - truth.reindex(base.index)).abs().reindex(est.index)
        r["vs constant: mean |err| diff"] = d.mean(); r["p"] = wilcoxon(d.dropna()).pvalue
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--features", default="freqmed"); a = ap.parse_args()
    d = WORK / f"m012_{a.features}"
    P = pd.read_csv(d / "visit_predictions.csv", index_col=0); Wt = pd.read_csv(d / "fold_weights.csv")
    Wt["woman"] = Wt.woman.astype(str); P["woman"] = P.woman.astype(str)
    truth = P.groupby("woman").GA_del.first(); pre = P.groupby("woman").preterm.first()
    have = P.groupby("woman").visit.apply(set)
    rows = []
    for setname, need in [("women with G1+G2+G3", {"G1", "G2", "G3"}), ("women with G1+G2", {"G1", "G2"})]:
        ws = have[have.apply(lambda s: need <= s)].index
        Q = P[P.woman.isin(ws)]
        base = Q.groupby("woman").constant.first()
        rows.append({"set": setname, "model": "constant (39.0 wk)", "visits": "-",
                     **{**scores(base, truth, pre), "Spearman": np.nan, "AUROC preterm": np.nan}})
        rows.append({"set": setname, "model": "clinical", "visits": "-", **scores(Q.groupby("woman").clinical.first(), truth, pre, base)})
        seqs = [["G1"], ["G2"], ["G1", "G2"]] + ([["G3"], ["G1", "G2", "G3"]] if "G3" in need else [])
        for var in ("dev", "raw"):
            for vs in seqs:
                rows.append({"set": setname, "model": f"{var}, mean", "visits": "+".join(vs),
                             **scores(combine(Q, f"m1_{var}", vs), truth, pre, base)})
                if len(vs) > 1:
                    rows.append({"set": setname, "model": f"{var}, bayes", "visits": "+".join(vs),
                                 **scores(combine(Q, f"m1_{var}", vs, Wt[Wt.variant == var]), truth, pre, base)})
    R = pd.DataFrame(rows); R.to_csv(d / "scores.csv", index=False)
    # leak-free landmarks: visits before t weeks, women still pregnant at t
    L = []
    for t in (30, 33, 35):
        ok = truth[truth > 7 * t].index
        Q = P[P.woman.isin(ok) & (P.EGA < t)]
        if Q.empty: continue
        base = Q.groupby("woman").constant.first()
        L.append({"landmark (wk)": t, "model": "constant", **{**scores(base, truth, pre), "Spearman": np.nan, "AUROC preterm": np.nan}})
        for var in ("dev", "raw"):
            L.append({"landmark (wk)": t, "model": f"{var}, mean of visits before t",
                      **scores(Q.groupby("woman")[f"m1_{var}"].mean(), truth, pre, base)})
    L = pd.DataFrame(L); L.to_csv(d / "landmarks.csv", index=False)
    pd.set_option("display.width", 220)
    print(R.round(3).to_string(index=False)); print(); print(L.round(3).to_string(index=False))
    print("\nBayes weights (median over folds): error SD per visit, days:")
    print(np.sqrt(Wt.groupby(["variant", "visit"]).err_var.median()).round(1).unstack(), "\nprior SD", round(np.sqrt(Wt.prior_var.median()), 1))


if __name__ == "__main__":
    main()
