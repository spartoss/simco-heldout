"""M3: all of a woman's visits at once, as trajectory features, leave-one-woman-out.

Women with G1 and G2 (59; 51 also have G3). Per feature, detrended on gestational age within the training fold
(lgt.m012.detrend), the woman's G1 value and her change G2 - G1 (and G3 - G2 when present; missing -> 0 after
standardising, i.e. "no change beyond the population"). ElasticNetCV (inner 5-fold), target = GA at delivery.
Variants: g1g2 (G1 level + G1->G2 change, all 59 women), g1g2g3 (adds G2->G3, the 51 women with three visits).
    python -m lgt.m3 --features freqmed -> WORK/m3_<features>/predictions.csv
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold

from lgt.data import WORK, features, visits
from lgt.m012 import L1, detrend, standardise


def woman_table(R, V, visits_needed):
    D = pd.DataFrame(R, index=V.index)
    out = {}
    for w, g in V.groupby("woman"):
        vv = dict(zip(g.visit, g.index))
        if not set(visits_needed) <= set(vv): continue
        parts = [D.loc[vv["G1"]].to_numpy(), D.loc[vv["G2"]].to_numpy() - D.loc[vv["G1"]].to_numpy()]
        if "G3" in visits_needed: parts.append(D.loc[vv["G3"]].to_numpy() - D.loc[vv["G2"]].to_numpy())
        out[w] = np.concatenate(parts)
    return pd.DataFrame(out).T


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--features", default="freqmed"); a = ap.parse_args()
    V = visits(); X = features(a.features).reindex(V.index); X = X.loc[:, X.notna().all(0)]
    truth = V.groupby("woman").GA_del.first(); rows = []
    for var, need in [("g1g2", ["G1", "G2"]), ("g1g2g3", ["G1", "G2", "G3"])]:
        ws = [w for w, g in V.groupby("woman") if set(need) <= set(g.visit)]
        for w in ws:
            tr = V.woman.isin(ws) & (V.woman != w); te = V.woman == w
            Rtr, Rte = detrend(X[tr].to_numpy(float), V[tr].EGA.to_numpy(), X[te].to_numpy(float), V[te].EGA.to_numpy())
            Ttr = woman_table(Rtr, V[tr], need); Tte = woman_table(Rte, V[te], need)
            ytr = truth.reindex(Ttr.index).to_numpy(); mu = ytr.mean()
            Ztr, Zte = standardise(Ttr.to_numpy(), Tte.to_numpy())
            m = ElasticNetCV(l1_ratio=L1, n_alphas=30, cv=KFold(5, shuffle=True, random_state=0), max_iter=5000).fit(Ztr, ytr - mu)
            rows.append({"variant": var, "woman": w, "pred": m.predict(Zte)[0] + mu, "constant": mu, "truth": truth[w],
                         "n_nonzero": int((m.coef_ != 0).sum())})
        R = pd.DataFrame(rows); R = R[R.variant == var]
        e, e0 = R.pred - R.truth, R.constant - R.truth
        print(f"{var}: {len(R)} women, RMSE {np.sqrt((e ** 2).mean()):.2f} vs constant {np.sqrt((e0 ** 2).mean()):.2f} d, "
              f"Spearman {spearmanr(R.pred, R.truth)[0]:.3f}, non-zero coefs median {R.n_nonzero.median():.0f}", flush=True)
    o = WORK / f"m3_{a.features}"; o.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(o / "predictions.csv", index=False)


if __name__ == "__main__":
    main()
