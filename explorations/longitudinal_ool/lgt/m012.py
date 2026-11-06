"""M0-M2: predict each woman's gestational age at delivery from her visits, leave-one-woman-out.

Per outer fold (one held-out woman, all her visits):
  per-visit model (M1)   ElasticNetCV on the training visits, target = the woman's GA at delivery (days), inner CV
                         grouped by woman. Two input variants:
                           dev   features detrended on gestational age (per-feature linear fit vs EGA on training
                                 visits; the "deviation from the clock" representation)
                           raw   features as measured + EGA
  updating (M2)          the held-out woman's per-visit predictions combined over the visits seen so far:
                           mean   equal weights
                           bayes  precision-weighted with a population prior: prior mean / variance = training women's
                                  GA at delivery; each visit type's error variance from out-of-fold predictions on the
                                  training women (assumes independent errors across visits, which overcounts if a
                                  woman's errors are correlated; the mean is the safe version)
  baselines (M0)         constant = training women's mean; clinical = linear on Age, BMI, Gravida, Para
Writes WORK/m012/visit_predictions.csv (one row per held-out visit) and the per-fold Bayes weights.
    python -m lgt.m012 --features freqmed
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV, LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_predict

from lgt.data import CLIN, WORK, features, visits

L1 = [0.5, 0.9, 1.0]


def detrend(Xtr, ega_tr, Xte, ega_te):
    A = np.c_[np.ones(len(ega_tr)), ega_tr]
    coef, *_ = np.linalg.lstsq(A, Xtr, rcond=None)
    return Xtr - A @ coef, Xte - np.c_[np.ones(len(ega_te)), ega_te] @ coef


def standardise(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0); keep = sd > 1e-8
    return (Xtr[:, keep] - mu[keep]) / sd[keep], (Xte[:, keep] - mu[keep]) / sd[keep]


def fit_predict(Xtr, ytr, gtr, Xte, inner=5, seed=0):
    cv = list(GroupKFold(n_splits=inner).split(Xtr, ytr, gtr))
    m = ElasticNetCV(l1_ratio=L1, n_alphas=30, cv=cv, max_iter=5000, n_jobs=1, random_state=seed).fit(Xtr, ytr)
    oof = cross_val_predict(ElasticNet(alpha=m.alpha_, l1_ratio=m.l1_ratio_, max_iter=5000), Xtr, ytr, cv=cv)
    return m.predict(Xte), oof, m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--features", default="freqmed"); a = ap.parse_args()
    V = visits(); X = features(a.features).reindex(V.index)
    keep = X.notna().all(0); X = X.loc[:, keep]
    print(f"{len(V)} visits, {V.woman.nunique()} women, {X.shape[1]} features ({a.features})", flush=True)
    out, wts, t0 = [], [], time.time()
    for k, w in enumerate(sorted(V.woman.unique(), key=int)):
        tr, te = V.woman != w, V.woman == w
        Vtr, Vte = V[tr], V[te]
        ytr, gtr = Vtr.GA_del.to_numpy(), Vtr.woman.to_numpy()
        prior_mu = Vtr.groupby("woman").GA_del.first().mean(); prior_var = Vtr.groupby("woman").GA_del.first().var()
        rec = Vte[["woman", "visit", "EGA", "GA_del", "preterm"]].copy()
        rec["constant"] = prior_mu
        C = LinearRegression().fit(Vtr[CLIN].fillna(Vtr[CLIN].median()), ytr)
        rec["clinical"] = C.predict(Vte[CLIN].fillna(Vtr[CLIN].median()))
        for var in ("dev", "raw"):
            Xtr, Xte = X[tr].to_numpy(float), X[te].to_numpy(float)
            if var == "dev":
                Xtr, Xte = detrend(Xtr, Vtr.EGA.to_numpy(), Xte, Vte.EGA.to_numpy())
            else:
                Xtr, Xte = np.c_[Xtr, Vtr.EGA.to_numpy()], np.c_[Xte, Vte.EGA.to_numpy()]
            Xtr, Xte = standardise(Xtr, Xte)
            pred, oof, m = fit_predict(Xtr, ytr - prior_mu, gtr, Xte)
            rec[f"m1_{var}"] = pred + prior_mu
            res = pd.Series(oof - (ytr - prior_mu), index=Vtr.index)
            for v in ("G1", "G2", "G3"):
                rv = res[Vtr.visit == v]
                wts.append({"woman": w, "variant": var, "visit": v, "err_var": float((rv ** 2).mean()),
                            "prior_var": prior_var, "alpha": m.alpha_, "l1": m.l1_ratio_, "n_nonzero": int((m.coef_ != 0).sum())})
        out.append(rec)
        if (k + 1) % 10 == 0: print(f"[FOLD ] {k + 1}/{V.woman.nunique()} ({time.time() - t0:.0f}s)", flush=True)
    o = WORK / f"m012_{a.features}"; o.mkdir(parents=True, exist_ok=True)
    pd.concat(out).to_csv(o / "visit_predictions.csv"); pd.DataFrame(wts).to_csv(o / "fold_weights.csv", index=False)
    print(f"[DONE ] {o} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
