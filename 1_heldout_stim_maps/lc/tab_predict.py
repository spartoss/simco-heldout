"""Tabular residual models on the learning-curve design (features and target: lc/tab_features.py).

Models (all fitted on the k training donors of a repeat only):
- lgbm     LightGBM, one model per (k, stim) over all its celltypes; small trees, rounds chosen by donor-grouped CV
- tabpfn   TabPFN v2 regressor (in-context, no training), one model per (k, stim, celltype)
- ridge    RidgeCV per (k, stim) on the same columns (linear reference)
Writes WORK/tab/<model>_<layer>/repeat_<r>/: maps.csv (stim, celltype, k) and rhat.npy (M, n_test, 5, 13), the
predicted residual (NaN where no prediction); lc/tab_ens.py turns it into shift + lambda * rhat and ensembles.
    python -m lc.tab_predict --model lgbm --layers L0 L1 L2 L2perm --repeat 0 [--ks 15 --stims IFNa]
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from lc.paths import PAIRS, WORK
from lc.tab_features import CAT, build

KS = [3, 5, 8, 15]
OUT = WORK / "tab"
LGB = dict(objective="l2", learning_rate=0.03, num_leaves=7, min_data_in_leaf=200, feature_fraction=0.5,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=8, seed=0)


def fit_lgbm(Xtr, y, groups, Xte):
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    ug = np.unique(groups); nf = min(5, len(ug))
    ds = lgb.Dataset(Xtr, y, categorical_feature=[c for c in CAT if c in Xtr], free_raw_data=False)
    cv = lgb.cv(LGB, ds, num_boost_round=1000, folds=GroupKFold(nf).split(Xtr, y, groups),
                callbacks=[lgb.early_stopping(50, verbose=False)])
    n = len(cv["valid l2-mean"])
    return lgb.train(LGB, ds, num_boost_round=n).predict(Xte), n


def fit_ridge(Xtr, y, Xte):
    from sklearn.linear_model import RidgeCV
    Xtr = pd.get_dummies(Xtr, columns=[c for c in CAT if c in Xtr], dtype=float)
    Xte = pd.get_dummies(Xte, columns=[c for c in CAT if c in Xte], dtype=float).reindex(columns=Xtr.columns, fill_value=0)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    m = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit((Xtr - mu) / sd, y)
    return m.predict((Xte - mu) / sd)


_PFN = {}


def fit_tabpfn(Xtr, y, Xte):
    from tabpfn import TabPFNRegressor
    if "m" not in _PFN:
        _PFN["m"] = TabPFNRegressor(device="cuda" if __import__("torch").cuda.is_available() else "cpu",
                                    random_state=0, ignore_pretraining_limits=True)
    m = _PFN["m"]
    Xtr = Xtr.drop(columns="celltype").astype(float); Xte = Xte.drop(columns="celltype").astype(float)
    keep = Xtr.columns[Xtr.std(0) > 0]                               # constant columns carry nothing
    return m.fit(Xtr[keep].to_numpy(), y).predict(Xte[keep].to_numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lgbm", "tabpfn", "ridge"], required=True)
    ap.add_argument("--layers", nargs="+", default=["L0", "L1", "L2", "L2perm", "Lperm"])
    ap.add_argument("--repeat", type=int, required=True)
    ap.add_argument("--ks", nargs="*", type=int, default=KS)
    ap.add_argument("--stims", nargs="*", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    dz = [x for x in json.load(open(WORK / "design.json")) if x["repeat"] == a.repeat][0]
    tests = dz["test"]
    stims = a.stims or sorted({s for s, _ in PAIRS})
    maps = [(s, c, k) for s, c in PAIRS for k in a.ks if s in stims]
    pos = {m: i for i, m in enumerate(maps)}
    R = {L: np.full((len(maps), len(tests), 5, 13), np.nan) for L in a.layers}
    log, done = [], set()
    # checkpoint after every (k, stim): a rerun of the same command resumes (a time-out loses one map at most)
    ck = OUT / f"partial_{a.model}{a.tag}_r{a.repeat}.npz"
    if ck.exists():
        z = np.load(ck, allow_pickle=True)
        if list(z["layers"]) == a.layers and [tuple(m) for m in z["maps"]] == [tuple(map(str, m)) for m in maps]:
            R = {L: z[f"R_{L}"] for L in a.layers}; done = {tuple(x) for x in z["done"]}; log = list(z["log"])
            print(f"[RESUM] {len(done)} (k, stim) from {ck}", flush=True)
    for k in a.ks:
        for stim in stims:
            if (str(k), stim) in done: continue
            t0 = time.time()
            D, cols = build(a.repeat, k, stim, dz["order"][:k], tests)
            tr = D[~D.is_test]; te = D[D.is_test]
            if not len(tr): continue
            for L in a.layers:
                X = D[cols[L]]
                if a.model == "lgbm":
                    p, n = fit_lgbm(X[~D.is_test], tr.target.to_numpy(), tr.donor.to_numpy(), X[D.is_test])
                    p = pd.Series(p, index=te.index); info = n
                elif a.model == "ridge":
                    p = pd.Series(fit_ridge(X[~D.is_test], tr.target.to_numpy(), X[D.is_test]), index=te.index); info = 0
                else:
                    p = pd.Series(np.nan, index=te.index); info = 0
                    for c, g in tr.groupby("celltype", observed=True):
                        gi = te.index[te.celltype == c]
                        if len(gi): p[gi] = fit_tabpfn(X.loc[g.index], g.target.to_numpy(), X.loc[gi])
                for (c, d), g in te.groupby(["celltype", "donor"], observed=True):
                    R[L][pos[(stim, c, k)], tests.index(d), g.qi, g.mi] = p[g.index]
                log.append({"k": k, "stim": stim, "layer": L, "n_train": len(tr), "info": info})
            done.add((str(k), stim)); tmp = ck.with_suffix(".tmp.npz")
            np.savez(tmp, layers=np.array(a.layers), maps=np.array(maps, dtype=str), done=np.array(sorted(done)),
                     log=np.array(log, dtype=object), **{f"R_{L}": R[L] for L in a.layers})
            tmp.replace(ck)
            print(f"[MAP  ] r{a.repeat} k{k} {stim} {len(tr)} rows {time.time() - t0:.0f}s", flush=True)
    for L in a.layers:
        o = OUT / f"{a.model}{a.tag}_{L}" / f"repeat_{a.repeat}"; o.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(maps, columns=["stim", "celltype", "k"]).to_csv(o / "maps.csv", index=False)
        np.save(o / "rhat.npy", R[L])
    pd.DataFrame(list(log)).to_csv(OUT / f"log_{a.model}{a.tag}_r{a.repeat}.csv", index=False)
    ck.unlink(missing_ok=True)
    print(f"[DONE ] -> {OUT}")


if __name__ == "__main__":
    main()
