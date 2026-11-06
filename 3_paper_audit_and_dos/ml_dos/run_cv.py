"""Run one (arm, model) over the 100 paper-protocol splits; one CSV per split, resumable.

    python ml_dos/run_cv.py --arm qtl --model rf --n-jobs 16

Writes <OUT>/preds/<arm>/<model>/split_###.csv (sample, y, pred, split) and a matching
``.json`` with the inner-CV choice and fit time; ``imp_###.csv`` holds the top-200 feature
importances (|coef| or impurity/gain) for models that expose them.

Reference arms (the thing to beat), same splits and exclusions:
  --arm ref_cellot       --model stabl_p43   CellOT table + its Stabl ALasso selections + params_43 XGBoost
  --arm ref_comb_cellot  --model stabl_p43   CellOT + measured (3204 cols) + its Stabl selections + params_43
"""
import argparse
import ast
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, GroupKFold

import common as C
import models as M

REFS = {
    "ref_cellot": (C.PAPER_TABLE, C.O / "foldfeats_GSS"),
    "ref_comb_cellot": (C.O / "paper_protocol/tables/comb_cellot.csv", C.O / "paper_protocol/stabl_n100/comb_cellot"),
}


def importances(est, names):
    p = est.best_estimator_ if hasattr(est, "best_estimator_") else est
    if not hasattr(p, "named_steps") or "sel" in p.named_steps:
        return None
    m = p.named_steps["m"]
    keep = names[p.named_steps["var"].get_support()] if "var" in p.named_steps else names
    if hasattr(m, "feature_importances_"):
        v = m.feature_importances_
    elif hasattr(m, "coef_") and np.ndim(m.coef_) == 1:
        v = np.abs(m.coef_)
    else:
        return None
    s = pd.Series(v, index=keep)
    s = s[s > 0].sort_values(ascending=False).head(200)
    return s.rename("importance").rename_axis("feature").reset_index()


def one_split(k, tr, te, X, y, g, arm, model, outdir, sel=None):
    f = outdir / f"split_{k:03d}.csv"
    if f.exists():
        return
    warnings.filterwarnings("ignore")
    t0 = time.time()
    info = {}
    Xtr, Xte, ytr = X.iloc[tr], X.iloc[te], y.iloc[tr].values
    if arm in REFS:
        feats = sel[k]
        if feats:
            est = M.pipe(M.XGB(**json.load(open(M.P43))))
            est.fit(Xtr[feats].values, ytr)
            pred = est.predict(Xte[feats].values)
        else:
            pred = np.full(len(te), ytr.mean())
        info["n_features"] = len(feats)
    else:
        if model == "stack":
            icv = list(GroupKFold(5).split(Xtr, ytr, g[tr]))
            est = M.stack(icv).fit(Xtr.values, ytr)
            info["weights"] = dict(zip(M.STACK_BASE, map(float, est.final_estimator_.coef_)))
        else:
            fam, base, grid = M.zoo()[model]
            if grid:
                est = GridSearchCV(clone(base), grid, cv=GroupKFold(5), scoring="neg_mean_squared_error",
                                   n_jobs=1, refit=True, error_score=np.nan)
                est.fit(Xtr.values, ytr, groups=g[tr])
                info["best_params"] = {kk: (vv if isinstance(vv, (int, float, str, type(None))) else str(vv))
                                       for kk, vv in est.best_params_.items()}
                info["inner_rmse"] = float(-est.best_score_) ** .5
            else:
                est = clone(base).fit(Xtr.values, ytr)
            imp = importances(est, X.columns.values)
            if imp is not None:
                imp.assign(split=k).to_csv(outdir / f"imp_{k:03d}.csv", index=False)
        pred = est.predict(Xte.values)
    info["seconds"] = time.time() - t0
    json.dump(info, open(outdir / f"split_{k:03d}.json", "w"))
    pd.DataFrame({"sample": X.index[te], "y": y.iloc[te].values, "pred": np.ravel(pred), "split": k}
                 ).to_csv(f.with_suffix(".tmp"), index=False)
    os.replace(f.with_suffix(".tmp"), f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--splits", default="0-99")
    a = ap.parse_args()
    lo, hi = map(int, a.splits.split("-"))
    idx = C.rows()
    sp = C.splits(idx)
    sel = None
    if a.arm in REFS:
        tab, seldir = REFS[a.arm]
        X = pd.read_csv(tab, index_col=0).loc[idx].astype(float)
        sel = pd.read_csv(Path(seldir) / "Training CV/Selected Features STABL ALasso.csv", index_col=0)
        sel = sel["Fold selected features"].apply(ast.literal_eval).tolist()
        y, g = C.outcome(idx), C.groups(idx)
    else:
        X, y, g = C.load_arm(a.arm)
    outdir = C.OUT / "preds" / a.arm / a.model
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO ] {a.arm}/{a.model}: X {X.shape}, splits {lo}-{hi}, n_jobs {a.n_jobs}", flush=True)
    t0 = time.time()
    Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(one_split)(k, *sp[k], X, y, g, a.arm, a.model, outdir, sel)
                                         for k in range(lo, hi + 1))
    pl = pd.concat([pd.read_csv(f) for f in sorted(outdir.glob("split_*.csv"))])
    sc, _ = C.pooled(pl)
    print(f"[DONE ] {a.arm}/{a.model}: {pl.split.nunique()} splits, pooled R2 {sc['R2']:.3f} RMSE {sc['RMSE']:.1f} "
          f"median split R2 {C.per_split(pl).median():.3f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
