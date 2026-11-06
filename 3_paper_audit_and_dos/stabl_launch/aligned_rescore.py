"""Rescore stored 100-split Stabl selections with XGBoost, rows and patient groups ALIGNED.

The lab's rescore (`get_foldinfo_on_existing_run.py` -> `cv_on_existing_feats`) takes `y` in the
outcome-table order and `groups` in the feature-table order, so GroupShuffleSplit's test sets are not
whole women and do not match the splits Stabl selected on (CLAUDE.md, Landmines). Its effect is ~0 for
CellOT (0.415 -> 0.418) but inflated gauss_ot / measured features by 0.02-0.04. This refits the same
second stage (median-impute + StandardScaler on train women, `XGBRegressor(**params, verbosity=0)`)
on the splits Stabl actually used: feature-table order, groups aligned.

For one arm and every params_*.json: pooled out-of-fold R2 (median per sample, as Stabl) and the
100 per-split R2, written to <out>/<arm>/sweep.csv and <out>/<arm>/folds.csv.
    python stabl_launch/aligned_rescore.py --arm cellot --features <csv> --selections <dir> --out <dir>
"""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

O = Path("/scratch/users/bmerwan/simco_work/onset_test")
ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--features", required=True)
ap.add_argument("--selections", required=True, help="dir holding 'Training CV/Selected Features STABL ALasso.csv'")
ap.add_argument("--out", default=str(O / "paper_protocol/aligned"))
ap.add_argument("--params-glob", default=str(O / "param_grid_files/params_*.json"))
ap.add_argument("--model", default="STABL ALasso", choices=["STABL ALasso", "STABL Lasso", "STABL ElasticNet"],
                help="whose Stabl selections to rescore (all three are stored by run_regression_cv)")
ap.add_argument("--n-splits", type=int, default=100,
                help="use the first N splits (GroupShuffleSplit draws are a strict prefix, so the first 20 of a "
                     "100-split selection file are exactly a 20-split run)")
ap.add_argument("--exclude", nargs="*", default=[],
                help="samples dropped from XGBoost training and scoring, splits unchanged (Stabl selections were still "
                     "made with them); adds R2_with = same kept rows scored, outliers kept in training")
a = ap.parse_args()

X = pd.read_csv(a.features, index_col=0)
y0 = pd.read_csv(O / "outcome_table_all_pre.csv", index_col=0)["DOS"]
rows = X.index.intersection(y0.index)
X, y = X.loc[rows], y0.loc[rows]
sel = pd.read_csv(Path(a.selections) / f"Training CV/Selected Features {a.model}.csv", index_col=0)
sel = sel["Fold selected features"].apply(ast.literal_eval).tolist()
g = rows.to_series().str.split("_").str[0].values
splits = list(GroupShuffleSplit(a.n_splits, test_size=0.2, random_state=42).split(np.zeros(len(rows)), y, g))
assert len(sel) >= a.n_splits, (len(sel), a.n_splits)
sel = sel[:a.n_splits]
# preprocess once per split (independent of the XGBoost config)
EX = np.isin(rows, a.exclude); assert EX.sum() == len(a.exclude), (EX.sum(), a.exclude)
prep, prep_with = [], []
for (tr, te), f in zip(splits, sel):
    if a.exclude:
        te = te[~EX[te]]
        if f:
            pw = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())])
            prep_with.append((tr, te, pw.fit_transform(X.iloc[tr][f]), pw.transform(X.iloc[te][f])))
        else:
            prep_with.append((tr, te, None, None))
        tr = tr[~EX[tr]]
    if not f:
        prep.append((tr, te, None, None)); continue
    pipe = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())])
    prep.append((tr, te, pipe.fit_transform(X.iloc[tr][f]), pipe.transform(X.iloc[te][f])))

out = Path(a.out) / a.arm; out.mkdir(parents=True, exist_ok=True)
done = pd.read_csv(out / "sweep.csv") if (out / "sweep.csv").exists() else pd.DataFrame()
rows_out, fold_out = done.to_dict("records"), []
for pf in sorted(Path().glob(a.params_glob) if not a.params_glob.startswith("/") else
                 Path(a.params_glob).parent.glob(Path(a.params_glob).name)):
    name = pf.stem
    if len(done) and name in set(done.params):
        continue
    params = json.load(open(pf))

    def run(pp):
        P = {r: [] for r in rows}; fr = []
        for tr, te, Xtr, Xte in pp:
            pr = (np.full(len(te), y.iloc[tr].mean()) if Xtr is None else
                  XGBRegressor(**params, verbosity=0).fit(Xtr, y.iloc[tr]).predict(Xte))
            for r, v in zip(rows[te], pr): P[r].append(v)
            fr.append(r2_score(y.iloc[te], pr))
        med = pd.Series({r: np.median(v) for r, v in P.items() if v})
        return med, fr
    med, fr = run(prep)
    yy = y.loc[med.index]
    rows_out.append({"params": name, "R2": r2_score(yy, med), "RMSE": mean_squared_error(yy, med) ** .5,
                     "median_split_R2": float(np.median(fr))})
    if a.exclude:
        mw, fw = run(prep_with)
        rows_out[-1].update({"R2_with": r2_score(y.loc[mw.index], mw), "median_split_R2_with": float(np.median(fw)),
                             "n_scored": len(med)})
    fold_out.append(pd.Series(fr, name=name))
    pd.DataFrame(rows_out).sort_values("RMSE").to_csv(out / "sweep.csv", index=False)
    f_old = pd.read_csv(out / "folds.csv") if (out / "folds.csv").exists() else pd.DataFrame()
    pd.concat([f_old, pd.DataFrame({name: fr})], axis=1).to_csv(out / "folds.csv", index=False)
    print(f"[{a.arm}] {name}: R2 {rows_out[-1]['R2']:.3f}", flush=True)
t = pd.DataFrame(rows_out).sort_values("RMSE")
print(f"[DONE ] {a.arm}: {len(t)} configs; params_43 R2 "
      f"{t.set_index('params').loc['params_43', 'R2']:.3f}; best {t.iloc[0].params} R2 {t.iloc[0].R2:.3f}")
