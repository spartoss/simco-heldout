"""Which features carry a drug's predicted DOS shift? Exact per-feature attribution with TreeSHAP.

For each fold model (aligned re-implementation of the paper's drug second stage, as in
drug_arm_checks.py) and each test woman, XGBoost's `pred_contribs` gives per-feature contributions
that sum exactly to the prediction, so
    shift = pred(treated) - pred(untreated) = sum_j [contrib_j(treated) - contrib_j(untreated)].
Averaging those per-feature differences over test women and folds gives each feature's share of the
drug's mean shift (features not selected in a fold contribute 0 there).

Writes <out>/attribution_<model>.csv (feature, drug, mean contribution to the shift in days,
selection frequency over the 100 folds) and prints the top features per drug and model, plus the
overlap between models at the feature and at the (stim, celltype) level.
    python stabl_launch/drug_shift_attribution.py [--drugs THF SALPZ MAP]
"""
import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

O = Path("/scratch/users/bmerwan/simco_work/onset_test")
M = O / "drug_models"
MODELS = {
    "CellOT": (O / "ina_13OG_final_long_allstims_filtered.csv",
               O / "foldfeats_GSS/Training CV/Selected Features STABL ALasso.csv",
               "/scratch/users/bmerwan/simco_work/cellot_results/ina_13OG_{drug}_pred_unstim.csv"),
    "gauss_ot": (O / "gaussot_13OG_both.csv",
                 O / "paper_protocol/stabl_n100/gaussot_both/Training CV/Selected Features STABL ALasso.csv",
                 str(M / "tables/{drug}_gaussot.csv")),
}
ap = argparse.ArgumentParser()
ap.add_argument("--drugs", nargs="+", default=["THF", "SALPZ", "MAP"])
ap.add_argument("--out", default=str(M / "attribution"))
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
params = json.load(open(O / "param_grid_files/params_43.json"))
y0 = pd.read_csv(O / "outcome_table_all_pre.csv", index_col=0)["DOS"]

res = {}
for name, (fp, sp, dp) in MODELS.items():
    X = pd.read_csv(fp, index_col=0); rows = X.index.intersection(y0.index); X, y = X.loc[rows], y0.loc[rows]
    XD = {d: pd.read_csv(dp.format(drug=d), index_col=0).reindex(index=rows, columns=X.columns) for d in a.drugs}
    sel = pd.read_csv(sp, index_col=0)["Fold selected features"].apply(ast.literal_eval).tolist()
    g = rows.to_series().str.split("_").str[0].values
    splits = GroupShuffleSplit(100, test_size=0.2, random_state=42).split(np.zeros(len(rows)), y, g)
    contrib = {d: defaultdict(float) for d in a.drugs}; nsel = defaultdict(int); n_test = 0; total = defaultdict(float)
    for (tr, te), f in zip(splits, sel):
        if not f: continue
        for c in f: nsel[c] += 1
        pipe = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())])
        Xtr = pipe.fit_transform(X.iloc[tr][f]); Xte = pipe.transform(X.iloc[te][f])
        m = xgb.XGBRegressor(**params, verbosity=0).fit(Xtr, y.iloc[tr])
        b = m.get_booster()
        c0 = b.predict(xgb.DMatrix(Xte), pred_contribs=True)[:, :-1]          # last column = bias
        for d in a.drugs:
            cd = b.predict(xgb.DMatrix(pipe.transform(XD[d].iloc[te][f])), pred_contribs=True)[:, :-1]
            dc = (cd - c0).sum(0)                                          # summed over test women
            for j, c in enumerate(f): contrib[d][c] += dc[j]
            total[d] += (cd - c0).sum()
        n_test += len(te)
    rows_out = []
    for d in a.drugs:
        for c, v in contrib[d].items():
            rows_out.append({"feature": c, "drug": d, "shift_days": v / n_test, "sel_freq": nsel[c] / 100,
                             "stim": c.split("_")[-1], "celltype": "_".join(c.split("_")[:-2]),
                             "marker": c.split("_")[-2]})
    t = pd.DataFrame(rows_out); t.to_csv(out / f"attribution_{name}.csv", index=False); res[name] = t
    print(f"\n=== {name}: mean shift per drug (days, sum of attributions): "
          + ", ".join(f"{d} {total[d] / n_test:+.2f}" for d in a.drugs))
    for d in a.drugs:
        td = t[t.drug == d].reindex(t[t.drug == d].shift_days.abs().sort_values(ascending=False).index)
        tot = td.shift_days.sum()
        top = td.head(8).copy(); top["share_of_shift"] = top.shift_days / tot
        print(f"--- {name} / {d}: top features (share of the mean shift {tot:+.2f} d)")
        print(top[["feature", "shift_days", "share_of_shift", "sel_freq"]].round(3).to_string(index=False))

print("\n=== overlap between models")
for d in a.drugs:
    tops = {}
    for name, t in res.items():
        td = t[t.drug == d].copy(); td["abs"] = td.shift_days.abs(); td = td.sort_values("abs", ascending=False)
        cum = td["abs"].cumsum() / td["abs"].sum()
        tops[name] = td[cum <= 0.8]                                           # features carrying 80 % of |shift|
    A, B = (set(tops[n].feature) for n in MODELS)
    Ap, Bp = (set(zip(tops[n].stim, tops[n].celltype)) for n in MODELS)
    As, Bs = (tops[n].groupby("stim").shift_days.sum() for n in MODELS)
    print(f"{d}: features carrying 80% of |shift| — CellOT {len(A)}, gauss_ot {len(B)}, shared features {len(A & B)} "
          f"(Jaccard {len(A & B) / max(1, len(A | B)):.2f}); shared (stim, celltype) {len(Ap & Bp)} of {len(Ap | Bp)}")
    print("   shift by stim block (days):", pd.DataFrame({"CellOT": As, "gauss_ot": Bs}).fillna(0).round(2).T.to_dict("index"))
