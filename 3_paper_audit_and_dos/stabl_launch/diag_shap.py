"""TreeSHAP use of each selected feature in the paper-protocol XGBoost second stage (params_43, rows
aligned) for the measured-only and combined models. Writes simco_work/diagnosis/shap_by_feature.csv
(model, feature, sum of |contribution| over the 100 splits' test women). Read by notebooks/diagnosis.ipynb.
    python stabl_launch/diag_shap.py            # measured-only and combined models
    python stabl_launch/diag_shap.py --simq     # combined + simulated quantiles -> shap_by_feature_simq.csv
"""
import ast, json
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

W = Path("/scratch/users/bmerwan/simco_work"); O = W / "onset_test"; S = O / "paper_protocol/stabl_n100"
ARMS = {"measured only (freq + med + qtl)": (O / "night2/distrib_all_v2.csv", S / "d_qtl"),
        "CellOT stims + measured": (O / "paper_protocol/tables/comb_cellot.csv", S / "comb_cellot"),
        "gauss_ot stims + measured": (O / "paper_protocol/tables/comb_gaussot.csv", S / "comb_gaussot")}
import sys
if "--simq" in sys.argv:      # the combined models with simulated quantiles (100-split selections)
    ARMS = {"CellOT stims + measured + sim. quantiles": (O / "paper_protocol/tables/comb_cellot_simq.csv", S / "comb_cellot_simq"),
            "gauss_ot stims + measured + sim. quantiles": (O / "paper_protocol/tables/comb_gaussot_simq.csv", S / "comb_gaussot_simq")}
params = json.load(open(O / "param_grid_files/params_43.json"))
y0 = pd.read_csv(O / "outcome_table_all_pre.csv", index_col=0)["DOS"]
rows = []
for name, (fp, run) in ARMS.items():
    sel = pd.read_csv(run / "Training CV/Selected Features STABL ALasso.csv", index_col=0)["Fold selected features"].apply(ast.literal_eval).tolist()
    X = pd.read_csv(fp, index_col=0); r = X.index.intersection(y0.index); X, y = X.loc[r], y0.loc[r]
    g = r.to_series().str.split("_").str[0].values; acc = {}
    for (tr, te), f in zip(GroupShuffleSplit(100, test_size=.2, random_state=42).split(np.zeros(len(r)), y, g), sel):
        if not f: continue
        pipe = Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())])
        b = xgb.XGBRegressor(**params, verbosity=0).fit(pipe.fit_transform(X.iloc[tr][f]), y.iloc[tr]).get_booster()
        c = np.abs(b.predict(xgb.DMatrix(pipe.transform(X.iloc[te][f])), pred_contribs=True)[:, :-1]).sum(0)
        for j, feat in enumerate(f): acc[feat] = acc.get(feat, 0) + c[j]
    rows += [{"model": name, "feature": k, "abs_shap": v} for k, v in acc.items()]
    print(f"[DONE ] {name}: {len(acc)} features", flush=True)
out = W / ("diagnosis/shap_by_feature_simq.csv" if "--simq" in sys.argv else "diagnosis/shap_by_feature.csv"); pd.DataFrame(rows).to_csv(out, index=False); print("->", out)
