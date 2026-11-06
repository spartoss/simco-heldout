"""How much of DOS does gestational age at sampling explain, on Fig 5b's own splits?

DOS = days from blood draw to labor onset (negative). EGA = estimated gestational age at
the draw, in weeks. Most pregnancies end near 39-40 wk, so DOS ~ 7 * (EGA - 39.5), and an
EGA-only model is the reference every immune feature table should be read against.

Same protocol as ``run_regression_cv.py``: rows in the order of the feature table (the
GroupShuffleSplit draw depends on row order), groups = patient id before ``_``,
``GroupShuffleSplit(n_splits, test_size=0.2, random_state=42)``, and the pooled out-of-fold
score of ``stablVMax/stabl/single_omic.py:295-329`` -- per sample, the median of its test-set
predictions across splits, then one R2 / RMSE / MAE over the 177 samples.

    python stabl_launch/clinical_baselines.py [--n_splits 100]
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

ONSET = "/scratch/users/bmerwan/simco_work/onset_test"

p = argparse.ArgumentParser()
p.add_argument("--n_splits", type=int, default=100)
p.add_argument("--row_order", default=f"{ONSET}/ina_13OG_final_long_allstims_filtered.csv")
p.add_argument("--out", default=f"{ONSET}/clinical_baselines.csv")
args = p.parse_args()

order = pd.read_csv(args.row_order, index_col=0, usecols=[0]).index
y = pd.read_csv(f"{ONSET}/outcome_table_all_pre.csv", index_col=0)["DOS"]
clin = pd.read_csv(f"{ONSET}/Preprocessed_OOL_Clinical_new.csv", index_col=0)
rows = order.intersection(y.index)
y = y.loc[rows].astype(float)
clin = clin.reindex(rows)
groups = rows.to_series().str.split("_").str[0].values
print(f"{len(rows)} samples, {len(set(groups))} patients; "
      f"missing: {clin[['EGA', 'Age', 'BMI', 'Gravida', 'Para']].isna().sum().to_dict()}")
print(f"corr(EGA, DOS) = {pearsonr(clin['EGA'].fillna(clin['EGA'].mean()), y)[0]:.3f}")

MODELS = {
    "mean_only": [],
    "EGA": ["EGA"],
    "EGA+Age+BMI+Gravida+Para": ["EGA", "Age", "BMI", "Gravida", "Para"],
}
cv = GroupShuffleSplit(n_splits=args.n_splits, test_size=0.2, random_state=42)
out = []
for name, cols in MODELS.items():
    preds = {r: [] for r in rows}
    per_split = []
    for tr, te in cv.split(np.zeros(len(rows)), y, groups):
        if cols:
            X = clin[cols]
            X = X.fillna(X.iloc[tr].median())  # impute from the train side only
            m = LinearRegression().fit(X.iloc[tr], y.iloc[tr])
            yh = m.predict(X.iloc[te])
        else:
            yh = np.full(len(te), y.iloc[tr].mean())
        for r, v in zip(rows[te], yh):
            preds[r].append(v)
        per_split.append(r2_score(y.iloc[te], yh))
    ok = [r for r in rows if preds[r]]
    yh = np.array([np.median(preds[r]) for r in ok])
    yt = y.loc[ok].values
    out.append(dict(model=name, n=len(ok),
                    R2=r2_score(yt, yh),
                    RMSE=mean_squared_error(yt, yh) ** 0.5,
                    MAE=mean_absolute_error(yt, yh),
                    pearson=pearsonr(yt, yh)[0] if np.std(yh) > 0 else np.nan,
                    R2_split_median=np.median(per_split)))
res = pd.DataFrame(out)
print(res.round(3).to_string(index=False))
res.to_csv(args.out, index=False)
print(f"-> {args.out}")
