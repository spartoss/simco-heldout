"""Checks a-d on the paper's drug arm (Fig 5d/5e/5g): are the predicted DOS shifts drug effects,
or artefacts of applying a DOS model outside the data it was fit on?

Re-implements the paper's second stage exactly (``cross_validation_drug_vs_dmso.cv_drug_vs_dmso``):
the lab's stored 100-fold STABL ALasso selections (``foldfeats_GSS`` == ``comp_9drugs_long_
unstimpred_treated/run_43/results_no_treatment``), median-impute + StandardScaler fit on the train
women, ``XGBRegressor(**params_43, verbosity=0)``, and for each drug the test women's selected
features swapped for their drug-treated values (``cellot_results/ina_13OG_<drug>_pred_unstim.csv``).
Validated against the lab's ``prediction_drugs_final.csv`` before any check is reported.

  a. range        drug-treated values of the selected features vs the untreated training range;
                  drug-induced change in units of the between-woman SD
  b. plausibility is the treated feature vector a "plausible woman"? nearest-untreated-neighbour
                  distance of the treated vector vs that of the untreated vector itself
  c. null         the same per-feature change magnitudes with random signs (K draws per fold):
                  predicted shift of a size-matched random perturbation vs the drug's
  d. consistency  sign of each drug's mean predicted shift in each of the 100 fold models

    python stabl_launch/drug_arm_checks.py --out <dir> [--k-null 20]
    # same checks for another DOS model (its own selections + drug-treated tables):
    python stabl_launch/drug_arm_checks.py --features <untreated.csv> --selections <Selected Features STABL ALasso.csv>
        --drug-table '<dir>/{drug}_gaussot.csv' --out <dir> --no-validate
"""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

O = Path("/scratch/users/bmerwan/simco_work/onset_test")
LAB = Path("/home/groups/gbrice/ptb-drugscreen/ool_stabl/onset_test/"
           "comp_9drugs_long_unstimpred_treated/run_43/results_no_treatment")
DRUGS = ["SA", "RIF", "SALPZ", "CHT", "THF", "LPZ", "MAP", "PRA", "MF"]

p = argparse.ArgumentParser()
p.add_argument("--out", default=str(O / "drug_checks"))
p.add_argument("--k-null", type=int, default=20)
p.add_argument("--features", default=str(O / "ina_13OG_final_long_allstims_filtered.csv"))
p.add_argument("--selections", default=str(O / "foldfeats_GSS/Training CV/Selected Features STABL ALasso.csv"))
p.add_argument("--drug-table", default="/scratch/users/bmerwan/simco_work/cellot_results/ina_13OG_{drug}_pred_unstim.csv")
p.add_argument("--exclude", nargs="*", default=[],
               help="samples dropped from XGBoost training and testing; splits unchanged (as aligned_rescore.py)")
p.add_argument("--no-validate", action="store_true", help="skip the comparison with the lab's CellOT file")
args = p.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

X0 = pd.read_csv(args.features, index_col=0)
y = pd.read_csv(O / "outcome_table_all_pre.csv", index_col=0)["DOS"].astype(int)
rows = X0.index.intersection(y.index)
X0, y = X0.loc[rows], y.loc[rows]
XD = {d: pd.read_csv(args.drug_table.format(drug=d), index_col=0).reindex(index=rows, columns=X0.columns)
      for d in DRUGS}
sel = pd.read_csv(args.selections, index_col=0)
sel = sel["Fold selected features"].apply(ast.literal_eval).tolist()
params = json.load(open(O / "param_grid_files/params_43.json"))
groups = rows.to_series().str.split("_").str[0].values
splits = list(GroupShuffleSplit(100, test_size=0.2, random_state=42).split(np.zeros(len(rows)), y, groups))
assert len(sel) == len(splits) == 100
EX = np.isin(rows, args.exclude); assert EX.sum() == len(args.exclude), args.exclude
splits = [(tr[~EX[tr]], te[~EX[te]]) for tr, te in splits]
missing = {f for l in sel for f in l} - set(X0.columns)
assert not missing, f"selected features absent from --features: {sorted(missing)[:5]}"
nan_d = {d: int(XD[d][sorted({f for l in sel for f in l})].isna().sum().sum()) for d in DRUGS}
print("[CHECK] NaN in drug tables over selected features:", nan_d)

pred = {k: {r: [] for r in rows} for k in ["No_treat"] + DRUGS}
fold_rows, a_rows, b_rows, c_rows = [], [], [], []
for k, ((tr, te), feats) in enumerate(zip(splits, sel)):
    tri, tei = rows[tr], rows[te]
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("std", StandardScaler())])
    Xtr = pipe.fit_transform(X0.loc[tri, feats]); Xte = pipe.transform(X0.loc[tei, feats])
    m = XGBRegressor(**params, verbosity=0).fit(Xtr, y.loc[tri])
    p0 = m.predict(Xte)
    for r, v in zip(tei, p0): pred["No_treat"][r].append(v)
    lo, hi = Xtr.min(0), Xtr.max(0)
    # nearest-untreated-training-neighbour distances (standardised selected-feature space)
    def nn(Z): return np.sqrt(((Z[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)).min(1)
    nn0 = nn(Xte)
    for d in DRUGS:
        Xd = pipe.transform(XD[d].loc[tei, feats])
        pdg = m.predict(Xd)
        for r, v in zip(tei, pdg): pred[d][r].append(v)
        delta = Xd - Xte                                           # change in between-woman SD units
        a_rows.append({"fold": k, "drug": d, "n_feats": len(feats),
                       "median_abs_change_sd": np.median(np.abs(delta)),
                       "frac_values_outside_train_range": ((Xd < lo) | (Xd > hi)).mean(),
                       "frac_untreated_outside_train_range": ((Xte < lo) | (Xte > hi)).mean()})
        b_rows.append({"fold": k, "drug": d, "median_nn_untreated": np.median(nn0),
                       "median_nn_treated": np.median(nn(Xd)),
                       "frac_treated_beyond_p95_untreated": (nn(Xd) > np.percentile(nn0, 95)).mean()})
        shift = pdg - p0
        null = [m.predict(Xte + np.abs(delta) * rng.choice([-1, 1], size=delta.shape)) - p0
                for _ in range(args.k_null)]
        null_mean = np.array([n.mean() for n in null])
        c_rows.append({"fold": k, "drug": d, "drug_mean_shift": shift.mean(),
                       "drug_mean_abs_shift": np.abs(shift).mean(),
                       "null_mean_abs_shift": np.mean([np.abs(n).mean() for n in null]),
                       "null_sd_of_mean_shift": null_mean.std(),
                       "frac_samples_unchanged": (np.abs(shift) < 1e-6).mean()})
        fold_rows.append({"fold": k, "drug": d, "mean_shift": shift.mean()})
    print(f"[FOLD ] {k + 1}/100", flush=True)

# validation against the lab's stored medians
med = pd.DataFrame({key: pd.Series({r: np.median(v) for r, v in d.items() if v}) for key, d in pred.items()})
if not args.no_validate:
    # NB the lab's rescore has a row-order bug (CLAUDE.md, Landmines); ours uses aligned orders, so
    # per-sample predictions differ (up to ~20 d) while R2 (0.418 vs 0.415) and drug ranking agree.
    lab = pd.read_csv(LAB / "prediction_drugs_final.csv", index_col=1).drop(columns=["Unnamed: 0"], errors="ignore")
    lab = lab.rename(columns={"0": "No_treat"})
    common = med.index.intersection(lab.index)
    valid = {c: float(np.abs(med.loc[common, c] - lab.loc[common, c]).max()) for c in ["No_treat"] + DRUGS}
    json.dump(valid, open(out / "validation_max_abs_diff_vs_lab.json", "w"), indent=1)
    print("[CHECK] max |ours - lab| per column (days):", {k: round(v, 3) for k, v in valid.items()})
from sklearn.metrics import r2_score
print("[CHECK] untreated R2 (params_43, aligned orders): %.3f" % r2_score(y.loc[med.index], med["No_treat"]))

med.to_csv(out / "median_predictions.csv")
for name, rs in [("a_range", a_rows), ("b_plausibility", b_rows), ("c_null", c_rows), ("d_folds", fold_rows)]:
    pd.DataFrame(rs).to_csv(out / f"{name}.csv", index=False)

A = pd.DataFrame(a_rows).groupby("drug").mean(numeric_only=True).drop(columns="fold")
B = pd.DataFrame(b_rows).groupby("drug").mean(numeric_only=True).drop(columns="fold")
C = pd.DataFrame(c_rows).groupby("drug").mean(numeric_only=True).drop(columns="fold")
F = pd.DataFrame(fold_rows)
D = F.groupby("drug").mean_shift.agg(mean="mean", frac_folds_negative=lambda s: (s < 0).mean(),
                                     frac_folds_positive=lambda s: (s > 0).mean())
shift_med = (med[DRUGS].sub(med["No_treat"], axis=0))
D["median_pred_shift_days (lab-style)"] = shift_med.mean()
summary = pd.concat([A, B, C, D], axis=1).round(3)
summary.to_csv(out / "summary_by_drug.csv")
with pd.option_context("display.width", 250, "display.max_columns", 30):
    print(summary)
print(f"[DONE ] -> {out}")
