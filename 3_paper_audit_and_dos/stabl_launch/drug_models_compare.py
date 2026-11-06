"""Compare the drug arm across DOS models (drug maps fixed = the paper's CellOT drug maps).

Reads drug_checks/ (paper CellOT tables, aligned orders) and drug_models/checks_<model>/, writes
drug_models/compare_by_drug.csv and prints: per drug and model the mean predicted shift (days,
negative = labor later), fraction of the 100 fold models agreeing in sign, the shift in units of the
size-matched random-sign null SD, the share of implausible treated women (check b), and each model's
untreated R2 + the Spearman agreement of its drug ranking with CellOT's.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

O = Path("/scratch/users/bmerwan/simco_work/onset_test")
M = O / "drug_models"
models = {"CellOT (paper tables)": O / "drug_checks", "CellOT (rebuilt)": M / "checks_cellot_rebuilt",
          "gauss_ot": M / "checks_gaussot", "gauss_ot + gauss_ot drug maps": M / "checks_gaussot_full",
          "measured + quantiles": M / "checks_meas_qtl",
          "measured freq + medians": M / "checks_meas_freqmed"}
y = pd.read_csv(O / "outcome_table_all_pre.csv", index_col=0)["DOS"]
rows, rank = [], {}
for name, d in models.items():
    if not (d / "summary_by_drug.csv").exists():
        print(f"[MISS ] {name}"); continue
    s = pd.read_csv(d / "summary_by_drug.csv", index_col=0)
    c = pd.read_csv(d / "c_null.csv").groupby("drug").mean(numeric_only=True)
    med = pd.read_csv(d / "median_predictions.csv", index_col=0)
    r2 = r2_score(y.loc[med.index], med["No_treat"])
    for drug, r in s.iterrows():
        agree = max(r["frac_folds_negative"], r["frac_folds_positive"])
        rows.append({"model": name, "untreated R2": round(r2, 3), "drug": drug,
                     "mean shift (d)": r["mean"], "folds same sign": agree,
                     "shift / null SD": r["mean"] / c.loc[drug, "null_sd_of_mean_shift"],
                     "frac implausible (b)": r["frac_treated_beyond_p95_untreated"],
                     "frac out of range (a)": r["frac_values_outside_train_range"]})
    rank[name] = s["mean"]
t = pd.DataFrame(rows).round(3)
t.to_csv(M / "compare_by_drug.csv", index=False)
with pd.option_context("display.width", 220, "display.max_rows", 200):
    print(t.pivot_table(index="drug", columns="model", values="mean shift (d)").round(1)
          .sort_values(list(models)[0] if list(models)[0] in rank else t.model.iloc[0]))
    print(t.pivot_table(index="drug", columns="model", values="folds same sign").round(2))
    print(t.pivot_table(index="drug", columns="model", values="shift / null SD").round(1))
    print(t.groupby("model")[["untreated R2", "frac implausible (b)", "frac out of range (a)"]].mean().round(3))
ref = rank.get("CellOT (paper tables)")
if ref is not None:
    for name, v in rank.items():
        print(f"Spearman(drug ranking, CellOT paper) {name:26s} {spearmanr(ref, v.loc[ref.index])[0]:+.2f}")
print("[DONE ] ->", M / "compare_by_drug.csv")
