"""Summary of the held-out drug scores (heldout.score_drugs): which pipeline predicts a NEW donor's drug effect?

Cases = (repeat, test donor, stim, celltype, drug, q, marker). A case is scored only where the effect is real:
|true| > FLOOR (the technical floor) and |true| > 2 * noise (the donor's own split-half noise). For each prediction
column, err = |pred - true| and skill = 1 - err / |true| (1 = perfect, 0 = no better than predicting "no effect").
Tables (WORK/drugs/analysis/):
  overall.csv      median skill and median err per column, all real cases, used features, effect > 0.1
  by_q.csv         median skill per column and quantile
  by_stim.csv      median skill per column and stim (Unstim = drug without stimulation)
  paired.csv       paired comparisons (Wilcoxon on err, share of cases where A beats B) for the pairs that matter
  confound.csv     the same with LPZ/SA under PI (the DMSO1 plate) separated
"Used" = (stim, celltype, marker) of the Fig 5b model selected in >= 5 % of folds (simco drug_choice), at q = 50.
    python -m heldout.analyze_drugs [--scores DIR]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from heldout.common import FLOOR, SIMCO_WORK, WORK

PAIRS_TO_TEST = [("chain:CellOT>CellOT", "feature:CellOT>quantile_map"),
                 ("chain:CellOT>CellOT", "feature:CellOT>median_shift"),
                 ("feature:CellOT>quantile_map", "feature:gauss_ot>quantile_map"),
                 ("feature:CellOT>quantile_map", "feature:CellOT>median_shift"),
                 ("chain:CellOT>CellOT", "chain:gauss_ot>gauss_ot"),
                 ("step:CellOT", "step:quantile_map"), ("step:CellOT", "step:median_shift"),
                 ("step:CellOT", "step:gauss_ot"), ("step:quantile_map", "step:median_shift")]


def skill(D, cols):
    t = D["true"].abs()
    return pd.DataFrame({c: 1 - (D[c] - D["true"]).abs() / t for c in cols}, index=D.index)


def med(S, by=None):
    return (S.median() if by is None else S.groupby(by).median()).round(3)


def paired(D, a, b):
    d = pd.concat([(D[a] - D["true"]).abs(), (D[b] - D["true"]).abs()], axis=1).dropna()
    if len(d) < 10: return {"A": a, "B": b, "n": len(d)}
    ea, eb = d.iloc[:, 0], d.iloc[:, 1]
    return {"A": a, "B": b, "n": len(d), "A_wins": round(float((ea < eb).mean()), 3),
            "median_err_A": round(float(ea.median()), 4), "median_err_B": round(float(eb.median()), 4),
            "p": float(wilcoxon(ea, eb).pvalue)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scores", default=str(WORK / "drugs/scores"))
    ap.add_argument("--out", default=str(WORK / "drugs/analysis")); a = ap.parse_args()
    D = pd.concat([pd.read_csv(f) for f in sorted(Path(a.scores).glob("r*.csv.gz"))], ignore_index=True)
    cols = [c for c in D if c.split(":")[0] in ("step", "chain", "feature")]
    n_all = len(D)
    D = D[(D["true"].abs() > FLOOR) & (D["true"].abs() > 2 * D["noise"])].copy()
    print(f"{n_all} cases, {len(D)} real (|true| > {FLOOR} and > 2 x split-half noise); "
          f"{D.repeat.nunique()} repeats, {D.donor.nunique()} test donors")
    used = pd.read_csv(SIMCO_WORK / "drug_choice/selected_features_fig5b.csv")
    used = set(map(tuple, used[used.sel_freq >= 0.05][["stim", "celltype", "marker"]].to_numpy()))
    D["used"] = [(s, c, m) in used and q == 50 for s, c, m, q in zip(D.stim, D.celltype, D.marker, D.q)]
    D["dmso1_PI"] = D.drug.isin(["LPZ", "SA"]) & (D.stim == "PI")
    S = skill(D, cols)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    subsets = {"all real": D.index, "effect > 0.1": D.index[D["true"].abs() > 0.1],
               "used features (q50)": D.index[D.used], "without LPZ/SA under PI": D.index[~D.dmso1_PI]}
    ov = pd.DataFrame({f"skill | {k}": med(S.loc[i]) for k, i in subsets.items()})
    ov["median err | all real"] = (D[cols].sub(D["true"], axis=0)).abs().median().round(4)
    ov.insert(0, "n cases", D[cols].notna().sum())
    ov = ov.sort_values("skill | all real", ascending=False); ov.to_csv(out / "overall.csv")
    print(f"\ncases per subset: " + ", ".join(f"{k} {len(i)}" for k, i in subsets.items()))
    print(ov.to_string())
    bq = med(S, D.q).T; bq.to_csv(out / "by_q.csv"); print("\nmedian skill by quantile\n" + bq.to_string())
    bs = med(S, D.stim).T; bs.to_csv(out / "by_stim.csv"); print("\nmedian skill by stim\n" + bs.to_string())
    P = []
    for k, i in subsets.items():
        for x, y in PAIRS_TO_TEST:
            if x in D and y in D: P.append({"subset": k, **paired(D.loc[i], x, y)})
    P = pd.DataFrame(P); P.to_csv(out / "paired.csv", index=False)
    print("\npaired (A_wins = share of cases where A's error is smaller; cases are not independent)\n" + P.to_string(index=False))
    cf = pd.DataFrame({g: med(S[D.dmso1_PI == g]) for g in (True, False)}).rename(
        columns={True: "LPZ/SA under PI", False: "everything else"})
    cf.to_csv(out / "confound.csv"); print("\nDMSO1 plate under PI vs the rest\n" + cf.to_string())


if __name__ == "__main__":
    main()
