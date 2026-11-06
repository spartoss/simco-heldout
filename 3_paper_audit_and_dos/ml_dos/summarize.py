"""Collect every (arm, model) run into the tables the notebook reads.

    python ml_dos/summarize.py

<OUT>/summary/
  scores.csv        arm, model, family, pooled R2/RMSE/MAE (Stabl's per-sample-median convention), median and
                    mean per-split R2, n_splits; plus paired tests vs both references
  per_split.csv     arm, model, split, R2  (wide-able; the paired tests come from here)
  oof.csv           arm, model, sample, median out-of-fold prediction, y
  inner_choice.csv  arm, model, split, inner RMSE, best params (json)
  auto_<arm>.csv    "auto": per split, the model with the lowest inner GroupKFold RMSE -- a model choice made on
                    the training women only, i.e. the honest answer to "the best model of the zoo"
  ega_context.csv   partial correlation of each model's OOF prediction with DOS given EGA (EGA used for
                    evaluation only, never as an input)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import common as C

FAMILY = {"mean": "null", "ridge": "linear", "lasso": "linear", "enet": "linear", "pls": "linear", "pcr": "linear",
          "svr_lin": "linear", "svr_rbf": "kernel", "krr_rbf": "kernel", "knn": "kernel", "gp": "kernel",
          "rf": "bagging", "et": "bagging", "bag_tree": "bagging", "bag_ridge": "bagging",
          "xgb_p43": "boosting", "xgb": "boosting", "hgb": "boosting", "gbr": "boosting", "ada": "boosting",
          "mlp": "neural", "dnn": "neural", "dnn_wide": "neural", "deepsets": "neural (cells)",
          "stack": "ensemble", "auto": "ensemble", "stabl_p43": "reference"}
# models whose inner RMSE comes from the same GroupKFold(5) on the same training rows -> comparable
AUTO_POOL = [m for m, f in FAMILY.items() if f in ("linear", "kernel", "bagging", "boosting", "neural")
             and m not in ("deepsets", "xgb_p43")]


def load_runs():
    runs = {}
    for d in sorted((C.OUT / "preds").glob("*/*")):
        fs = sorted(d.glob("split_*.csv"))
        if not fs:
            continue
        runs[(d.parent.name, d.name)] = pd.concat([pd.read_csv(f) for f in fs])
    return runs


def inner(d):
    rows = []
    for f in sorted(d.glob("split_*.json")):
        j = json.load(open(f))
        rows.append({"split": int(f.stem[-3:]), "inner_rmse": j.get("inner_rmse", np.nan),
                     "seconds": j.get("seconds", np.nan), "params": json.dumps(j.get("best_params", j.get("weights", {})))})
    return pd.DataFrame(rows)


def paired(a, b):
    """a, b: per-split R2 Series on the same split index. Positive diff = a better."""
    k = a.index.intersection(b.index)
    d = (a.loc[k] - b.loc[k])
    p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
    return dict(mean_diff=d.mean(), median_diff=d.median(), wins=int((d > 0).sum()), n_paired=len(k), p=p)


def main():
    out = C.OUT / "summary"
    out.mkdir(exist_ok=True)
    runs = load_runs()
    # honest "best of the zoo": per arm and split, the model with the lowest inner RMSE
    # (only on the four main arms, where the whole zoo ran; the ablation arms run a subset)
    for arm in ["freqmed", "qtl", "full"]:
        ch = []
        for m in AUTO_POOL:
            if (arm, m) in runs:
                ch.append(inner(C.OUT / "preds" / arm / m).assign(model=m))
        # ignore_index: each frame is indexed 0..99, and duplicate labels made .loc[idxmin] return every
        # model (bug until 2026-10-07: "auto" was the median over all models, not a per-split choice)
        ch = pd.concat(ch, ignore_index=True).dropna(subset=["inner_rmse"])
        best = ch.loc[ch.groupby("split").inner_rmse.idxmin()]
        pl = pd.concat([runs[(arm, m)][runs[(arm, m)].split == s] for s, m in zip(best.split, best.model)])
        runs[(arm, "auto")] = pl
        best.to_csv(out / f"auto_{arm}.csv", index=False)

    idx = C.rows()
    clin = pd.read_csv(C.O / "Preprocessed_OOL_Clinical_new.csv", index_col=0)
    ps = {k: C.per_split(v) for k, v in runs.items()}
    refs = {"cellot": ps[("ref_cellot", "stabl_p43")], "comb_cellot": ps[("ref_comb_cellot", "stabl_p43")]}
    rows, oof, ega = [], [], []
    for (arm, m), pl in runs.items():
        sc, med = C.pooled(pl)
        r = dict(arm=arm, model=m, family=FAMILY.get(m, "?"), n_splits=pl.split.nunique(), **sc,
                 median_split_R2=ps[(arm, m)].median(), mean_split_R2=ps[(arm, m)].mean())
        for rn, rv in refs.items():
            for kk, vv in paired(ps[(arm, m)], rv).items():
                r[f"vs_{rn}_{kk}"] = vv
        rows.append(r)
        y = pl.groupby("sample").y.first().loc[med.index]
        oof.append(pd.DataFrame({"arm": arm, "model": m, "sample": med.index, "pred": med.values, "y": y.values}))
        e = clin.loc[med.index, "EGA"].values
        ry = y.values - LinearRegression().fit(e[:, None], y.values).predict(e[:, None])
        rp = med.values - LinearRegression().fit(e[:, None], med.values).predict(e[:, None])
        ega.append(dict(arm=arm, model=m, r_pred_DOS=pearsonr(med, y)[0], r_pred_EGA=pearsonr(med, e)[0],
                        partial_r_given_EGA=pearsonr(rp, ry)[0], partial_p=pearsonr(rp, ry)[1]))
    S = pd.DataFrame(rows)
    # Holm over the valid comparisons only: measured Unstim arms + DeepSets on Unstim cells. Feature sets with the
    # OOL 15-min stims or clinical variables are excluded (user, 2026-10-07: only the data CellOT + Stabl used), and
    # the CellOT-table control arms are a separate question.
    VALID = ["freqmed", "qtl", "full", "cells"]
    for rn in refs:
        m = S.arm.isin(VALID)
        p = S.loc[m, f"vs_{rn}_p"].values
        o = np.argsort(p)
        adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o]).clip(max=1)
        h = np.empty_like(adj); h[o] = adj
        S.loc[m, f"vs_{rn}_p_holm"] = h
    S.sort_values("R2", ascending=False).to_csv(out / "scores.csv", index=False)
    pd.concat([s.rename("R2").rename_axis("split").reset_index().assign(arm=a, model=m)
               for (a, m), s in ps.items()]).to_csv(out / "per_split.csv", index=False)
    pd.concat(oof).to_csv(out / "oof.csv", index=False)
    pd.DataFrame(ega).to_csv(out / "ega_context.csv", index=False)
    ic = []
    for (arm, m) in runs:
        d = C.OUT / "preds" / arm / m
        if d.exists():
            ic.append(inner(d).assign(arm=arm, model=m))
    pd.concat(ic).to_csv(out / "inner_choice.csv", index=False)
    imp = []
    for (arm, m) in runs:
        fs = sorted((C.OUT / "preds" / arm / m).glob("imp_*.csv")) if (C.OUT / "preds" / arm / m).exists() else []
        if fs:
            d = pd.concat([pd.read_csv(f) for f in fs])
            d["importance"] = d.groupby("split").importance.transform(lambda v: v / v.sum())
            g = d.groupby("feature").agg(mean_share=("importance", lambda v: v.sum() / len(fs)),
                                         n_splits=("split", "nunique")).reset_index()
            imp.append(g.assign(arm=arm, model=m))
    if imp:
        pd.concat(imp).to_csv(out / "importances.csv", index=False)
    show = S.sort_values("R2", ascending=False)[["arm", "model", "R2", "RMSE", "median_split_R2",
                                                 "vs_cellot_mean_diff", "vs_cellot_wins", "vs_cellot_p"]]
    print(show.head(40).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
