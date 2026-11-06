"""Preterm detection from the stored out-of-fold DOS predictions -- nothing is refit.

    python ml_dos/preterm_auc.py      # -> <OUT>/summary/preterm_auc.csv, preterm_scores.csv

Rule (fixed, no parameter learnt on the preterm label):
  predicted pregnancy length L = 7 * EGA - predicted DOS      (days; EGA in weeks, DOS negative)
  score = -L (shorter predicted pregnancy = more preterm); AUROC over visits, preterm = `Delivery` == preterm
  threshold version: preterm if L < 259 d (37 weeks), i.e. predicted delivery > 21 days before the due date.
References on the same visits:
  due-date rule   L = 280 for everyone (predicted DOS = 7 (EGA - 40))   -> AUROC 0.5 by construction
  constant DOS    the training-mean prediction (`mean` model)           -> what timing of the visits alone gives:
                  preterm women deliver early, so they are sampled earlier and L = 7 EGA - c is lower for them.
Also reported: G1 only (one visit per woman, timing roughly matched), and the score with its linear EGA trend removed
(residual of L on EGA over all visits, labels unused) -- what is left of the score once the visit timing is taken out.
Uncertainty is by woman (5 preterm women): permutation of the preterm label across women, bootstrap over women.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import common as C

MODELS = [("ref_cellot", "stabl_p43", "CellOT + Stabl (paper)"),
          ("ref_comb_cellot", "stabl_p43", "CellOT + measured + Stabl"),
          ("full", "dnn_wide", "wide deep network"), ("full", "dnn", "deep network"),
          ("full", "stack", "stacked ensemble"), ("full", "auto", "auto"),
          ("full", "svr_rbf", "SVR (RBF)"), ("full", "krr_rbf", "kernel ridge"), ("full", "gp", "Gaussian process"),
          ("full", "mlp", "sklearn MLP"), ("cells", "deepsets", "DeepSets (raw cells)"),
          ("full", "mean", "constant DOS (no information)")]
N_PERM, N_BOOT = 10000, 2000


def woman_stats(score, pre, woman, rng):
    auc = roc_auc_score(pre, score)
    w = pd.Series(pre, index=woman).groupby(level=0).first()
    perm = []
    for _ in range(N_PERM):
        lab = pd.Series(rng.permutation(w.values), index=w.index).loc[woman].values
        perm.append(roc_auc_score(lab, score))
    p = (np.sum(np.array(perm) >= auc) + 1) / (N_PERM + 1)
    ws = w.index.values
    boot = []
    for _ in range(N_BOOT):
        pick = rng.choice(ws, len(ws), replace=True)
        idx = np.concatenate([np.flatnonzero(woman == x) for x in pick])
        if pre[idx].min() != pre[idx].max():
            boot.append(roc_auc_score(pre[idx], score[idx]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return auc, p, lo, hi


def main():
    rng = np.random.default_rng(0)
    oof = pd.read_csv(C.OUT / "summary/oof.csv")
    clin = pd.read_csv(C.O / "Preprocessed_OOL_Clinical_new.csv", index_col=0)
    rows, scores = [], []
    for arm, model, label in MODELS:
        d = oof.query("arm == @arm and model == @model").set_index("sample")
        if d.empty:
            continue
        c = clin.loc[d.index]
        L = 7 * c.EGA.values - d.pred.values
        pre = (c.Delivery == "preterm").values.astype(int)
        woman = c.Patient.astype(str).values
        g1 = (c.Timepoint == "G1").values
        Lres = L - np.polyval(np.polyfit(c.EGA.values, L, 1), c.EGA.values)
        r = dict(model=label, arm=arm, key=model, n_visits=len(L), n_preterm_visits=int(pre.sum()),
                 n_preterm_women=len(set(woman[pre == 1])))
        for tag, s, m in [("all", -L, slice(None)), ("G1", -L, g1), ("all_detrended", -Lres, slice(None)),
                          ("G1_detrended", -Lres, g1)]:
            auc, p, lo, hi = woman_stats(s[m], pre[m], woman[m], rng)
            r.update({f"AUROC_{tag}": auc, f"p_{tag}": p, f"CI_lo_{tag}": lo, f"CI_hi_{tag}": hi})
        flag = L < 259
        r.update(sens_259=flag[pre == 1].mean(), spec_259=1 - flag[pre == 0].mean(),
                 women_flagged_259=len(set(woman[(pre == 1) & flag])),
                 mean_L_preterm=L[pre == 1].mean(), mean_L_term=L[pre == 0].mean(),
                 slope_L_per_EGA_week=np.polyfit(c.EGA.values, L, 1)[0])
        rows.append(r)
        scores.append(pd.DataFrame({"model": label, "sample": d.index, "EGA": c.EGA.values, "true_DOS": d.y.values,
                                    "pred_DOS": d.pred.values, "pred_length": L, "true_length": 7 * c.EGA.values - d.y.values,
                                    "preterm": pre, "visit": c.Timepoint.values}))
    T = pd.DataFrame(rows)
    T.to_csv(C.OUT / "summary/preterm_auc.csv", index=False)
    pd.concat(scores).to_csv(C.OUT / "summary/preterm_scores.csv", index=False)
    show = ["model", "AUROC_all", "CI_lo_all", "CI_hi_all", "p_all", "AUROC_G1", "p_G1", "AUROC_all_detrended",
            "p_all_detrended", "AUROC_G1_detrended", "p_G1_detrended", "sens_259", "spec_259", "women_flagged_259",
            "slope_L_per_EGA_week"]
    print(T[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
