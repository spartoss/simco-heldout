"""Fig 5b protocol (Stabl ALasso, 100 splits, aligned XGBoost rescore): the new best held-out model's simulated stims
(tab_both, mix_both; atlas_heldout/lc/ool_tab.py) against CellOT (paper), gauss_ot and measured quantiles.
Prints and writes paper_protocol/ool_tab_summary.csv: pooled R2 / RMSE at params_43 (all 177 and without
36_G1 / 52_G1), best-of-97 (aligned, optimistic), and per-split paired R2 differences vs CellOT (Wilcoxon).
    python stabl_launch/ool_tab_summary.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

P = Path("/scratch/users/bmerwan/simco_work/onset_test/paper_protocol")
ARMS = {"cellot": "CellOT (paper Fig 5b)", "gaussot_both": "gauss_ot", "d_qtl": "measured freq+med+qtl",
        "tab_both": "shift + TabPFN (new)", "mix_both": "0.4 CellOT + 0.6 (shift + TabPFN) (new)",
        "tab_raw_both": "shift + TabPFN, raw OOL (as CellOT)", "mix_raw_both": "0.4 CellOT + 0.6 tab, raw OOL"}


def load(sub, arm):
    s, f = P / sub / arm / "sweep.csv", P / sub / arm / "folds.csv"
    if not s.exists(): return None, None
    S = pd.read_csv(s).set_index("params"); F = pd.read_csv(f)
    return S, F


rows = []
ref = {sub: load(sub, "cellot")[1] for sub in ["aligned", "aligned_excl"]}
for arm, lab in ARMS.items():
    r = {"arm": arm, "features": lab}
    for sub, tag in [("aligned", "all177"), ("aligned_excl", "excl")]:
        S, F = load(sub, arm)
        if S is None or "params_43" not in S.index: continue
        r[f"R2_p43_{tag}"] = S.loc["params_43", "R2"]; r[f"RMSE_p43_{tag}"] = S.loc["params_43", "RMSE"]
        r[f"median_split_R2_{tag}"] = S.loc["params_43", "median_split_R2"]
        if len(S) > 1: r[f"best97_R2_{tag}"] = S.R2.max()
        if arm != "cellot" and ref[sub] is not None:
            d = (F["params_43"] - ref[sub]["params_43"]).to_numpy()
            r[f"diff_vs_cellot_{tag}"] = d.mean(); r[f"wins_vs_cellot_{tag}"] = f"{int((d > 0).sum())}/{len(d)}"
            r[f"p_vs_cellot_{tag}"] = wilcoxon(d).pvalue
    rows.append(r)
T = pd.DataFrame(rows)
T.to_csv(P / "ool_tab_summary.csv", index=False)
pd.set_option("display.width", 250)
print(T.round(3).to_string(index=False))
