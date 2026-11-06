"""Summary of the quantile re-test (heldout.quantile_arms -> Stabl 100 splits -> aligned 97-config rescore).

Per feature set: pooled R2 at params_43 and best / median over the 97 XGBoost settings (aligned rescore, 36_G1 / 52_G1
already removed from the tables), Stabl's median number of selected features and CVS (ALasso), what the selections are
made of, and per-split paired Wilcoxon tests against ref852 and rule_med at params_43 (and best vs best).
A set whose sweep is incomplete is flagged; its params_43 numbers are final, its best / median are provisional.
    python -m heldout.analyze_quantile_arms -> WORK/quantile_arms/summary/
"""
import ast
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from heldout.common import STIMS, WORK

Q = WORK / "quantile_arms"
ARMS = ["ref852", "rule_med", "rule_rawq", "rule_selq", "rule_shape", "rule_rep", "rule_corr", "selq_corr", "selq_corr_eff"]
N_CONFIGS = 97


def family(c):
    last = c.split("_")[-1]
    if last in STIMS: return "sim median"
    if re.search(r"q$", last) and last[:-1] in STIMS: return "sim quantile / spread"
    if c.endswith("_freq"): return "measured freq"
    if c.endswith("_med"): return "measured median"
    return "measured quantile / spread"


def main():
    out = Q / "summary"; out.mkdir(exist_ok=True)
    rows, folds, fams = [], {}, []
    for a in ARMS:
        sw, fo = Q / f"aligned/{a}/sweep.csv", Q / f"aligned/{a}/folds.csv"
        if not sw.exists(): continue
        s, f = pd.read_csv(sw), pd.read_csv(fo)
        best = s.sort_values("R2").iloc[-1].params
        folds[a] = {"p43": f["params_43"], "best": f[best]}
        sel = pd.read_csv(Q / f"stabl/{a}/Training CV/Selected Features STABL ALasso.csv", index_col=0)
        lists = [ast.literal_eval(x) for x in sel["Fold selected features"]]
        sc = pd.read_csv(Q / f"stabl/{a}/Summary/Scores training CV.csv", index_col=0)
        n_tab = pd.read_csv(Q / f"tables/{a}.csv", nrows=1, index_col=0).shape[1]
        allsel = pd.Series([family(c) for l in lists for c in l])
        rows.append({"set": a, "columns": n_tab, "sweep": f"{len(s)}/{N_CONFIGS}",
                     "R2 params_43": s.set_index("params").R2["params_43"], "R2 best of sweep": s.R2.max(),
                     "R2 median config": s.R2.median(), "median split R2 (params_43)": f["params_43"].median(),
                     "N features (median)": np.median([len(l) for l in lists]),
                     "CVS": sc.loc["STABL ALasso", "CVS"].split(" ")[0],
                     **{f"sel: {k}": v for k, v in allsel.value_counts(normalize=True).round(3).items()}})
        for k, v in allsel.value_counts(normalize=True).items(): fams.append({"set": a, "family": k, "share": v})
    T = pd.DataFrame(rows).set_index("set"); T.to_csv(out / "sets.csv")
    P = []
    for ref in ["ref852", "rule_med"]:
        for a in folds:
            if a == ref: continue
            for kind in ["p43", "best"]:
                d = (folds[a][kind] - folds[ref][kind]).dropna()
                P.append({"set": a, "vs": ref, "config": "params_43" if kind == "p43" else "best vs best",
                          "mean per-split diff": d.mean(), "wins /100": int((d > 0).sum()), "Wilcoxon p": wilcoxon(d).pvalue})
    P = pd.DataFrame(P); P.to_csv(out / "paired.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(3).T.to_string()); print(); print(P.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
