"""Leave-one-donor-out summary (design_lodo.json: 19 folds, train on 18 donors, test the 19th).

Reads the case table of lc/tab_compare.py --dump (k = 18) and reports, per case set (floor / rule): MAE, MSE and
between_R2 for every method and for mixes built here from the same cases:
  <cellot>+shift = 0.5 CellOT + 0.5 shift;  w<cellot>+<model> = w CellOT + (1 - w) model  (w in --weights).
between_R2 is POOLED over folds: per (stim, celltype, q, marker), predictions and truths are centred across the 19
left-out donors (each from its own fold), so it measures whether predicted donor differences follow the real ones.
Paired: % of cases closer than --ref and Wilcoxon over (donor, stim, celltype) mean |error|.
    python -m lc.lodo_summary --cases WORK/tab/lodo_compare/cases.pkl --cellots CellOT_orig CellOT_paired \
        --models pfn_L1cat ... --ref CellOT_paired+shift
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--cellots", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--weights", nargs="*", type=float, default=[0.4])
    ap.add_argument("--ref", default=None)
    a = ap.parse_args()
    D = pd.read_pickle(a.cases)
    names = ["shift"] + a.cellots + a.models
    for c in a.cellots:
        D[f"{c}+shift"] = 0.5 * D[c] + 0.5 * D["shift"]; names.append(f"{c}+shift")
        for w in a.weights:
            for m in a.models:
                n = f"{w:g}{c}+{m}"; D[n] = w * D[c] + (1 - w) * D[m]; names.append(n)
    key = ["stim", "celltype", "q", "marker"]
    rows, pr = [], []
    for s, g in D.groupby("set"):
        tc = g.true - g.groupby(key).true.transform("mean"); nd = g.groupby(key).true.transform("size"); ok = nd >= 3
        for m in names:
            e = g[m] - g.true; pc = g[m] - g.groupby(key)[m].transform("mean")
            rows.append({"set": s, "method": m, "MAE": e.abs().mean(), "MSE": (e ** 2).mean(),
                         "between_R2": 1 - ((pc - tc)[ok] ** 2).sum() / (tc[ok] ** 2).sum(), "n": len(g)})
            if a.ref and m != a.ref:
                ea, eb = e.abs(), (g[a.ref] - g.true).abs()
                gm = pd.DataFrame({"a": ea, "b": eb, "donor": g.donor, "stim": g.stim, "celltype": g.celltype}) \
                    .groupby(["donor", "stim", "celltype"])[["a", "b"]].mean()
                pr.append({"set": s, "method": m, "pct_closer_than_ref": 100 * (ea < eb).mean(),
                           "p_wilcoxon": wilcoxon(gm.a, gm.b).pvalue})
    R = pd.DataFrame(rows)
    if pr: R = R.merge(pd.DataFrame(pr), on=["set", "method"], how="left")
    out = Path(a.cases).parent / "lodo_summary.csv"; R.round(5).to_csv(out, index=False)
    pd.set_option("display.width", 220)
    for s, g in R.groupby("set"):
        print(f"=== {s}  (n = {int(g.n.iloc[0])} cases, 19 left-out donors)")
        print(g.drop(columns=["set", "n"]).set_index("method").sort_values("MSE", ascending=False).round(4).to_string())
    print(f"[DONE ] -> {out}")


if __name__ == "__main__":
    main()
