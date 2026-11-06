"""Summaries for lineage round 2 from lc/tab_compare.py --dump case tables (one per comparison set).

For every model column: alone and as w * CellOT + (1 - w) * model (w = 0.4; CellOT = the column named --cellot), MAE and
MSE pooled over cases and averaged per donor (each donor once, its tests averaged), and against --ref (same kind:
alone vs alone, mix vs mix) the number of donors better and a Wilcoxon p over donors. Grouped vs per-celltype: for each
column G<set>_<L> with a matching ct_<L>, the % MSE change per (celltype, stim) and per celltype, with Wilcoxon p over
donors. Rule cases only by default. Writes <dir>/lin_summary.csv, lin_group_vs_ct.csv.
    python -m lc.lineage2_summary --cases WORK/tab/lin2_<tag>/cases.pkl --cellot CellOT --ref ct_L1cat
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def per_donor(g, col, f):
    return g.assign(_e=f(g[col] - g.true)).groupby(["donor", "repeat"])._e.mean().groupby(level="donor").mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--cellot", default="CellOT")
    ap.add_argument("--ref", default="ct_L1cat")
    ap.add_argument("--w", type=float, default=0.4)
    ap.add_argument("--set", default="rule")
    a = ap.parse_args()
    D = pd.read_pickle(a.cases); D = D[D.set.str.startswith(a.set)].copy()
    if "CellOT+shift" not in D: D["CellOT+shift"] = 0.5 * D[a.cellot] + 0.5 * D["shift"]
    base = ["shift", a.cellot, "CellOT+shift"]
    models = [c for c in D.columns if c.startswith(("ct_", "G", "lgb_", "pfn_", "old_"))]
    for m in models: D[f"mix_{m}"] = a.w * D[a.cellot] + (1 - a.w) * D[m]
    rows = []
    for m in base + models + [f"mix_{m}" for m in models]:
        e = D[m] - D.true; r = {"method": m, "MAE": e.abs().mean(), "MSE": (e ** 2).mean(),
                                "MAE_donor": per_donor(D, m, np.abs).mean(), "MSE_donor": per_donor(D, m, np.square).mean()}
        ref = ("mix_" + a.ref) if m.startswith("mix_") else a.ref
        if ref in D and m != ref and m not in base:
            x, y = per_donor(D, m, np.square), per_donor(D, ref, np.square)
            r.update({"vs": ref, "donors_better": int((x < y).sum()), "n_donors": len(x), "p": wilcoxon(x - y).pvalue})
        rows.append(r)
    S = pd.DataFrame(rows); out = Path(a.cases).parent; S.round(5).to_csv(out / "lin_summary.csv", index=False)
    gv = []
    for m in [c for c in models if c.startswith("G")]:
        ct = "ct_" + m.split("_", 1)[1]
        if ct not in D: continue
        for keys, g in list(D.groupby(["celltype", "stim"])) + [((c, "ALL"), g) for c, g in D.groupby("celltype")]:
            x, y = per_donor(g, m, np.square), per_donor(g, ct, np.square)
            gv.append({"group_model": m, "ct_model": ct, "celltype": keys[0], "stim": keys[1],
                       "pct_change": 100 * (((g[m] - g.true) ** 2).mean() / ((g[ct] - g.true) ** 2).mean() - 1),
                       "donors_better": int((x < y).sum()), "n_donors": len(x),
                       "p": wilcoxon(x - y).pvalue if len(x) > 5 and (x != y).any() else np.nan})
    G = pd.DataFrame(gv); G.round(4).to_csv(out / "lin_group_vs_ct.csv", index=False)
    pd.set_option("display.width", 230)
    print(f"=== {a.cases}: {len(D)} {a.set} cases, {D.donor.nunique()} donors")
    print(S.set_index("method").sort_values("MSE_donor").round(4).to_string())
    if len(G): print(G[G.stim == "ALL"].pivot(index="celltype", columns="group_model", values="pct_change").round(1).to_string())
    print(f"[DONE ] -> {out}")


if __name__ == "__main__":
    main()
