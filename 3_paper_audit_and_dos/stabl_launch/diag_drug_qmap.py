"""quantile_map vs CellOT / gauss_ot on the drug arm's per-quantile errors, and where the gaps are.

Atlas pooled drug benchmark with quantile errors (simco_work/baselines/pooled_drug_q, cell-weighted), cases
(celltype, stim, drug, marker, quantile) whose observed effect |q_drug - q_DMSO| exceeds the 0.0273 technical
floor. Paired Wilcoxon per quantile, both per case and on per-(celltype, stim, drug) means (less optimistic:
cases within a pair are not independent). Then the CellOT - quantile_map gap by stim / celltype / marker / drug,
and whether the (celltype, marker, stim) features with the largest gaps are the ones Stabl selects: selection
frequency over the 100 folds of the paper's Fig 5b ALasso run (foldfeats_GSS) and of CellOT + measured
(paper_protocol/stabl_n100/comb_cellot), plus |SHAP| rank in CellOT + measured (diagnosis/shap_by_feature.csv).
    python stabl_launch/diag_drug_qmap.py   -> simco_work/diagnosis/drug_qmap/*.csv
"""
import ast
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

W = Path("/scratch/users/bmerwan/simco_work"); O = W / "onset_test"
OUT = W / "diagnosis/drug_qmap"; OUT.mkdir(parents=True, exist_ok=True)
FLOOR = 0.0273
QL = ["q10", "q25", "median", "q75", "q90"]
M = {"cellot_insample": "CellOT", "gauss_ot": "gauss_ot", "quantile_map": "quantile_map", "identity": "identity"}

raw = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(str(W / "baselines/pooled_drug_q/*.csv"))))
raw = raw[raw.weighting.isin(["cell", "invariant"]) & raw.method.isin(M)]
mk = [c[len("median_error__"):] for c in raw.columns if c.startswith("median_error__")]
keys = ["celltype", "stim", "drug"]
parts = []
for q in QL:
    pre = "median_error__" if q == "median" else f"{q}_error__"
    m = raw.melt(id_vars=keys + ["method"], value_vars=[pre + x for x in mk], var_name="marker", value_name="err")
    m["marker"] = m.marker.str[len(pre):]; m["q"] = q; parts.append(m)
w = pd.concat(parts).pivot_table(index=keys + ["marker", "q"], columns="method", values="err").rename(columns=M)
w = w[w.identity > FLOOR].dropna(subset=["CellOT", "gauss_ot", "quantile_map"]).reset_index()
print(f"cases above the floor: {len(w)} ({w.groupby(keys).ngroups} pairs)")

# 1. paired table
rows = []
for q in QL + ["all"]:
    s = w if q == "all" else w[w.q == q]
    for other in ["CellOT", "gauss_ot"]:
        d = s[other] - s.quantile_map                      # > 0: quantile_map better
        dp = (s.assign(d=d).groupby(keys).d.mean())         # per-pair mean
        rows.append({"quantile": q, "quantile_map vs": other, "n cases": len(s),
                     "qmap better %": 100 * (d > 0).mean(), "median err qmap": s.quantile_map.median(),
                     f"median err other": s[other].median(), "median diff (other - qmap)": d.median(),
                     "p (cases)": wilcoxon(d).pvalue, "n pairs": len(dp),
                     "qmap better % (pairs)": 100 * (dp > 0).mean(), "p (pairs)": wilcoxon(dp).pvalue})
T = pd.DataFrame(rows); T.to_csv(OUT / "paired_table.csv", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 20):
    print(T.round(4).to_string(index=False))

# 2. where is the CellOT - quantile_map gap (positive = CellOT worse)
w["gap"] = w.CellOT - w.quantile_map
w["gap_rel"] = w.gap / w.identity                       # in units of the observed effect
for g in ["stim", "celltype", "marker", "drug", "q"]:
    t = w.groupby(g).agg(n=("gap", "size"), qmap_better=("gap", lambda x: 100 * (x > 0).mean()),
                         median_gap=("gap", "median"), sum_gap=("gap", "sum"), median_effect=("identity", "median"))
    t["share_of_total_gap"] = 100 * t.sum_gap / w.gap.sum()
    t.sort_values("sum_gap", ascending=False).to_csv(OUT / f"gap_by_{g}.csv")
    print(f"\n--- gap by {g} (sum over cases; share of total CellOT excess error)")
    print(t.sort_values("sum_gap", ascending=False).round(4).head(12).to_string())

# 3. per Stabl feature (celltype_marker_stim) and link to selection
f = w.assign(feature=w.celltype + "_" + w.marker + "_" + w.stim).groupby("feature").agg(
    n=("gap", "size"), qmap_better=("gap", lambda x: 100 * (x > 0).mean()), sum_gap=("gap", "sum"),
    median_gap=("gap", "median"), median_effect=("identity", "median"))


def selfreq(path):
    s = pd.read_csv(path, index_col=0)["Fold selected features"].apply(ast.literal_eval)
    return pd.Series([x for l in s for x in l]).value_counts() / len(s)


f["sel_fig5b"] = selfreq(O / "foldfeats_GSS/Training CV/Selected Features STABL ALasso.csv").reindex(f.index).fillna(0)
f["sel_comb_cellot"] = selfreq(O / "paper_protocol/stabl_n100/comb_cellot/Training CV/Selected Features STABL ALasso.csv"
                               ).reindex(f.index).fillna(0)
sh = pd.read_csv(W / "diagnosis/shap_by_feature.csv")
sh = sh[sh.model == "CellOT stims + measured"].set_index("feature").abs_shap
f["shap_share_comb_cellot_%"] = (100 * sh / sh.sum()).reindex(f.index).fillna(0)
paper_cols = set(pd.read_csv(O / "ina_13OG_final_long_allstims_filtered_no_Unstim.csv", nrows=1, index_col=0).columns)
f["in_paper_table"] = f.index.isin(paper_cols)
f = f.sort_values("sum_gap", ascending=False); f.to_csv(OUT / "gap_by_feature.csv")
print(f"\nfeatures with drug cases above floor: {len(f)}, of which in the paper's 852-col table: {f.in_paper_table.sum()}")
print("\n--- top 20 features by CellOT excess error (sum over drugs x quantiles)")
print(f.head(20).round(3).to_string())
sel = f[(f.sel_fig5b > 0) | (f.sel_comb_cellot > 0)].sort_values("sel_fig5b", ascending=False)
print(f"\n--- features ever selected (Fig 5b or CellOT + measured): {len(sel)}")
print(sel.round(3).to_string())
for c in ["sel_fig5b", "sel_comb_cellot"]:
    r, p = spearmanr(f.sum_gap, f[c]); print(f"spearman(sum_gap, {c}) = {r:.3f}, p = {p:.2g}")
    s1, s0 = f[f[c] >= 0.2], f[f[c] < 0.2]
    print(f"  selected in >=20% folds: n {len(s1)}, qmap better {np.average(s1.qmap_better, weights=s1.n) if len(s1) else np.nan:.0f}% "
          f"(median gap {s1.median_gap.median() if len(s1) else np.nan:.4f}); others: qmap better "
          f"{np.average(s0.qmap_better, weights=s0.n):.0f}% (median gap {s0.median_gap.median():.4f})")
