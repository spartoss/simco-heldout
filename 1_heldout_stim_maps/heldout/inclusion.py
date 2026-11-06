"""Which (stim, celltype, marker) earn a simulated feature: the paper's median rule vs the any-quantile rule.

Paper rule (simco/baselines/stim_effect_medians.py, reproduced there 168/168): keep a (stim, celltype) pair if one of
the 13 markers has mean-over-patients |per-patient median(stim) - median(Unstim)| > 0.1, with 0.01 for IL33.
New rule (asked 2026-09-28): the same statistic at each of q10/q25/q50/q75/q90, keep a (stim, celltype, marker) if
ANY quantile exceeds 0.1, and IL33 gets no special threshold. DMSO wells pooled, uncorrected (the stim arm's data),
from cache_stim_all (<= 10k cells per donor and condition).
Writes WORK/features/inclusion.csv, one row per (stim, celltype, marker), and prints the counts.
    python -m heldout.inclusion
"""
import numpy as np
import pandas as pd

from heldout.common import ALL_PAIRS, CACHE_STIM_ALL, SIMCO_WORK, WORK
from lc.paths import MARKERS, PAIRS

QS = (10, 25, 50, 75, 90)


def main():
    rows = []
    for ct in sorted({c for _, c in ALL_PAIRS}):
        z = np.load(CACHE_STIM_ALL / f"{ct}.npz")
        donors = sorted({k.split("|")[1] for k in z.files})
        for st in sorted({s for s, c in ALL_PAIRS if c == ct}):
            d = []
            for p in donors:
                if f"Unstim|{p}" in z.files and f"{st}|{p}" in z.files:
                    d.append(np.abs(np.percentile(z[f"{st}|{p}"], QS, 0) - np.percentile(z[f"Unstim|{p}"], QS, 0)))
            mean_abs = np.mean(d, 0)                          # (5 quantiles, 13 markers)
            for j, m in enumerate(MARKERS):
                rows.append({"stim": st, "celltype": ct, "marker": m, "n_donors": len(d),
                             **{f"shift_q{q}": mean_abs[i, j] for i, q in enumerate(QS)}})
    T = pd.DataFrame(rows)
    thr_paper = np.where(T.stim == "IL33", 0.01, 0.1)
    T["median_rule_marker"] = T.shift_q50 > thr_paper
    T["pair_in_paper_grid"] = [(s, c) in set(PAIRS) for s, c in zip(T.stim, T.celltype)]
    T["pair_median_rule"] = T.groupby(["stim", "celltype"]).median_rule_marker.transform("any")
    qcols = [f"shift_q{q}" for q in QS]
    T["any_quantile_rule"] = (T[qcols] > 0.1).any(axis=1)
    T["quantiles_passing"] = [",".join(f"q{q}" for q in QS if r[f"shift_q{q}"] > 0.1) for _, r in T.iterrows()]
    paper = pd.read_csv(SIMCO_WORK / "onset_test/ina_13OG_final_long_allstims_filtered_no_Unstim.csv", nrows=1, index_col=0).columns
    T["feature"] = T.celltype + "_" + T.marker + "_" + T.stim
    T["in_paper_852"] = T.feature.isin(set(paper))
    out = WORK / "features"; out.mkdir(parents=True, exist_ok=True); T.to_csv(out / "inclusion.csv", index=False)
    pr = T.groupby(["stim", "celltype"]).agg(paper=("pair_in_paper_grid", "first"), med=("pair_median_rule", "first"),
                                              anyq=("any_quantile_rule", "any"))
    print(f"pairs: paper grid {pr.paper.sum()}, median rule on this cache {pr.med.sum()} "
          f"(agreement with the grid {(pr.paper == pr.med).mean():.1%}), any-quantile rule {pr.anyq.sum()}")
    print(f"pairs added by the any-quantile rule: {int((pr.anyq & ~pr.paper).sum())}; pairs of the grid it drops: "
          f"{int((~pr.anyq & pr.paper).sum())} (IL33: {int((~pr.anyq & pr.paper)[pr.index.get_level_values(0) == 'IL33'].sum())})")
    print(f"(stim, celltype, marker) features: paper 852; any-quantile rule {int(T.any_quantile_rule.sum())} "
          f"(of which in the paper's 852: {int((T.any_quantile_rule & T.in_paper_852).sum())}; "
          f"paper features it drops: {int((~T.any_quantile_rule & T.in_paper_852).sum())})")
    new = T[T.any_quantile_rule & ~T.in_paper_852]
    print("new features by stim:", new.stim.value_counts().to_dict())
    print("which quantiles make them pass:", new.quantiles_passing.value_counts().head(8).to_dict())
    print("dropped paper features by stim:", T[~T.any_quantile_rule & T.in_paper_852].stim.value_counts().to_dict())


if __name__ == "__main__":
    main()
