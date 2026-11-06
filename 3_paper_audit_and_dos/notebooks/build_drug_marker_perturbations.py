"""Generate notebooks/drug_marker_perturbations.ipynb.

Reselects the drug arm on *per-marker* perturbation instead of whole-distribution MMD,
then asks which map predicts those specific marker shifts best.
"""
import nbformat as nbf

cells = []
md = lambda t: cells.append(nbf.v4.new_markdown_cell(t.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Drug arm, reselected on per-marker perturbation

`summarize.py` splits the drug arm into `live` / `null` on **identity MMD ≥ 0.005**, a
whole-distribution criterion, and only **48 of 1512** pairs survive it.

That is the wrong filter for the question that matters downstream. A Stabl feature is a
**per-marker median**, so the quantity that has to be non-trivial is the *per-marker* shift
between control and drug — not the distance between two 13-dimensional clouds. A pair can
move one marker substantially and still have a small MMD, and such a pair produces a real
feature.

So here the selection is: **keep a (celltype, stim, drug) pair if at least one of the 13
markers shifts by more than a threshold**, which is exactly the form of the paper's own
stim-selection rule (§"The 168-pair grid": keep a pair if ≥1 marker has arcsinh median
difference > 0.1). Then, on those marker shifts specifically, compare the methods.

Nothing is re-run: the pooled drug CSVs already carry `median_error__<marker>` for all 13
markers, and the `identity` row of that column **is** the observed perturbation
(`identity` means pred = control cells, so `median_error__M = |median(control_M) −
median(drug_M)|`).

Companion to `baselines_results.ipynb` Part C §11, which holds the MMD-based view.
""")

code(r"""
import glob, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200, "display.max_columns", 60)

DRUG = Path("/scratch/users/bmerwan/simco_work/baselines/pooled_drug")
raw = pd.concat([pd.read_csv(f) for f in sorted(DRUG.glob("*.csv"))], ignore_index=True)
MARKERS = [c[len("median_error__"):] for c in raw.columns if c.startswith("median_error__")]
KEYS = ["celltype", "stim", "drug"]

# `identity` needs no fitting, so it is recorded once with weighting="invariant"; the fitted
# methods appear under both "cell" and "unweighted". Take the cell arm, which matches
# CellOT's cell-level sampling, and carry identity across.
print(raw.groupby(["method", "weighting"]).size().to_string())
long = (raw[raw.weighting.isin(["cell", "invariant"])]
        .melt(id_vars=KEYS + ["method"],
              value_vars=[f"median_error__{m}" for m in MARKERS],
              var_name="marker", value_name="err"))
long["marker"] = long.marker.str[len("median_error__"):]
print(f"\n{len(MARKERS)} markers x {raw[raw.method=='identity'].shape[0]} pairs "
      f"= {len(long[long.method=='identity'])} (pair, marker) cells")
""")

md(r"""
## §1 — The per-marker perturbation, and how many survive each threshold

`pert(celltype, stim, drug, marker)` = |median(control) − median(drug)| for that marker,
on the pooled test cells.
""")

code(r"""
pert = (long[long.method == "identity"]
        .rename(columns={"err": "pert"})[KEYS + ["marker", "pert"]])
mx = pert.groupby(KEYS).pert.max()

print("per-marker perturbation, quantiles over all 19656 (pair, marker) cells")
print(pert.pert.quantile([.25, .5, .75, .9, .99, 1]).round(4).to_string())

FLOOR = 0.0273   # technical noise floor: DMSO2 vs DMSO3, median_error at n~20.8k (notebook 10.7)
rows = []
for t in [FLOOR, 0.05, 0.1, 0.2]:
    rows.append({"threshold": t,
                 "pairs kept": f"{int((mx > t).sum())}/{len(mx)}",
                 "% pairs": round(100*(mx > t).mean(), 1),
                 "(pair,marker) kept": f"{int((pert.pert > t).sum())}/{len(pert)}",
                 "% cells": round(100*(pert.pert > t).mean(), 1)})
print("\n" + pd.DataFrame(rows).to_string(index=False))
print(f"\nfor reference, the MMD live cut (identity MMD >= 0.005) keeps 48/1512 pairs = 3.2%")
""")

md(r"""
Two things to note.

**The per-marker view is far less restrictive than MMD.** At the paper's own stim threshold
of 0.1, **301 of 1512** pairs carry at least one moving marker, against 48 under the MMD cut.
MMD averages a single-marker shift away across 13 dimensions; the feature does not.

**The technical floor is the honest lower bound.** Two identical control wells (`DMSO2` vs
`DMSO3`) differ by `median_error` 0.0273 at n≈20.8k — so a marker shift below that is not
distinguishable from plate noise. Caveat carried from §10.7 of `baselines_results.ipynb`: that
floor was measured at n≈20.8k cells on *uncorrected* data, while the drug arm is
batch-corrected at load time and pooled over a different number of cells, so it is indicative,
not matched.

## §2 — Where the moving markers are

The MMD cut concentrated brutally: 46 of its 48 pairs were `PI`, and 39 were the two `DMSO1`
drugs (`LPZ`, `SA`) — which is what made it suspect, since `DMSO1` is an outlier under `PI`.
Does the per-marker selection do the same?
""")

code(r"""
THR = 0.1     # the paper's own stim-selection threshold
keep_pairs = mx[mx > THR].reset_index()
print(f"threshold {THR}: {len(keep_pairs)} pairs\n")
for k in ["stim", "drug"]:
    tot = mx.reset_index().groupby(k).size()
    sel = keep_pairs.groupby(k).size().reindex(tot.index).fillna(0).astype(int)
    t = pd.DataFrame({"kept": sel, "of": tot, "%": (100*sel/tot).round(1)}).sort_values("%", ascending=False)
    print(f"--- by {k}\n{t.to_string()}\n")
sel_markers = pert[pert.pert > THR].groupby("marker").size().sort_values(ascending=False)
print("--- which markers actually move\n" + sel_markers.to_string())
""")

md(r"""
## §3 — On those marker shifts, who predicts best?

For every retained `(celltype, stim, drug, marker)` cell, each method has its own
`median_error__<marker>`: the absolute gap between the marker median it predicts and the
observed one. Lower is better. `identity` is the null — "the drug does nothing" — and its
error *is* the perturbation, so any method that fails to beat `identity` has extracted
nothing.
""")

code(r"""
def head_to_head(thr):
    sel = pert[pert.pert > thr][KEYS + ["marker"]]
    j = long.merge(sel, on=KEYS + ["marker"], how="inner")
    wide = j.pivot_table(index=KEYS + ["marker"], columns="method", values="err")
    wide = wide.dropna()
    methods = [m for m in wide.columns]
    best = wide.idxmin(axis=1)
    out = pd.DataFrame({
        "median |err|":  wide.median().round(4),
        "mean |err|":    wide.mean().round(4),
        "best in N":     best.value_counts().reindex(methods).fillna(0).astype(int),
        "best %":        (100*best.value_counts().reindex(methods).fillna(0)/len(wide)).round(1),
        "beats identity %": (100*wide.lt(wide["identity"], axis=0).mean()).round(1),
        "beats cellot %":   (100*wide.lt(wide["cellot_insample"], axis=0).mean()).round(1),
    })
    return len(wide), out.sort_values("median |err|")

n, tbl = head_to_head(THR)
print(f"threshold {THR} -- {n} (pair, marker) cells\n")
print(tbl.to_string())
""")

md(r"""
**The marginal medians in that table are not a ranking.** They are 7 separate column medians;
the cells are paired, so the question "is CellOT better than X" has to be asked on the paired
differences. It changes the answer.
""")

code(r"""
from scipy.stats import wilcoxon

sel = pert[pert.pert > THR][KEYS + ["marker"]]
w = (long.merge(sel, on=KEYS + ["marker"])
         .pivot_table(index=KEYS + ["marker"], columns="method", values="err").dropna().reset_index())
print(f"{int((pert.pert > THR).sum())} cells above threshold; {len(w)} after dropping the 4 pairs "
      f"where linear_median* is missing")
print(f"  drawn from {w.groupby(KEYS).ngroups} pairs / {w.celltype.nunique()} celltypes / "
      f"{w.stim.nunique()} stims / {w.drug.nunique()} drugs")
print(f"  markers per pair: median {w.groupby(KEYS).size().median():.0f}, max {w.groupby(KEYS).size().max()}"
      "   <- mostly one marker per pair, so the cells are close to independent\n")

c = w["cellot_insample"].values
for m in ["quantile_map", "median_shift", "linear_median_ridge", "gauss_ot", "linear_median", "identity"]:
    b = w[m].values; d = b - c        # positive = CellOT better
    print(f"  CellOT vs {m:20} median diff {np.median(d):+.4f}   CellOT wins {100*(d>0).mean():4.1f}%"
          f"   Wilcoxon p = {wilcoxon(c, b).pvalue:.3f}")
""")

code(r"""
# robustness: the same table at the technical floor, which keeps 11x more cells
n2, tbl2 = head_to_head(FLOOR)
print(f"threshold {FLOOR} (technical floor) -- {n2} (pair, marker) cells\n")
print(tbl2.to_string())
""")

md(r"""
## §4 — Does the advantage depend on how big the perturbation is?

The drug arm selects its checkpoints on **MMD**, not on `median_error` (`data.metric` is absent
from all 1512 drug configs, so `train.py:190-193` falls back to mmd). So CellOT is not being
scored here on the criterion that picked it — unlike the stim arm, where `median_error` *was*
the selection metric. Worth keeping in mind when reading the row above.

Below, the win rate as a function of how large the perturbation is: if a method only wins on
the near-null cells, that is a statement about noise, not about prediction.
""")

code(r"""
sel = pert[pert.pert > FLOOR].copy()
sel["bin"] = pd.cut(sel.pert, [FLOOR, 0.05, 0.1, 0.2, np.inf],
                    labels=["floor-0.05", "0.05-0.1", "0.1-0.2", ">0.2"])
j = long.merge(sel[KEYS + ["marker", "bin"]], on=KEYS + ["marker"], how="inner")
wide = j.pivot_table(index=KEYS + ["marker", "bin"], columns="method", values="err").dropna()
wide = wide.reset_index().set_index(KEYS + ["marker"])
rows = []
for b, g in wide.groupby("bin", observed=True):
    m = g.drop(columns="bin")
    best = m.idxmin(axis=1).value_counts()
    rows.append(pd.Series({"n cells": len(m),
                           **{f"{k} best %": round(100*best.get(k, 0)/len(m), 1) for k in m.columns}},
                          name=b))
print("share of cells where each method is best, by perturbation size\n")
print(pd.DataFrame(rows).to_string())
""")

code(r"""
# and the absolute scale: median |err| per bin, so "winning" can be read against what is
# actually being predicted
rows = []
for b, g in wide.groupby("bin", observed=True):
    m = g.drop(columns="bin")
    rows.append(pd.Series({"n": len(m), **m.median().round(4).to_dict()}, name=b))
print("median |err| per method, by perturbation size\n")
print(pd.DataFrame(rows).to_string())
""")

code(r"""
# the >0.2 bin is the thinnest slice in the notebook -- size it and test it before quoting it
big = pert[pert.pert > 0.2][KEYS + ["marker"]]
wb = (long.merge(big, on=KEYS + ["marker"])
          .pivot_table(index=KEYS + ["marker"], columns="method", values="err").dropna().reset_index())
print(f"{len(wb)} cells, from {wb.groupby(KEYS).ngroups} pairs / {wb.celltype.nunique()} celltypes / "
      f"{wb.drug.nunique()} drugs / {wb.stim.nunique()} stims")
print("  drugs:", dict(wb.drug.value_counts()))
cb = wb["cellot_insample"].values
for m in ["quantile_map", "median_shift", "gauss_ot"]:
    d = wb[m].values - cb
    print(f"  CellOT vs {m:14} wins {100*(d>0).mean():4.1f}%   Wilcoxon p = {wilcoxon(cb, wb[m].values).pvalue:.3f}")
print("\n-> nothing significant here. The lower median |err| for CellOT in the >0.2 row above")
print("   is a marginal statistic over 39 non-independent cells; it does not survive pairing.")
""")

md(r"""
## §5 — Reading this

Limits to carry with any number above:

* The **`DMSO1` confound is not removed here.** `LPZ` and `SA` are referenced to `DMSO1`, which
  is an outlier under `PI` in every celltype (`baselines_results.ipynb` §10.7 and
  `baselines/control_swap.py`). Any concentration of retained pairs on those drugs under `PI`
  is suspect for the same reason as before — reselecting on per-marker shift does not fix a bad
  control.
* `median_error__<marker>` uses **all** cells of the pair and is n-dependent, so the technical
  floor quoted here (0.0273, measured at n≈20.8k on uncorrected data) is indicative, not
  matched to these pairs.
* The comparison is on the **`cell` weighting arm**; the `unweighted` arm exists on disk and
  moved several drug-arm conclusions substantially before, so any headline should be re-checked
  there before being quoted.
* `cellot_insample` is one checkpoint applied unchanged — it never saw a patient held out, and
  its checkpoint was chosen by an argmin on the same test cells (MMD, not this metric).

Rebuild: `python notebooks/build_drug_marker_perturbations.py` then
`jupyter nbconvert --to notebook --execute --inplace notebooks/drug_marker_perturbations.ipynb`.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python", "version": "3.9"}}
out = "/scratch/users/bmerwan/simco/notebooks/drug_marker_perturbations.ipynb"
nbf.write(nb, out)
print("wrote", out, f"({len(cells)} cells)")
