"""Generate notebooks/cellot_vs_gaussot.ipynb -- CellOT vs gauss_ot, everything in one place:
compute cost, atlas modelling (stim, drug), DOS prediction, combined features, drug arm.

Same convention as the other notebooks: every table is recomputed from result files.
    python notebooks/build_cellot_vs_gaussot.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/cellot_vs_gaussot.ipynb
"""
import nbformat as nbf

cells = []
md = lambda t: cells.append(nbf.v4.new_markdown_cell(t.strip()))
code = lambda t: cells.append(nbf.v4.new_code_cell(t.strip()))

md(r"""
# CellOT vs gauss_ot — cost, modelling, prediction, drugs

**gauss_ot** is the optimal-transport map between two Gaussians: T(x) = μ_t + A (x − μ_s), with A the
unique symmetric 13×13 matrix mapping the source covariance onto the target's. It is fitted in closed
form from a mean and a covariance per condition — no training loop, no hyperparameters, no seed.
**CellOT** is the paper's model: two input-convex neural networks per condition, trained by
stochastic gradient descent, checkpoint chosen on test cells.

Sections: §1 compute cost · §2 modelling stimulations on the atlas · §3 modelling drugs on the atlas ·
§4 predicting DOS (20 splits and paper protocol) · §5 combined with measured features · §6 drug arm
(virtual twins) · §7 summary.

Nothing is retrained here. CellOT results use the paper's own checkpoints.
""")

code(r"""
import warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
warnings.filterwarnings("ignore")
pd.set_option("display.width", 180, "display.max_columns", 40)
W = Path("/scratch/users/bmerwan/simco_work"); O = W / "onset_test"; B = W / "baselines"
first = lambda x: float(str(x).split()[0])
""")

md(r"""
## 1. Compute cost

**CellOT training times are measured from the paper's own files**: `config.yaml` is written when a
training run starts and `cache/last.pt` when it ends (`simco_work/timing/cellot_train_minutes.csv`,
recorded 2026-09-24 with `stat` on the three results trees). Each map ran on 1 CPU core for 50 000
steps (stim) — these are wall-clock durations, so they include any slowdown of the node.
The 30 h / 25 h figures in the lab's launchers (`train_cellot_CV_jakob.sh`, `train_drug_jakob.sh`)
are **time limits**, not measurements.
""")

code(r"""
t = pd.read_csv(W / "timing/cellot_train_minutes.csv")
lab = {"stim": "stim maps (Unstim → stim)", "drug": "drug maps (DMSO → drug, within a stim)",
       "drug_unstim": "drug maps on Unstim"}
summ = t.groupby("kind").minutes.agg(maps="size", median_min="median",
                                     p90_min=lambda s: s.quantile(.9), max_min="max", total_min="sum")
summ["median per map"] = (summ.median_min / 60).round(2).astype(str) + " h"
summ["total CPU-hours"] = (summ.total_min / 60).round(0).astype(int)
summ.index = summ.index.map(lab)
display(summ[["maps", "median_min", "p90_min", "max_min", "median per map", "total CPU-hours"]])
print(f"All CellOT maps: {len(t)} maps, {t.minutes.sum() / 60:,.0f} CPU-hours "
      f"({t.minutes.sum() / 60 / 24 / 365:.2f} CPU-years)")
""")

code(r"""
g = pd.read_csv(W / "timing/gaussot_drug_fit_seconds.csv")
print(f"gauss_ot drug maps: 1764 maps (1512 stim×drug + 252 Unstim×drug), 28 celltype tasks, "
      f"{g.cpus.iloc[0]} cores each")
print(f"  sum of task wall times {g.elapsed_seconds.sum() / 3600:.2f} h  ->  at most "
      f"{(g.elapsed_seconds * g.cpus).sum() / 3600:.1f} core-hours; longest task "
      f"{g.elapsed_seconds.max() / 60:.0f} min ({g.loc[g.elapsed_seconds.idxmax(), 'celltype']}), "
      f"median {g.elapsed_seconds.median() / 60:.1f} min — dominated by reading the h5ad files")
cell_drug_h = t[t.kind != "stim"].minutes.sum() / 60
print(f"  CellOT drug maps: {cell_drug_h:,.0f} CPU-hours  ->  ratio ≈ "
      f"{cell_drug_h / ((g.elapsed_seconds * g.cpus).sum() / 3600):,.0f}×")
print("gauss_ot stim maps (168): ~10 min in total (baselines/simco_features.py fit, 2026-09-22; not re-timed)")
""")

md(r"""
Inference is cheap for both (a matrix product per cell for gauss_ot, a gradient pass through a small
network for CellOT) and dominated by reading the cells: building the 168-map feature table for the
177 women takes ~10 min (gauss_ot) vs ~15–25 min (CellOT); the 9 drug-treated tables 52 min for all
drugs (gauss_ot) vs 1.3–2.6 h per drug job (CellOT chain). **With Stabl the cost is identical** —
Stabl only sees the 177 × 1216 table (~4 h for 100 splits on 16 cores either way).

## 2. Modelling stimulations on the atlas

127 stim × celltype pairs with a real effect (identity MMD ≥ 0.005), test cells of CellOT's own split,
cell-weighted arm (`simco_work/baselines/summary_pooled_cell/all_rows.csv`).
""")

code(r"""
r = pd.read_csv(B / "summary_pooled_cell/all_rows.csv"); r = r[r.weighting.isin(["cell", "invariant"])]
idm = r[r.method == "identity"].set_index(["celltype", "stim"]).mmd
r = r.set_index(["celltype", "stim"]).loc[idm[idm >= 0.005].index].reset_index()
rows = []
for m in ["mean_error", "median_error", "w1", "mmd", "corr_error"]:
    w = r.pivot_table(index=["celltype", "stim"], columns="method", values=m)
    d = w.gauss_ot - w.cellot_insample
    rows.append({"metric": m, "CellOT median": w.cellot_insample.median(), "gauss_ot median": w.gauss_ot.median(),
                 "gauss_ot beats CellOT": f"{(d < 0).mean():.0%}", "Wilcoxon p": wilcoxon(d).pvalue})
pd.DataFrame(rows).round(4)
""")

md(r"""
gauss_ot wins the marker-wise and correlation metrics; CellOT wins the whole-distribution ones (MMD,
W1), the objective it is trained on.

## 3. Modelling drugs on the atlas

Per (drug, stim, celltype, marker) with a drug effect above the noise floor (0.0273, the median-error
difference between two identical control wells DMSO2 vs DMSO3) or above the paper's 0.1
(`simco_work/baselines/pooled_drug/`, cell arm).
""")

code(r"""
raw = pd.concat([pd.read_csv(f) for f in sorted((B / "pooled_drug").glob("*.csv"))], ignore_index=True)
MK = [c[len("median_error__"):] for c in raw.columns if c.startswith("median_error__")]
long = (raw[raw.weighting.isin(["cell", "invariant"])]
        .melt(id_vars=["celltype", "stim", "drug", "method"], value_vars=[f"median_error__{m}" for m in MK],
              var_name="marker", value_name="err"))
wide = long.pivot_table(index=["celltype", "stim", "drug", "marker"], columns="method", values="err") \
           .dropna(subset=["cellot_insample", "gauss_ot", "identity"])
out = []
for thr, lab_ in [(0.0273, "above noise floor"), (0.1, "above 0.1 (paper's rule)")]:
    s = wide[wide.identity > thr]; d = s.gauss_ot - s.cellot_insample
    out.append({"cases": lab_, "n": len(s), "CellOT median err": s.cellot_insample.median(),
                "gauss_ot median err": s.gauss_ot.median(), "gauss_ot beats CellOT": f"{(d < 0).mean():.0%}",
                "Wilcoxon p": wilcoxon(d).pvalue})
pd.DataFrame(out).round(4)
""")

md(r"""
On drugs the picture flips: near the noise floor gauss_ot is slightly better, on the larger effects
CellOT is (the per-marker baselines `median_shift` / `quantile_map` tie CellOT there — see
`drug_marker_perturbations.ipynb`).

## 4. Predicting DOS

### 4a. 20 splits (`run_regression_cv.py`, rows aligned — no bug)
""")

code(r"""
s = pd.read_csv(O / "null_maps/null_maps_scores.csv"); s["R2v"] = s.R2.map(first)
tab = s[s.arm.isin(["cellot", "gauss_ot"])].pivot_table(index=["table", "model"], columns="arm", values="R2v")
pp = pd.read_csv(O / "null_maps/null_maps_paired.csv"); pp = pp[pp.arm == "gauss_ot"].set_index(["table", "model"])
tab["mean Δ (CellOT − gauss_ot)"] = pp.mean_diff_cellot_minus_arm; tab["Wilcoxon p"] = pp.wilcoxon_p
tab["CellOT wins /20"] = pp.cellot_wins
tab.round(3)
""")

md(r"""
### 4b. Paper protocol — 100 splits, `params_43`, and all 97 XGBoost settings

Two versions of the second stage: **the lab's rescore script as published** (`paper_protocol/rescore/`)
and **the same with rows and patient groups aligned** (`paper_protocol/aligned/`). The published
script has a row-order bug (94 % of test samples have their own woman in training) that leaves CellOT
unchanged but inflates gauss_ot by ~0.02.
""")

code(r"""
def lab(arm, p="params_43"):
    return first(pd.read_csv(O / f"paper_protocol/rescore/{arm}/{p}/Summary/Scores training CV.csv",
                             index_col=0).loc["STABL ALasso", "R2"])
def labmae(arm):
    return first(pd.read_csv(O / f"paper_protocol/rescore/{arm}/params_43/Summary/Scores training CV.csv",
                             index_col=0).loc["STABL ALasso", "MAE"])
AL = O / "paper_protocol/aligned"
sw = {a: pd.read_csv(AL / a / "sweep.csv") for a in ["cellot", "gaussot_both"]}
fo = {a: pd.read_csv(AL / a / "folds.csv") for a in ["cellot", "gaussot_both"]}
rows = []
for a, name, la in [("cellot", "CellOT", "cellot_both"), ("gaussot_both", "gauss_ot", "gaussot_both")]:
    t_ = sw[a].set_index("params")
    rows.append({"model": name, "lab script R² (params_43)": lab(la), "lab script MAE (d)": labmae(la),
                 "aligned R² params_43": t_.loc["params_43", "R2"], "aligned best of 97": t_.R2.max(),
                 "aligned median of 97": t_.R2.median(), "aligned worst of 97": t_.R2.min()})
display(pd.DataFrame(rows).round(3))
for mode, pa, pc in [("params_43", "params_43", "params_43"),
                     ("each its best", sw["gaussot_both"].params.iloc[0], sw["cellot"].params.iloc[0])]:
    d = fo["gaussot_both"][pa] - fo["cellot"][pc]
    print(f"aligned, {mode:14s}: gauss_ot − CellOT per split {d.mean():+.3f}, gauss_ot wins "
          f"{int((d > 0).sum())}/100, Wilcoxon p {wilcoxon(d).pvalue:.4f}")
""")

md(r"""
## 5. Combined with measured features

One Stabl model on the simulated stims + measured cell frequencies, 31-marker medians and
functional-marker quantiles (3204 columns).
""")

code(r"""
rows = []
for a, name in [("comb_cellot", "CellOT stims + measured"), ("comb_gaussot", "gauss_ot stims + measured")]:
    t_ = pd.read_csv(AL / a / "sweep.csv").set_index("params")
    rows.append({"model": name, "lab script R² (params_43)": lab(a), "lab script MAE (d)": labmae(a),
                 "aligned R² params_43": t_.loc["params_43", "R2"], "aligned best of 97": t_.R2.max()})
display(pd.DataFrame(rows).round(3))
fc = {a: pd.read_csv(AL / a / "folds.csv") for a in ["comb_cellot", "comb_gaussot", "cellot"]}
for a in ["comb_cellot", "comb_gaussot"]:
    d = fc[a]["params_43"] - fc["cellot"]["params_43"]
    print(f"aligned params_43, {a} − CellOT alone: {d.mean():+.3f}, wins {int((d > 0).sum())}/100, p {wilcoxon(d).pvalue:.4f}")
d = fc["comb_gaussot"]["params_43"] - fc["comb_cellot"]["params_43"]
print(f"aligned params_43, gauss_ot+measured − CellOT+measured: {d.mean():+.3f}, wins {int((d > 0).sum())}/100, p {wilcoxon(d).pvalue:.4f}")
""")

md(r"""
## 6. Drug arm — virtual twins

Three complete pipelines, each with its own DOS model (100-split selections, `params_43`, aligned):
**CellOT** stim + CellOT drug maps (the paper); **gauss_ot** stim + **gauss_ot** drug maps (no neural
network anywhere; drug maps fitted on the same control wells, batch correction and 80/20 cell split as
CellOT's, `baselines/gauss_drug_maps.py`); and the earlier hybrid gauss_ot stim + CellOT drug maps.
Shift = mean predicted change in DOS (days; negative = labor later); *folds* = share of the 100 fold
models agreeing in sign; *implausible* = share of drug-treated women farther from every untreated
training woman than the untreated 95th percentile (the model extrapolates for them).
""")

code(r"""
cmp = pd.read_csv(O / "drug_models/compare_by_drug.csv")
M3 = ["CellOT (paper tables)", "gauss_ot + gauss_ot drug maps", "gauss_ot"]
sh = cmp.pivot_table(index="drug", columns="model", values="mean shift (d)")[M3].round(1)
ag = cmp.pivot_table(index="drug", columns="model", values="folds same sign")[M3].round(2)
nz = cmp.pivot_table(index="drug", columns="model", values="shift / null SD")[M3].round(1)
out = pd.concat({"shift (d)": sh, "folds same sign": ag, "shift / null SD": nz}, axis=1)
display(out.sort_values(("shift (d)", "CellOT (paper tables)")))
from scipy.stats import spearmanr
print("Spearman of the drug ranking vs CellOT:",
      {m: round(spearmanr(sh["CellOT (paper tables)"], sh[m])[0], 2) for m in M3[1:]})
cmp.groupby("model")[["untreated R2", "frac implausible (b)", "frac out of range (a)"]].mean().loc[M3].round(3)
""")

md(r"""
With no neural network at all, THF stays first (−7.2 d, 98 % of fold models) and SALPZ holds; MAP
disappears; RIF and CHT gain. The ranking correlates +0.80 with the paper's, and only ~10 % of
drug-treated women are implausible (vs 56 % with CellOT) — the gauss_ot model extrapolates far less.

**Which features carry THF / SALPZ** (TreeSHAP, CellOT vs the hybrid gauss_ot stim + CellOT drug maps;
`drug_models/attribution/`): the same axis — pSTAT3 in naive CD4 T cells — through partly different
stim columns.
""")

code(r"""
A = O / "drug_models/attribution"
for d in ["THF", "SALPZ"]:
    parts = []
    for m in ["CellOT", "gauss_ot"]:
        a = pd.read_csv(A / f"attribution_{m}.csv"); a = a[a.drug == d]
        a = a.reindex(a.shift_days.abs().sort_values(ascending=False).index).head(5)
        parts.append(a.assign(model=m)[["model", "feature", "shift_days"]])
    print(f"--- {d}"); display(pd.concat(parts).round(2))
""")

md(r"""
## 7. The two mishandled visits (36_G1, 52_G1)

Their "Unstim" tubes look stimulated. Each has one visit with 134 / 44 features more than 5 SD from the cohort,
against ≤ 2 on the same woman's other visits, all up, on pSTAT, and correlating with the IFNa / LPS response.
Here they are dropped from XGBoost training and scoring; the splits are unchanged (`aligned_rescore.py --exclude`).
Stabl's feature *selection* still saw them, so a full removal would mean rerunning Stabl.
100 splits, `params_43`:
""")

code(r"""
rows = []
for a in ["cellot", "comb_cellot", "comb_gaussot", "gaussot_both", "d_qtl", "d_freqmed", "crossswap_s0_both", "rscaled_s0_both"]:
    r0 = pd.read_csv(AL / a / "sweep.csv").set_index("params").loc["params_43"]
    r1 = pd.read_csv(O / "paper_protocol/aligned_excl" / a / "sweep.csv").set_index("params").loc["params_43"]
    rows.append({"arm": a, "R² all 177": r0.R2, "R² 175, visits kept in training": r1.R2_with, "R² visits removed": r1.R2})
ex = pd.DataFrame(rows).set_index("arm")
c = pd.read_csv(O / "paper_protocol/aligned_excl/cellot/folds.csv")["params_43"]
for a in ex.index.drop("cellot"):
    d = pd.read_csv(O / "paper_protocol/aligned_excl" / a / "folds.csv")["params_43"] - c
    ex.loc[a, "Δ vs CellOT per split"] = d.mean(); ex.loc[a, "p"] = wilcoxon(d).pvalue
ex.round(3)
""")

md(r"""
CellOT barely moves (0.418 → 0.415); gauss_ot and the measured feature sets lose 0.02–0.035. So the two visits
were *helping* CellOT's competitors, and every conclusion above holds with a slightly larger CellOT lead.

## 8. Simulated quantiles as extra Stabl features

The 10/25/75/90th percentiles of the simulated stimulated cells (CellOT or gauss_ot maps) for the paper's 852
(stim, celltype, marker) columns: 3408 more columns, one Stabl block per stim (`<stim>q`), added to the combined
tables (6612 columns). Run on 20 splits for time (a strict prefix of the 100). Rescored aligned with `params_43`,
with and without the two visits.
""")

code(r"""
N20 = O / "paper_protocol"
arms = ["cellot", "gaussot_both", "d_qtl", "comb_cellot", "comb_gaussot", "comb_cellot_simq", "comb_gaussot_simq"]
rows = []
for tag, lab_ in [("aligned_n20", "all 177"), ("aligned_n20_excl", "36_G1, 52_G1 removed")]:
    Fd = {a: pd.read_csv(N20 / tag / a / "folds.csv")["params_43"] for a in arms}
    for a in arms:
        r = pd.read_csv(N20 / tag / a / "sweep.csv").set_index("params").loc["params_43"]
        row = {"samples": lab_, "arm": a, "R²": r.R2, "median split R²": r.median_split_R2}
        if "simq" in a:
            ref = "comb_cellot" if "cellot" in a else "comb_gaussot"
            d = Fd[a] - Fd[ref]
            row.update({"Δ vs same set w/o quantiles": d.mean(), "p": wilcoxon(d).pvalue, "splits better": f"{(d > 0).sum()}/20"})
        rows.append(row)
T8 = pd.DataFrame(rows)
# the same reference arms on all 100 splits (params_43, aligned): NOT comparable to the 20-split column
for tag, lab_ in [("aligned", "all 177"), ("aligned_excl", "36_G1, 52_G1 removed")]:
    for a in arms:
        f100 = N20 / tag / a / "sweep.csv"
        if f100.exists():
            T8.loc[(T8.samples == lab_) & (T8.arm == a), "R² at 100 splits"] = pd.read_csv(f100).set_index("params").loc["params_43", "R2"]
display(T8.round(3))
""")

md(r"""
**20 vs 100 splits.** The pooled R² depends on the number of splits. Each sample's prediction is the median of its
out-of-fold predictions: about 4 with 20 splits, about 20 with 100. So a 20-split R² and a 100-split R² are
different measurements; e.g. CellOT gives 0.410 at 20 splits and 0.418 at 100. That is why every comparison in this
section is made on the **same 20 splits**. The reference arms are rescored on the first 20 of their 100-split Stabl
selections. This equals a 20-split run: the `GroupShuffleSplit` draws are a strict prefix, and Stabl uses a fixed
`random_state` per split. It was checked directly: the quantile runs, run natively on 20 splits, select exactly
the reference model's non-quantile features in 20/20 splits (next cell). The last column above is there for
orientation only; never subtract it from the 20-split column.
""")

code(r"""
import ast, re
arms_txt = {l.split("|")[0]: l.split("|")[2] for l in open(N20 / "aligned_n20/arms.txt").read().split("\n") if l}
for a in ["comb_cellot_simq", "comb_gaussot_simq"]:
    sel = pd.read_csv(Path(arms_txt[a]) / "Training CV/Selected Features STABL ALasso.csv", index_col=0)
    sel = sel["Fold selected features"].apply(ast.literal_eval)[:20]
    allf = [f for l in sel for f in l]; q = [f for f in allf if re.search(r"_q\d+_[A-Za-z0-9]+q$", f)]
    print(f"{a}: {len(allf) / 20:.1f} features per split, of which simulated quantiles {len(q) / 20:.1f} ({100 * len(q) / len(allf):.0f} %)")
    vc = pd.Series(q).value_counts()
    display(vc[vc >= 8].rename("splits selected (of 20)").to_frame())
""")

md(r"""
**Why selected ≠ useful.** Two checks for each model:
- Stabl runs its selection separately in each block. So the non-quantile features chosen per split are exactly those of
  the model without quantiles, and the quantiles are pure additions.
- The quantiles Stabl keeps are near-copies of median features already in the model. The table below gives, for each
  quantile selected in ≥ 8 of 20 splits: its correlation with the median of the same (celltype, marker, stim), how often
  that median is itself selected, and its highest |r| with any selected non-quantile feature.
""")

code(r"""
isq = lambda f: re.search(r"_q\d+_[A-Za-z0-9]+q$", f) is not None
for a, ref in [("comb_cellot_simq", "comb_cellot"), ("comb_gaussot_simq", "comb_gaussot")]:
    X = pd.read_csv(O / f"paper_protocol/tables/{a}.csv", index_col=0)
    sel = pd.read_csv(Path(arms_txt[a]) / "Training CV/Selected Features STABL ALasso.csv", index_col=0)
    sel = sel["Fold selected features"].apply(ast.literal_eval)[:20]
    sel0 = pd.read_csv(Path(arms_txt[ref]) / "Training CV/Selected Features STABL ALasso.csv", index_col=0)
    sel0 = sel0["Fold selected features"].apply(ast.literal_eval)[:20]
    same = sum(set(f for f in l if not isq(f)) == set(l0) for l, l0 in zip(sel, sel0))
    print(f"{a}: non-quantile features per split {np.mean([sum(not isq(f) for f in l) for l in sel]):.1f} vs "
          f"{np.mean([len(l) for l in sel0]):.1f} in {ref}; identical non-quantile set in {same}/20 splits")
    vc = pd.Series([f for l in sel for f in l]).value_counts()
    others = [f for f in vc.index if not isq(f) and f in X]
    rows = []
    for f in [f for f in vc.index if isq(f) and vc[f] >= 8]:
        m = re.match(r"(.+)_q\d+_([A-Za-z0-9]+)q$", f); base = f"{m.group(1)}_{m.group(2)}"
        rows.append({"quantile feature": f, "splits selected": vc[f],
                     "r with its own median": X[f].corr(X[base]) if base in X else np.nan,
                     "own median selected (splits)": int(vc.get(base, 0)),
                     "max |r| with a selected non-quantile feature": X[others].corrwith(X[f]).abs().max()})
    display(pd.DataFrame(rows).set_index("quantile feature").round(2))
""")

md(r"""
Stabl selects the simulated quantiles a lot: ~41 % of the CellOT model's selections (29 per split) and 36 % of
gauss_ot's. Several are chosen in every split, e.g. `CD4Tnaive_pSTAT3_q75_PIq` and `CD4Tnaive_pSTAT5_q90_PIq`.
**But R² does not improve.** CellOT + measured + quantiles vs CellOT + measured: −0.014 (p 0.57) with all samples
and +0.012 (p 0.41) without the two visits. gauss_ot: −0.005 / −0.027. The selected quantiles sit on the same axis as
the median features they replace (naive CD4 T pSTAT3/5, Granulocytes pS6), mostly upper tails (q75/q90). They are
different columns carrying the same information, not new signal. This is the table above: the CellOT model keeps
exactly the same non-quantile features with or without quantiles (40.9 per split), and the quantiles it adds correlate
0.88–0.99 with their own median, which is usually selected too. XGBoost gets two columns carrying the same
information, so R² does not move. A quantile would add something only if a tail varied between women independently of
the median. The few that do (e.g. `CD8Tcm_CCR2pos_HLADR_q90_IL33q`, r 0.33) are not selected consistently. Without the two visits, CellOT + measured and
gauss_ot + measured are tied at 20 splits (0.417 vs 0.423).
""")

md(r"""
### 8b. The same comparison at the paper's resolution: 100 splits (2026-09-28)

Stabl re-run on all 100 splits for both simulated-quantile models (`stabl_launch/simq_stabl_n100.sbatch`), then the
same aligned rescoring as every other 100-split arm: `params_43`, the 97 XGBoost settings, and without the two
mishandled visits. Everything here is at 100 splits, so the comparison with the models without quantiles is like for
like.
""")

code(r"""
rows = []
for tag, lab_ in [("aligned", "all 177"), ("aligned_excl", "36_G1, 52_G1 removed")]:
    for a, ref in [("comb_cellot_simq", "comb_cellot"), ("comb_gaussot_simq", "comb_gaussot")]:
        sa = pd.read_csv(N20 / tag / a / "sweep.csv").set_index("params"); sr = pd.read_csv(N20 / tag / ref / "sweep.csv").set_index("params")
        d = pd.read_csv(N20 / tag / a / "folds.csv")["params_43"] - pd.read_csv(N20 / tag / ref / "folds.csv")["params_43"]
        r = {"samples": lab_, "model": a, "R² with quantiles": sa.loc["params_43", "R2"], "R² without": sr.loc["params_43", "R2"],
             "per-split Δ": d.mean(), "p": wilcoxon(d).pvalue, "splits better": f"{(d > 0).sum()}/100"}
        if len(sa) == 97 and len(sr) == 97:
            r.update({"best of 97 with / without": f"{sa.R2.max():.3f} / {sr.R2.max():.3f}",
                      "median of 97 with / without": f"{sa.R2.median():.3f} / {sr.R2.median():.3f}"})
        rows.append(r)
display(pd.DataFrame(rows).round(3))
sel100 = {a: pd.read_csv(N20 / "stabl_n100" / a / "Training CV/Selected Features STABL ALasso.csv", index_col=0)["Fold selected features"].apply(ast.literal_eval)
          for a in ["comb_cellot_simq", "comb_cellot", "comb_gaussot_simq", "comb_gaussot"]}
for a, ref in [("comb_cellot_simq", "comb_cellot"), ("comb_gaussot_simq", "comb_gaussot")]:
    same = sum(set(f for f in la if not isq(f)) == set(lb) for la, lb in zip(sel100[a], sel100[ref]))
    nq = np.mean([sum(isq(f) for f in l) for l in sel100[a]])
    print(f"{a}: {nq:.1f} simulated quantiles selected per split; non-quantile selection identical to {ref} in {same}/100 splits")
""")

md(r"""
At 100 splits the verdict is the same: **the simulated quantiles add nothing**.
- CellOT + measured: 0.452 with quantiles vs 0.449 without (all samples), and 0.449 vs 0.448 without the two visits.
  Per split, p 0.50–0.54.
- gauss_ot + measured: 0.447 vs 0.441, and 0.445 vs 0.446 (p 0.33–0.83).
- They are again pure additions (non-quantile selections identical in 100/100 splits), ~28 per split for CellOT.

Everything about quantiles, measured and simulated, is gathered in `notebooks/quantiles.ipynb`.

## 9. Drug arm with quantile_map drug maps

A quantile map is increasing for each marker, so the median of the transported cells equals the map applied to the
median. The treated feature is therefore a 1-D function of the untreated feature value: no cells are transported,
and one OT map (the stim map) sits before Stabl. Maps are fitted with the same control wells, batch correction and
80/20 split as the CellOT drug maps (`gauss_drug_maps.py --method quantile_map`, `baselines/drug_features_qmap.py`).
On the atlas they beat CellOT's drug maps on the features Stabl selects (`diagnosis.ipynb` Part 2b). There are four
(stim model × drug map) combinations, all with 36_G1 and 52_G1 excluded. *z* = mean shift / SD of the
random-direction control.
""")

code(r"""
E = O / "drug_models/excl"
combos = {"cellot_cellotdrug": "CellOT + CellOT drug (paper)", "cellot_qmapdrug": "CellOT + quantile_map",
          "gauss_gaussdrug": "gauss_ot + gauss_ot drug", "gauss_qmapdrug": "gauss_ot + quantile_map"}
parts = {}
for k, n in combos.items():
    t = pd.read_csv(E / k / "summary_by_drug.csv", index_col=0)
    parts[n] = pd.DataFrame({"shift (d)": t["mean"], "folds agree": t[["frac_folds_negative", "frac_folds_positive"]].max(1),
                             "z": t["mean"] / t["null_sd_of_mean_shift"], "implausible": t["frac_treated_beyond_p95_untreated"],
                             "out of range": t["frac_values_outside_train_range"]})
D4 = pd.concat(parts, axis=1)
display(D4.xs("shift (d)", axis=1, level=1).join(D4.xs("folds agree", axis=1, level=1), rsuffix=" | agree")
        .sort_values("CellOT + quantile_map").round(2))
display(D4.xs("implausible", axis=1, level=1).round(2).style.set_caption("share of implausible treated women"))
from scipy.stats import spearmanr
sh = D4.xs("shift (d)", axis=1, level=1)
print("Spearman of drug rankings vs the paper's:", {c: round(spearmanr(sh.iloc[:, 0], sh[c])[0], 2) for c in sh.columns[1:]})
""")

md(r"""
- **THF** is the strongest effect everywhere, and cleanest with quantile_map: −10.4 d, 99 % of folds, z ≈ −8.
- **SALPZ** holds: −5.2 d, 98 %.
- **MAP** loses its effect (−1.8 d / 79 % with CellOT stims, −0.5 d with gauss_ot). Its effect in the paper came
  from the CellOT drug maps.
- **CHT** becomes a consistent positive (labor predicted earlier), 88–96 % of folds.
- **Implausible treated women** fall from 48–81 % (paper chain) to 7–10 %. The exceptions are SA and LPZ (20–38 %),
  the two drugs referenced to the DMSO1 plate, i.e. the plate confound.
- The THF / SALPZ result therefore needs neither the CellOT drug maps nor, with gauss_ot stims, any neural network.

""")

md(r"""
## 10. Summary

| | CellOT | gauss_ot |
|---|---|---|
| training, 1932 maps | ~12 560 CPU-hours (median 55 min per stim map, ~4 h per drug map) | ~10 min (stim) + ≤ 12 core-hours (drug, mostly file reading) — ~1000× less; closed form, no seed |
| atlas, stim | best on MMD / W1 | best on mean, median, correlation errors |
| atlas, drug | better on large effects | slightly better near the noise floor |
| DOS, 20 splits | 0.29–0.30 | 0.27–0.32, no significant difference |
| DOS, paper protocol (aligned) | 0.418 | 0.399 (p ≈ 0.002 per split) |
| DOS, lab's published script | 0.415 | 0.418 |
| + measured features (aligned) | **0.449** (beats CellOT alone, p = 0.014) | 0.441 (vs CellOT alone p = 0.25; vs CellOT + measured −0.027, p = 0.007) |
| drug arm, top drug | THF | THF (ρ = 0.80 with the paper's ranking) |
| implausible virtual twins | 56 % | 10 % |
| DOS without 36_G1/52_G1 (100 splits) | 0.415 | 0.364 |
| + simulated quantiles (20 splits) | no gain (p 0.41–0.57) | no gain |
| drug arm with quantile_map drug maps | THF −10.4 d (99 %), SALPZ −5.2 d, MAP lost; 7–10 % implausible | THF −10.8 d (100 %), SALPZ −4.7 d |

At a fraction of a percent of the compute, a two-moment linear map reproduces most of what CellOT
delivers downstream; CellOT's measurable edge is small (~0.02 R² on the paper's protocol once the
evaluation bug is fixed) and comes with far more extrapolation in the drug arm.
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python", "version": "3.9"}}
out = "/scratch/users/bmerwan/simco/notebooks/cellot_vs_gaussot.ipynb"
nbf.write(nb, out); print("wrote", out, f"({len(cells)} cells)")
