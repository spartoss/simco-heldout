"""Interview figures for the SIMCO work, all regenerated from the source result tables (nothing typed by hand
except where a comment says so). Each figure also writes the exact numbers it plots to data/<name>.csv.

    module load python/3.9.0; source /scratch/users/bmerwan/venv-cellot/bin/activate
    python make_figures.py            # -> figures/*.png, data/*.csv

Sources:
- DOS (paper protocol, 100 splits, params_43, row-order bug fixed = "aligned"):
  simco_work/onset_test/paper_protocol/aligned/<arm>/sweep.csv, atlas_heldout/work/heldout/quantile_arms/summary/sets.csv,
  simco_work/onset_test/ml_dos/summary/scores.csv
- Stim maps on new donors (leave-one-donor-out, 19 folds, PI / IL33 out, q50-q90 selected features):
  atlas_heldout/work/lc/tab/final/{metrics_rule,by_donor_rule,variance_rule,cases}.*, work/lc/tab/red_lodo/lin_summary.csv
- Learning curve: atlas_heldout/work/lc/summary/curve.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

HERE = Path(__file__).parent
FIG, DAT = HERE / "figures", HERE / "data"
FIG.mkdir(exist_ok=True); DAT.mkdir(exist_ok=True)
SC = Path("/scratch/users/bmerwan")
PP = SC / "simco_work/onset_test/paper_protocol/aligned"
LC = SC / "atlas_heldout/work/lc"
FIN = LC / "tab/final"

# palette (dataviz reference instance): blue = our result / highlight, orange = second series, greys = reference
BLUE, ORANGE, INK, INK2, MUTED, GRID, REF, BG = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#8a8a85", "#e6e5e0", "#b9b8b2", "#fcfcfb"
LIGHTBLUE = "#9ec5f4"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK, "text.color": INK, "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "axes.titlesize": 13, "axes.titleweight": "bold", "font.family": "DejaVu Sans"})


def title(fig, main, sub=None):
    fig.suptitle(main, x=0.012, y=0.985, ha="left", va="top", fontweight="bold", fontsize=15)
    if sub:
        fig.text(0.012, 0.915, sub, ha="left", va="top", color=INK2, fontsize=11)


def footnote(fig, text):
    fig.text(0.012, 0.012, text, ha="left", va="bottom", color=MUTED, fontsize=9)


def save(fig, name, df):
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)
    df.to_csv(DAT / f"{name}.csv", index=False)
    print("wrote", name)


def hbar(ax, labels, values, colors, fmt, xmax, ref=None):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.62, edgecolor=BG, linewidth=2)
    for yi, v, s in zip(y, values, fmt):
        ax.text(v + xmax * 0.01, yi, s, va="center", color=INK, fontsize=11.5, zorder=4,
                bbox=dict(boxstyle="square,pad=0.15", fc=BG, ec="none"))
    ax.set_yticks(y, labels); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, xmax); ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
    if ref is not None:
        ax.plot([ref[0]], [len(labels) - 0.45], marker="v", ms=9, color=MUTED, clip_on=False)
        ax.text(ref[0], len(labels) - 0.3, f"{ref[1]}: {ref[0]:.4f}", color=MUTED, fontsize=10, ha="center", va="bottom")


# ---------------------------------------------------------------- 1. DOS: baselines / null maps
def p43(arm):
    s = pd.read_csv(PP / arm / "sweep.csv").set_index("params")
    return s.loc["params_43", "R2"], s.loc["params_43", "RMSE"]

sets = pd.read_csv(SC / "atlas_heldout/work/heldout/quantile_arms/summary/sets.csv").set_index("set")
sc = pd.read_csv(SC / "simco_work/onset_test/ml_dos/summary/scores.csv")
auto = sc[(sc.arm == "full") & (sc.model == "auto")].iloc[0]
rc = sets.loc["rule_corr", "R2 params_43"]
rows = [("untrained random network (ICNN)", "rscaled_s0_both"), ("maps of the wrong stimulation / cell type", "crossswap_s0_both"),
        ("measured resting cells only\n(frequencies, medians, quantiles)", "d_qtl"), ("CellOT (paper)", "cellot")]
d1 = pd.DataFrame([(l, a, *p43(a), "177 visits") for l, a in rows], columns=["features", "arm", "R2", "RMSE_days", "note"])
d1 = pd.concat([d1, pd.DataFrame([
    ("CellOT + measured features, simulated\nquantiles, correlation filter", "rule_corr", rc,
     31.3 * np.sqrt(1 - rc),  # RMSE derived from R2 as in the meeting summary (31.3 d = constant-prediction error); not saved by the run
     "175 visits, RMSE derived from R²"),
    ("measured resting cells only,\ndense supervised model (no simulation)", "ml_dos_auto", auto.R2, auto.RMSE,
     "175 visits, model chosen inside each split")], columns=d1.columns)], ignore_index=True)
fig, ax = plt.subplots(figsize=(11.5, 6.4)); fig.subplots_adjust(left=0.33, right=0.97, top=0.82, bottom=0.12)
hbar(ax, d1.features, d1.R2, [REF, REF, REF, MUTED, MUTED, BLUE],
     [f"R² {r:.2f}  ·  {e:.1f} days" for r, e in zip(d1.R2, d1.RMSE_days)], 0.7)
ax.axhline(1.5, color=GRID, lw=1.5)   # separates the paper and its baselines (top) from our improvements (bottom two)
ax.set_xlabel("R² of days-to-labour prediction")
title(fig, "Predicting labour onset: the gain comes from the measured cells, not the simulation",
      f"Simple inputs come close to CellOT; using the measured cells richly cuts the error from "
      f"{d1.RMSE_days.iloc[3]:.1f} to {d1.RMSE_days.iloc[-1]:.1f} days (better in {int(auto.vs_cellot_wins)}/100 splits)")
footnote(fig, "Paper protocol (Stabl + XGBoost params_43, 100 splits), row-order bug fixed. 177 visits; last two rows 175 (2 mishandled visits removed).")
save(fig, "01_dos", d1)

# ---------------------------------------------------------------- 3. stim: CellOT vs the average shift, per new donor
BD = pd.read_csv(FIN / "by_donor_rule.csv", index_col=0)
d3 = BD[["shift", "CellOT_paper"]].reset_index()
M = pd.read_csv(FIN / "metrics_rule.csv").set_index("method")
win = int((d3.CellOT_paper < d3["shift"]).sum())
gain = 100 * (1 - M.loc["CellOT_paper", "MSE"] / M.loc["shift", "MSE"])
fig, ax = plt.subplots(figsize=(7.5, 6.6)); fig.subplots_adjust(left=0.14, right=0.96, top=0.82, bottom=0.12)
lim = (0, max(d3[["shift", "CellOT_paper"]].max()) * 1.08)
ax.plot(lim, lim, color=MUTED, ls="--", lw=1.2); ax.set_xlim(lim); ax.set_ylim(lim)
ax.scatter(d3["shift"], d3.CellOT_paper, s=80, color=BLUE, edgecolor=BG, lw=2, zorder=3)
ax.text(lim[1] * 0.97, lim[1] * 0.9, "CellOT worse", ha="right", color=MUTED, fontsize=10, rotation=0)
ax.text(lim[1] * 0.97, lim[1] * 0.05, "CellOT better", ha="right", color=MUTED, fontsize=10)
ax.set_xlabel("error of 'add the average response' (MSE)"); ax.set_ylabel("error of CellOT (MSE)")
ax.grid(color=GRID); ax.set_axisbelow(True); ax.set_aspect("equal")
title(fig, "On new donors, CellOT barely beats the average response",
      f"One dot per held-out donor: CellOT {gain:.0f} % lower error on average, better for {win}/19 donors (n.s.)")
footnote(fig, "Leave-one-donor-out (train 18, test 1), 5 stims (PI, IL33 out), selected features q50-q90.")
save(fig, "03_stim_cellot_vs_shift", d3)

# ---------------------------------------------------------------- 4. learning curve (RMSE), from lc_rmse.py
LR = pd.read_csv(DAT / "lc_rmse_cases.csv")
d4 = LR.groupby(["method", "k"]).mse.mean().pipe(np.sqrt).rename("RMSE").reset_index()
names = {"CellOT_paper": ("CellOT, paper training", LIGHTBLUE, "-"),
         "CellOT_pcblood": ("CellOT + whole-blood covariate", ORANGE, "-"),
         "CellOT_cluster": ("CellOT + phenotype-subcluster covariate", BLUE, "-")}
fig, ax = plt.subplots(figsize=(11, 5.8)); fig.subplots_adjust(left=0.09, right=0.97, top=0.76, bottom=0.13)
for m, (lab, c, ls) in names.items():
    x = d4[d4.method == m].sort_values("k")
    ax.plot(x.k, x.RMSE, color=c, ls=ls, lw=2.4, marker="o", ms=6, label=lab)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=10.5)
ax.set_xticks([2, 3, 4, 5, 6, 8, 10, 12, 15]); ax.set_xlim(1.6, 15.3)
ax.set_xlabel("number of training donors"); ax.set_ylabel("error on new donors (RMSE)")
ax.grid(color=GRID); ax.set_axisbelow(True)
r = d4.pivot(index="method", columns="k", values="RMSE")
title(fig, "The subcluster covariate helps from 5 donors; the whole-blood covariate needs ~15",
      f"k = 3: paper {r.loc['CellOT_paper', 3]:.3f} vs covariates {r.loc['CellOT_cluster', 3]:.3f} / {r.loc['CellOT_pcblood', 3]:.3f};  "
      f"k = 15: paper {r.loc['CellOT_paper', 15]:.3f} vs {r.loc['CellOT_cluster', 15]:.3f} / {r.loc['CellOT_pcblood', 15]:.3f}")
footnote(fig, "Held-out donors (4 per repeat, repeats 1-4; repeat 0 set the covariate penalty), 5 stims (PI, IL33 out), q50-q90, "
              "effect > 0.0273. RMSE = sqrt(mean per-donor MSE).")
save(fig, "04_learning_curve", d4)

# ---------------------------------------------------------------- 5. stim ladder (MSE)
steps = [("CellOT, paper training", "CellOT_paper"), ("+ paired batches (one donor per batch)", "CellOT_paired"),
         ("+ phenotype-subcluster covariate", "CellOT_cluster"), ("+ average with the mean shift (50/50)", "CellOT_cluster+shift"),
         ("+ TabPFN donor model (0.4 CellOT / 0.6 TabPFN)", "0.4 CellOT_cluster + 0.6 pfn_L1allM")]
d5 = pd.DataFrame([(l, m, M.loc[m, "MSE"]) for l, m in steps], columns=["step", "method", "MSE"])
d5["change_vs_previous_%"] = 100 * (d5.MSE / d5.MSE.shift() - 1)
d5["donors_better_than_previous"] = [np.nan] + [int((BD[b] < BD[a]).sum()) for a, b in zip(d5.method[:-1], d5.method[1:])]
fig, ax = plt.subplots(figsize=(11.5, 5.8)); fig.subplots_adjust(left=0.36, right=0.97, top=0.8, bottom=0.14)
lab = [f"{v:.4f}" + ("" if np.isnan(c) else f"   {c:+.0f} %  ({int(w)}/19 donors)")
       for v, c, w in zip(d5.MSE, d5["change_vs_previous_%"], d5.donors_better_than_previous.fillna(0))]
hbar(ax, d5.step, d5.MSE, [REF] * 4 + [BLUE], lab, 0.058)
ax.set_xlabel("error on new donors (MSE)")
tot = 100 * (1 - d5.MSE.iloc[-1] / d5.MSE.iloc[0])
title(fig, f"Predicting a new donor's stimulated cells: −{tot:.0f} % error vs the paper's CellOT",
      "Each step kept only if it helps on donors never seen in training (% and donors vs the step above)")
footnote(fig, "Leave-one-donor-out, 19 donors, 5 stims (PI, IL33 out), q50-q90 selected features. "
              f"With the whole-blood covariate instead: {M.loc['0.4 CellOT_pcblood + 0.6 pfn_L1allM', 'MSE']:.4f} (equivalent).")
save(fig, "05_stim_ladder", d5)

# ---------------------------------------------------------------- 6. TabPFN feature engineering
R = pd.read_csv(LC / "tab/red_lodo/lin_summary.csv").set_index("method")
feat = [("no donor information", "pfn_S0", M), ("+ 5 PCs of the donor's own cell-type profile", "pfn_S1_5", M),
        ("raw own profile instead (13 markers × 5 quantiles)", "pfn_L1cat", M),
        ("+ cell-type frequencies, pregnant / not, blood PCs", "pfn_L1all_raw", M),
        ("+ same marker in the donor's 27 other cell types", "pfn_L1allM", M),
        ("compact version (72 columns)", "pfn_C_ms_lin", R)]
d6 = pd.DataFrame([(l, m, t.loc[m, "MSE"]) for l, m, t in feat], columns=["features", "method", "MSE"])
fig, ax = plt.subplots(figsize=(11.5, 5.8)); fig.subplots_adjust(left=0.40, right=0.97, top=0.8, bottom=0.14)
hbar(ax, d6.features, d6.MSE, [REF, REF, REF, REF, BLUE, BLUE], [f"{v:.4f}" for v in d6.MSE], 0.05)
ax.set_xlabel("error on new donors, TabPFN alone (MSE)")
title(fig, "TabPFN: raw features and the donor's other cell types carry the signal",
      "Target = the donor's deviation from the average response; PCs lose information, other cell types add it")
footnote(fig, "Leave-one-donor-out, 19 donors, 5 stims, q50-q90 selected features. TabPFN v2 regressor, one table per (stim, cell type).")
save(fig, "06_tabpfn_features", d6)

# ---------------------------------------------------------------- 7-8. per lineage
cases = pd.read_pickle(FIN / "cases.pkl"); cases = cases[cases.set.str.startswith("rule")].copy()
LIN = {"CD4 T": ["CD4Tcm", "CD4Tcm_CCR2pos", "CD4Teff", "CD4Tem", "CD4Tem_CCR2pos", "CD4Tnaive", "CD4Tregs"],
       "CD8 T": ["CD8Tcells_Th1", "CD8Tcm", "CD8Tcm_CCR2pos", "CD8Teff", "CD8Tem", "CD8Tem_CCR2pos", "CD8Tnaive"],
       "NKT / DN T": ["NKT", "CD4negCD8negTcells"],
       "NK": ["CD56hiCD16negNK", "CD56loCD16posNK", "NK_cells_CD11cneg", "NK_cells_CD11cpos"],
       "B": ["Bcells"], "Monocytes": ["cMCs", "intMCs", "ncMCs"], "DCs": ["mDCs", "pDCs"], "Granulocytes / MDSCs": ["Granulocytes", "MDSCs"]}
cases["lineage"] = cases.celltype.map({c: l for l, cs in LIN.items() for c in cs})
assert cases.lineage.notna().all()
cases["se_prev"] = (0.5 * cases.CellOT_paired + 0.5 * cases["shift"] - cases.true) ** 2
cases["se_best"] = (0.4 * cases.CellOT_cluster + 0.6 * cases.pfn_L1allM - cases.true) ** 2

def by(g):
    pdn = cases.groupby(g + ["donor"])[["se_prev", "se_best"]].mean().reset_index()
    out = pdn.groupby(g).agg(prev=("se_prev", "mean"), best=("se_best", "mean")).reset_index()
    out["gain_%"] = 100 * (1 - out.best / out.prev)
    w = pdn.assign(b=pdn.se_best < pdn.se_prev).groupby(g).b.sum().reset_index(name="donors_better")
    return out.merge(w, on=g)

d7 = by(["lineage"]).sort_values("gain_%")
allg = 100 * (1 - cases.groupby("donor").se_best.mean().mean() / cases.groupby("donor").se_prev.mean().mean())
allw = int((cases.groupby("donor").se_best.mean() < cases.groupby("donor").se_prev.mean()).sum())
fig, ax = plt.subplots(figsize=(11, 6)); fig.subplots_adjust(left=0.2, right=0.97, top=0.78, bottom=0.12)
y = np.arange(len(d7))
ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
ax.hlines(y, d7.best, d7.prev, color=GRID, lw=4)
ax.scatter(d7.prev, y, s=90, facecolor=BG, edgecolor=MUTED, lw=2.2, zorder=3, label="CellOT + average response (previous best)")
ax.scatter(d7.best, y, s=90, color=BLUE, edgecolor=BG, lw=2, zorder=4, label="CellOT + TabPFN (best)")
for yi, (_, r) in zip(y, d7.iterrows()):
    ax.text(r.prev + 0.0012, yi, f"−{r['gain_%']:.0f} %  ({int(r.donors_better)}/19 donors)", va="center", color=INK2, fontsize=10.5)
ax.set_yticks(y, d7.lineage); ax.tick_params(axis="y", length=0); ax.set_xlim(0.018, 0.072)
ax.set_xlabel("error on new donors (MSE)")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=10.5)
title(fig, f"The gain holds in every immune lineage (overall −{allg:.0f} %, {allw}/19 donors)")
footnote(fig, "Leave-one-donor-out, 19 donors, 5 stims, q50-q90 selected features; 28 cell types grouped into 8 lineages.")
save(fig, "07_lineage_gain", d7)

d8 = by(["lineage", "stim"])
H = d8.pivot(index="lineage", columns="stim", values="gain_%").loc[d7.lineage[::-1]]
cmap = LinearSegmentedColormap.from_list("blue", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
fig, ax = plt.subplots(figsize=(9, 6)); fig.subplots_adjust(left=0.22, right=0.95, top=0.82, bottom=0.1)
im = ax.imshow(H.clip(lower=0).values, cmap=cmap, vmin=0, vmax=35, aspect="auto")
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        v = H.iat[i, j]
        ax.text(j, i, "–" if np.isnan(v) else f"{v:.0f}", ha="center", va="center",
                color=MUTED if np.isnan(v) else ("white" if v > 20 else INK), fontsize=11.5)
ax.set_xticks(range(H.shape[1]), H.columns); ax.set_yticks(range(H.shape[0]), H.index)
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
ax.set_xticks(np.arange(-.5, H.shape[1]), minor=True); ax.set_yticks(np.arange(-.5, H.shape[0]), minor=True)
ax.grid(False); ax.grid(which="minor", color=BG, lw=3); ax.tick_params(which="minor", length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("% lower error vs previous best", color=INK2); cb.outline.set_visible(False)
title(fig, "Where TabPFN helps most: % lower error, lineage × stimulation", "– = no selected feature for this pair")
save(fig, "08_lineage_stim_heatmap", d8)

# ---------------------------------------------------------------- 9. plate design schematic (no data)
fig = plt.figure(figsize=(12, 5.6)); ax = fig.add_axes([0, 0, 1, 0.86]); ax.set_xlim(0, 12); ax.set_ylim(0, 5.6); ax.axis("off"); ax.set_aspect("equal")
title(fig, "How the atlas was measured", "Every well holds all 19 donors (barcoded); each condition is repeated on 3 plates")
for p, x0 in enumerate([0.3, 3.0, 5.7]):
    ax.add_patch(FancyBboxPatch((x0, 1.0), 2.4, 3.6, boxstyle="round,pad=0.05,rounding_size=0.15", fc="#f0efec", ec=MUTED, lw=1.5))
    ax.text(x0 + 1.2, 4.8, f"plate {p + 1}  (DMSO{p + 1} control)", ha="center", fontsize=11, color=INK)
    for r in range(6):
        for c in range(4):
            hl = (r == 1 and c == 1)
            ax.add_patch(Circle((x0 + 0.45 + c * 0.5, 4.1 - r * 0.55), 0.19, fc=BLUE if hl else "white", ec=BLUE if hl else MUTED, lw=1.2))
    ax.text(x0 + 1.2, 0.6, "wells: resting, 7 stims, 9 drugs", ha="center", fontsize=9.5, color=INK2)
ax.annotate("", xy=(9.0, 3.4), xytext=(6.95, 3.55), arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.8))
ax.add_patch(Circle((10.2, 3.0), 1.5, fc="white", ec=BLUE, lw=2))
rng = np.random.default_rng(1)
cols = plt.cm.tab20(np.linspace(0, 1, 19))
for i in range(19):
    a = 2 * np.pi * i / 19; rr = 1.0 if i % 2 else 0.55
    ax.add_patch(Circle((10.2 + rr * np.cos(a), 3.0 + rr * np.sin(a)), 0.13, fc=cols[i], ec="white", lw=0.8))
ax.text(10.2, 5.0, "one well = one condition (e.g. IFNa)", ha="center", fontsize=11)
ax.text(10.2, 1.05, "19 donors, barcoded, measured together", ha="center", fontsize=10.5, color=INK2)
ax.text(10.2, 0.6, "→ well effects are shared by all donors", ha="center", fontsize=10.5, color=INK2)
save(fig, "09_plate_design", pd.DataFrame({"note": ["schematic, no data"]}))

# ---------------------------------------------------------------- 10c. share of donor differences explained (backup)
V = pd.read_csv(FIN / "variance_rule.csv")
V = V[V.method == "0.4 CellOT_cluster + 0.6 pfn_L1allM"].copy()
V["average response"] = V.baseline_shift
V["added by our model"] = V.explained - V.baseline_shift
V["noise (plate × donor)"] = V.noise
V["real, not yet explained"] = 1 - V.explained - V.noise
parts = [("average response", REF, None), ("added by our model", BLUE, None),
         ("real, not yet explained", ORANGE, None), ("noise (plate × donor)", "#f0efec", "///")]
order = ["ALL"] + sorted(set(V.stim) - {"ALL"})
V = V.set_index("stim").loc[order].reset_index()
d10 = V[["stim"] + [p for p, _, _ in parts]]
fig, ax = plt.subplots(figsize=(11.5, 5.8)); fig.subplots_adjust(left=0.1, right=0.97, top=0.76, bottom=0.12)
y = np.arange(len(V))[::-1]; left = np.zeros(len(V))
for p, c, h in parts:
    v = V[p].values
    ax.barh(y, v, left=left, color=c, hatch=h, edgecolor=BG if h is None else MUTED, linewidth=0 if h is None else 0.5,
            height=0.66, label=p)
    for yi, l, w in zip(y, left, v):
        if w > 0.03:
            ax.text(l + w / 2, yi, f"{100 * w:.0f} %", ha="center", va="center", fontsize=10.5,
                    color="white" if c in (BLUE, ORANGE) else INK)
    left += v
ax.set_yticks(y, ["all stims" if s == "ALL" else s for s in V.stim]); ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 1); ax.set_xticks([0, .25, .5, .75, 1], ["0", "25 %", "50 %", "75 %", "100 %"])
ax.set_xlabel("share of the differences between donors in the stimulated cells")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=10.5)
a = V.set_index("stim").loc["ALL"]
title(fig, f"Only {100 * a['noise (plate × donor)']:.0f} % of donor differences is noise: "
           f"{100 * a['real, not yet explained']:.0f} % is real but still unexplained")
footnote(fig, "Noise = donor × plate interaction across the 3 control plates. Leave-one-donor-out, best model, q50-q90 selected features.")
save(fig, "10c_variance_explained", d10)

# ---------------------------------------------------------------- 10a. noise components per stim (3-plate design)
# x[d, p] = average + a[d] (donor, real) + w[p] (well/plate effect shared by all donors) + e[d, p] (donor x well noise),
# per (cell type, feature) on the 3 DMSO plates; medians over cell types as in atlas_heldout dmso_plate.ipynb Part 3b.
PV = pd.read_csv(LC / "dmso/classify/plate_variance.csv")
PV = PV[PV.kind == "response"].groupby("condition")[["mean_response", "SD_donor", "SD_well", "SD_dxw"]].median()
for c in ["donor", "well", "dxw"]:
    PV["v_" + c] = PV["SD_" + c] ** 2
tot = PV[["v_donor", "v_well", "v_dxw"]].sum(1)
PV["donor (real)"] = PV.v_donor / tot
PV["well / plate, shared by all donors"] = PV.v_well / tot
PV["donor × well noise"] = PV.v_dxw / tot
PV["real share, 3-plate mean"] = PV.v_donor / (PV.v_donor + PV.v_dxw / 3)
PV = PV.sort_values("real share, 3-plate mean")
d10a = PV.reset_index().rename(columns={"condition": "stim"})
parts = [("donor (real)", BLUE, None), ("well / plate, shared by all donors", MUTED, None), ("donor × well noise", "#f0efec", "///")]
fig, ax = plt.subplots(figsize=(12, 6)); fig.subplots_adjust(left=0.09, right=0.72, top=0.76, bottom=0.12)
y = np.arange(len(PV)); left = np.zeros(len(PV))
for p_, c, h in parts:
    v = PV[p_].values
    ax.barh(y, v, left=left, color=c, hatch=h, edgecolor=BG if h is None else MUTED, linewidth=0 if h is None else 0.5,
            height=0.66, label=p_)
    for yi, l, w in zip(y, left, v):
        if w > 0.06:
            ax.text(l + w / 2, yi, f"{100 * w:.0f} %", ha="center", va="center", fontsize=10.5, color="white" if c != "#f0efec" else INK)
    left += v
ax.set_yticks(y, PV.index); ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 1); ax.set_xticks([0, .25, .5, .75, 1], ["0", "25 %", "50 %", "75 %", "100 %"])
ax.set_xlabel("share of the variance of a donor's response measured in one well")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=10.5)
ax.text(1.04, len(PV) - 0.35, "average\nresponse", transform=ax.get_yaxis_transform(), fontsize=10, color=INK2, ha="left", va="bottom")
ax.text(1.2, len(PV) - 0.35, "real share,\n3-plate mean", transform=ax.get_yaxis_transform(), fontsize=10, color=INK2, ha="left", va="bottom")
for yi, (_, r) in zip(y, PV.iterrows()):
    ax.text(1.04, yi, f"{r.mean_response:.2f}", transform=ax.get_yaxis_transform(), va="center", fontsize=11)
    ax.text(1.2, yi, f"{100 * r['real share, 3-plate mean']:.0f} %", transform=ax.get_yaxis_transform(), va="center", fontsize=11,
            color=ORANGE if r["real share, 3-plate mean"] < 0.6 else INK, fontweight="bold")
title(fig, "Where the noise is: each donor × stim was measured on 3 plates",
      "Strong stims (LPS, IFNa, IL246): donor differences are > 90 % real once the 3 plates are pooled; IL33 is mostly noise; PI has a plate effect")
footnote(fig, "Response = stim − resting, same plate; medians over 28 cell types (arcsinh units). Shared well effect is learnable; donor × well noise is not.")
save(fig, "10a_noise_components", d10a)

# ---------------------------------------------------------------- 10d. the floor, step by step (MSE, same cases as 05)
# Error a perfect model would still make against the pooled 3-plate truth (atlas_heldout lc/noise_floor_lodo.py):
# floor_samp = cell sampling (split-half), floor_dxw = + donor x well noise of the stimulated wells (shared well effect
# removed, learnable), floor_resp = + the same noise in the resting wells the predictions read (realistic floor),
# floor_all = stimulated wells with the shared well effect NOT removed.
cs = cases.merge(pd.read_pickle(FIN / "noise_floor.pkl"), on=["stim", "celltype", "donor", "q", "marker"], how="left")
pdm = lambda col: cs.groupby("donor")[col].mean().mean()          # per-donor mean, then mean over donors (as the models)
fl = {c: pdm(c) for c in ["floor_samp", "floor_dxw", "floor_resp", "floor_all"]}
steps = [("previous estimate: split one measurement\nin two halves (cell sampling only)", fl["floor_samp"], BG, "///"),
         ("+ donor × well noise, stimulated wells", fl["floor_dxw"], REF, None),
         ("+ donor × well noise, resting wells\n= realistic floor", fl["floor_resp"], MUTED, None),
         ("well effect shared by all donors not learned\n(stimulated wells only)", fl["floor_all"], REF, None),
         ("CellOT + TabPFN (ours)", M.loc["0.4 CellOT_cluster + 0.6 pfn_L1allM", "MSE"], BLUE, None),
         ("CellOT, paper training", M.loc["CellOT_paper", "MSE"], LIGHTBLUE, None),
         ("average response only", M.loc["shift", "MSE"], REF, None)]
d10d = pd.DataFrame([(l, v) for l, v, _, _ in steps], columns=["step", "MSE"])
fig, ax = plt.subplots(figsize=(11.5, 6.6)); fig.subplots_adjust(left=0.34, right=0.97, top=0.8, bottom=0.12)
yy = np.arange(len(steps))[::-1]
for yi, (l, v, c, h) in zip(yy, steps):
    ax.barh(yi, v, color=c, hatch=h, edgecolor=MUTED if h else BG, linewidth=1 if h else 0, height=0.62)
    ax.text(v + 0.0005, yi, f"{v:.4f}", va="center", fontsize=11.5)
ax.set_yticks(yy, d10d.step); ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 0.048); ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
ax.set_xlabel("error on new donors (MSE)")
best = M.loc["0.4 CellOT_cluster + 0.6 pfn_L1allM", "MSE"]
title(fig, f"Noise explains only {100 * fl['floor_resp'] / best:.0f} % of our remaining error",
      f"Realistic noise floor {fl['floor_resp']:.4f} vs our best model {best:.4f}: most of the error is real donor signal still missed")
footnote(fig, "Leave-one-donor-out, 19 donors, 5 stims (PI, IL33 out), q50-q90 selected features. Floors from the 3 replicate wells "
              "per donor × stim, error of a perfect model vs the pooled truth.")
save(fig, "10d_noise_floor_mse", d10d)

# ---------------------------------------------------------------- 10e. everything against ONE well (replicate = re-measuring the donor)
# Per-well stim quantiles of the 3 DMSO plate wells (well_ceiling/work/quant, the same cells as the learning-curve cache).
# Target = one single well; scores averaged over the wells of each (donor, stim, cell type), then per donor, then over donors.
# perfect model = the noise of one well around the donor's value = half the squared gap between two wells.
import sys
sys.path.insert(0, str(SC / "atlas_heldout"))
from lc.paths import MARKERS, QS
QW = SC / "well_ceiling/work/quant"; WELLS = ("DMSO1", "DMSO2", "DMSO3")
qi = {q: i for i, q in enumerate(QS)}; mi = {m: i for i, m in enumerate(MARKERS)}
cases["best"] = 0.4 * cases.CellOT_cluster + 0.6 * cases.pfn_L1allM
meth = {"best": "best", "CellOT_paper": "CellOT_paper", "shift": "shift"}
parts_ = []
for ct, g in cases.groupby("celltype"):
    z = np.load(QW / f"{ct}.npz"); keys = set(z.files)
    for (st, dn), h in g.groupby(["stim", "donor"]):
        ws = [w for w in WELLS if f"{st}|{dn}|w:{w}" in keys and float(z[f"{st}|{dn}|w:{w}|n"]) >= 20]
        if len(ws) < 2:
            continue
        ii, jj = h.q.map(qi).values, h.marker.map(mi).values
        v = {w: z[f"{st}|{dn}|w:{w}"][ii, jj] for w in ws}
        pairs = [(a, b) for k, a in enumerate(ws) for b in ws[k + 1:]]
        rep = np.mean([(v[a] - v[b]) ** 2 for a, b in pairs], 0)
        out = {"donor": dn, "stim": st, "another well of the same donor": rep, "perfect model (noise of one well)": rep / 2}
        for c in meth:
            out[c] = np.mean([(h[c].values - v[w]) ** 2 for w in ws], 0)
        parts_.append(pd.DataFrame(out))
OW = pd.concat(parts_)
cols = ["perfect model (noise of one well)", "another well of the same donor", "best", "CellOT_paper", "shift"]
labels = {"best": "CellOT + TabPFN (ours)", "CellOT_paper": "CellOT, paper training", "shift": "average response only"}
tot = OW.groupby("donor")[cols].mean().mean()
bys = OW.groupby(["stim", "donor"])[cols].mean().groupby("stim").mean()
d10e = pd.concat([tot.rename("ALL").to_frame().T, bys]).reset_index().rename(columns={"index": "stim"})
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6.4), gridspec_kw={"width_ratios": [1.15, 1]})
fig.subplots_adjust(left=0.215, right=0.98, top=0.78, bottom=0.12, wspace=0.33)
lab = [labels.get(c, c) for c in cols]; colc = [REF, MUTED, BLUE, LIGHTBLUE, REF]
yy = np.arange(len(cols))[::-1]
ax.barh(yy, tot.values, color=colc, height=0.62)
for yi, v_ in zip(yy, tot.values):
    ax.text(v_ + 0.0008, yi, f"{v_:.4f}", va="center", fontsize=11.5)
ax.axvline(tot["another well of the same donor"], color=MUTED, ls="--", lw=1.2, zorder=0)
ax.set_yticks(yy, lab); ax.tick_params(axis="y", length=0); ax.set_xlim(0, 0.05)
ax.grid(axis="x", color=GRID); ax.set_axisbelow(True); ax.set_xlabel("error against one well (MSE)")
ax.set_title("all stims", loc="left", fontsize=12)
stims_ = list(bys.index); x = np.arange(len(stims_)); wdt = 0.38
ax2.bar(x - wdt / 2, bys["another well of the same donor"], wdt, color=MUTED, label="another well of the same donor")
ax2.bar(x + wdt / 2, bys["best"], wdt, color=BLUE, label="CellOT + TabPFN (ours)")
ax2.set_xticks(x, stims_); ax2.grid(axis="y", color=GRID); ax2.set_axisbelow(True); ax2.set_ylabel("error against one well (MSE)")
ax2.legend(loc="upper left", fontsize=10); ax2.set_title("by stimulation", loc="left", fontsize=12)
ratio = tot["best"] / tot["another well of the same donor"]
title(fig, f"Predicting an unseen donor: {ratio:.1f}× the gap between two measurements of the same donor",
      "Same target for everyone: one single well. For GMCSF and TNFa our prediction is about as close as re-measuring the donor")
footnote(fig, "Leave-one-donor-out, 19 donors, 5 stims (PI, IL33 out), q50-q90 selected features. 3 DMSO-plate wells per donor × stim; "
              "perfect model = noise of one well (half the squared gap between two wells).")
save(fig, "10e_vs_one_well", d10e)

# ---------------------------------------------------------------- 11. what did not work (table figure)
K15 = pd.read_csv(DAT / "k15_methods_cases.csv").pivot_table(index=["repeat", "donor"], columns="method", values="mse")
km = K15.mean(); ref = km["CellOT_paired"]
pct = lambda m: f"{100 * (km[m] / ref - 1):+.0f} % MSE"
rows = [
    ("Cytometry foundation model (ImmuneFM) as donor encoder", f"MSE {km['ctx_ft_immunefm']:.4f} vs {km['ctx_ft_scratch']:.4f} (random init)", "= random initialisation"),
    ("Meta Flow Matching (built for unseen patients)", pct("mfm"), "worse"),
    ("Flow matching instead of optimal transport", pct("fm"), "worse"),
    ("One conditional network for all 168 maps (CondOT)", pct("condot_global"), "worse"),
    ("Donor covariate without regularisation", pct("ctx_pc_noreg"), "memorises donors"),
    ("Per-donor normalised inputs", pct("dnorm"), "worse"),
    ("PCs / PLS of the donor profile for TabPFN", f"MSE {R.loc['pfn_R_ownPC10', 'MSE']:.4f} / {R.loc['pfn_R_ownPLS5', 'MSE']:.4f} vs {R.loc['pfn_C_ms_lin', 'MSE']:.4f}", "raw columns win"),
    ("Lineage-group tables for TabPFN", "−2 to +3 % MSE", "no gain"),  # STATUS 2026-10-14, lineage round 2
    ("Predicting donor-specific drug effects", "two-step = mean effect", "not predictable"),  # STATUS 2026-10-16
    ("Better stim maps plugged into labour prediction", f"R² 0.423 vs 0.418", "n.s. (49/100)"),  # ool_tab_summary.csv
]
d11 = pd.DataFrame(rows, columns=["idea", "result vs reference", "verdict"])
fig = plt.figure(figsize=(12, 6.2)); ax = fig.add_axes([0.01, 0.06, 0.98, 0.78]); ax.axis("off")
title(fig, "What did not work, and why it is informative",
      "With 19 donors, donor-specific effects have 19 data points: complex models overfit, simple ones win")
ys = np.linspace(0.95, 0.03, len(d11) + 1)
for x, h in zip([0.0, 0.56, 0.84], ["idea", "result vs reference", "verdict"]):
    ax.text(x, ys[0], h, fontweight="bold", fontsize=11.5, color=INK2, va="center")
for i, r in d11.iterrows():
    yy = ys[i + 1]
    if i % 2 == 0: ax.add_patch(Rectangle((-0.01, yy - 0.04), 1.02, 0.08, fc="#f0efec", ec="none"))
    ax.text(0.0, yy, r.idea, fontsize=11.5, va="center")
    ax.text(0.56, yy, r["result vs reference"], fontsize=11, va="center", color=INK2)
    ax.text(0.84, yy, r.verdict, fontsize=11, va="center", color=ORANGE if r.verdict in ("worse", "memorises donors") else INK)
footnote(fig, "First 6 rows: MSE vs paired CellOT, 15 training donors, 16 held-out donor-repeats (k15_methods_mse.py). "
              "TabPFN rows: leave-one-donor-out, vs the compact raw feature set.")
save(fig, "11_what_did_not_work", d11)
