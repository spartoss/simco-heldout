"""Second set of slide figures (2026-09-24): atlas modelling power (stim, drug), the prediction
story (null maps, measured enrichment) on the paper protocol as run by the lab's rescore script.
Every value is recomputed from result files. Output: simco_work/presentation/fig[A-E]*.png.
    python notebooks/make_presentation_figures_2.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

W = Path("/scratch/users/bmerwan/simco_work")
O, B = W / "onset_test", W / "baselines"
OUT = W / "presentation"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b3"
plt.rcParams.update({"font.size": 12, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})
NAMES = {"cellot_insample": "CellOT", "median_shift": "median shift", "quantile_map": "quantile map",
         "gauss_ot": "gauss_ot", "identity": "identity (no effect)"}
COL = {"CellOT": BLUE, "median shift": ORANGE, "quantile map": AQUA, "gauss_ot": GREY}
first = lambda x: float(str(x).split()[0])


def hbars(ax, labels, vals, colors, fmt="{:.3f}", err=None, pad=.006):
    """err: 2 x n array (minus, plus) as matplotlib expects; pad in data units."""
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, color=colors, height=.62, edgecolor=SURF, linewidth=2,
            xerr=err, error_kw={"ecolor": INK2, "capsize": 3, "lw": 1})
    plus = err[1] if err is not None else np.zeros(len(vals))
    for yi, v, e in zip(y, vals, plus):
        right = v >= 0 or abs(v) < .05
        ax.text(max(v, 0) + e + pad if right else v - pad, yi, fmt.format(v), va="center",
                ha="left" if right else "right", fontsize=10.5)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.axvline(0, color=MUTED, lw=1); ax.grid(axis="x", color=GRID, lw=.8); ax.set_axisbelow(True)


caps = []
# ---------- Fig A: stim modelling on the atlas (127 live pairs, cell-weighted) --------------------
r = pd.read_csv(B / "summary_pooled_cell/all_rows.csv")
r = r[r.weighting.isin(["cell", "invariant"])]
idm = r[r.method == "identity"].set_index(["celltype", "stim"]).mmd
live = idm[idm >= 0.005].index
r = r.set_index(["celltype", "stim"]).loc[live].reset_index()
meths = ["cellot_insample", "median_shift", "quantile_map", "gauss_ot"]
fig, axes = plt.subplots(1, 2, figsize=(13, 3.4), gridspec_kw={"wspace": 0.9})
for ax, metric, title in [(axes[0], "median_error", "per-marker median error  (what Stabl uses)"),
                          (axes[1], "mmd", "whole-distribution error (MMD)")]:
    w = r.pivot_table(index=["celltype", "stim"], columns="method", values=metric)
    med = [w[m].median() for m in meths]
    beat = [np.nan if m == "cellot_insample" else (w[m] < w["cellot_insample"]).mean() for m in meths]
    labs = [NAMES[m] + ("" if m == "cellot_insample" else f"  (beats CellOT {b:.0%})") for m, b in zip(meths, beat)]
    hbars(ax, labs, med, [COL[NAMES[m]] for m in meths], fmt="{:.4f}", pad=max(med) * .03)
    ax.set_xlabel("median over 127 pairs (lower = better)"); ax.set_title(title, loc="left", fontsize=12.5)
    ax.set_xlim(0, max(med) * 1.35)
fig.suptitle("Stimulations: simple per-marker maps beat CellOT on marker medians; CellOT keeps the distribution",
             x=0.01, ha="left", fontsize=13.5, y=1.04)
fig.tight_layout(); fig.savefig(OUT / "figA_stim_atlas.png"); plt.close(fig)
caps.append(("figA_stim_atlas.png", f"Atlas (19 donors), 127 stim × celltype pairs with a real effect, test cells, cell-weighted. Left: error on per-marker medians — the quantity every Stabl feature is — median shift (x + constant) beats CellOT in {(r.pivot_table(index=['celltype','stim'],columns='method',values='median_error').median_shift < r.pivot_table(index=['celltype','stim'],columns='method',values='median_error').cellot_insample).mean():.0%} of pairs, even though median error is CellOT's own checkpoint-selection metric. Right: on the whole distribution (MMD) CellOT is best. Unweighted, the median-shift win rate is 61 %."))

# ---------- Fig B: drug modelling on the atlas ------------------------------------------------------
raw = pd.concat([pd.read_csv(f) for f in sorted((B / "pooled_drug").glob("*.csv"))], ignore_index=True)
MK = [c[len("median_error__"):] for c in raw.columns if c.startswith("median_error__")]
K = ["celltype", "stim", "drug"]
long = (raw[raw.weighting.isin(["cell", "invariant"])]
        .melt(id_vars=K + ["method"], value_vars=[f"median_error__{m}" for m in MK], var_name="marker", value_name="err"))
wide = long.pivot_table(index=K + ["marker"], columns="method", values="err").dropna(subset=meths + ["identity"])
pert = wide["identity"]
FLOOR = 0.0273
bins = [(FLOOR, .05, "0.027–0.05"), (.05, .1, "0.05–0.1"), (.1, .2, "0.1–0.2"), (.2, 9, "> 0.2")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 3.8), gridspec_kw={"width_ratios": [1, 1.25], "wspace": .45})
q = np.log10(pert.clip(lower=1e-4))
a1.hist(q, bins=60, color=GREY, edgecolor=SURF)
for t, lab_, yf, ha, dx in [(FLOOR, "noise between two\nidentical control wells", .9, "right", -.05),
                            (0.1, "paper's\nthreshold", .6, "left", .05)]:
    a1.axvline(np.log10(t), color=INK2, ls="--", lw=1)
    a1.text(np.log10(t) + dx, a1.get_ylim()[1] * yf, lab_, fontsize=9, color=INK2, ha=ha)
a1.set_xticks([-3, -2, -1]); a1.set_xticklabels(["0.001", "0.01", "0.1"])
a1.set_xlabel("drug effect on a marker's median\n(19 656 cases)"); a1.set_yticks([])
a1.set_title(f"{(pert < FLOOR).mean():.0%} of drug × marker effects are below the noise", loc="left", fontsize=12.5)
groups = [(FLOOR, "above the noise floor"), (0.1, "above the paper's threshold (0.1)")]
yb = np.arange(len(meths))[::-1]; hb = .38
for k, (thr, lab_) in enumerate(groups):
    sel = wide.loc[pert > thr]
    vals = [sel[m].median() for m in meths]
    a2.barh(yb + (hb / 2 if k == 0 else -hb / 2), vals, hb, color=[COL[NAMES[m]] for m in meths],
            alpha=1 if k == 0 else .55, edgecolor=SURF, linewidth=2)
    for yi, m, v in zip(yb, meths, vals):
        beat = "" if m == "cellot_insample" else f"  beats CellOT {(sel[m] < sel.cellot_insample).mean():.0%}"
        a2.text(v + .0015, yi + (hb / 2 if k == 0 else -hb / 2), f"{v:.4f}{beat}", va="center", fontsize=9)
a2.set_yticks(yb); a2.set_yticklabels([NAMES[m] for m in meths]); a2.set_xlim(0, .105)
a2.grid(axis="x", color=GRID, lw=.8)
n_floor, n_01 = int((pert > FLOOR).sum()), int((pert > .1).sum())
a2.set_xlabel(f"median error on the moving markers (lower = better)\ndark: above noise floor (n={n_floor}) · light: above 0.1 (n={n_01})")
a2.set_title("Simple per-marker maps equal or beat CellOT on drugs", loc="left", fontsize=12.5)
fig.tight_layout(); fig.savefig(OUT / "figB_drug_atlas.png"); plt.close(fig)
sm = wide.loc[pert > FLOOR]
caps.append(("figB_drug_atlas.png", f"Drugs (1512 drug × stim × celltype pairs, per marker). Left: drug effects are tiny — {(pert < FLOOR).mean():.0%} of marker effects are below 0.027, the difference between two control wells (DMSO2 vs DMSO3) that should be identical. Right: above that noise floor ({len(sm)} cases), median shift and quantile map have ~16 % lower median error than CellOT and beat it in {(sm.median_shift < sm.cellot_insample).mean():.0%} of cases; at the paper's threshold (> 0.1) it is a tie (paired p = 0.69 for both)."))

# ---------- Fig C / D: paper protocol, 100 splits, params_43, the LAB'S rescore script -------------------
# (get_foldinfo_on_existing_run.py as published; its row-order bug inflates the non-CellOT arms by
#  0.015-0.04 and leaves CellOT unchanged -- see CLAUDE.md Landmines and paper_protocol/aligned/)
def lab43(d):
    return first(pd.read_csv(d / "Summary/Scores training CV.csv", index_col=0).loc["STABL ALasso", "R2"])
def mae43(d):
    return first(pd.read_csv(d / "Summary/Scores training CV.csv", index_col=0).loc["STABL ALasso", "MAE"])
def with_mae(ax, dirs, vals):
    # replace hbars' R2 labels by "R2 · MAE x d"
    for t in list(ax.texts): t.remove()
    y = np.arange(len(vals))[::-1]
    for yi, v, d in zip(y, vals, dirs):
        ax.text(max(v, 0) + .006, yi, f"{v:.3f}   MAE {mae43(d):.1f} d", va="center", fontsize=10.5)
PR = O / "paper_protocol/rescore"
AXL = "R² for DOS — paper protocol: 100 splits, params_43, lab's rescore script"
unst = lab43(O / "rescore_params43/unstim_only")
cel = lab43(PR / "cellot_both/params_43")

cdirs = [O / "rescore_params43/unstim_only", PR / "rscaled_s0_both/params_43", PR / "crossswap_s0_both/params_43",
         PR / "gaussot_both/params_43", PR / "cellot_both/params_43"]
order = [("measured Unstim only (paper's block)", unst, GREY),
         ("untrained ICNN, CellOT-sized", lab43(cdirs[1]), GREY),
         ("trained maps, wrong stim & celltype", lab43(cdirs[2]), GREY),
         ("gauss_ot (13×13 linear map)", lab43(cdirs[3]), GREY),
         ("CellOT (paper)", cel, BLUE)]
fig, ax = plt.subplots(figsize=(9.5, 3.2))
hbars(ax, [o[0] for o in order], [o[1] for o in order], [o[2] for o in order], fmt="{:.3f}")
with_mae(ax, cdirs, [o[1] for o in order])
ax.set_xlim(0, 0.62); ax.set_xlabel(AXL)
ax.set_title("Any CellOT-sized nonlinear map recovers most of the gain — the right perturbation is not needed",
             loc="left", fontsize=12.5)
fig.savefig(OUT / "figC_null_maps.png"); plt.close(fig)
caps.append(("figC_null_maps.png", f"Paper protocol (100 splits, params_43, lab's rescore script), only the map that produces the simulated-stim features changes. Measured Unstim alone {unst:.3f}; an untrained random ICNN scaled to CellOT's displacement {order[1][1]:.3f}; CellOT maps trained for another stimulation and another celltype {order[2][1]:.3f}; a 13×13 linear map {order[3][1]:.3f}; CellOT {cel:.3f}. The random ICNN recovers ~80 % of CellOT's gain over measured Unstim. Numbers from the lab's rescore script as published (row-order bug; with rows aligned the non-CellOT arms drop 0.015–0.04, CellOT does not)."))

ddirs = [O / "rescore_params43/unstim_only", PR / "d_freqmed/params_43", PR / "d_qtl/params_43",
         PR / "gaussot_both/params_43", PR / "cellot_both/params_43"]
rows = [("measured Unstim, paper's block (13 functional medians)", unst, GREY),
        ("frequencies + medians of all 31 markers", lab43(ddirs[1]), AQUA),
        ("+ quantiles of functional markers", lab43(ddirs[2]), AQUA),
        ("gauss_ot simulated stims", lab43(ddirs[3]), GREY),
        ("CellOT simulated stims (paper)", cel, BLUE)]
fig, ax = plt.subplots(figsize=(9.5, 3.3))
hbars(ax, [x[0] for x in rows], [x[1] for x in rows], [x[2] for x in rows], fmt="{:.3f}")
with_mae(ax, ddirs, [x[1] for x in rows])
ax.set_xlim(0, 0.62); ax.set_xlabel(AXL)
ax.set_title("Summarising the unstimulated cells properly matches CellOT — no simulation needed",
             loc="left", fontsize=12.5)
fig.savefig(OUT / "figD_measured_enrichment.png"); plt.close(fig)
caps.append(("figD_measured_enrichment.png", f"Paper protocol (100 splits, params_43, lab's rescore script). The paper's measured baseline keeps only the 13 functional-marker medians ({unst:.3f}, MAE {mae43(ddirs[0]):.1f} d). Adding the 18 phenotypic markers and log cell frequencies — no simulation, no neural network — gives {rows[1][1]:.3f} (MAE {mae43(ddirs[1]):.1f} d), and with functional-marker quantiles {rows[2][1]:.3f} (MAE {mae43(ddirs[2]):.1f} d), vs gauss_ot {rows[3][1]:.3f} and CellOT {cel:.3f} (MAE {mae43(ddirs[4]):.1f} d). Lab's rescore script as published (row-order bug; aligned: measured 0.384 / 0.397, gauss_ot 0.399, CellOT 0.418)."))

# ---------- Fig F: combining simulated stims with the measured features ------------------------------
fdirs = [PR / "d_qtl/params_43", PR / "gaussot_both/params_43", PR / "cellot_both/params_43",
         PR / "comb_gaussot/params_43", PR / "comb_cellot/params_43"]
frows = [("measured: freq + medians + quantiles", lab43(fdirs[0]), AQUA),
         ("gauss_ot simulated stims", lab43(fdirs[1]), GREY),
         ("CellOT simulated stims (paper)", lab43(fdirs[2]), BLUE),
         ("gauss_ot stims + measured (no neural net)", lab43(fdirs[3]), GREY),
         ("CellOT stims + measured", lab43(fdirs[4]), BLUE)]
fig, ax = plt.subplots(figsize=(9.5, 3.3))
hbars(ax, [x[0] for x in frows], [x[1] for x in frows], [x[2] for x in frows], fmt="{:.3f}")
with_mae(ax, fdirs, [x[1] for x in frows])
ax.axhline(1.5, color=MUTED, lw=.8, ls=":")
ax.set_xlim(0, 0.72); ax.set_xlabel(AXL)
ax.set_title("Combining measured features with simulated stims beats CellOT alone", loc="left", fontsize=12.5)
fig.savefig(OUT / "figF_combined.png"); plt.close(fig)
caps.append(("figF_combined.png", f"Paper protocol (100 splits, params_43, lab's rescore script). One Stabl model on simulated stim features + measured frequencies / 31-marker medians / quantiles (3204 columns, 10 blocks). CellOT + measured {frows[4][1]:.3f} (MAE {mae43(fdirs[4]):.1f} d) and gauss_ot + measured {frows[3][1]:.3f} (MAE {mae43(fdirs[3]):.1f} d), vs {frows[2][1]:.3f} for CellOT alone (MAE {mae43(fdirs[2]):.1f} d). The two families carry partly different information (prediction correlation 0.66–0.73). This gain survives the bug fix: aligned, CellOT + measured 0.449 vs CellOT 0.418, +0.041 per split, 61/100, p = 0.014 (gauss_ot + measured 0.441, p = 0.25)."))

with open(OUT / "captions.md", "a") as f:
    f.write("\n\n## Second set\n\n" + "\n\n".join(f"**{a}** — {b}" for a, b in caps) + "\n")
print("wrote", [c[0] for c in caps])
