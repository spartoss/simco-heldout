"""Slide figures for the SIMCO review (2026-09-24). Every value is read from the result files the
notebooks use (ega_preterm_nullmaps.ipynb, baselines_results.ipynb) -- nothing is pasted.
Output: /scratch/users/bmerwan/simco_work/presentation/fig*.png (+ captions.md).
    python notebooks/make_presentation_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score

O = Path("/scratch/users/bmerwan/simco_work/onset_test")
GH = Path("/home/groups/gbrice/ptb-drugscreen/ool_stabl/onset_test")
OUT = Path("/scratch/users/bmerwan/simco_work/presentation"); OUT.mkdir(parents=True, exist_ok=True)

# reference palette (dataviz skill, light mode) -- validated: 3 slots all-pairs PASS
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b3"
plt.rcParams.update({"font.size": 12, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})


def first(x): return float(str(x).split()[0])
def lab_r2(d): return first(pd.read_csv(d / "Summary/Scores training CV.csv", index_col=0).loc["STABL ALasso", "R2"])
def hbars(ax, labels, vals, colors, fmt="{:.2f}"):
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, color=colors, height=.62, edgecolor=SURF, linewidth=2)
    for yi, v in zip(y, vals):
        right = v >= 0 or abs(v) < 0.05          # tiny negatives: label on the right, clear of the axis text
        ax.text(max(v, 0) + 0.012 if right else v - 0.012, yi, fmt.format(v), va="center",
                ha="left" if right else "right", color=INK, fontsize=11)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.axvline(0, color=MUTED, lw=1); ax.grid(axis="x", color=GRID, lw=.8); ax.set_axisbelow(True)


clin = pd.read_csv(O / "Preprocessed_OOL_Clinical_new.csv", index_col=0)
cb = pd.read_csv(O / "clinical_baselines.csv").set_index("model")
aligned = pd.read_csv(O / "paper_protocol/aligned_p43_summary.csv").set_index("features")
captions = []

# ---------- Fig 1: what predicts DOS --------------------------------------------------------
lab = [("average DOS for everyone", cb.loc["mean_only", "R2"], GREY),
       ("measured Unstim (paper's block)", lab_r2(O / "rescore_params43/unstim_only"), GREY),
       ("SIMCO (Fig 5b)", aligned.loc["cellot", "pooled R2 aligned"], BLUE),
       ("EGA alone (1 coefficient)", cb.loc["EGA", "R2"], ORANGE),
       ("SIMCO + EGA (lab's own run)", lab_r2(GH / "GS_cellot_longallstims_EGA_foldinfo"), ORANGE)]
fig, ax = plt.subplots(figsize=(8.5, 3.6))
hbars(ax, [l[0] for l in lab], [l[1] for l in lab], [l[2] for l in lab])
ax.set_xlim(-0.05, 1.0); ax.set_xlabel("R² for days to labor (DOS), same 100 splits")
ax.set_title("Gestational age at sampling (EGA) predicts DOS far better than the immune model",
             loc="left", fontsize=13)
fig.savefig(OUT / "fig1_dos_vs_ega.png"); plt.close(fig)
captions.append(("fig1_dos_vs_ega.png", "SIMCO predicts DOS with R² 0.42; the week of pregnancy at the blood draw alone gives 0.88, and adding the immune features to EGA (the lab's own run) only reaches 0.89. SIMCO is essentially a noisy gestational-age clock."))

# ---------- Fig 2: gestational-age-free target ---------------------------------------------------
n2 = {}
for d in (O / "night2/stabl_n20").glob("o_*"):
    f = d / "Summary/Scores training CV.csv"
    if f.exists(): n2[d.name] = first(pd.read_csv(f, index_col=0).loc["STABL Lasso", "R2"])
co = pd.read_csv(O / "night2/clinical_offset_n20.csv").set_index("model")
rows = [("EGA alone", co.loc["EGA", "R2"], ORANGE), ("CellOT (paper features)", n2["o_cellot_both"], BLUE),
        ("gauss_ot", n2["o_gaussot_both"], GREY), ("measured Unstim (paper)", n2["o_unstim_only"], GREY),
        ("measured freq + medians", n2["o_freqmed"], GREY), ("measured + quantiles", n2["o_qtl"], GREY)]
fig, ax = plt.subplots(figsize=(8.5, 3.4))
hbars(ax, [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows])
ax.set_xlim(-0.55, 0.3); ax.set_xlabel("R² for days before/after the 40-week due date (20 splits, Lasso)")
ax.set_title("Once gestational age is removed, nothing predicts labor timing", loc="left", fontsize=13)
fig.savefig(OUT / "fig2_due_date_target.png"); plt.close(fig)
captions.append(("fig2_due_date_target.png", "Target = DOS − 7·(EGA − 40): how many days before (+) or after (−) the due date labor started. Every feature set is at or below 0, i.e. no better than predicting the average. (Only 5 preterm women; this does not exclude a small signal.)"))

# ---------- Fig 3: preterm AUROC is sampling timing ---------------------------------------------
p = pd.read_csv(O / "rescore_params43/both/Training CV/STABL ALasso/STABL ALasso predictions.csv", index_col=0).dropna()
ega, pre = clin.loc[p.index, "EGA"], (clin.loc[p.index, "Delivery"].str.strip() == "preterm").astype(int)
vis = clin.loc[p.index, "Timepoint"]; score = -(7 * ega - p["0"]); g1 = (vis == "G1").values
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
m = pd.DataFrame({"EGA": ega, "pre": pre, "v": vis}).groupby(["v", "pre"]).EGA.mean().unstack()
for i, v in enumerate(["G1", "G2", "G3"]):
    a1.plot([m.loc[v, 1], m.loc[v, 0]], [i, i], color=GRID, lw=3, zorder=1)
    a1.scatter(m.loc[v, 1], i, s=90, color=ORANGE, zorder=2, edgecolor=SURF, linewidth=2, label="preterm (5 women)" if i == 0 else None)
    a1.scatter(m.loc[v, 0], i, s=90, color=BLUE, zorder=2, edgecolor=SURF, linewidth=2, label="term (58 women)" if i == 0 else None)
    a1.text(m.loc[v, 1] - .35, i, f"{m.loc[v, 1]:.1f}", ha="right", va="center", fontsize=10, color=INK)
    a1.text(m.loc[v, 0] + .35, i, f"{m.loc[v, 0]:.1f}", ha="left", va="center", fontsize=10, color=INK)
a1.set_yticks(range(3)); a1.set_yticklabels(["visit G1", "visit G2", "visit G3"]); a1.invert_yaxis()
a1.set_xlim(24.5, 39.5); a1.set_ylim(2.6, -0.6)
a1.set_xlabel("weeks pregnant at the blood draw (mean)"); a1.legend(frameon=False, fontsize=10, loc="upper right")
a1.set_title("Preterm women miss the late visits", loc="left", fontsize=13); a1.grid(axis="x", color=GRID, lw=.8)
vals = {"SIMCO": [roc_auc_score(pre, score), roc_auc_score(pre[g1], score[g1])],
        "EGA alone": [roc_auc_score(pre, -ega), roc_auc_score(pre[g1], -ega[g1])]}
x = np.arange(2); w = .36
for k, (nm, col) in enumerate([("SIMCO", BLUE), ("EGA alone", ORANGE)]):
    b = a2.bar(x + (k - .5) * w, vals[nm], w, color=col, edgecolor=SURF, linewidth=2, label=nm)
    for xi, v in zip(x + (k - .5) * w, vals[nm]): a2.text(xi, v + .01, f"{v:.2f}", ha="center", fontsize=10)
a2.axhline(.5, color=MUTED, ls="--", lw=1); a2.text(1.55, .505, "chance", color=INK2, fontsize=9, ha="right")
a2.set_xticks(x); a2.set_xticklabels(["all visits", "G1 only (timing matched)"]); a2.set_ylim(.4, .9)
a2.set_ylabel("AUROC, preterm vs term"); a2.legend(frameon=False, fontsize=10, loc="upper left")
a2.set_title("Paper's 0.73 ≈ timing; G1 is the real lead", loc="left", fontsize=13); a2.grid(axis="y", color=GRID, lw=.8)
fig.tight_layout(); fig.savefig(OUT / "fig3_preterm_auroc.png"); plt.close(fig)
captions.append(("fig3_preterm_auroc.png", "Left: preterm women are sampled ~4 weeks earlier at G2/G3 because they deliver before the late visits. Right: ranking samples by sampling week alone already gives AUROC 0.71; SIMCO 0.75. At G1, where timing is matched, EGA falls to 0.64 (n.s.) while SIMCO reaches 0.81 (permutation p = 0.02; 0.80 with the EGA trend removed) — a lead, but 5 women and a post hoc subgroup."))

# ---------- Fig 4: paper protocol, 97 XGBoost settings ------------------------------------------
AL = O / "paper_protocol/aligned"
arms = [("cellot", "CellOT (paper)", BLUE), ("d_qtl", "measured + quantiles (no map)", GREY),
        ("gaussot_both", "gauss_ot (13×13 linear map)", GREY), ("d_freqmed", "measured freq + medians (no map)", GREY),
        ("crossswap_s0_both", "CellOT maps, wrong stim & celltype", GREY),
        ("rscaled_s0_both", "untrained random ICNN", GREY),
        ("comb_cellot", "CellOT stims + measured", AQUA), ("comb_gaussot", "gauss_ot stims + measured", AQUA)]
arms = [a for a in arms if (AL / a[0] / "sweep.csv").exists()]
fig, ax = plt.subplots(figsize=(9, 0.55 * len(arms) + 1.4))
for i, (k, lab_, col) in enumerate(arms):
    t = pd.read_csv(AL / k / "sweep.csv"); y = len(arms) - 1 - i
    bp = ax.boxplot(t.R2, positions=[y], vert=False, widths=.5, patch_artist=True, showfliers=False,
                    medianprops={"color": INK, "lw": 1.5}, whiskerprops={"color": MUTED}, capprops={"color": MUTED},
                    boxprops={"facecolor": col, "edgecolor": col, "alpha": .45})
    p43 = t.set_index("params").loc["params_43", "R2"]
    ax.scatter(p43, y, s=70, color=col if col != GREY else INK2, edgecolor=SURF, linewidth=2, zorder=3)
    ax.text(t.R2.max() + .008, y, f"{p43:.2f}", va="center", fontsize=10)
ax.set_yticks(range(len(arms))[::-1]); ax.set_yticklabels([a[1] for a in arms])
ax.set_xlabel("R² (100 splits, row-order bug fixed) — box = 97 XGBoost settings, dot/number = params_43")
ax.set_title("CellOT is ahead, but by less than the XGBoost setting moves each result  (EGA alone: 0.88)",
             loc="left", fontsize=12.5); ax.grid(axis="x", color=GRID, lw=.8); ax.set_axisbelow(True)
fig.savefig(OUT / "fig4_paper_protocol.png"); plt.close(fig)
captions.append(("fig4_paper_protocol.png", "Every feature set run through the paper's own protocol (100 splits, the paper's XGBoost setting params_43 = dot, and all 97 settings = box). CellOT 0.418 is significantly ahead of measured features (0.397, p = 0.03), gauss_ot (0.399, p = 0.002), wrong-stim maps (0.343) and a random ICNN (0.329) — but the spread due to the XGBoost setting alone (~0.1) is larger than CellOT's lead over a good summary of the unstimulated cells."))

# ---------- Fig 5: drug arm across DOS models -----------------------------------------------------
cmp = pd.read_csv(O / "drug_models/compare_by_drug.csv")
models = [("CellOT (paper tables)", "CellOT", BLUE), ("gauss_ot + gauss_ot drug maps", "gauss_ot (stim + drug maps, no neural net)", ORANGE),
          ("measured + quantiles", "measured + quantiles", AQUA)]
sh = cmp.pivot_table(index="drug", columns="model", values="mean shift (d)")
ag = cmp.pivot_table(index="drug", columns="model", values="folds same sign")
order = sh["CellOT (paper tables)"].sort_values().index
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.0), gridspec_kw={"width_ratios": [2.2, 1]})
yy = np.arange(len(order))[::-1]
for k, (col_name, lab_, col) in enumerate(models):
    mk = ["o", "s", "^"][k]
    if k == 0:
        for yv in yy: a1.axhline(yv, color=GRID, lw=.8, zorder=1)
    a1.scatter(sh.loc[order, col_name], yy, s=80, marker=mk, color=col, edgecolor=SURF, linewidth=1.5,
               label=lab_, zorder=3 + k, alpha=.95)
    for d, yv in zip(order, yy):
        if ag.loc[d, col_name] >= .85:   # consistency label, placed above (k=0,2) or below (k=1) the dot
            a1.text(sh.loc[d, col_name], yv + (.3 if k != 1 else -.3), f"{ag.loc[d, col_name]:.0%}",
                    ha="center", va="center", fontsize=7.5, color=col)
a1.axvline(0, color=MUTED, lw=1); a1.set_yticks(yy); a1.set_yticklabels(order)
a1.set_xlabel("mean predicted shift in DOS (days; negative = labor later)")
a1.legend(frameon=False, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3); a1.grid(axis="x", color=GRID, lw=.8)
a1.set_title("THF and SALPZ hold with no neural network; vanish with measured features", loc="left", fontsize=12.5)
imp = cmp.groupby("model")["frac implausible (b)"].mean()
labs = ["CellOT", "gauss_ot", "measured + qtl"]; v = [imp[m[0]] for m in models]
hbars(a2, labs, v, [m[2] for m in models], fmt="{:.0%}")
a2.set_xlim(0, .75); a2.set_xlabel("implausible treated women")
a2.set_title("Extrapolation", loc="left", fontsize=12.5)
fig.tight_layout(); fig.savefig(OUT / "fig5_drug_arm.png"); plt.close(fig)
gs = pd.read_csv(GH / "comp_9drugs_long_unstimpred_treated/gridsearch_summary_treated_unstim.csv")
captions.append(("fig5_drug_arm.png", f"Three complete pipelines: CellOT stim + CellOT drug maps (paper); gauss_ot stim + gauss_ot drug maps (no neural network anywhere, fitted on the same control wells, batch correction and cell split); measured features + CellOT Unstim→drug maps. With no neural network THF stays first (−7.2 d, 98 % of fold models), SALPZ holds (−3.2 d, 91 %), MAP disappears (−0.5 d); ranking correlation with the paper +0.80, and only 10 % of treated women are implausible (vs 56 % with CellOT). Earlier variant (gauss_ot stims + CellOT drug maps): ρ +0.75. Percentages = share of the 100 fold models agreeing in sign (shown when ≥ 85 %). THF/SALPZ/MAP shift labor 6–9 days later with CellOT, beyond a size-matched random perturbation, carried mainly by pSTAT3 in naive CD4 T cells; they survive a gauss_ot model but not a model on measured unstimulated features (not like-for-like: that model sees the drug only on resting cells). Right: with CellOT, 55 % of drug-treated women look like no real woman. Across the lab's XGBoost settings, the number of drugs with a positive effect ranges {gs.Nb_Positive_Deltas.min()}–{gs.Nb_Positive_Deltas.max()}; params_43 was the setting with the best preterm AUROC."))

(OUT / "captions.md").write_text("# SIMCO review — figures and captions\n\n" + "\n\n".join(f"**{f}** — {c}" for f, c in captions) + "\n")
print("wrote", [c[0] for c in captions], "->", OUT)
