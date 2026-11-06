"""Build notebooks/ceiling.ipynb (then: jupyter nbconvert --execute --inplace notebooks/ceiling.ipynb)."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# The replicate-well ceiling: is the 0.60 plateau a model limit or a data limit?

**Question.** On held-out donors, every Unstim → stim map (CellOT, gauss_ot, quantile_map, median_shift) plateaus at a
skill of about **0.60** at k = 15 training donors (`atlas_heldout`, learning curve). The ceiling quoted there, **0.87**,
splits *one* measurement into two random halves, so it only contains cell-sampling noise. Every (donor, stim) was
measured in **three plate-control wells (DMSO1/2/3)**. How well does one well predict another well of the same donor
and stim? That is the replicate-well ceiling.

**Skill, as in atlas_heldout (`lc/score.py`, reimplemented in `wc/ceiling.py`).** A case is (test donor, stim,
celltype, quantile q10/25/50/75/90, one of 13 functional markers). Truth = that quantile of the donor's stimulated
cells, identity = the same quantile of her Unstim cells (both: the three wells pooled, ≤ 10,000 cells), effect =
|truth − identity|, skill = 1 − |prediction − truth| / effect. Only cases with effect > 0.0273 (the atlas technical
floor) count. Headline = median over the cases of one repeat (4 test donors × 168 pairs), averaged over the 5 repeats.
1 = perfect, 0 = no better than "the stim does nothing".

**Data.** `wc/extract.py` re-reads the atlas h5ad files and takes *exactly* the learning-curve cells (same seed, same
order; checked: all quantiles equal to 2e-7), now keeping the well label. Under PI, DMSO1 is a plate outlier (simco
CLAUDE.md): it is left out of the well-vs-well numbers and shown separately.
""")

code(r"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import Image, display
S = Path("/scratch/users/bmerwan/well_ceiling/work/summary"); F = Path("figures"); F.mkdir(exist_ok=True)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
BLUE, ORANGE, AQUA, YELLOW, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10, "axes.grid": True, "grid.alpha": .25,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
lc = pd.read_csv("/scratch/users/bmerwan/atlas_heldout/work/lc/summary/curve.csv")
print("atlas_heldout k = 15 plateau and ceiling (curve.csv):")
lc[lc.k == 15][["method", "skill", "ceiling"]].round(3)
""")

md(r"""
## 1. Validation: my code reproduces the split-half ceiling

`wc/ceiling.py`, variant `split_half`, must give atlas_heldout's 0.872. The other rows are the new ceilings
(explained below).
""")
code(r"""
o = pd.read_csv(S / "ceiling_overall.csv").set_index("variant")
o = o.loc[["split_half", "well", "well_big", "pseudo_well", "pseudo_well_big", "well_src", "delta_transfer"]]
o.index = ["split half of the pooled cells (lc ceiling)", "one well → another well (lc cells)",
           "one well → another well (≤ 30k cells/well)", "pseudo-wells: pooled cells re-split at well sizes",
           "pseudo-wells, ≤ 30k cells/well", "well → well, identity = target well's own Unstim",
           "donor response from well A applied to well B's Unstim"]
o[["skill", "skill_sd", "skill_all_donors"]].round(3)
""")
md(r"""
**Reading.**
- **The split-half ceiling reproduces: 0.872** (atlas_heldout 0.872).
- **One well predicting another well of the same donor scores 0.56** — *below* the 0.60 model plateau.
- It is not cell sampling: random "pseudo-wells" of the same sizes score 0.83, and giving each well up to 30,000 cells
  changes nothing (0.561). The loss from 0.83 to 0.56 is **well/plate technical variation**.
- Letting the source change well too (identity = Unstim of the target's own well) does not help (0.56). Carrying the
  donor's response measured on plate A over to plate B's Unstim gives 0.60: a small part of the plate effect is shared
  between the Unstim and stim wells of a plate, so a map fed with the matching Unstim would inherit it.
""")

md(r"""
## 2. The literal number cannot be compared to 0.60: the targets differ

The plateau scores a model against the **pooled** measurement (three wells, ~10k cells). A replicate well is scored
against **one other well**, and the replicate brings its own well noise too, so it carries two wells' worth of noise
against the model's (at most) one third. To compare like with like, `wc/compare.py` scores everything on the same
targets, using atlas_heldout's k = 15 predictions (`work/lc/where_skill/cases.pkl`, read-only):
- *target = one well*: the models, another well, the mean of the other two wells;
- *target = pooled*: the models and the split-half replicate;
- the **ideal model** (knows the donor's true response, has no noise of its own), estimated under "each well = truth +
  independent noise of equal variance": against one well its error is |well A − well B| / √2, against the pool of
  three wells |well A − well B| / √6. This is an estimate, checked in §3.
""")
code(r"""
R = pd.read_csv(S / "same_target_comparison.csv")
display(R.round(3))
order = {"pooled": ["model: quantile_map", "model: gauss_ot", "model: CellOT", "ideal model, sampling + well noise",
                    "replicate: split half (lc ceiling)", "ideal model, sampling noise only"],
         "one well": ["model: quantile_map", "model: gauss_ot", "model: CellOT", "replicate: another well",
                      "mean of the other two wells", "ideal model, sampling + well noise"]}
col = lambda p: BLUE if p.startswith("model") else (AQUA if p.startswith("ideal") else ORANGE)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
for ax, (tgt, preds) in zip(axs, order.items()):
    r = R[R.target == tgt].set_index("predictor").loc[preds]
    y = np.arange(len(r))
    ax.errorbar(r.skill, y, xerr=r.skill_sd, fmt="none", ecolor=MUTED, lw=1, capsize=2)
    ax.scatter(r.skill, y, s=60, c=[col(p) for p in r.index], zorder=3, edgecolor="#fcfcfb", lw=2)
    for yi, v in zip(y, r.skill): ax.text(v, yi - 0.3, f"{v:.2f}", ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels([p.replace("model: ", "") for p in r.index]); ax.set_ylim(len(r) - 0.5, -0.8)
    ax.axvline(0.60, color=BLUE, ls=":", lw=1); ax.axvline(0.872, color=ORANGE, ls=":", lw=1)
    ax.set_title(f"target = {'the pooled 3 wells (as in the learning curve)' if tgt == 'pooled' else 'one single well'}",
                 fontsize=10, loc="left")
    ax.set_xlim(0.3, 1.0); ax.set_xlabel("skill (k = 15, mean over 5 repeats ± sd)")
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], marker="o", ls="", color=c) for c in (BLUE, ORANGE, AQUA)],
           ["map trained on 15 other donors", "a replicate measurement", "ideal model (estimate)"],
           loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Same targets, same skill. Dotted lines: plateau 0.60 (blue), split-half ceiling 0.87 (orange)", fontsize=10, color=MUTED, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(F / "same_target.png", dpi=150, bbox_inches="tight"); plt.close(fig)
display(Image(filename=str(F / "same_target.png")))
""")
md(r"""
**Reading.**
- **On the same single-well target, the models are clearly worse than a replicate well:** CellOT 0.44, gauss_ot 0.44,
  quantile_map 0.34, against 0.56 for another well of the same donor, 0.61 for the mean of her two other wells and
  ~0.69 for the ideal model. A map trained on 15 other donors is not yet as good as re-measuring the donor.
- **On the pooled target (what the learning curve scores), the realistic ceiling is ~0.73, not 0.87.** The pooled
  measurement still carries its three wells' plate noise, which no model can predict. That is about half of the
  0.60 → 0.87 gap; the other half (0.60 → ~0.73) is room the models leave.
""")

md(r"""
## 3. Check of the ideal-model estimate: averaging more wells

Under the independent-noise assumption, predicting well B with the mean of m other wells has an error proportional
to √(1 + 1/m): 1.41 for one well, 1.22 for two, 1 for the ideal (m → ∞). Fit the scale on m = 1 and check that it
predicts the measured m = 2.
""")
code(r"""
r = R[R.target == "one well"].set_index("predictor").skill
s1, s2, sinf = r["replicate: another well"], r["mean of the other two wells"], r["ideal model, sampling + well noise"]
k = (1 - s1) / np.sqrt(2)
print(f"m = 1 measured {s1:.3f};  m = 2 measured {s2:.3f}, predicted {1 - k*np.sqrt(1.5):.3f};  m = inf (ideal) {1 - k:.3f}")
m = np.linspace(1, 12, 100)
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(1/m, 1 - k*np.sqrt(1 + 1/m), color=AQUA, lw=2, label="independent noise, scale fitted at m = 1")
ax.scatter([1, 0.5], [s1, s2], s=60, color=ORANGE, zorder=3, label="measured (mean of m other wells)")
ax.scatter([0], [1 - k], s=60, color=AQUA, zorder=3)
ax.axhline(r["model: CellOT"], color=BLUE, ls=":", lw=1); ax.text(0.98, r["model: CellOT"] + .01, "CellOT", color=MUTED, ha="right", fontsize=8)
ax.set_xlabel("1 / m  (m = number of other wells averaged; 0 = ideal model)"); ax.set_ylabel("skill vs one well")
ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(F / "scaling_check.png", dpi=150); plt.close(fig)
display(Image(filename=str(F / "scaling_check.png")))
""")
md(r"""
**Reading.** The independent-noise model fitted on one well predicts the two-well mean within ~0.01, so the ideal
ceilings in §2 (0.69 against one well, ~0.73 against the pool) are credible to about that precision. If part of the
plate effect is shared between a plate's Unstim and stim wells (§1, `delta_transfer`), a map that reads the donor's
Unstim could predict a little of it, so the true pooled ceiling would be slightly higher, not lower.
""")

md("## 4. Per stimulation")
code(r"""
T = pd.read_csv(S / "same_target_by_stim.csv", header=[0, 1], index_col=0)
st = pd.DataFrame({"CellOT (pooled target)": T[("pooled", "model: CellOT")],
                   "ideal model (pooled target)": T[("pooled", "ideal model, sampling + well noise")],
                   "split half (lc ceiling)": T[("pooled", "replicate: split half (lc ceiling)")],
                   "replicate well (one-well target)": T[("one well", "replicate: another well")],
                   "CellOT (one-well target)": T[("one well", "model: CellOT")]})
w = pd.read_csv(S / "ceiling_by_stim.csv"); w = w[w.variant == "well"].set_index("stim").median_skill
st.loc["PI (with DMSO1)", "replicate well (one-well target)"] = w.get("PI (with DMSO1)")
display(st.round(3))
def dots(tab, cols, colors, fname, h):
    tab = tab[cols].dropna(how="all"); y = np.arange(len(tab))
    fig, ax = plt.subplots(figsize=(8, h))
    for c, colr, mk in zip(cols, colors, ["o", "D", "s", "^"]):
        ax.scatter(tab[c], y, s=50, color=colr, marker=mk, label=c, zorder=3, edgecolor="#fcfcfb", lw=1.5)
    ax.set_yticks(y); ax.set_yticklabels(tab.index); ax.set_xlabel("skill"); ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.tight_layout(); fig.savefig(F / fname, dpi=150); plt.close(fig); display(Image(filename=str(F / fname)))
cols = ["CellOT (pooled target)", "ideal model (pooled target)", "split half (lc ceiling)", "replicate well (one-well target)"]
dots(st.drop(index="PI (with DMSO1)", errors="ignore").sort_values("ideal model (pooled target)"), cols,
     [BLUE, AQUA, YELLOW, ORANGE], "by_stim.png", 3.8)
""")
md(r"""
**Reading.**
- **IFNa, LPS, IL246, GMCSF**: the replicate well (0.55–0.69) is 0.07–0.11 above CellOT on the same one-well target,
  and the pooled ideal (0.76–0.83) is 0.13–0.20 above CellOT's pooled skill. Clear room in the four stims with large
  effects.
- **PI**: CellOT reaches 0.71 on the pooled target, at the (DMSO1-limited) ideal of 0.70; scored against the clean
  DMSO2/DMSO3 wells it drops to 0.40 vs 0.65 for a replicate — the maps learned the DMSO1 plate artefact that is in
  their training target. With DMSO1, one well predicting another falls to ~0.2–0.5 (see the well-pair table).
- **TNFa** (0.42 vs ideal 0.62) and **IL33** (0.11 vs 0.36; a replicate is *negative*): small effects, mostly plate
  noise; nothing much to gain there.
""")

md("## 5. Per celltype")
code(r"""
T = pd.read_csv(S / "same_target_by_celltype.csv", header=[0, 1], index_col=0)
ct = pd.DataFrame({"CellOT (pooled target)": T[("pooled", "model: CellOT")],
                   "ideal model (pooled target)": T[("pooled", "ideal model, sampling + well noise")],
                   "split half (lc ceiling)": T[("pooled", "replicate: split half (lc ceiling)")],
                   "replicate well (one-well target)": T[("one well", "replicate: another well")]})
ct["room (ideal − CellOT)"] = ct["ideal model (pooled target)"] - ct["CellOT (pooled target)"]
display(ct.sort_values("ideal model (pooled target)").round(3))
dots(ct.sort_values("ideal model (pooled target)"), cols, [BLUE, AQUA, YELLOW, ORANGE], "by_celltype.png", 8)
print("room, median over celltypes:", round(ct["room (ideal − CellOT)"].median(), 3))
""")
md(r"""
**Reading.** The room left (ideal − CellOT, pooled target) is 0.07–0.20 in every celltype (median 0.14): smallest in
CD4 naive / CD4 central-memory / CD8 naive T cells (0.07–0.08, CellOT already at 0.67–0.74), largest in Granulocytes,
mDCs, CD8Tcm CCR2+ and pDCs (0.18–0.20). In the smallest populations (pDCs, CD56hi NK) the split-half and ideal
ceilings nearly coincide, because cell sampling dominates; in the large ones the split half sits 0.1–0.2 above the
ideal, i.e. the 0.87 ceiling overstated the room most in large celltypes, where plate noise dominates.
""")

md("## 6. Which wells disagree (well-vs-well skill, ordered pairs)")
code(r"""
w = pd.read_csv(S / "ceiling_by_wellpair.csv"); w = w[w.variant == "well"]
w.pivot_table(index="stim", columns=["src", "tgt"], values="median_skill").round(2)
""")
md(r"""
**Reading.** DMSO2 ↔ DMSO3 is the best-agreeing pair for most stims; DMSO1 → DMSO3 is the worst. Under PI, any pair
involving DMSO1 collapses (down to −0.46), the known plate-1 artefact. IL33 is negative everywhere: the stim effect is
smaller than the well-to-well difference.
""")

md(r"""
## 7. Answer and decision

| | skill |
|---|---|
| split-half ceiling (atlas_heldout; reproduced here) | 0.872 (0.872) |
| **replicate-well ceiling** (one well → another, same donor & stim, PI without DMSO1) | **0.56** |
| models at k = 15, pooled target: CellOT / gauss_ot / quantile_map | 0.60 / 0.59 / 0.54 |
| same models scored against one single well | 0.44 / 0.44 / 0.34 |
| ideal model (estimate), one-well target / pooled target | 0.69 / **0.73** |

- **The literal replicate-well ceiling is 0.56, at or below the plateau** — but that compares a replicate scored
  against one noisy well with models scored against the pooled three wells. On a common target, the models are
  **clearly below a replicate** (0.44 vs 0.56 against one well), and the realistic ceiling for the learning curve's own
  target is **~0.73**, not 0.87.
- So about half of the apparent 0.60 → 0.87 gap is plate/well technical noise that no model can recover (0.73 → 0.87),
  and about half (0.60 → 0.73) is room the models leave. **The plateau is not (yet) the data limit.**
- **Decision: the "clearly above 0.6" branch applies.** Do not stop modelling; the next test is **cross-stim transfer**
  (the donor's measured response to one stim as input to predict another), with the realistic target ~0.73 on the
  pooled score (≈ 0.69 against single wells), and with the expectation that IL33/TNFa have almost no room and that PI
  needs DMSO1 handled before training.

Limits: the ideal ceilings are estimates under independent, equal-variance well noise (checked within ~0.01 in §3);
the pooled ideal for PI uses the DMSO1 outlier as ordinary noise; and the cases are the learning curve's own (live
filter on the pooled effect), so per-well denominators can be small.
""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "cellot", "display_name": "CellOT (py3.9)", "language": "python"}
nbf.write(nb, Path(__file__).with_name("ceiling.ipynb"))
print("wrote ceiling.ipynb")
