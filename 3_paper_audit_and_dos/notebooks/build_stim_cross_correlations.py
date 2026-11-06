"""Generate notebooks/stim_cross_correlations.ipynb: does the response of a marker in one celltype co-vary with the
response of another marker in another celltype (same donor, same well)? The granulocyte (paracrine) question.

Every table is recomputed from the CSVs of baselines/stim_cross_correlations.py (same convention as the others).
    python baselines/stim_corr_wells.py              # per-well medians from atlas_heldout/work/heldout/pheno
    python baselines/stim_cross_correlations.py      # ~10 min, 3 threads -> simco_work/stim_correlations/
    python notebooks/build_stim_cross_correlations.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/stim_cross_correlations.ipynb
"""
import nbformat as nbf

cells = []
md = lambda t: cells.append(nbf.v4.new_markdown_cell(t.strip()))
code = lambda t: cells.append(nbf.v4.new_code_cell(t.strip()))

md(r"""
# Cross-celltype coupling of stimulation responses

**Question.** All the cells of a donor sit in the same well during the 2 h stimulation. Is the rise of a marker in one
celltype associated with the rise of another marker in another celltype? In particular, do granulocytes (~60 % of the
cells in a well) act on the others, e.g. by secreting cytokines?

**What can be measured here.** The atlas has one time point and no depletion, so causation cannot be shown. What can be
measured is **co-variation**:
- **across donors**: if donor A's granulocytes respond more than donor B's, do A's T cells also respond more?
- **across wells of the same donor**: each donor has three plate-control wells per stim (DMSO1/2/3). If one well's
  granulocytes respond more than the donor's other wells, do the other cells *of that same well* also respond more?
  This is the closest the data gets to "same well", but it is also where technical well effects are strongest.

**Summary (plain terms).** *Filled in from the results below; see §10 for the full caveats.*

- **Is a marker's response in one celltype associated with another marker's response in another celltype?** Yes, a lot
  across donors for the strong stims (IFNa, LPS, PI): ~20 % of different-marker and ~50 % of same-marker cross-celltype
  pairs exceed chance (5 %). But **most of it is donor-level**: one "global responsiveness" factor (which for PI, IFNa and
  IL246 is largely *pregnant vs non-pregnant*) halves it, down to what phenotypic markers that do not respond also show.
- **What remains is mostly "same pathway, related celltypes"**: T subsets with T subsets, NK with T, the same STAT in
  several celltypes under IFNa, the NF-κB/MK2 axis under TNFa/LPS. IFNa and LPS keep a clear excess over the technical
  reference (218 and 117 FDR-surviving different-marker pairs vs 3–15 for the reference); PI a small one; TNFa, GMCSF,
  IL33, IL246 none.
- **Granulocytes are not a hub.** Ranked against the 27 other celltypes as a source of coupling, granulocytes are last or
  near-last in raw correlations and mid-pack after the global factor, never in the top third, although they are the best
  measured cells. Their clearest couplings are with other myeloid cells (shared lineage and gating), plus some T/B links
  under LPS and PI that are no stronger than those of other celltypes.
- **Granulocyte share** of the well is associated with weaker responses elsewhere, but every pregnant donor has more
  granulocytes than every non-pregnant donor (67–79 % vs 34–56 %), and the association vanishes when pregnancy is
  controlled. It cannot be attributed to granulocytes.
- **Within a donor, well-to-well** (the literal "same well" test): wells co-vary across all celltypes for *all* markers,
  phenotypic included and Unstim wells included (a technical well effect, ~25 % of pairs above chance). Stimulated
  functional markers add nothing beyond it except a little for LPS and clearly for PI, whose stimulation strength varies
  between wells. Granulocyte-involving pairs are not more coupled than the rest.
- **Caveats**: 19 donors (|r| must exceed ~0.8 to survive FDR individually), one 2 h time point, no depletion, the
  pregnancy confound, technical well coupling, and gating that moves under PI/LPS. These are candidate couplings, not
  mechanisms. Testing the granulocyte hypothesis needs an intervention (granulocyte depletion / add-back, conditioned
  medium, cytokine blockade or secretion inhibitors, and a time course) — see §10.
""")

md(r"""
## 0. Data and design

- **Input**: `atlas_heldout/work/heldout/pheno/<celltype>.csv.gz` — per well, the median of every marker over ≤ 20 000
  cells, raw arcsinh (no batch correction), plus the exact cell count. Only the plate controls are used
  (no drug): a **well** = (stim, plate DMSO1/2/3, donor), 19 donors (8 HV, 11 PV).
- **Feature** = median of one marker in one celltype. **Functional set** = 13 markers × 28 celltypes = 364.
- **Response Δ** = stim well − Unstim well of the same donor and plate. Per donor: the mean over plates.
  **PI uses DMSO2/3 only** — its DMSO1 wells respond much more strongly than the other two plates (CLAUDE.md).
- **Technical reference (tref)**: phenotypic markers (CD3, CD45, CD19, CD66, …) that are expressed in that celltype
  (Unstim median > 1) and that the stim does not move (|mean Δ| < 0.1, the paper's own threshold). Their Δ should be
  zero biologically. Whatever cross-celltype correlation they show is therefore **not signalling**: it is donor- or
  well-level coupling (staining, acquisition, sample handling). It is the yardstick for the functional markers.
- **Correlation** = Spearman across the 19 donors. Four variants:
  `raw`; `pc1` = after removing the first principal component of the functional Δ matrix (a "global donor
  responsiveness" factor); `tech` = after removing the first PC of the tref Δ matrix; `pc1tech` = both.
- **Null** = donors shuffled independently per celltype (200 permutations). This keeps each celltype's own marker
  correlations and breaks only the link between celltypes. p-values from that pooled null, Benjamini–Hochberg over the
  cross-celltype pairs of each (stim, variant).
- **Scale of the multiple-testing problem**: ~60 000 cross-celltype pairs per stim, 19 donors. Under the null, the
  95th percentile of |r| is already **0.46**; one needs |r| ≳ 0.8 to survive FDR when the signal is sparse.
""")

code(r"""
import warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl
from scipy.cluster.hierarchy import linkage, leaves_list
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 80)
mpl.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
W = Path("/scratch/users/bmerwan/simco_work/stim_correlations")
STIMS = ["IFNa", "IL246", "IL33", "LPS", "PI", "TNFa", "GMCSF"]
CT = ["Granulocytes", "MDSCs", "cMCs", "intMCs", "ncMCs", "mDCs", "pDCs", "Bcells",
      "CD56hiCD16negNK", "CD56loCD16posNK", "NK_cells_CD11cneg", "NK_cells_CD11cpos", "NKT", "CD4negCD8negTcells",
      "CD4Tnaive", "CD4Tcm", "CD4Tcm_CCR2pos", "CD4Tem", "CD4Tem_CCR2pos", "CD4Teff", "CD4Tregs",
      "CD8Tnaive", "CD8Tcm", "CD8Tcm_CCR2pos", "CD8Tem", "CD8Tem_CCR2pos", "CD8Teff", "CD8Tcells_Th1"]
SHORT = {"CD56hiCD16negNK": "NK CD56hi", "CD56loCD16posNK": "NK CD56lo", "NK_cells_CD11cneg": "NK CD11c-",
         "NK_cells_CD11cpos": "NK CD11c+", "CD4negCD8negTcells": "DN T", "CD8Tcells_Th1": "CD8T Th1",
         "CD4Tcm_CCR2pos": "CD4Tcm CCR2+", "CD4Tem_CCR2pos": "CD4Tem CCR2+", "CD8Tcm_CCR2pos": "CD8Tcm CCR2+",
         "CD8Tem_CCR2pos": "CD8Tem CCR2+", "Granulocytes": "Granulocytes"}
sh = lambda c: SHORT.get(c, c)
LINEAGE = {**{c: "myeloid" for c in CT[:7]}, "Bcells": "B", **{c: "NK" for c in CT[8:13]}, **{c: "T" for c in CT[13:]}}
meta = pd.read_csv(W / "features_meta.csv"); dist = pd.read_csv(W / "dist_summary.csv")
blocks = pd.read_csv(W / "blocks.csv"); hubs = pd.read_csv(W / "hubs.csv"); summ = pd.read_csv(W / "stim_summary.csv")
pairs = pd.read_csv(W / "pairs_cross.csv.gz"); gfreq = pd.read_csv(W / "granulocyte_freq.csv")
bpairs = pd.read_csv(W / "baseline_to_response_pairs.csv.gz"); bblocks = pd.read_csv(W / "baseline_to_response_blocks.csv")
well = pd.read_csv(W / "well_level.csv"); counts = pd.read_csv(W / "wells_counts.csv.gz")
blocks["excess"] = blocks.mean_abs_r - blocks.null_mean
hubs["excess"] = hubs.hub - hubs.null_mean
print(len(pairs), "pair rows;", meta.groupby("kind").size().to_dict())
""")

md(r"""
## 1. Which responses can carry a donor-level signal at all?

A correlation across donors needs the feature to *differ reproducibly between donors*. Each donor has 2–3 plates, so the
reproducibility is measurable: the Spearman correlation across donors between the Δ measured on two plates, and its
Spearman–Brown extension to the plate mean (`rel_plate_mean`, the reliability of the feature actually correlated).
A feature with reliability 0.3 cannot correlate above √0.3 ≈ 0.55 with anything.
""")

code(r"""
f = meta.copy(); f["responding"] = f.mean_delta.abs() > 0.1
t = (f.groupby(["stim", "kind"]).agg(n=("marker", "size"), responding=("responding", "sum"),
                                     rel_median=("rel_plate_mean", "median"))
       .unstack("kind").reindex(STIMS))
t.columns = [f"{a}_{b}" for a, b in t.columns]
t = t[["n_func", "responding_func", "rel_median_func", "n_tref", "rel_median_tref"]].round(2)
fr = f[(f.kind == "func") & f.responding].groupby("stim").rel_plate_mean.median().reindex(STIMS).round(2)
t["rel_median_func_responding"] = fr
display(t)

fig, ax = plt.subplots(figsize=(9, 3.9))
pos = np.arange(len(STIMS))
for k, (kind, col, lab) in enumerate([("func", "#2a6fbb", "functional (13 markers)"), ("tref", "#9a9a9a", "technical reference (phenotypic, unmoved)")]):
    data = [f[(f.stim == s) & (f.kind == kind)].rel_plate_mean.dropna().values for s in STIMS]
    bp = ax.boxplot(data, positions=pos + (k - 0.5) * 0.35, widths=0.3, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black"))
    for b in bp["boxes"]: b.set(facecolor=col, alpha=0.75, edgecolor="none")
    ax.plot([], [], "s", color=col, label=lab)
ax.axhline(0, color="#cccccc", lw=0.8); ax.set_xticks(pos); ax.set_xticklabels(STIMS)
ax.set_ylabel("reliability of donor Δ\n(plate mean, Spearman–Brown)"); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8)
ax.set_title("How reproducible is each donor's response across plates?", loc="left")
plt.tight_layout(); plt.savefig(W / "fig1_reliability.png", dpi=130); plt.show()
""")

md(r"""
**Reading.** The functional responses to **PI, LPS and IFNa are donor traits** (median reliability 0.73–0.80, and
0.88–0.94 for the features the stim actually moves): the same donor ranks the same on different plates. **IL246** is
intermediate (0.54); **TNFa, GMCSF and IL33** are mostly noise at the donor level (0.27–0.44), so any correlation
involving them is attenuated toward zero. **IL33 moves no functional feature by more than 0.1 on average** — there is
nothing to correlate there beyond noise and technical variation. The technical-reference markers have low reliability
(0.2–0.36), as expected for features with no biological response — except under PI (0.79): PI moves most phenotypic
markers, only 156 survive the filter, and those still carry reproducible donor differences, so PI's reference is not
"pure noise".
""")

md(r"""
## 2. How much cross-celltype coupling is there, compared with chance and with the technical reference?

For each stim: the fraction of pairs whose |r| exceeds the 95th percentile of the permutation null (5 % expected by
chance). Pairs are split into **within-celltype** (two markers of the same celltype — kept intact by the null, so always
5 %: a check), **cross-celltype, same marker** (e.g. pSTAT1 in NK vs pSTAT1 in B cells) and **cross-celltype, different
marker** (the paracrine-style question). The grey bars are the same statistic for the technical reference.
""")

code(r"""
d = dist.copy()
show = [("func", "raw", "functional, raw"), ("func", "pc1", "functional, global factor removed"),
        ("func", "pc1tech", "functional, global + technical factor removed"), ("tref", "raw", "technical reference, raw")]
cols = ["#9ec3e6", "#2a6fbb", "#123d6b", "#9a9a9a"]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, catg, title in zip(axes, ["cross_same_marker", "cross_diff_marker"],
                           ["cross-celltype, same marker", "cross-celltype, different marker"]):
    for k, ((st, var, lab), c) in enumerate(zip(show, cols)):
        v = d[(d.set == st) & (d.variant == var) & (d.category == catg)].set_index("stim").frac_above_null_q95.reindex(STIMS)
        ax.bar(np.arange(len(STIMS)) + (k - 1.5) * 0.2, v.values, width=0.18, color=c, label=lab)
    ax.axhline(0.05, color="black", lw=0.8, ls="--", label="chance (5 %)")
    ax.set_xticks(range(len(STIMS))); ax.set_xticklabels(STIMS); ax.set_title(title, loc="left")
axes[0].set_ylabel("fraction of pairs with |r| above\nnull 95th percentile")
axes[1].legend(frameon=False, fontsize=8, loc="upper right")
plt.tight_layout(); plt.savefig(W / "fig2_excess_coupling.png", dpi=130); plt.show()
tab = d[d.category != "within"].pivot_table(index="stim", columns=["set", "variant", "category"], values="frac_above_null_q95").reindex(STIMS)
display(tab.round(3))
display(d[(d.category == "within")].groupby(["set", "variant"]).frac_above_null_q95.agg(["min", "max"]).round(3))
""")

md(r"""
**Reading.**
- **There is far more cross-celltype coupling than chance** for the strong stims. On IFNa, LPS and PI about half of the
  same-marker pairs and ~20 % of different-marker pairs exceed the null 95th percentile (5 % expected).
- **Most of it is a donor-level, same-marker effect.** Same-marker coupling (a donor whose pSTAT1 rises in one celltype
  sees it rise in the others) is 2–3× the different-marker coupling — what one expects from donor-level traits, not
  specifically from one celltype acting on another.
- **Removing one global donor factor (PC1) roughly halves it**: different-marker excess IFNa 18 → 10 %, LPS 21 → 8 %,
  PI 23 → 8 %, i.e. **down to the level of the technical reference** (grey, 6.5–10 %: phenotypic markers with no reason
  to respond). Same-marker coupling stays above the reference for IFNa (35 % vs 20 %) and LPS (31 % vs 20 %), but not for
  PI (25 % vs 35 %, PI's reference is itself strongly coupled).
- For **TNFa, GMCSF, IL33** the functional coupling is no larger than the technical reference even before any control:
  nothing distinguishes their response coupling from technical co-variation.
- Removing the technical factor on top of PC1 (`pc1tech`) changes little. Without PC1, the technical factor alone
  (`tech`) matters only for PI (different-marker 23 → 13.5 %).
- The within-celltype rows are at exactly 5 %: the null is calibrated by construction.
""")

md(r"""
## 3. The global donor factor

PC1 of the 364 functional Δ across the 19 donors, per stim: how much variance it carries, and how it relates to the
technical factor and to the granulocytes' own response (`granulocyte score` = mean of the granulocytes' standardised Δ).
""")

code(r"""
s = summ.pivot(index="stim", columns="quantity", values="value").reindex(STIMS)
s = s[["n_func_features", "pc1_var_explained_func", "corr_pc1func_pc1tref", "corr_pc1func_granulocyte_score",
       "n_tref_features", "pc1_var_explained_tref"]]
s.columns = ["n functional", "PC1 var. explained", "|r| PC1 vs technical PC1", "|r| PC1 vs granulocyte score",
             "n technical ref.", "technical PC1 var."]
display(s.round(2))
""")

md(r"""
**Reading.** One donor factor explains 16–33 % of all response variance (one of 19 components would carry ~5 % by
chance); largest for PI, LPS, IFNa, IL246. Under **PI, GMCSF, IL33 and IL246** it is substantially aligned with the
technical factor (|r| 0.56–0.76), so part of the "global responsiveness" there is the same donor-level shift that unmoved
phenotypic markers show. Under **IFNa and LPS** it is not (|r| 0.09). The granulocytes' own response is not what
drives it (|r| 0.02–0.17 for PI/LPS/IL246/GMCSF, 0.5 for IFNa/IL33; |r| ≈ 0.45 is the 95 % chance range with 19 donors).

**What the global factor mostly is: pregnancy.** The table below (from `baselines/stim_corr_granfreq.py`) correlates
PC1 with the donor group (8 non-pregnant HV vs 11 pregnant PV) and with the Unstim granulocyte share. For PI, IFNa and
IL246, PC1 is largely the HV/PV split (|r| 0.62–0.76). So "removing the global factor" here mostly means removing the
pregnancy difference in responsiveness — real biology, but a donor-level one, not cell-to-cell coupling.
""")

code(r"""
pg = pd.read_csv(W / "pc1_vs_group.csv").set_index("stim").reindex(STIMS)
display(pg.abs().round(2).rename(columns={"r_pc1_vs_pregnant": "|r| PC1 vs pregnant (PV)", "r_pc1_vs_gran_share": "|r| PC1 vs granulocyte share"}))
display(pd.read_csv(W / "granfreq_group.csv").round(3))
""")

md(r"""
## 4. Which celltypes are coupled? Celltype × celltype maps

Each cell of the map = mean |r| over the 13 × 13 marker pairs of two celltypes, **minus its permutation-null mean**
(the null mean is ~0.19, the value 13 × 13 independent random correlations give with 19 donors). Global donor factor
removed (`pc1`). Fixed order (myeloid → B → NK → T) so the stims can be compared; diagonal = within-celltype, masked.
""")

code(r"""
def block_matrix(stim, var="pc1", st="func", col="excess"):
    b = blocks[(blocks.stim == stim) & (blocks.variant == var) & (blocks.set == st)]
    M = b.pivot(index="ct_a", columns="ct_b", values=col).reindex(index=CT, columns=CT)
    M = (M + M.T) / 2
    np.fill_diagonal(M.values, np.nan)
    return M

vmax = 0.12
fig, axes = plt.subplots(2, 4, figsize=(17, 9.4))
for ax, s in zip(axes.flat, STIMS):
    M = block_matrix(s)
    im = ax.imshow(M.values, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_title(s, loc="left", fontsize=11)
    ax.set_xticks(range(len(CT))); ax.set_yticks(range(len(CT)))
    ax.set_xticklabels([sh(c) for c in CT], rotation=90, fontsize=6); ax.set_yticklabels([sh(c) for c in CT], fontsize=6)
    for x in (6.5, 7.5, 12.5): ax.axhline(x, color="white", lw=1.2); ax.axvline(x, color="white", lw=1.2)
    ax.spines[:].set_visible(False)
axes.flat[-1].axis("off")
cax = fig.add_axes([0.80, 0.12, 0.012, 0.3]); fig.colorbar(im, cax=cax, label="mean |r| above null\n(global factor removed)")
plt.subplots_adjust(wspace=0.45, hspace=0.45, left=0.06, right=0.97, top=0.95, bottom=0.10)
plt.savefig(W / "fig3_block_maps.png", dpi=130); plt.show()

rows = []
for s in STIMS:
    for var in ("raw", "pc1"):
        M = block_matrix(s, var); v = M.values[np.triu_indices(len(CT), 1)]
        P = block_matrix(s, var, col="p_perm").values[np.triu_indices(len(CT), 1)]
        rows.append((s, var, np.nanmedian(v), np.nanmax(v), np.mean(P < 0.005)))
display(pd.DataFrame(rows, columns=["stim", "variant", "median excess", "max excess", "frac blocks p<0.005"])
        .pivot(index="stim", columns="variant").reindex(STIMS).round(3))
""")

md(r"""
**Reading.** The maps are dominated by **lineage blocks**: T-cell subsets are coupled with each other (close relatives,
same signalling machinery, and gated from one another), and NK subsets likewise; monocytes/MDSCs form a smaller block.
Lymphoid ↔ myeloid blocks are mostly pale. The **granulocyte row (top) is unremarkable**: pale under IFNa, IL246, IL33
and GMCSF, moderately coupled under LPS (to B cells and the monocytes) and PI (to many T subsets) — but at the level of
many other rows, not above them (§6 quantifies this). TNFa, GMCSF, IL33 maps are flat except a few T/NK blocks,
consistent with their low reliability. Because a block p < 0.005 (200 permutations) flags 14–49 % of blocks after
PC1, "above chance" is not a useful bar here: what matters is which blocks stand out from the lineage pattern.
""")

code(r"""
def clustered(stim, var="pc1"):
    M = block_matrix(stim, var).fillna(0)
    D = 1 - (M.values - M.values.min()) / (M.values.max() - M.values.min())
    np.fill_diagonal(D, 0)
    from scipy.spatial.distance import squareform
    order = leaves_list(linkage(squareform(D, checks=False), "average"))
    return block_matrix(stim, var).iloc[order, order]

fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
for ax, s in zip(axes, ["IFNa", "LPS"]):
    M = clustered(s)
    im = ax.imshow(M.values, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(M))); ax.set_yticks(range(len(M)))
    lab = [sh(c) for c in M.index]
    ax.set_xticklabels(lab, rotation=90, fontsize=7); ax.set_yticklabels(lab, fontsize=7)
    for tl in ax.get_yticklabels() + ax.get_xticklabels():
        if tl.get_text() == "Granulocytes": tl.set_fontweight("bold"); tl.set_color("#b2182b")
    ax.set_title(f"{s}: celltype blocks, clustered (global factor removed)", loc="left")
    ax.spines[:].set_visible(False)
fig.colorbar(im, ax=axes, shrink=0.6, label="mean |r| above null")
plt.savefig(W / "fig4_block_clustered.png", dpi=130, bbox_inches="tight"); plt.show()
""")

md(r"""
**Reading.** Clustering recovers the lineages (T subsets together, NK subsets together, monocytes with MDSCs/mDCs). The
coupling structure of the *responses* is mostly the structure of *cell identity*: related celltypes respond alike within a
donor. NK and T clusters are clean; the myeloid cluster is looser. Granulocytes (red label) do not sit in a cluster of
their own: under IFNa they group with mDCs and CD4Teff (the granulocyte–CD4Teff block is one of the darker
cross-lineage blocks, a single block among 378), under LPS next to B cells. Neither places them at the centre of the
map.
""")

md(r"""
### 4b. The full 364 × 364 matrix for IFNa and LPS

Spearman r of the response across donors, global factor removed. Rows/columns grouped by celltype (lineage order), the
13 markers inside each block in a fixed order. Lines separate celltypes.
""")

code(r"""
FUNC = ["CD25", "HLADR", "IkB", "pCREB", "pERK", "pMK2", "pNFkB", "pS6", "pSTAT1", "pSTAT3", "pSTAT5", "pSTAT6", "pp38"]
def full(stim, var="pc1"):
    A = pd.read_csv(W / f"mat_func_{stim}_{var}.csv.gz", index_col=0)
    keys = [f"{c}|{m}" for c in CT for m in FUNC if f"{c}|{m}" in A.index]
    return A.loc[keys, keys]
fig, axes = plt.subplots(1, 2, figsize=(17, 8.6))
for ax, s in zip(axes, ["IFNa", "LPS"]):
    A = full(s); ct = [k.split("|")[0] for k in A.index]
    A2 = A.values.copy(); np.fill_diagonal(A2, np.nan)
    im = ax.imshow(A2, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    starts = [i for i in range(len(ct)) if i == 0 or ct[i] != ct[i - 1]]
    for x in starts[1:]: ax.axhline(x - 0.5, color="#555555", lw=0.3); ax.axvline(x - 0.5, color="#555555", lw=0.3)
    mids = [(a + b) / 2 for a, b in zip(starts, starts[1:] + [len(ct)])]
    ax.set_yticks(mids); ax.set_yticklabels([sh(ct[i]) for i in starts], fontsize=6.5)
    ax.set_xticks(mids); ax.set_xticklabels([sh(ct[i]) for i in starts], fontsize=6.5, rotation=90)
    ax.set_title(f"{s}: 364 × 364 response correlations (global factor removed)", loc="left")
    ax.spines[:].set_visible(False)
fig.colorbar(im, ax=axes, shrink=0.6, label="Spearman r across 19 donors")
plt.savefig(W / "fig5_full_matrix.png", dpi=130, bbox_inches="tight"); plt.show()
""")

md(r"""
**Reading.** Within-celltype blocks (on the diagonal) and T↔T / NK↔NK blocks are the dense areas. Inside off-diagonal
blocks, the strongest entries are the **same marker across celltypes** (a stripe parallel to the diagonal, period 13) —
the dominant form of cross-celltype coupling. Negative (blue) entries are partly created by the PC1 removal itself,
which makes residuals slightly anti-correlated on average.
""")

md(r"""
## 5. The strongest individual cross-celltype pairs

Pairs surviving FDR (q < 0.05, permutation p, BH over ~60 000 cross-celltype pairs per stim). First the counts — and the
**same count for the technical reference**, which tells how many "significant" cross-celltype pairs the method produces
from markers that do not respond.
""")

code(r"""
cnt = (pairs[pairs.q_bh < 0.05].groupby(["stim", "set", "variant", "category"]).size()
       .unstack(["set", "variant", "category"]).reindex(STIMS).fillna(0).astype(int))
cnt = cnt[[c for c in cnt.columns if c[1] in ("raw", "pc1", "pc1tech")]]
tot = pairs.groupby(["stim", "set", "variant"]).size().unstack(["set", "variant"]).reindex(STIMS)
display(cnt)
print("pairs tested per stim (functional, raw):", tot[("func", "raw")].to_dict())
""")

code(r"""
fm = meta[meta.kind == "func"].set_index(["stim", "celltype", "marker"])
P = pairs[(pairs.set == "func") & (pairs.variant == "pc1") & (pairs.q_bh < 0.05)].copy()
for side in ("a", "b"):
    ix = list(zip(P.stim, P[f"ct_{side}"], P[f"mk_{side}"]))
    P[f"meanΔ_{side}"] = fm.loc[ix, "mean_delta"].values
    P[f"rel_{side}"] = fm.loc[ix, "rel_plate_mean"].values
P["both_respond"] = (P["meanΔ_a"].abs() > 0.1) & (P["meanΔ_b"].abs() > 0.1)
P["both_reliable"] = (P.rel_a > 0.5) & (P.rel_b > 0.5)
P["lineages"] = [f"{LINEAGE[a]}–{LINEAGE[b]}" for a, b in zip(P.ct_a, P.ct_b)]
print("q<0.05 after global factor:", len(P), "| both responding:", P.both_respond.sum(), "| both reliable:", P.both_reliable.sum())
print("\nby lineage pair, different marker only:")
display(P[P.category == "cross_diff_marker"].groupby(["stim", "lineages"]).size().unstack(fill_value=0).reindex(STIMS).fillna(0).astype(int))
top = P[P.category == "cross_diff_marker"].sort_values("r", key=np.abs, ascending=False)
cols = ["stim", "ct_a", "mk_a", "ct_b", "mk_b", "r", "q_bh", "meanΔ_a", "meanΔ_b", "rel_a", "rel_b"]
print("\nTop 30 different-marker cross-celltype pairs (global factor removed):")
display(top[cols].head(30).round(3).reset_index(drop=True))
print("\nTop 15 cross-lineage, different-marker pairs, both features responding and reliable:")
tl = top[(top.lineages.str.split("–").str[0] != top.lineages.str.split("–").str[1]) & top.both_respond & top.both_reliable]
display(tl[cols].head(15).round(3).reset_index(drop=True))
""")

md(r"""
**Reading.**
- **Counts.** After the global factor, the functional set yields many FDR-surviving cross-celltype pairs for **IFNa
  (218 different-marker / 614 same-marker)** and **LPS (117 / 400)**, against 3–15 for the technical reference on the
  same stims — a clear excess. For **PI** (41 / 266 vs 19 / 96 on a 5× smaller reference set) the excess per tested pair
  is small. For **TNFa, GMCSF, IL33, IL246** the functional counts are of the same order as the reference's
  (e.g. TNFa 89 vs 81, IL33 4 vs 25): no evidence of specific coupling.
- **Where.** The surviving different-marker pairs are mostly **T–T and NK–T**, then myeloid–T (IFNa 41, LPS 24 pairs).
  Many share a pathway: STATs with STATs under IFNa (e.g. pSTAT3 in CD4Tcm CCR2+ with pSTAT5 in Tregs, r 0.91),
  IkB / pNFkB / pMK2 under TNFa and LPS.
- **Not all survivors are interpretable.** The top-30 list includes features that do not respond on average or have
  near-zero plate reliability (e.g. IL246 `DN T HLADR` / `Tregs pCREB`, reliability −1.1). One recurring source,
  `NK CD56hi pERK` (mean Δ 0.03), correlates with pSTAT1 in many celltypes under IFNa — a single noisy feature lining
  up with the donor's IFN axis. The second table keeps only features that respond (|mean Δ| > 0.1) and are reliable
  (> 0.5) and crosses lineages: these are the best candidates, all |r| ≈ 0.77–0.84 with q 0.006–0.04.
- Read every row as a *candidate*: n = 19 donors, one time point, and the PC1 control is crude.
""")

md(r"""
## 6. The granulocyte hypothesis

If granulocytes act on the other cells within the 2 h stimulation, donors (or wells) whose granulocytes respond more
should show stronger responses elsewhere, and **granulocytes should be more coupled to the other celltypes than a
typical celltype is**. Three tests:
1. **Hub score**: for each source celltype, the mean (over the 27 other celltypes) of the block mean |r|, minus its
   permutation-null mean. Granulocytes vs the 27 other possible sources.
2. **Granulocyte frequency** in the well (log share of the 28 celltypes' cells): does a donor with more granulocytes
   show stronger (or weaker) responses in other celltypes?
3. **Within-donor, well-to-well** (§9): in the wells where a donor's granulocytes respond more, do her other cells too?
""")

code(r"""
h = hubs[(hubs.set == "func")].copy()
h["rank"] = h.groupby(["stim", "variant"]).excess.rank(ascending=False)
g = h[h.source == "Granulocytes"].pivot(index="stim", columns="variant", values="rank").reindex(STIMS)[["raw", "pc1", "tech", "pc1tech"]]
print("Rank of granulocytes among 28 source celltypes by hub excess (1 = most coupled to the others):")
display(g.astype(int))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
hp = h[h.variant == "pc1"]
for i, s in enumerate(STIMS):
    v = hp[hp.stim == s]
    others = v[v.source != "Granulocytes"]
    ax.scatter(np.full(len(others), i) + np.random.default_rng(i).uniform(-0.18, 0.18, len(others)), others.excess,
               s=14, color="#9a9a9a", alpha=0.8, label="other source celltypes" if i == 0 else None)
    gv = v[v.source == "Granulocytes"].excess.values[0]
    ax.scatter([i], [gv], s=70, color="#b2182b", marker="D", label="Granulocytes" if i == 0 else None, zorder=3)
ax.set_xticks(range(len(STIMS))); ax.set_xticklabels(STIMS)
ax.set_ylabel("hub score above null\n(mean |r| with the 27 others)"); ax.legend(frameon=False, fontsize=8)
ax.set_title("Is any celltype a hub? (global factor removed)", loc="left")
ax = axes[1]
rel = meta[(meta.kind == "func")].groupby(["stim", "celltype"]).rel_plate_mean.median().rename("rel").reset_index()
x = hp.merge(rel, left_on=["stim", "source"], right_on=["stim", "celltype"])
ax.scatter(x.rel, x.excess, s=12, color="#9a9a9a")
xg = x[x.source == "Granulocytes"]; ax.scatter(xg.rel, xg.excess, s=60, color="#b2182b", marker="D")
for _, r in xg.iterrows(): ax.annotate(r.stim, (r.rel, r.excess), xytext=(5, -3), textcoords="offset points", fontsize=7, color="#b2182b")
ax.set_xlabel("median reliability of the source's 13 responses"); ax.set_ylabel("hub score above null")
ax.set_title("Hub score follows reliability", loc="left")
plt.tight_layout(); plt.savefig(W / "fig6_granulocyte_hub.png", dpi=130); plt.show()
print("Spearman(hub excess, reliability) across all (stim, source):", round(x[["rel", "excess"]].corr("spearman").iloc[0, 1], 2))
""")

md(r"""
**Reading.** **Granulocytes are not a hub.** Ranked against the 27 other celltypes as a source of coupling, they are
**last or second-to-last in raw correlations for IFNa, LPS, PI and GMCSF**, and mid-pack (rank 10–13) for LPS, PI,
IL246 and TNFa after the global factor — never in the top 9. That is despite being the most precisely measured
celltype (tens of thousands of cells per well). Hub scores mostly follow how reproducible a celltype's responses are
(right panel, Spearman 0.56) and lineage (T subsets have many related T subsets to correlate with); the granulocyte
points lie at or below that trend.
""")

code(r"""
gb = blocks[(blocks.set == "func") & (blocks.ct_a == "Granulocytes") & (blocks.ct_b != "Granulocytes")]
G = gb[gb.variant == "pc1"].pivot(index="ct_b", columns="stim", values="excess").reindex(index=CT[1:], columns=STIMS)
Gp = gb[gb.variant == "pc1"].pivot(index="ct_b", columns="stim", values="p_perm").reindex(index=CT[1:], columns=STIMS)
fig, ax = plt.subplots(figsize=(6.5, 7.5))
im = ax.imshow(G.values, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
for i in range(G.shape[0]):
    for j in range(G.shape[1]):
        if Gp.values[i, j] < 0.005: ax.text(j, i, "•", ha="center", va="center", color="white" if G.values[i, j] > 0.07 else "black", fontsize=9)
ax.set_xticks(range(len(STIMS))); ax.set_xticklabels(STIMS); ax.set_yticks(range(len(G))); ax.set_yticklabels([sh(c) for c in G.index], fontsize=8)
ax.set_title("Granulocyte response vs each other celltype\n(mean |r| above null, global factor removed; • p < 0.005)", loc="left", fontsize=9)
ax.spines[:].set_visible(False); fig.colorbar(im, ax=ax, shrink=0.5, label="mean |r| above null")
plt.tight_layout(); plt.savefig(W / "fig7_granulocyte_targets.png", dpi=130); plt.show()
""")

md(r"""
**Reading.** Where granulocyte responses co-vary with another celltype beyond chance (•), it is mostly with the **other
myeloid cells** (MDSCs, cMCs, ncMCs, mDCs — shared lineage, receptors, and gating boundaries; MDSCs are partly
granulocyte-like), plus CD4Teff under IFNa, B cells and naive T cells under LPS, and several T subsets under PI. The excesses (≤ ~0.08 in
mean |r|) are of the same size as the T↔T blocks of other sources, not larger. A granulocyte-driven effect on
lymphocytes would appear as a dark column of T/NK/B rows across stims; there is a hint of it only for LPS and PI, and
there the granulocytes are no more coupled to T cells than e.g. monocytes or B cells are (§6 ranks).
""")

code(r"""
gf = gfreq.copy()
gf["excess"] = gf.mean_abs_r_obs - gf.null_mean
gt = gf.pivot_table(index="stim", columns=["feature", "variant"], values=["excess", "p_perm", "n_abs_r_gt_0.6"]).reindex(STIMS)
display(gt.round(3))
fr = counts.assign(tot=counts.groupby(["stim", "plate", "donor"]).n_cells.transform("sum"))
fr = fr[fr.celltype == "Granulocytes"].assign(share=lambda x: x.n_cells / x.tot)
print("Granulocyte share of the 28 celltypes' cells, per stim (median [min, max] over wells):")
display(fr.groupby("stim").share.describe()[["50%", "min", "max"]].reindex(["Unstim"] + STIMS).round(3))
""")

md(r"""
**Reading (first table, from the main script).** Donors differ a lot in granulocyte share (33–80 % of the cells in
Unstim wells; PI wells lose granulocytes, median 56 %). The Unstim granulocyte share is associated with other celltypes'
responses beyond chance in most stims, and part of it survives the global factor (p ≤ 0.05 in every stim
but LPS; resolution 1/200). The *change* in share under the stim is associated with others' responses only through the
global factor (null-level after PC1).

**But granulocyte share is pregnancy status here** (next tables). Every pregnant donor has a higher share than every
non-pregnant one (HV 34–56 %, PV 67–79 %; Mann–Whitney p = 3e-5). The raw association is negative on average (a donor
with more granulocytes responds *less* in other celltypes: mean signed r −0.26 IFNa, −0.28 PI). Controlling for the HV/PV
group alone removes the association almost entirely (`group` row: mean |r| ≈ null, p 0.07–0.55). With 19 donors whose
granulocyte share is perfectly split by pregnancy, **a granulocyte-share effect cannot be separated from a pregnancy
effect** — and the plainer reading is that pregnancy changes both the blood composition and the responses.
""")

code(r"""
gs = pd.read_csv(W / "granfreq_summary.csv")
gs["excess"] = gs.mean_abs_r - gs.null_mean
display(gs.pivot(index="stim", columns="variant", values=["excess", "p_perm", "mean_signed_r"]).reindex(STIMS)
        [[(a, b) for a in ("excess", "p_perm", "mean_signed_r") for b in ("raw", "pc1", "group", "pc1_group")]].round(3))
""")

md(r"""
## 7. Does one celltype's baseline predict another's response?

Partial Spearman r between the Unstim value of feature X (celltype A) and the response Δ of feature Y (celltype B ≠ A),
**controlling for Y's own baseline** (Δ always correlates with its own Unstim value, partly by regression to the mean).
Predictors also include the Unstim log-frequency of each celltype. Null: donors permuted on the predictor side.
""")

code(r"""
bp = bpairs.copy()
c = bp[bp.q_bh < 0.05].assign(freq=lambda x: x.pred_mk == "freq").groupby(["stim", "freq"]).size().unstack(fill_value=0).reindex(STIMS).fillna(0).astype(int)
c.columns = ["marker predictors", "frequency predictors"][: len(c.columns)] if len(c.columns) == 2 else c.columns
c["pairs tested"] = bp.groupby("stim").size().reindex(STIMS)
display(c)
print("Top 20 baseline -> response pairs:")
display(bp.sort_values("p_perm").assign(absr=lambda x: x.r_partial.abs()).sort_values(["q_bh", "absr"], ascending=[True, False])
        .head(20)[["stim", "pred_ct", "pred_mk", "resp_ct", "resp_mk", "r_partial", "q_bh"]].round(3).reset_index(drop=True))
gr = bp[(bp.pred_ct == "Granulocytes")]
print("Granulocyte baseline (markers + frequency) as predictor: pairs with q<0.05 per stim")
display(gr[gr.q_bh < 0.05].groupby(["stim", "pred_mk"]).size().unstack(fill_value=0).reindex(STIMS).fillna(0).astype(int))
""")

md(r"""
**Reading.** Once the target's own baseline is controlled, **no stim except PI** gives any FDR-surviving
baseline → response pair. Under PI, 30 pairs survive, nearly all one pattern: the Unstim **pS6** level of many T / NK /
monocyte subsets predicts the **pSTAT6** response of DN T and memory T cells (r ≈ 0.86–0.91) — a donor-level "baseline
pS6" trait, shared by all celltypes, predicting a T-cell response; not a cell-to-cell link. **Granulocyte baseline
(markers or share) predicts nothing** in any stim. No global-factor control is applied here (the predictor is a
baseline), so these remain donor-level associations.
""")

md(r"""
## 8. Unstim–Unstim structure as reference

If the response coupling were just the donor's baseline structure carried over, the cross-celltype correlation matrix of
the responses would resemble that of the Unstim medians. Spearman correlation between the two matrices' cross-celltype
entries:
""")

code(r"""
sp = summ.pivot(index="stim", columns="quantity", values="value").reindex(STIMS)
display(sp[["corr_unstim_vs_delta_matrix_raw", "corr_unstim_vs_delta_matrix_pc1"]]
        .rename(columns=lambda c: "Unstim vs Δ matrix (" + c.rsplit("_", 1)[1] + ")").round(2))
""")

md(r"""
**Reading.** The resemblance is weak (r 0.15–0.25 raw, 0.05–0.20 after the global factor). The coupling of responses is
**not** simply the donor's baseline co-variation re-expressed; it is organised by lineage and pathway (§4–5).
""")

md(r"""
## 9. Within-donor, well-to-well coupling (the "same well" test)

Each donor has 2–3 wells per stim (one per plate). After removing donor and plate means, the remaining well-to-well
deviations are correlated across celltypes (Spearman, 57 wells; 38 for PI). If granulocytes in a given well respond more
and act on the others *in that well*, functional markers in the stim wells should co-vary across celltypes **more than**
(a) the same markers in the Unstim wells and (b) the technical-reference markers.
""")

code(r"""
wl = well.copy()
wl["excess"] = wl.median_abs_r_cross - wl.null_median
wt = wl.pivot_table(index="stim", columns=["wells_of_kind", "set"] if False else ["set"], values="excess")
wl["wells"] = np.where(wl.wells_of == "Unstim", "Unstim wells", "stim wells")
wt = wl.pivot_table(index="stim", columns=["wells", "set"], values=["median_abs_r_cross", "frac_above_null_q95", "median_abs_r_cross_granulocyte"]).reindex(STIMS)
display(wt.round(3))
fig, ax = plt.subplots(figsize=(9, 3.4))
combos = [("stim wells", "func", "#2a6fbb", "stim wells, functional"), ("Unstim wells", "func", "#9ec3e6", "Unstim wells, functional"),
          ("stim wells", "tref", "#6b6b6b", "stim wells, technical ref."), ("Unstim wells", "tref", "#c4c4c4", "Unstim wells, technical ref.")]
for k, (w_, st, c, lab) in enumerate(combos):
    v = wl[(wl.wells == w_) & (wl.set == st)].set_index("stim").frac_above_null_q95.reindex(STIMS)
    ax.bar(np.arange(len(STIMS)) + (k - 1.5) * 0.2, v.values, width=0.18, color=c, label=lab)
ax.axhline(0.05, color="black", lw=0.8, ls="--"); ax.set_xticks(range(len(STIMS))); ax.set_xticklabels(STIMS)
ax.set_ylabel("fraction of cross-celltype pairs\nabove null 95th pct."); ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
ax.set_title("Well-to-well coupling within donors", loc="left"); ax.set_ylim(0, 0.45)
plt.tight_layout(); plt.savefig(W / "fig8_well_level.png", dpi=130); plt.show()
""")

md(r"""
**Reading.** Within a donor, wells differ in a way that is shared by **all** markers of all celltypes: ~20–25 % of
cross-celltype pairs exceed the null, for functional *and* phenotypic markers, in stimulated *and* Unstim wells alike.
That is a well-level technical factor (staining intensity, acquisition, cell handling) — it is there without any
stimulation. Against that baseline, the functional markers of the stimulated wells show **no additional coupling** for
IFNa, IL246, IL33, TNFa and GMCSF, a small excess for LPS (0.28 vs 0.25 in Unstim wells), and a clear excess only for
**PI** (0.34 vs 0.26), the stim whose strength varies most between plates. Granulocyte-involving pairs are not more
coupled than the average pair. So at the level where "same well" is literally tested, there is no sign of a
granulocyte-driven paracrine coupling beyond the technical well effect; PI's extra coupling is most simply read as
well-to-well variation in stimulation strength hitting all celltypes at once.
""")

md(r"""
## 10. Caveats, and what would test causation

- **Power.** 19 donors. The null 95th percentile of |r| is 0.46 and FDR needs |r| ≳ 0.8. Real couplings of moderate
  size (|r| ~ 0.3–0.5) cannot be resolved individually; only aggregated statements (blocks, hubs, fractions above null)
  are informative.
- **Pregnancy.** The donors are 8 non-pregnant and 11 pregnant. Pregnancy is the main axis of the global factor for PI,
  IFNa and IL246 (§3) and perfectly separates granulocyte share (§6). Any donor-level association with granulocyte
  share is therefore also an association with pregnancy; the two cannot be separated with these donors.
- **Donor-level confounding.** Any trait that makes a donor respond more everywhere (genotype, sample age, time to
  processing, pregnancy, inflammation) creates cross-celltype correlation without any cell acting on another. One PC is
  a crude control: it removes some real coupling too (if granulocytes drove everyone, their effect would *be* PC1 — but
  §3 shows the granulocyte score is not aligned with PC1 for most stims).
- **Technical coupling.** Phenotypic markers that the stim does not move show 6–10 % excess coupling across donors and
  ~20–25 % across wells within a donor. This is the floor for any "same-well" interpretation. Barcoding and staining
  layout of the wells are not documented here; if wells were stained separately, same-marker cross-celltype coupling
  partly reflects the staining of that well.
- **Gating.** Celltypes are gated on markers that some stims move (CD62L, CCR2, CD14, CD16 under PI/LPS; monocytes
  are lost under PI/LPS). Responses of neighbouring gates (T memory subsets, monocyte subsets) are partly the same cells
  reassigned, which inflates within-lineage coupling.
- **Cell counts.** Rare celltypes (pDCs, mDCs ~80–120 cells per well) have noisy medians, which lowers their
  reliability and therefore their apparent coupling; granulocytes are the opposite (most precise), which makes their
  low hub score more, not less, convincing.
- **Correlation is not causation.** Everything here is a candidate coupling. A direct test of the granulocyte
  hypothesis needs an intervention: **granulocyte depletion or add-back** (stimulate PBMC vs whole blood vs PBMC +
  autologous granulocytes, compare lymphocyte responses), **transwell / conditioned-medium transfer** (supernatant from
  stimulated granulocytes onto unstimulated PBMC), **cytokine blockade** (e.g. anti-IL-6, anti-TNF, IL-1RA during the
  stim) or a **protein-transport inhibitor** to stop secretion, and a **time course** (secondary signalling appears later
  than primary; the atlas has one 2 h time point).
""")

md(r"""
## 11. Commands
```
module load python/3.9.0; source /scratch/users/bmerwan/venv-stabl/bin/activate; cd /scratch/users/bmerwan/simco
python baselines/stim_corr_wells.py                        # -> simco_work/stim_correlations/wells_{q50,counts}.csv.gz
OMP_NUM_THREADS=3 python baselines/stim_cross_correlations.py --nperm 200   # ~10 min
python notebooks/build_stim_cross_correlations.py
jupyter nbconvert --to notebook --execute --inplace notebooks/stim_cross_correlations.ipynb
```
Input `atlas_heldout/work/heldout/pheno/` is `heldout/pheno_summary.py` of that repo (≤ 20k cells per well, raw arcsinh).
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "notebooks/stim_cross_correlations.ipynb")
print("wrote notebooks/stim_cross_correlations.ipynb")
