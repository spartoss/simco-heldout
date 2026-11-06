"""Write notebooks/phenotype_shifts.ipynb: how stims and drugs move the phenotypic (gating) markers, and how that
changes population sizes without any biological change in cell numbers.
    python -m heldout.build_phenotype_notebook && jupyter nbconvert --to notebook --execute --inplace notebooks/phenotype_shifts.ipynb
"""
import nbformat as nbf

from lc.paths import REPO

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Phenotypic markers under stimulation and drugs, and populations that change size without changing biologically

Every population in the atlas (the 28 celltypes) is defined by **fixed gates on phenotypic markers** (CD14, CD16,
CD11b, CD45RA, CD62L, CCR2, …), applied identically to every well. If a stimulation or a drug *moves* those markers,
cells cross gates: a population can lose or gain cells although no cell divided or died. This notebook measures:

1. **How each stim moves the phenotypic markers** of each population (§2).
2. **How population sizes change under each stim** (§3), and whether the change is a transfer between populations of
   the same lineage (the lineage total is conserved) rather than a real gain or loss (§4).
3. **Case studies**: mDCs and monocytes under LPS and PI; T-cell memory subsets under PI (§5).
4. **The same for the drugs** (§6).
5. **Which simulated features, stim maps and Stabl selections are affected** (§7).

Data: `heldout/pheno_summary.py` (job per celltype) → `work/heldout/pheno/<celltype>.csv.gz`: for every well
(stim × treatment × donor; treatment = one of the 3 plate controls DMSO1/2/3 or one of the 9 drugs), the exact number
of cells and, for all 38 markers, the mean and q10/q50/q90 over ≤ 20,000 random cells. Raw arcsinh values, no batch
correction.

**Definitions.**
- *Frequency* of a population in a well = its cells / all cells of the 28 populations in that well (so differences in
  how many cells a well yielded cancel out).
- *Stim effect* (per donor) = the stim well minus the Unstim well **of the same plate control**, averaged over the
  three plates (DMSO1/2/3); for frequencies, log2(freq stim / freq Unstim). Donor-level effects are then summarised by
  the median over the 19 donors.
- *Technical floor* = the same statistic between two identical wells, DMSO2 vs DMSO3 under the same stim
  (DMSO1 differs under PI, see simco's CLAUDE.md).
""")

code(r"""
import sys; sys.path.insert(0, "/scratch/users/bmerwan/atlas_heldout")
import glob, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display
from heldout.common import WORK, STIMS, DRUGS, SIMCO_WORK
from lc.paths import CELLTYPES, MARKERS as FUNCTIONAL
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
BLUE, ORANGE = "#2a78d6", "#eb6834"
DIV = LinearSegmentedColormap.from_list("div", [ORANGE, "#f2f1ec", BLUE])
SEQ = LinearSegmentedColormap.from_list("seq", ["#f2f1ec", "#86b6ef", "#1c5cab", "#0d366b"])
P = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(WORK / "pheno/*.csv.gz")))], ignore_index=True)
PHENO = ["CD45", "CD66", "CD3", "CD4", "CD8", "CD7", "CD19", "CD56", "CD16", "CD14", "CD11b", "CD11c", "CD123",
         "CD45RA", "CD62L", "CCR2", "CD44", "CD36", "FoxP3", "Tbet"]
OTHER = sorted(set(P.marker) - set(PHENO) - set(FUNCTIONAL))
LINEAGE = {**{c: "CD4 T" for c in CELLTYPES if c.startswith("CD4T")}, **{c: "CD8 T" for c in CELLTYPES if c.startswith("CD8T")},
           "CD4negCD8negTcells": "other T", "NKT": "other T",
           **{c: "NK" for c in ["CD56hiCD16negNK", "CD56loCD16posNK", "NK_cells_CD11cneg", "NK_cells_CD11cpos"]},
           **{c: "monocytes + mDCs" for c in ["cMCs", "intMCs", "ncMCs", "mDCs", "MDSCs"]},
           "pDCs": "pDCs", "Bcells": "B cells", "Granulocytes": "Granulocytes"}
print(f"{P.celltype.nunique()} celltypes, {P.groupby(['stim','treatment','donor']).ngroups} wells, {P.marker.nunique()} markers")
print("phenotypic:", PHENO); print("functional (the 13 of the maps):", FUNCTIONAL); print("other:", OTHER)
""")

md(r"""
## 1. Cells per well, and population frequencies

A well's total cell count varies (loading, losses during fixation/permeabilisation, acquisition), which is why raw
counts cannot be compared between wells and frequencies are used below.
""")
code(r"""
W = P[P.marker == "CD45"][["celltype", "stim", "treatment", "donor", "n_cells"]].copy()
tot = W.groupby(["stim", "treatment", "donor"]).n_cells.transform("sum"); W["freq"] = W.n_cells / tot; W["total"] = tot
T = W.drop_duplicates(["stim", "treatment", "donor"])
ctrl = T[T.treatment.str.startswith("DMSO")]
display(ctrl.groupby("stim").total.describe()[["min", "25%", "50%", "75%", "max"]].astype(int)
        .style.set_caption("Total cells per plate-control well (28 populations), by stim"))
""")

md("## 2. How each stim moves the phenotypic markers")
code(r"""
def stim_effect(value):
    # per (celltype, stim, donor, marker): stim well - Unstim well of the same plate control, mean over the 3 plates
    C = P[P.treatment.str.startswith("DMSO")]
    U = C[C.stim == "Unstim"].set_index(["celltype", "treatment", "donor", "marker"])[value]
    S = C[C.stim != "Unstim"].copy()
    S["d"] = S[value].to_numpy() - U.reindex(pd.MultiIndex.from_frame(S[["celltype", "treatment", "donor", "marker"]])).to_numpy()
    return S.groupby(["celltype", "stim", "donor", "marker"]).d.mean().reset_index()

def floor_markers(value):
    C = P[P.treatment.isin(["DMSO2", "DMSO3"])].pivot_table(index=["celltype", "stim", "donor", "marker"],
                                                              columns="treatment", values=value)
    return (C.DMSO2 - C.DMSO3).abs().groupby(level=["celltype", "stim", "marker"]).median()

E50 = stim_effect("q50")
M = E50.groupby(["celltype", "stim", "marker"]).d.median().rename("shift").reset_index()
M["floor"] = floor_markers("q50").reindex(pd.MultiIndex.from_frame(M[["celltype", "stim", "marker"]])).to_numpy()
M["kind"] = np.where(M.marker.isin(PHENO), "phenotypic", np.where(M.marker.isin(FUNCTIONAL), "functional", "other"))
S = M.assign(a=M["shift"].abs()).groupby(["kind", "stim"]).a.median().unstack().round(3)
display(S.style.background_gradient(cmap=SEQ, axis=None).set_caption(
    "Median over celltypes and markers of |median shift| (arcsinh), by marker kind and stim"))
H = M[M.kind == "phenotypic"].assign(a=lambda d: d["shift"].abs()).pivot_table(index="marker", columns="stim", values="a", aggfunc="max")
display(H.reindex(PHENO).round(2).style.background_gradient(cmap=SEQ, axis=None)
        .set_caption("Largest |median shift| of each phenotypic marker in any celltype, by stim"))
""")
code(r"""
fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
for ax, s in zip(axs, ["LPS", "PI"]):
    X = M[(M.stim == s) & (M.kind == "phenotypic")].pivot_table(index="celltype", columns="marker", values="shift")
    X = X.reindex(index=CELLTYPES, columns=PHENO)
    im = ax.imshow(X.to_numpy(), cmap=DIV, vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_xticks(range(len(PHENO))); ax.set_xticklabels(PHENO, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(CELLTYPES))); ax.set_yticklabels(CELLTYPES, fontsize=8); ax.set_title(f"{s}: median shift vs Unstim", fontsize=10)
fig.colorbar(im, ax=axs, fraction=0.02).set_label("arcsinh (blue: up under the stim, orange: down)")
fig.suptitle("Phenotypic markers under LPS and PI, per population (median over donors)"); plt.show()
top = M[(M.kind == "phenotypic")].assign(ratio=lambda d: d["shift"].abs() / d["floor"].clip(lower=0.01))
top = top.reindex(top["shift"].abs().sort_values(ascending=False).index).head(20)
display(top[["stim", "celltype", "marker", "shift", "floor", "ratio"]].round(2).style.hide(axis="index")
        .set_caption("The 20 largest phenotypic-marker shifts (floor = DMSO2 vs DMSO3, same stim; ratio = shift / floor)"))
""")

md(r"""
## 3. How population frequencies change under each stim

log2(frequency in the stim well / frequency in the Unstim well of the same plate), median over donors: +1 = the
population doubles, −1 = it halves. The technical floor is the same log2 ratio between DMSO2 and DMSO3 wells.
""")
code(r"""
F = W[W.treatment.str.startswith("DMSO")].copy()
Fu = F[F.stim == "Unstim"].set_index(["celltype", "treatment", "donor"]).freq
Fs = F[F.stim != "Unstim"].copy()
Fs["l2"] = np.log2(Fs.freq.to_numpy() / Fu.reindex(pd.MultiIndex.from_frame(Fs[["celltype", "treatment", "donor"]])).to_numpy())
FD = Fs.replace([np.inf, -np.inf], np.nan).groupby(["celltype", "stim", "donor"]).l2.mean().reset_index()
FR = FD.groupby(["celltype", "stim"]).l2.median().unstack()
F23 = W[W.treatment.isin(["DMSO2", "DMSO3"])].pivot_table(index=["celltype", "stim", "donor"], columns="treatment", values="freq")
floor_f = np.log2(F23.DMSO2 / F23.DMSO3).replace([np.inf, -np.inf], np.nan).abs().groupby(level=["celltype", "stim"]).median().unstack()
fig, ax = plt.subplots(figsize=(8, 9))
X = FR.reindex(index=CELLTYPES, columns=[s for s in STIMS])
im = ax.imshow(X.to_numpy(), cmap=DIV, vmin=-2, vmax=2, aspect="auto")
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        v = X.iat[i, j]
        if abs(v) >= 0.5: ax.text(j, i, f"{2 ** v:.1f}×", ha="center", va="center", fontsize=7, color="white" if abs(v) > 1.2 else "#222222")
ax.set_xticks(range(len(STIMS))); ax.set_xticklabels(STIMS); ax.set_yticks(range(len(CELLTYPES))); ax.set_yticklabels(CELLTYPES, fontsize=8)
fig.colorbar(im, ax=ax, fraction=0.04).set_label("log2 frequency ratio, stim / Unstim (median over donors)")
ax.set_title("Population size under each stim (labelled when ≥ 1.4× or ≤ 0.7×)", fontsize=10); plt.show()
print(f"technical floor |log2(DMSO2/DMSO3)|: median over populations and stims {np.nanmedian(floor_f.to_numpy()):.2f} "
      f"(= {2 ** np.nanmedian(floor_f.to_numpy()):.2f}×); 90th percentile {np.nanpercentile(floor_f.to_numpy(), 90):.2f}")
big = FR.stack().rename("log2").reset_index(); big["floor"] = floor_f.stack().reindex(pd.MultiIndex.from_frame(big[["celltype", "stim"]])).to_numpy()
big = big[big.log2.abs() >= 0.5].assign(fold=lambda d: 2 ** d.log2, over_floor=lambda d: d.log2.abs() / d.floor)
display(big.sort_values("log2").round(2).style.hide(axis="index").set_caption(
    "Populations whose frequency changes ≥ 1.4× under a stim (median over donors)"))
""")

md(r"""
## 4. Transfer between populations, or a real change? Lineage totals

If cells only cross gates *within* a lineage (for example monocytes becoming "mDCs", or naive T cells becoming
"effector"), the lineage's total frequency is conserved while its members change in opposite directions. A change of
the lineage total instead means cells left or entered the lineage (real loss, or crossing into another lineage).
""")
code(r"""
W["lineage"] = W.celltype.map(LINEAGE)
L = W[W.treatment.str.startswith("DMSO")].groupby(["lineage", "stim", "treatment", "donor"]).freq.sum().reset_index()
Lu = L[L.stim == "Unstim"].set_index(["lineage", "treatment", "donor"]).freq
Ls = L[L.stim != "Unstim"].copy()
Ls["l2"] = np.log2(Ls.freq.to_numpy() / Lu.reindex(pd.MultiIndex.from_frame(Ls[["lineage", "treatment", "donor"]])).to_numpy())
LT = Ls.groupby(["lineage", "stim", "donor"]).l2.mean().groupby(["lineage", "stim"]).median().unstack()
display((2 ** LT[STIMS]).round(2).style.background_gradient(cmap=DIV, vmin=0.25, vmax=1.75, axis=None)
        .set_caption("Lineage total frequency, stim / Unstim (fold, median over donors)"))
rows = []
for lin, g in pd.Series(LINEAGE).groupby(pd.Series(LINEAGE)):
    for s in STIMS:
        mem = FR.loc[FR.index.intersection(g.index), s]
        if len(mem) < 2: continue
        rows.append({"lineage": lin, "stim": s, "total fold": 2 ** LT.loc[lin, s], "largest gain": f"{mem.idxmax()} {2 ** mem.max():.2f}×",
                     "largest loss": f"{mem.idxmin()} {2 ** mem.min():.2f}×", "spread (max/min fold)": 2 ** (mem.max() - mem.min())})
R = pd.DataFrame(rows); R = R[R["spread (max/min fold)"] >= 1.5].sort_values("spread (max/min fold)", ascending=False)
display(R.round(2).style.hide(axis="index").set_caption(
    "Lineages whose members move apart (≥ 1.5× between the largest gain and the largest loss) under a stim"))
""")

md("## 5. Case studies")
md(r"""
### 5a. Monocytes and mDCs under LPS and PI

Per donor, the change in mDC frequency against the change in monocyte (cMCs + intMCs + ncMCs) frequency, in
percentage points of all cells. Points on the dashed line = every lost monocyte reappears as an "mDC". Then the
phenotype of the cells labelled mDCs: do stimulated "mDCs" look like Unstim mDCs, or like monocytes?
""")
code(r"""
C = W[W.treatment.str.startswith("DMSO")]
def pp(names):
    x = C[C.celltype.isin(names)].groupby(["stim", "treatment", "donor"]).freq.sum()
    u = x.xs("Unstim", level="stim")
    return {s: ((x.xs(s, level="stim") - u) * 100).groupby(level="donor").mean() for s in ["LPS", "PI"]}
dm, dmono = pp(["mDCs"]), pp(["cMCs", "intMCs", "ncMCs"])
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, s in zip(axs, ["LPS", "PI"]):
    ax.scatter(-dmono[s], dm[s], color=BLUE, s=40)
    lim = max(-dmono[s].min(), dm[s].max(), 0.1) * 1.1
    ax.plot([0, lim], [0, lim], color="#999999", ls="--", lw=1, label="all lost monocytes become 'mDCs'")
    ax.set_xlabel("monocytes lost (percentage points of all cells)"); ax.set_ylabel("'mDCs' gained (percentage points)")
    ax.set_title(f"{s}: per donor", fontsize=10); ax.grid(alpha=.3); ax.legend(frameon=False, fontsize=8)
    print(f"{s}: monocytes lose a median {-dmono[s].median():.2f} pp, mDCs gain {dm[s].median():.2f} pp "
          f"(ratio {dm[s].median() / -dmono[s].median():.2f})")
fig.suptitle("Where do the extra 'mDCs' come from?"); plt.show()
K = ["CD14", "CD16", "CD11b", "CD11c", "HLADR", "CD123", "CD36", "CCR2", "CD62L", "CD45RA"]
Q = P[P.treatment.str.startswith("DMSO") & P.marker.isin(K)].groupby(["celltype", "stim", "marker"]).q50.median().unstack("marker")[K]
rows = {"Unstim mDCs": Q.loc[("mDCs", "Unstim")], "LPS 'mDCs'": Q.loc[("mDCs", "LPS")], "PI 'mDCs'": Q.loc[("mDCs", "PI")]}
for c in ["cMCs", "intMCs", "ncMCs"]: rows[f"Unstim {c}"] = Q.loc[(c, "Unstim")]
for c in ["cMCs", "intMCs"]: rows[f"PI {c}"] = Q.loc[(c, "PI")]
display(pd.DataFrame(rows).T.round(2).style.background_gradient(cmap=SEQ, axis=0)
        .set_caption("Median phenotypic markers (median over donors and plates)"))
""")
md(r"""
### 5b. T-cell memory subsets under PI

The CD4 and CD8 subsets are gated on CD45RA, CD62L and CCR2, three markers that PI (PMA/ionomycin) can move (CD62L is
shed on activation). Same question: do the subsets exchange cells?
""")
code(r"""
for lin in ["CD4 T", "CD8 T"]:
    mem = [c for c in CELLTYPES if LINEAGE[c] == lin]
    t = pd.DataFrame({"fold vs Unstim (PI)": 2 ** FR.loc[mem, "PI"], "fold (LPS)": 2 ** FR.loc[mem, "LPS"]})
    for m in ["CD45RA", "CD62L", "CCR2"]:
        t[f"{m} shift (PI)"] = M[(M.stim == "PI") & (M.marker == m)].set_index("celltype")["shift"].reindex(mem)
    display(t.round(2).style.background_gradient(cmap=DIV, subset=["fold vs Unstim (PI)"], vmin=0.25, vmax=1.75)
            .set_caption(f"{lin}: subset sizes and gating-marker shifts under PI (lineage total PI {2 ** LT.loc[lin, 'PI']:.2f}×)"))
""")

md(r"""
## 6. Drugs

Drug well vs its own plate control (`use_DMSO` of batchcorrection_2.csv: LPZ/SA → DMSO1; CHT/MAP/MF/PRA → DMSO2;
RIF/SALPZ/THF → DMSO3), within the same stim. Same two quantities: phenotypic-marker shift and log2 frequency ratio,
against the control-vs-control floor.
""")
code(r"""
PLATE = {"LPZ": "DMSO1", "SA": "DMSO1", "CHT": "DMSO2", "MAP": "DMSO2", "MF": "DMSO2", "PRA": "DMSO2",
         "RIF": "DMSO3", "SALPZ": "DMSO3", "THF": "DMSO3"}
Wd = W.set_index(["celltype", "stim", "treatment", "donor"]).freq
rows = []
for d, c in PLATE.items():
    a = Wd.xs(d, level="treatment"); b = Wd.xs(c, level="treatment")
    l2 = np.log2(a / b.reindex(a.index)).replace([np.inf, -np.inf], np.nan)
    rows.append(l2.groupby(level=["celltype", "stim"]).median().rename(d))
DF = pd.concat(rows, axis=1)
fl = floor_f.stack()
share = pd.DataFrame({d: ((DF[d].abs() > 0.5) & (DF[d].abs() > 2 * fl.reindex(DF.index))).groupby(level="stim").mean()
                      for d in PLATE}).T
display((share * 100).round(1).style.background_gradient(cmap=SEQ, axis=None).set_caption(
    "Drugs: % of populations whose frequency changes ≥ 1.4× AND ≥ 2× the DMSO2/DMSO3 floor, by stim"))
Pd = P.set_index(["celltype", "stim", "treatment", "donor", "marker"]).q50
rows = []
for d, c in PLATE.items():
    a = Pd.xs(d, level="treatment"); b = Pd.xs(c, level="treatment")
    x = (a - b.reindex(a.index)).groupby(level=["celltype", "stim", "marker"]).median()
    rows.append(x.rename(d))
DM = pd.concat(rows, axis=1).reset_index()
DM = DM[DM.marker.isin(PHENO)]
mf = floor_markers("q50")
long = DM.melt(id_vars=["celltype", "stim", "marker"], var_name="drug", value_name="shift")
long["floor"] = mf.reindex(pd.MultiIndex.from_frame(long[["celltype", "stim", "marker"]])).to_numpy()
long["beyond"] = (long["shift"].abs() > 0.1) & (long["shift"].abs() > 2 * long["floor"])
display((long.groupby(["drug", "stim"]).beyond.mean().unstack() * 100).round(1).style.background_gradient(cmap=SEQ, axis=None)
        .set_caption("Drugs: % of (population, phenotypic marker) with a median shift > 0.1 AND > 2× the floor"))
display(long.reindex(long["shift"].abs().sort_values(ascending=False).index).head(15).round(2).style.hide(axis="index")
        .set_caption("Largest phenotypic-marker shifts caused by a drug"))
""")

md(r"""
## 7. What this touches downstream

A simulated feature `<celltype>_<marker>_<stim>` is the median of the cells the gate calls `<celltype>` in the stim
well. Where the population changes size a lot under that stim, part of the "response" the map learns is a change of
*who* is in the gate. Flag: |log2 frequency ratio| ≥ 0.5 (≥ 1.4× or ≤ 0.7×) and ≥ 2× the technical floor.
""")
code(r"""
flag = big[(big.log2.abs() >= 0.5) & (big.over_floor >= 2)][["celltype", "stim", "fold"]]
grid = pd.read_csv("/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/valid_jobs.txt",
                   header=None, names=["stim", "celltype", "evalkey"])
fl_set = set(map(tuple, flag[["stim", "celltype"]].to_numpy()))
print(f"flagged (stim, celltype) pairs: {len(fl_set)}; of the paper's 168: {sum((s, c) in fl_set for s, c in zip(grid.stim, grid.celltype))}")
sel = pd.read_csv(SIMCO_WORK / "drug_choice/selected_features_fig5b.csv")
sel["flagged"] = [(s, c) in fl_set for s, c in zip(sel.stim, sel.celltype)]
u = sel[sel.sel_freq >= 0.05]
print(f"Fig 5b features selected in >= 5 % of folds: {len(u)}; in a flagged pair: {int(u.flagged.sum())} "
      f"({u[u.flagged].sel_freq.sum() / u.sel_freq.sum():.0%} of their selections)")
display(u[u.flagged].sort_values("sel_freq", ascending=False).round(2).style.hide(axis="index")
        .set_caption("Fig 5b features that sit in a population changing size under their stim"))
display(flag.sort_values("fold").round(2).style.hide(axis="index").set_caption("Flagged (stim, celltype) pairs"))
""")

md(r"""
## 8. Summary (2026-09-30)

**Phenotypic markers.**
- **GMCSF, IFNa, IL-2/4/6 and IL33 leave the phenotypic markers alone** (median |shift| 0.006–0.010 arcsinh, the size
  of IL33's functional effect). **PI moves them a lot** (median 0.074, up to 2.6), and **LPS and TNFa move a few**.
- The markers that move are exactly the ones used to define populations: **CD62L is shed** under PI, LPS and TNFa
  (−1.4 to −2.5 in Granulocytes, B cells, MDSCs, pDCs, T cells), **CCR2** (±1.5–1.8 in monocytes and mDCs), **CD16,
  CD14, CD11b, CD36** in myeloid cells (up to 2.6), and **CD4, CD8, CD7** under PI (up to 1.4 in pDCs / T cells). All
  far above the DMSO2-vs-DMSO3 floor (5–50×).

**Population sizes.**
- **PI wells yield 41 % fewer cells** (median 80k vs 136k for Unstim), LPS 14 % fewer: cells are lost, not only relabelled.
- The technical floor for a population's frequency is small (median 1.08×), so changes of 1.4× and more are real.
  All 23 of them are under PI or LPS, except MDSCs under TNFa and GMCSF.
- **Monocytes are lost, not reclassified.** The monocyte + mDC lineage falls to 0.61× (PI) and 0.71× (LPS). Per donor,
  monocytes lose 2.3 (PI) / 1.4 (LPS) percentage points of all cells while "mDCs" gain only 0.28 / 0.07: at most
  12 % / 5 % of the lost monocytes could have become "mDCs". The rest leave the gated populations altogether
  (activated monocytes adhere to plastic or die; this data cannot say which). ncMCs keep only 17 % of their cells
  under PI, cMCs 57 %.
- **mDCs still grow 5× under PI and 1.75× under LPS, because they are a small population**: a few monocyte-derived
  cells are a large share of them. Their phenotype moves toward monocytes (CD11b 1.6 → 3.1, CD36 4.3 → 5.3, CD16
  1.5 → 2.3 under PI), consistent with CD14-low activated monocytes entering the mDC gate, although activated mDCs
  could also up-regulate these markers.
  *(Correction of what was said in the conversation of 2026-09-30: the raw count ratio 2.5× hid the 41 % loss of cells
  per PI well, and the monocytes do not mostly become mDCs.)*
- **T cells are reshuffled between memory subsets under PI.** Relative to all cells the T lineages grow 1.3–1.4×
  (mostly because myeloid cells and others are lost), but inside them CD4Tcm grows 1.8×, CD4Tcm CCR2+ 2.5×, CD8Tcm
  CCR2+ 4.2× while CD4Tem CCR2+ halves; CD62L falls by 1.0–1.6 in every subset and CCR2 moves by up to 0.7. Which way
  cells move depends on the lab's gate definitions (not in this repo); the point is that a PI "CD8Tcm CCR2+" population
  is not the same set of cells as the Unstim one.
- NK: CD56hi CD16− grows 1.65× under PI (CD16 is down-regulated).

**Drugs.** No drug changes a population's size or a phenotypic marker beyond the floor, **except LPZ and SA under PI**
(14 % of populations, 9–11 % of phenotypic markers), which is the DMSO1-under-PI plate artefact again: those two drugs
are compared to DMSO1, whose PI wells differ from all other wells.

**What it touches downstream.**
- 16 (stim, celltype) pairs change size ≥ 1.4× (and ≥ 2× the floor); **all 16 are in the paper's 168 maps**. For them,
  the map from Unstim to stim cells learns a change of *who is in the gate* as well as the signalling response.
- **8 of the 90 features the Fig 5b model uses sit in such a pair (10 % of its selections)**, including two of its most
  selected: `ncMCs_pMK2_PI` (80 % of folds; ncMCs keep 17 % of their cells under PI) and `CD8Tcm_CCR2pos_pNFkB_PI` (62 %;
  that population is 4× larger under PI).
- The measured features (frequencies, medians, quantiles of Unstim wells) are not affected; neither are the other
  six stims' maps, apart from MDSCs and mDCs.
""")
code(r"""
print("largest lineage-conserving transfers:"); print(R.head(8).round(2).to_string(index=False))
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
(REPO / "notebooks").mkdir(exist_ok=True)
nbf.write(nb, REPO / "notebooks/phenotype_shifts.ipynb"); print("wrote notebooks/phenotype_shifts.ipynb")
