"""Write notebooks/well_consistency.ipynb: is a donor's cell count and population frequency the same in every well?
    python -m heldout.well_consistency && python -m heldout.build_well_consistency_notebook && \
    jupyter nbconvert --to notebook --execute --inplace notebooks/well_consistency.ipynb
"""
import nbformat as nbf

from lc.paths import REPO

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Is a donor's cell count and population frequency the same in every well?

**Question (2026-10-08).** A donor's blood has one composition. Unstim, a stim, DMSO or a drug act for 2 h, too short
to change how many B cells or monocytes there are. So for one donor, **the frequency of each population should be the
same in every well**, apart from counting noise. It is not. This notebook measures by how much it varies, where the
variation comes from, and why the number of cells per donor and well varies so much.

**Design.** 96 wells = 8 stims (Unstim + 7) × 12 treatments (3 plate controls DMSO1/2/3 + 9 drugs). Every well holds
**all 19 donors, barcoded**, and is debarcoded afterwards. A treatment belongs to one plate (`use_DMSO` in
`batchcorrection_2.csv`): plate 1 = DMSO1, SA, LPZ; plate 2 = DMSO2, CHT, MAP, MF, PRA; plate 3 = DMSO3, RIF, SALPZ, THF.
Each well is then gated into 28 populations with fixed gates.

**Scope.**
- **PI is excluded** from everything below: it moves the gating markers (CD62L, CD4, CD8, CD14, CD16 …) and loses 41 %
  of the cells, so its populations are not the same sets of cells (`phenotype_shifts.ipynb`).
- **LPS is analysed on its own** (§7): it moves CD14/CD16/CCR2/CD62L in myeloid cells.
- **Main set** = Unstim, GMCSF, IFNa, IL-2/4/6, IL33, TNFa: 72 wells × 19 donors = 1368 donor-wells. These stims leave
  the phenotypic markers alone (median shift 0.006–0.010 arcsinh), except a few myeloid markers under TNFa/GMCSF.

**Definitions.**
- *Count* = cells of a population for one donor in one well; *total* = sum over the 28 populations (only gated cells
  are in the files, so this is not the number of events acquired).
- *Frequency* = count / total of that donor in that well.
- Everything is on the **log scale**, so a spread of 0.05 means about ±5 %, 0.7 means about ×2.
- **Decomposition** (`heldout/well_consistency.py`): for one population,
  `log y[donor, stim, treatment] = mean + well + donor + donor×stim + donor×treatment + residual`.
  - *well*: shared by all 19 donors in that well (a stim or drug effect, or plate staining / gating).
  - *donor*: the donor's own level. For a frequency, this is the biology we want to measure.
  - *donor×stim*, *donor×treatment*, *residual*: the donor's wells disagreeing with each other. Their combined
    size is called **within-donor** below.

  Each term is reported as its RMS. The **sampling floor** is the same decomposition after redrawing every count
  from a Poisson around `donor + well` (no donor-specific variation per well). It is the variation that counting a
  finite number of cells alone produces.

**How to read the tables: "typical ± %".** Each table answers "by how much does this part typically move the
number?". Worked example: a donor's B-cell frequency in 4 wells is 10.0 %, 11.0 %, 9.5 % and 10.5 % (average
10.25 %). The wells sit −2 %, +7 %, −7 % and +2 % from the donor's own average. Squaring, averaging and taking the
square root (an RMS, the same thing as a standard deviation around the average) gives **±5 %**: "this donor's B-cell
frequency typically moves by about 5 % from one well to the next". The same is done for each part:
- **well ±6 %** = a whole well (all 19 donors together) typically sits 6 % above or below the average well;
- **donor ±90 %** = a donor typically sits 90 % above or below the average donor (i.e. donors differ by ~×1.9);
- **within-donor ±7 %** = the same donor typically moves 7 % between its own wells, after removing the well and donor
  parts (this is the number that should be zero if a donor's composition were identical in every well);
- **counting floor** = the within-donor ± % expected from counting chance alone (next paragraph).

Technically the calculation is done on log values (so that ×2 and ×½ count as the same size), then converted back to
± % for display.

**The counting floor, in plain terms.** Even if a donor's blood had exactly the same composition in every well,
the counts would not be identical, because each well only holds a random sample of that donor's cells. Example:
pDCs are ~0.06 % of a donor's cells, so a well with 130 000 cells holds **about 80 pDCs**. By chance alone, one well
gets 72, the next 88: about **±9 cells, i.e. ±11 %**. (The rule for counts: the chance fluctuation is about the
square root of the count. √80 ≈ 9; this is called Poisson noise.) A big population is much steadier: ~6 800 naive
CD4 T cells fluctuate by √6 800 ≈ 82 cells, only **±1.2 %**. So the floor depends on how many cells are counted.

How it is computed here: build a fake copy of the data where every donor has *exactly* the same frequencies in all its
wells, draw the counts at random with the real well sizes, run the same analysis, and see how much within-donor
spread it finds. That is the spread to expect if nothing but counting were going on. Then compare with the real data:
- real ≈ floor (pDCs: ±16 % vs ±13 %) → the well-to-well differences are just counting chance;
- real ≫ floor (naive CD4 T: ±7 % vs ±1.4 %, 4.7×; Granulocytes ±2.4 % vs ±0.4 %, 6.8×) → something else changes the donor's frequency
  between wells (handling, plate, gating), and that is what this notebook tries to explain.

Data: `work/heldout/pheno/*.csv.gz` (exact cell counts, `heldout/pheno_summary.py`), cached as
`work/heldout/well_consistency/counts.csv.gz`.
""")

code(r"""
import sys; sys.path.insert(0, "/scratch/users/bmerwan/atlas_heldout")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display
from lc.paths import REPO
from scipy.stats import spearmanr
from heldout.well_consistency import counts, grid, decompose, rms, freq_table, PLATE, MAIN, TERMS
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8a85"
pct = lambda d: ((np.exp(d) - 1) * 100).round(0).astype(int)   # log-scale RMS -> "typical +- %"
P = counts(); P["plate"] = P.treatment.map(PLATE)
N_all = grid(P, MAIN + ["LPS", "PI"]); TOT = N_all.sum(1).rename("total")
T = TOT.reset_index(); T["plate"] = T.treatment.map(PLATE); T["cohort"] = T.donor.str[:2]
print(f"{P.celltype.nunique()} populations, {P.groupby(['stim','treatment']).ngroups} wells, {P.donor.nunique()} donors")
print("missing (donor, well, population) combinations:", int((N_all == 0).sum().sum()))
""")

md(r"""
## 1. Cells per donor and well: how much it varies, and why

Total cells per donor-well, over the 72 main-set wells. The first table splits the spread into the terms above. The
counting floor for a total of ~130 000 cells is 1/√130 000 ≈ ±0.3 %, so **none of this is counting noise**.
""")
code(r"""
Tm = T[T.stim.isin(MAIN)]
print("total cells per donor-well, main set: min %d, median %d, max %d (max/min %.1fx)"
      % (Tm.total.min(), Tm.total.median(), Tm.total.max(), Tm.total.max() / Tm.total.min()))
rows = {}
for name, st in [("main set", MAIN), ("LPS", ["LPS"]), ("PI (for reference)", ["PI"])]:
    D = decompose(np.log(TOT[TOT.index.get_level_values("stim").isin(st)])); rows[name] = rms(D)
C = pd.DataFrame(rows).T; C["within-donor"] = np.sqrt((C[TERMS[2:]] ** 2).sum(1))
display(pct(C).style.format(precision=1).set_caption("total cells per donor-well: typical ± % from each part"))
""")
md(r"""
**The donor is the dominant part (typically ±26 %).** Each donor gives a consistent number of cells
in every well. The well part (stim / drug / plate, ±7 %) and the donor's own well-to-well variation (±7 %) are about
3–4× smaller. LPS looks the same; PI is shown for reference only.

### 1a. The donor term: pregnant donors give more cells, and more granulocytes
""")
code(r"""
nm = T[T.stim.isin(MAIN + ["LPS"])]
don = nm.groupby("donor").total.apply(lambda x: np.exp(np.log(x).mean()))
gran = (N_all.Granulocytes / TOT)[TOT.index.get_level_values("stim") != "PI"].groupby(level="donor").mean()
nongran = (TOT - N_all.Granulocytes)[TOT.index.get_level_values("stim") != "PI"].groupby(level="donor").apply(lambda x: np.exp(np.log(x).mean()))
Dn = pd.DataFrame({"geo-mean total": don.round(0).astype(int), "relative": (don / np.exp(np.log(don).mean())).round(2),
                   "non-granulocytes": nongran.round(0).astype(int), "granulocyte freq": gran.round(2)}).sort_values("relative")
display(Dn.T)
print("range of the donor term: %.2fx (total), %.2fx (non-granulocyte cells)" % (don.max() / don.min(), nongran.max() / nongran.min()))
print(Dn.assign(cohort=Dn.index.str[:2]).groupby("cohort")[["relative", "granulocyte freq"]].mean().round(2))
from scipy.stats import spearmanr
print("Spearman(donor's total, granulocyte freq) = %.2f, p = %.3f" % spearmanr(Dn["relative"], Dn["granulocyte freq"]))
""")
md(r"""
Reading: donors differ by up to ~2.5× in the cells they contribute to every well. Pregnant donors (PV) give ~1.4× more
cells than non-pregnant ones (HV) and have more granulocytes (0.72 vs 0.47 of the gated cells). That is the known
neutrophilia of pregnancy: if each donor's sample is a fixed volume of blood, more leukocytes per ml means more cells
in every well. This term is **not a problem**: it is constant for a donor, and frequencies remove it.

### 1b. The well term: stim, plate and drug, shared by all donors
""")
code(r"""
we = np.log(Tm.groupby(["stim", "treatment"]).total.mean())
by_stim = np.exp(we.groupby(level=0).mean() - we.loc["Unstim"].mean()).round(2)
by_trt = np.exp(we.groupby(level=1).mean() - we.groupby(level=1).mean().mean()).round(2)
lps = np.exp(np.log(T[T.stim == "LPS"].total).mean() - np.log(T[T.stim == "Unstim"].total).mean())
display(by_stim.to_frame("cells vs Unstim").T)
display(by_trt.to_frame("cells vs average treatment").T)
print("LPS vs Unstim: %.2fx  (PI: %.2fx, excluded)" % (lps, np.exp(np.log(T[T.stim=='PI'].total).mean() - np.log(T[T.stim=='Unstim'].total).mean())))
""")
md(r"""
Reading: among the main-set stims, TNFa wells give 13 % fewer cells than Unstim and GMCSF / IL-2/4/6 5–6 % fewer;
LPS 12 % fewer (PI 41 % fewer). Treatments differ by up to ±6 %, DMSO3 lowest (0.90×). These are shared by every donor in the well, so they also
cancel in frequencies, unless the cells lost belong to particular populations (§4).

### 1c. Within a donor: how much the same donor's count varies between wells
""")
code(r"""
w = Tm.groupby("donor").total.agg(cv=lambda x: x.std() / x.mean(), max_over_min=lambda x: x.max() / x.min())
print(w.describe().loc[["min", "50%", "max"]].round(2).to_string())
Tm = Tm.assign(share=Tm.total / Tm.groupby(["stim", "treatment"]).total.transform("sum"))
ls = np.log(Tm.set_index(["stim", "treatment", "donor"]).share)
dev = (ls - ls.groupby(level="donor").transform("mean")).rename("log deviation").reset_index()
dev["x of usual"] = np.exp(dev["log deviation"]).round(2)
print("\nlargest deviations of a donor's share of a well from that donor's usual share:")
display(dev.reindex(dev["log deviation"].abs().sort_values(ascending=False).index).head(10).round(3))
DT = decompose(np.log(TOT[TOT.index.get_level_values("stim").isin(MAIN)]))
print("donor x treatment terms beyond 0.2 (x0.82 or x1.22):")
dxt = DT.donor_x_treatment.groupby(level=["donor", "treatment"]).first()
print(np.exp(dxt[dxt.abs() > 0.2]).round(2).to_string())
""")
md(r"""
Reading: a donor's count varies between its wells with a CV of 8 % (5–14 % depending on the donor; largest/smallest
well 1.3–2.5×). This is far
above counting noise (0.3 %), so it is **handling**: how many cells of that donor ended up in that well before the
donors were pooled (pipetting of the sample, losses during incubation, fixation and barcoding). The clearest case is **PV01 in the MF wells: about half its usual cells under GMCSF, IFNa and
IL33 (and LPS)**; it is the only donor × treatment term beyond ±20 %.

## 2. Population frequencies: within-donor variation against between-donor differences
""")
code(r"""
N = grid(P, MAIN); F = freq_table(N)
show = pd.DataFrame({"cells per donor-well (median)": F.median_cells.astype(int),
                     "between donors ± %": pct(F.donor), "whole well ± %": pct(F.well),
                     "within donor ± %": pct(F.within), "counting floor ± %": pct(F.null_within),
                     "within / floor": F.excess.round(1), "between / within": F.between_over_within.round(1)}
                    ).sort_values("cells per donor-well (median)", ascending=False)
display(show.style.format(precision=1).set_caption("population frequency, main set (72 wells): typical ± % from each part"))
print("median over populations: between-donor %.2f, within-donor %.3f, sampling floor %.3f, excess %.1fx, ratio between/within %.0f"
      % tuple(F[c].median() for c in ["donor", "within", "null_within", "excess", "between_over_within"]))
""")
code(r"""
S = F.sort_values("donor")
fig, ax = plt.subplots(figsize=(8, 8))
y = np.arange(len(S))
ax.scatter(S.donor, y, s=40, color=BLUE, label="between donors (donor term)", zorder=3)
ax.scatter(S.within, y, s=40, color=ORANGE, label="within a donor, across its 72 wells", zorder=3)
ax.scatter(S.null_within, y, s=40, facecolor="white", edgecolor=GREY, lw=1.5, label="within, expected from counting alone", zorder=3)
for i, (_, r) in enumerate(S.iterrows()): ax.plot([r.null_within, r.donor], [i, i], color="#e4e2dc", lw=1, zorder=1)
ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels(S.index, fontsize=8)
ax.set_xlabel("spread of log frequency (RMS; 0.05 ≈ ±5 %, 0.7 ≈ ×2)")
ax.set_title("Population frequency per donor: stable across wells, but noisier than counting alone\n(main set: Unstim, GMCSF, IFNa, IL-2/4/6, IL33, TNFa; PI and LPS excluded)", fontsize=10)
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.07), ncol=3, fontsize=8, frameon=False); ax.grid(axis="x", color="#eeeeea"); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); fig.savefig(REPO / "notebooks/fig_well_consistency_freq.png", dpi=150); plt.show()
""")
md(r"""
Reading:
- **Frequencies are consistent enough to rank donors.** For a typical population, a donor's 72 wells spread by ~7 %
  (RMS, median over populations), while donors typically differ by ±90 % (about ×1.9): a between/within ratio of 8 (on the log scale) (5–13 for most
  populations). The exceptions are **MDSCs (2.3), intMCs (3.2), mDCs (2.7) and cMCs (3.6)**: either donors differ
  little (cMCs, Granulocytes) or the population sits on a gate that moves (MDSCs, intMCs, mDCs, CD4Teff).
- **But they vary more than counting alone would: 2.3× the counting floor (median), 1.2–6.8× depending on the
  population.** The excess is largest for the big populations (Granulocytes 6.8×, cMCs 5.3×, CD4Tnaive 4.7×), whose
  counting noise is tiny: Granulocytes vary by 2.4 % against a floor of 0.4 %. So something donor-specific happens
  in each well, on top of counting.
- Small populations (pDCs, CD8Tcm CCR2+, CD56hi NK, ~80–200 cells per donor-well) are within 1.2–1.4× of their
  counting floor: for them, counting is most of the noise.

## 3. Which term carries the within-donor variation
""")
code(r"""
B = F[["well", "donor_x_stim", "donor_x_treatment", "residual"]].copy()
B.columns = ["well (shared by all donors)", "donor x stim", "donor x treatment", "residual"]
display(pct(B.sort_values("well (shared by all donors)", ascending=False).head(10))
        .style.format(precision=1).set_caption("population frequency, main set: typical ± % from each part (10 largest well parts)"))
print("median over populations (± %):"); print(pct(B.median()).to_string())
dxs = {c: decompose(np.log((N[c] + 0.5) / N.sum(1))).donor_x_stim.groupby(level="stim").apply(lambda x: np.sqrt((x ** 2).mean()))
       for c in N.columns}
dxs["TOTAL count"] = decompose(np.log(N.sum(1))).donor_x_stim.groupby(level="stim").apply(lambda x: np.sqrt((x ** 2).mean()))
display(pct(pd.DataFrame(dxs).T).style.format(precision=1).set_caption("donor x stim part, typical ± % per stim (last row: total cell count)"))
""")
md(r"""
Reading: for most populations the four terms are similar in size (medians 0.03–0.06). The well term is large only in
small, gate-sensitive populations: MDSCs (±25 %, mostly stim: 0.6–0.7× under TNFa and GMCSF), CD4Teff (±24 %, plate,
§5), intMCs, NKT, mDCs.

**Is the donor×stim term a donor-specific response to the stim?** No: it is as large for Unstim as for the stims
(table below). A donor's 12 Unstim wells share a deviation from that donor's other wells, although nothing was
added. So it is handling shared by the wells of one stim for one donor (e.g. the donor's sample being split per stim
before the treatments), not biology. Only MDSCs and mDCs under TNFa stand out, consistent with TNFa moving myeloid
gating markers.

## 4. When a donor has fewer cells in a well, are particular populations lost?

If a donor-well is short of cells because of pipetting, every population loses the same share, and frequencies do not
move. If cells are lost selectively (e.g. dying or sticking myeloid cells), the populations that are lost have a lower
frequency in the donor-wells with fewer cells. The test: regress each population's within-donor frequency deviation
on the same donor-well's within-donor deviation of total count (both on the log scale). A slope of 0 = uniform loss;
+0.3 = when the donor-well has 10 % fewer cells, this population's frequency is 3 % lower.
""")
code(r"""
Dt = decompose(np.log(N.sum(1))); ct = Dt[TERMS[2:]].sum(1)
sel = []
for c in N.columns:
    Dc = decompose(np.log((N[c] + 0.5) / N.sum(1))); fd = Dc[TERMS[2:]].sum(1)
    sel.append({"celltype": c, "slope": np.polyfit(ct, fd, 1)[0], "r": np.corrcoef(ct, fd)[0, 1]})
SEL = pd.DataFrame(sel).set_index("celltype").sort_values("slope")
display(SEL.round(2).T)
lym = [c for c in SEL.index if c.startswith(("CD4", "CD8", "Bcells", "CD56", "NK"))]
print("lymphocyte populations: median slope %.2f; Granulocytes %.2f" % (SEL.loc[lym, "slope"].median(), SEL.loc["Granulocytes", "slope"]))
# same split as counts: granulocytes vs non-granulocyte cells
g = decompose(np.log(N.Granulocytes))[TERMS[2:]].sum(1); ng = decompose(np.log(N.sum(1) - N.Granulocytes))[TERMS[2:]].sum(1)
print("within-donor SD of log count: granulocytes %.3f, non-granulocytes %.3f; correlation %.2f" % (g.std(), ng.std(), np.corrcoef(g, ng)[0, 1]))
""")
md(r"""
Reading: the loss is **not uniform, but the selectivity is mild.** When a donor-well has more cells than usual, the
extra is mostly lymphocytes and monocytes (slope +0.15 to +0.38, intMCs +0.64), while the granulocyte share falls
(−0.09). Within a donor, the non-granulocyte count varies more than the granulocyte count (10 % vs 7 %), and the two
are correlated (0.71) but not in lockstep. So part of a frequency's
within-donor variation is "how many lymphocytes vs granulocytes of that donor made it into that well", a handling
effect rather than a biological one.

## 5. Plate effects on frequencies, shared by all donors (main set)

Frequency per population by plate (mean over the plate's treatments and the main-set stims), relative to the average
of the three plates. Plates 1/2/3 = DMSO1, SA, LPZ / DMSO2, CHT, MAP, MF, PRA / DMSO3, RIF, SALPZ, THF.
""")
code(r"""
lf = np.log((N + 0.5).div(N.sum(1), axis=0))
wl = lf.groupby(level=["stim", "treatment"]).mean().reset_index(); wl["plate"] = wl.treatment.map(PLATE)
pl = wl.groupby("plate")[list(N.columns)].mean(); pl = np.exp(pl - pl.mean())
top = pl.T.assign(spread=lambda d: d.max(1) / d.min(1)).sort_values("spread", ascending=False)
display(top.head(10).round(2).style.format(precision=1).set_caption("frequency by plate relative to the 3-plate average (main set, no PI/LPS)"))
print("populations whose plates differ by > 1.2x:", int((top.spread > 1.2).sum()), "of", len(top))
""")
md(r"""
Reading: **plate 1 has more CD4Teff (1.26× the 3-plate average), NKT (1.13×), CD56hi NK and mDCs, and fewer intMCs
(0.88×); plate 3 has fewer CD4Teff (0.78×, so 1.6× between plates 1 and 3)**, for every donor at once, and without
PI. Only 4 of 28 populations differ by more than 1.2× between plates. These are small populations defined by gates on markers that vary
continuously (effector T cells, intermediate monocytes between classical and non-classical), so a small plate-wide
shift in staining moves cells across the gate. This is the same plate 1 that differs under PI (simco's CLAUDE.md); it
is a smaller effect outside PI, but it is there. Because it is shared by all donors in the well, it does not change
how donors rank, only the level.

## 6. Same donor, two replicate wells, compared with two different donors

Concrete numbers: Unstim DMSO2 vs Unstim DMSO3 for the same donor (two control wells that should be identical), and
the same wells for two different donors.
""")
code(r"""
u2 = np.log((N.loc[("Unstim", "DMSO2")] + 0.5).div(N.loc[("Unstim", "DMSO2")].sum(1), axis=0))
u3 = np.log((N.loc[("Unstim", "DMSO3")] + 0.5).div(N.loc[("Unstim", "DMSO3")].sum(1), axis=0))
same = (u2 - u3).abs()
dd = [(u2.loc[a] - u2.loc[b]).abs() for i, a in enumerate(u2.index) for b in u2.index[i + 1:]]
diff = pd.concat(dd, axis=1).median(1)
R = pd.DataFrame({"same donor, DMSO2 vs DMSO3 (median)": np.exp(same.median()) , "same donor (worst of 19)": np.exp(same.max()),
                  "two different donors (median)": np.exp(diff)}).round(2).sort_values("two different donors (median)")
display(R.style.format(precision=1).set_caption("fold difference in a population's frequency (1.05 = 5 %)"))
""")
md(r"""
Reading: two control wells of the same donor typically differ by 1–7 % in a population's frequency (10–19 % for
MDSCs, intMCs, NKT, pDCs, CD56hi NK; **30 % for CD4Teff**), while two donors typically differ by 1.3–2.5×. For
**CD4Teff, mDCs, MDSCs and cMCs the worst of the 19 replicate pairs is as large as or larger than a typical
difference between two donors** (CD4Teff 1.93× vs 1.68×). For those, a single well does not reliably rank a donor.

## 7. LPS, on its own

The same decomposition over the 12 LPS wells (one stim, so donor×stim does not exist and donor×treatment is the
whole within-donor term). The column `within vs Unstim` compares the within-donor variation under LPS with the same 12-well
decomposition under Unstim (same design, nothing added).
""")
code(r"""
NL = grid(P, ["LPS"]); FL = freq_table(NL)
L = FL[["median_cells", "donor", "well", "within", "null_within", "excess", "between_over_within"]].copy()
FU = freq_table(grid(P, ["Unstim"])); L["within vs Unstim"] = FL.within / FU.within
L["cells vs Unstim"] = (NL.median() / grid(P, ["Unstim"]).median()).round(2)
Ls = pd.DataFrame({"cells per donor-well (median)": L.median_cells.astype(int), "between donors ± %": pct(L.donor),
                   "whole well ± %": pct(L.well), "within donor ± %": pct(L.within), "counting floor ± %": pct(L.null_within),
                   "within / floor": L.excess.round(1), "within LPS / within Unstim": L["within vs Unstim"].round(1),
                   "cells vs Unstim": L["cells vs Unstim"]}).sort_values("within LPS / within Unstim", ascending=False)
display(Ls.style.format(precision=1).set_caption("population frequency under LPS (12 wells): typical ± % from each part"))
x = np.log((NL.intMCs + 0.5) / NL.sum(1)).unstack("treatment").droplevel(0)
print("intMCs under LPS, frequency relative to the donor's own LPS average (x):")
display(np.exp(x.sub(x.mean(1), axis=0)).round(2))
""")
md(r"""
Reading: under LPS, lymphoid populations are only slightly noisier than under Unstim (1.0–1.4×, median 1.2), but
the **monocyte gates break down**: intMCs vary within a donor by ±65 % (from 0.27× to 6× of the donor's own
LPS average), 4.1× the Unstim value against a counting floor of ±9 %; Granulocytes 1.9×, cMCs 1.6×, ncMCs 1.5×.
MDSCs and ncMCs lose 60–67 % of their cells and mDCs gain 1.6× (`cells vs Unstim`), without becoming noisier. LPS moves CD14 and
CD16 (`phenotype_shifts.ipynb`), the two markers that separate classical, intermediate and non-classical monocytes,
so how many cells land in the intMC gate depends on how strongly each donor's cells responded in that well. For LPS,
these myeloid populations are not comparable sets of cells across wells, and neither are their frequencies.

## 8. Is it debarcoding?

All donors share a well, so a cell assigned to the wrong donor would pull that donor's frequencies toward the mix of
the other donors in the well, more so where the donor has few cells. Test, per donor-well: regress the donor's
frequency deviation (after removing the well's shared shift) on (other donors' frequency in that well − the donor's
own frequency), over the 28 populations. The slope estimates the fraction of misassigned cells.
""")
code(r"""
f = N.div(N.sum(1), axis=0)
fd = f.groupby(level="donor").transform("mean")
pool = N.groupby(level=["stim", "treatment"]).transform("sum")
others = (pool - N).div((pool - N).sum(1), axis=0)
shift = f.groupby(level=["stim", "treatment"]).transform("mean") - f.groupby(level="donor").transform("mean").groupby(level=["stim","treatment"]).transform("mean")
x = (f - fd - shift); z = (others - fd)
alpha = (x * z).sum(1) / (z * z).sum(1)
share = N.sum(1) / N.sum(1).groupby(level=["stim", "treatment"]).transform("sum")
print("estimated misassigned fraction per donor-well: median %.3f, 5-95 %% [%.3f, %.3f]" % (alpha.median(), *alpha.quantile([.05, .95])))
print("Spearman(alpha, donor's share of the well) = %.2f" % spearmanr(alpha, share)[0])
print("PV01 in MF wells:", alpha.xs(("MF", "PV01"), level=["treatment", "donor"]).round(3).to_dict())
""")
md(r"""
Reading: **no sign of misassigned cells.** The estimated fraction is 0.7 % at the median and does not depend on how
large the donor's share of the well is (Spearman 0.10). PV01's half-empty MF wells show no pull toward the other donors
either. The estimate is noisy per donor-well (5–95 % range ±20 %, because granulocytes dominate the linear
frequencies and vary by handling), so it rules out systematic misassignment, not a few % in a given well.

## 9. Summary (2026-10-08)
""")
md(r"""
**Cell numbers per donor and well (main set: 58k to 226k gated cells, 3.9×).**
- Mostly **the donor** (±26 %, a 2.5× range between donors). Pregnant donors give ~1.4× more cells and have more
  granulocytes (0.73 vs 0.48 of gated cells), consistent with the neutrophilia of pregnancy if each sample is a fixed
  blood volume. Constant per donor, so frequencies remove it.
- Then **the well**, shared by all donors: TNFa −13 %, LPS −12 % (PI −41 %), treatments ±6 %.
- Then **the donor's own well-to-well handling**: CV 8 % (5–14 %), 25× counting noise. One clear accident: **PV01 in
  the MF wells has half its usual cells.**

**Frequencies, main set (Unstim, GMCSF, IFNa, IL-2/4/6, IL33, TNFa).**
- **Consistent enough to rank donors**: a donor's wells spread by ~7 %, donors typically differ by ±90 %, about ×1.9 (ratio 8). Two
  replicate control wells of a donor usually differ by 1–7 %.
- **But not as consistent as counting allows**: 2.3× the counting floor (up to 6.8× for Granulocytes). The excess is
  split about evenly between (i) **handling shared by a donor's wells of one stim** (as large for Unstim as for the
  stims, so not a response), (ii) per-treatment handling, (iii) per-well handling, and (iv) **well-level shifts
  shared by all donors**.
- **Mild selective loss**: when a donor-well has fewer cells than usual, it has lost relatively more lymphocytes and
  monocytes than granulocytes.
- **Plate effect on gate-sensitive populations, even without PI**: CD4Teff 1.6× between plates 1 and 3, NKT,
  intMCs, mDCs, CD56hi NK 1.2–1.3×. Plate 1 again.
- **Unreliable populations** (one well does not rank a donor): CD4Teff, MDSCs, intMCs, mDCs, and cMCs (donors differ
  little). The ~100-cell populations (pDCs, mDCs, CD56hi NK) are dominated by counting noise.
- **Not debarcoding**: no detectable pull toward the other donors in the well.

**LPS.** Lymphoid populations are almost as stable as under Unstim (1.2× noisier); **intMCs are not comparable between
wells (±65 % within a donor, 4× Unstim)**, cMCs / ncMCs / Granulocytes are 1.5–1.9× noisier, and MDSCs, ncMCs, mDCs
change size: LPS moves CD14/CD16 across the monocyte gates.

**What this means.** For the frequency features, averaging a donor over several wells (or using the 3 controls) is
worth it for the populations listed as unreliable; a single Unstim well is fine for the large lymphoid ones. The
myeloid populations under LPS (and everything under PI) should not be compared as the same populations across wells.
These are atlas donors; the OOL women's frequency features (one Unstim well each) carry the same per-well noise and
cannot be averaged.
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
nbf.write(nb, REPO / "notebooks/well_consistency.ipynb"); print("wrote notebooks/well_consistency.ipynb")
