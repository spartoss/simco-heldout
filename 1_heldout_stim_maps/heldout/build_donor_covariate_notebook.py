"""Write notebooks/donor_covariate.ipynb: a donor covariate from Unstim population frequencies, built to separate donors
and to stay the same across a donor's wells.
    python -m heldout.donor_covariate && python -m heldout.donor_covariate --final && \
    python -m heldout.build_donor_covariate_notebook && \
    jupyter nbconvert --to notebook --execute --inplace notebooks/donor_covariate.ipynb
"""
import nbformat as nbf

from lc.paths import REPO

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# A donor covariate from Unstim features: different between donors, the same within a donor

**Question (2026-10-08).** `well_consistency.ipynb` showed that a donor's population frequencies barely move between
its wells (±7 %) while donors differ a lot (±90 %). So frequencies can be a **donor signature**, e.g. the patient
covariate of a conditional map. Can we compress them into a few numbers (like a PCA), chosen so that they
**separate donors as much as possible** and **change as little as possible between a donor's Unstim and stim wells**?

**Method, in plain terms.**
- *Ordinary PCA* finds the directions along which the data vary most, whatever the reason. Here that mixes "donors
  differ" with "the same donor differs between wells".
- *What we want* is the direction with the largest ratio **(variance between donors) / (variance within a donor)**.
  That is a Fisher discriminant (LDA) with each donor as a class. Each axis is a weighted combination of features.
- With 15 training donors and 28 features, pure LDA can overfit, so the within-donor part is regularised by
  `gamma` (`W + gamma·I`). Small `gamma` = pure between/within; very large `gamma` = a PCA of the donor means, which
  ignores the within-donor part. The whole path is tested.

**Data.** For each of the 19 atlas donors × 96 wells (`work/heldout/pheno`): log frequency of the 28 populations
(`freq`), and in a second set also the median of the 31 markers that the OOL table has, per population (`freq+q50`,
896 features). Each well's shared offset (mean over the training donors) is removed first, so "within donor" means the
donor's wells disagreeing once the stim / drug / plate effects shared by all donors are taken out. Fit on the main set
(Unstim, GMCSF, IFNa, IL-2/4/6, IL33, TNFa × 12 treatments). **LPS and PI are never used in the fit**: they are the test
of whether the covariate survives conditions it has not seen.

**Test on held-out donors.** The 5 learning-curve repeats (15 training donors, 4 test donors). The axes are learnt on the
15. For each test donor, each of its stim wells is matched to the nearest of the **19 donor signatures**, where a
signature is **the donor's single Unstim DMSO2 well** (an OOL woman also has one Unstim well). The score is the share of
stim wells matched to the right donor; chance is 1/19 = 5 %. Distances are measured in units of the training
within-donor spread of each axis. Code: `heldout/donor_covariate.py` → `work/heldout/donor_covariate/`.
""")

code(r"""
import sys; sys.path.insert(0, "/scratch/users/bmerwan/atlas_heldout")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display
from lc.paths import REPO
from heldout.common import WORK
from heldout.donor_covariate import atlas_features, OOL_TABLE
pd.set_option("display.width", 220)
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8a85"
D = WORK / "donor_covariate"
E = pd.read_csv(D / "eval.csv")
E["method_label"] = np.where(E.method == "pca_unstim", "PCA on Unstim wells",
                    np.where(E.method == "all_features", "all features (no compression)",
                             "LDA gamma=" + E.gamma.map(lambda g: f"{g:g}" if pd.notna(g) else "")))
print(E.groupby(["features", "method_label"]).size().rename("rows").to_string())
""")

md(r"""
## 1. Can a held-out donor be recognised from its Unstim signature?

Share of a test donor's stim wells matched to the right donor among 19, mean over the 5 repeats × 4 test donors.
Columns: number of axes kept (K) and which stim wells are matched (main-set stims, LPS, PI).
""")
code(r"""
def table(fs):
    t = E[(E.features == fs) & (E.method != "all_features")].pivot_table(index="method_label", columns=["K", "evalset"],
                                                                         values="ident_1well")
    t = t.reindex(columns=["main", "LPS", "PI"], level=1)
    order = ["PCA on Unstim wells"] + [f"LDA gamma={g}" for g in ["10000", "10", "1", "0.1", "0.01", "0.001"]]
    return (100 * t.reindex(order)).round(0).astype(int)
for fs in ["freq", "freq+q50"]:
    display(table(fs).style.set_caption(f"{fs}: % of test-donor stim wells recognised (chance 5 %)"))
ref = E[E.method == "all_features"].pivot_table(index="features", columns="evalset", values="ident_1well")[["main", "LPS", "PI"]]
display((100 * ref).round(0).astype(int).style.set_caption("reference: all standardised features, no compression"))
""")
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
sel = E[E.features == "freq"]
for ax, ev, title in zip(axes, ["main", "LPS", "PI"], ["main-set stims", "LPS (not in fit)", "PI (not in fit)"]):
    for lab, col in [("LDA gamma=0.01", BLUE), ("PCA on Unstim wells", ORANGE)]:
        s = sel[(sel.method_label == lab) & (sel.evalset == ev)].groupby("K").ident_1well.mean() * 100
        ax.plot(s.index, s.values, marker="o", color=col, lw=2, label="between/within (LDA)" if "LDA" in lab else "ordinary PCA")
    ax.axhline(100 / 19, color=GREY, lw=1, ls="--"); ax.text(5, 100 / 19 + 2, "chance", color=GREY, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10); ax.set_xlabel("number of axes kept"); ax.set_xticks([1, 2, 3, 5])
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#eeeeea")
axes[0].set_ylabel("% stim wells matched to the right donor"); axes[0].set_ylim(0, 105)
axes[0].legend(frameon=False, fontsize=8, loc="center right")
fig.suptitle("Recognising a held-out donor from one Unstim well (population frequencies only)", fontsize=11)
plt.tight_layout(); fig.savefig(REPO / "notebooks/fig_donor_covariate_ident.png", dpi=150); plt.show()
""")
md(r"""
Reading (frequencies only; the numbers are recomputed above):
- **Donors are very recognisable.** Using all 28 frequencies with no compression, 100 % of main-set and LPS stim wells
  and 93 % of PI wells are matched to the right donor among 19, from a single Unstim well.
- **When the covariate is compressed to 2–3 numbers, the between/within criterion matters.** With 2 axes, LDA
  (`gamma` 0.001–0.01) recognises 98 % (main), 97–98 % (LPS) and 62–72 % (PI), against 81 % / 57 % / 32 % for an
  ordinary PCA of the Unstim wells. With 3 axes: 100 / 100 / 81–84 % vs 97 / 84 / 52 %.
- **Little regularisation is best** for the 28 frequencies (`gamma` 0.001–0.01); the large-`gamma` end, which ignores
  the within-donor part, behaves like PCA.
- **Adding marker medians (`freq+q50`) helps only under PI** (3 axes: 91 % vs 84 %), and marker medians are where the
  OOL cohort is off the atlas scale (`atlas_vs_ool.ipynb`). Frequencies alone are the better choice.
- PI is the hardest case, because it reshuffles populations (`phenotype_shifts.ipynb`), but the covariate still
  recognises most donors from their PI wells although PI was never in the fit.

## 2. What the axes are (final fit on all 19 donors, frequencies, `gamma` = 0.01, 3 axes)

Each number is the correlation, over the 19 donors, between the donor's axis score and a population's log frequency
(Unstim wells). Signs are set so that + means more granulocytes on each axis.
""")
code(r"""
L = pd.read_csv(D / "final_freq_g0.01_loadings.csv", index_col=0); L.index = L.index.str.replace("__freq", "")
A = pd.read_csv(D / "final_freq_g0.01_atlas_scores.csv", index_col=0)
for a in L.columns:
    s = L[a].sort_values()
    print(f"{a}:  low = " + ", ".join(f"{i} {v:+.2f}" for i, v in s.head(4).items())
          + "   |   high = " + ", ".join(f"{i} {v:+.2f}" for i, v in s.tail(4)[::-1].items()))
print("\nmean score by cohort (in units of the within-donor spread):")
display(A.groupby(A.index.str[:2]).mean().round(1))
""")
md(r"""
Reading: the axes are readable immune-composition contrasts.
- **Axis 1**: naive CD4 T (and CD4 effector / Treg) against double-negative T cells and CD8 memory.
- **Axis 2**: memory T cells (CD4/CD8 Tem, CD4 Tcm) against naive CD8 T and granulocytes.
- **Axis 3**: granulocytes against NK cells and pDCs.

All three also separate pregnant (PV) from non-pregnant (HV) donors, most clearly axis 3: pregnancy raises
granulocytes. In the atlas, being pregnant is part of a donor's identity. **Within the OOL cohort, where everyone is
pregnant, the axes have to separate women on other grounds** (next section).

## 3. Does it carry over to the OOL women?

The OOL table has the same 28 populations (frequencies sum to 1 over them), but a different protocol. First, how far
apart the two cohorts are:
""")
code(r"""
X = atlas_features(); fr = [c for c in X.columns if c.endswith("__freq")]
A0 = X.loc["Unstim", fr].groupby(level="donor").mean()
O0 = pd.read_csv(OOL_TABLE, usecols=["Individual"] + fr).set_index("Individual") * np.log(10)
PV = A0[A0.index.str.startswith("PV")]
C = pd.DataFrame({"OOL vs atlas PV donors (x)": np.exp(O0.mean() - PV.mean()),
                  "spread between OOL women / between atlas donors": O0.std() / A0.std()})
C.index = C.index.str.replace("__freq", "")
display(C.sort_values("OOL vs atlas PV donors (x)").round(2).T)
""")
md(r"""
Reading: even against the atlas's pregnant donors, OOL has very different levels for some populations (e.g. CD8 Th1
and Tregs several-fold lower, mDCs and ncMCs higher). That is gating / protocol, not biology we can use. The covariate
therefore **centres the OOL women on the OOL cohort mean** and uses the atlas weights: only a woman's deviation from
the other OOL women enters. That removes level offsets, not differences in how gates behave.

The test: OOL women have 2–3 visits (G1/G2/G3, weeks apart). A good donor signature should stay with the woman.
""")
code(r"""
rows = []
for name, lab in [("freq_g0.01", "LDA gamma=0.01"), ("freq_pca", "PCA on Unstim wells")]:
    O = pd.read_csv(D / f"final_{name}_ool_scores.csv", index_col=0); Aa = pd.read_csv(D / f"final_{name}_atlas_scores.csv", index_col=0)
    w = O.index.str.split("_").str[0]; v = O.index.str.split("_").str[1]
    multi = pd.Series(w).map(pd.Series(w).value_counts()).values > 1
    Om, wm = O[multi], w[multi]
    bv = Om.groupby(wm).mean().var(); wv = (Om - Om.groupby(wm).transform("mean")).var()
    g1 = O[v == "G1"]; g1.index = w[v == "G1"]; later = O[(v != "G1") & pd.Index(w).isin(g1.index)]
    lw = later.index.str.split("_").str[0]
    Dm = ((later.values[:, None, :] - g1.values[None]) ** 2).sum(-1)
    rows.append({"covariate (3 axes)": lab,
                 **{f"ICC across visits, {a}": round(float(bv[a] / (bv[a] + wv[a])), 2) for a in O.columns},
                 "later visit matched to own G1 (%)": round(100 * float(np.mean(g1.index.values[Dm.argmin(1)] == lw.values)), 0),
                 "women": len(g1), "later visits": len(later),
                 "OOL spread / atlas spread": round(float((O.std() / Aa.std()).mean()), 2)})
R = pd.DataFrame(rows).set_index("covariate (3 axes)"); display(R.T)
print("chance for matching: %.1f %%" % (100 / R.women.iloc[0]))
""")
md(r"""
Reading: **the covariate stays with the woman across pregnancy visits.** On the LDA axes, 92–95 % of the variance in OOL
scores is between women (ICC), and a woman's G2/G3 visit is matched to her own G1 among 63 women 59 % of the time
(chance 1.6 %), against 23 % for the PCA covariate, whose first axis (granulocytes vs lymphocytes) varies within a woman
during pregnancy (ICC 0.66). That is an independent check: other people, another protocol, and visits weeks apart, so
some of the remaining within-woman variation is real change with gestational age. OOL women spread over about
two-thirds of the atlas range.

## 4. Summary (2026-10-08)

- **A 2–3-number donor covariate built on (between-donor variance) / (within-donor variance) recognises held-out donors
  from a single Unstim well**: 98 % of main-set stim wells with 2 axes, 100 % with 3, and it survives conditions it never
  saw (LPS 97–100 %, PI 62–84 %). An ordinary PCA of the same frequencies needs more axes and fails under PI (32–52 %).
- **Population frequencies alone are enough**; marker medians add little and carry the atlas–OOL offset.
- **The axes are readable**: naive CD4 vs DN T / CD8 memory; memory T vs naive CD8 / granulocytes; granulocytes vs
  NK / pDCs. All three also separate HV from PV in the atlas.
- **It transfers to OOL**: women keep their covariate across visits (ICC 0.92–0.95; 59 % later-visit identification vs
  1.6 % chance), after centring OOL on its own cohort mean (levels differ by protocol / gating).
- **Not shown here: that the covariate predicts anything about responses.** It is a stable identity signature. Whether
  donors close on it respond alike is the conditional-map experiment (`README.md`, "Next experiment"); the learned
  encoders tried on 2026-10-06 did not help, and this covariate is the obvious next candidate.

Files: `work/heldout/donor_covariate/eval.csv` (held-out test), `final_freq_g0.01_{loadings,atlas_scores,ool_scores}.csv`
(the covariate to use; scores are in units of the within-donor spread), `final_freq_pca_*` (PCA baseline).
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
nbf.write(nb, REPO / "notebooks/donor_covariate.ipynb"); print("wrote notebooks/donor_covariate.ipynb")
