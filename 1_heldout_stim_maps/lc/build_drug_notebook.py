"""Write notebooks/drug_heldout.ipynb: predicting a new donor's drug effects from its Unstim profile (LODO, TabPFN).
Reads work/lc/drug_tab/score/ (lc/drug_score.py); every number in the text is computed.
    python -m lc.build_drug_notebook && jupyter nbconvert --to notebook --execute --inplace notebooks/drug_heldout.ipynb
    (env_cpu.sh)
"""
import nbformat as nbf

from lc.paths import REPO

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Drug arm: predicting a new donor's drug effects from its Unstim profile

**Data.** The paper's drug arm: batch-corrected drug-screen cells (`work/heldout/cache_drug`), 9 drugs (SA, RIF, SALPZ,
CHT, THF, LPZ, MAP, PRA, MF). Each drug is measured under a stim and compared with **its plate DMSO control** under the
same stim (DMSO1: SA, LPZ; DMSO2: CHT, MAP, MF, PRA; DMSO3: RIF, SALPZ, THF).

For a donor, stim *s*, celltype *c*, quantile *q* (q50 / q75 / q90) and marker:
- **D** = quantile of the drug + stim cells;
- **C** = quantile of the control + stim cells;
- **drug effect E = D − C**.

**Task.** For a donor never seen in training, predict E (and D) from the donor's **Unstim** profile only.
- Leave-one-donor-out: 19 folds, 18 training donors each.
- PI and IL33 left out.
- Scored on the **selected features** (`meta/feature_set.csv`, the same stim features as the stim work) × the 9 drugs.

**Methods.** TabPFN on the compact feature set of the stim work (72 columns), with one table per (fold, stim, celltype,
drug).
- **E0, mean training effect:** the drug's average effect in the 18 training donors. This is the drug-arm analogue of
  the shift: no donor-specific information.
- **Strategy A, direct:** predict D and C separately from Unstim; E_A = D̂ − Ĉ.
- **Strategy B, two-step:** predict C from Unstim, then predict E from Unstim plus the donor's control profile.
  - Training donors use their measured control; the test donor uses its predicted control.
  - Drug + stim is then D_B = Ĉ + Ê.
- **Drug + stim baselines:** D0 = mean training D + the donor's Unstim deviation (drug "shift"), and D_C0E = Ĉ + E0.

**Metrics.**
- MSE (main) and MAE, per donor, with 95 % CI over donors.
- **Share of the mean-effect error removed** = 1 − MSE / MSE(E0): how much of the donor-specific drug response is
  predicted.
- **between_R2**: whether predicted differences between donors in the drug effect follow the real ones.
- **Noise floor of E**: split-half sampling noise of D and C.
- Two case sets: all selected features, and the "real effects" (|E| > 0.0273 and > 2× its split-half noise).
""")

code(r"""
import sys; sys.path.insert(0, "/scratch/users/bmerwan/atlas_heldout")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, Markdown
from lc.paths import WORK
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
SC = WORK / "drug_tab" / "score"
M = pd.read_csv(SC / "metrics.csv"); B = pd.read_csv(SC / "by_drug_stim.csv"); R = pd.read_pickle(SC / "cases.pkl")
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, REF = "#0b0b0b", "#52514e", "#e6e5e0", "#8a8a85"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "legend.frameon": False,
                     "axes.titlesize": 11, "axes.titleweight": "bold"})
LAB = {"E0": "mean training effect (E0)", "E_A": "A: direct (D̂ − Ĉ)", "E_B": "B: two-step (Unstim + predicted control)",
       "D0": "drug shift (D0)", "D_hat": "A: direct D̂", "D_B": "B: Ĉ + Ê", "D_C0E": "Ĉ + mean effect", "C0": "control shift", "C_hat": "Ĉ"}
COL = {"E0": REF, "E_A": S2, "E_B": S1}
print(f"{len(R)} selected cases ({int(R.real.sum())} real effects), {R.donor.nunique()} donors, {R.drug.nunique()} drugs, "
      f"{R.stim.nunique()} stims, {R.celltype.nunique()} celltypes")
""")

md(r"""
## 1. Drug effect: does the donor's Unstim profile predict its drug response?
""")

code(r"""
fig, axs = plt.subplots(1, 2, figsize=(12, 3.4))
for ax, cs in zip(axs, ["selected", "selected, real effect"]):
    g = M[(M.set == cs) & (M.target == "E")].set_index("method").loc[["E0", "E_A", "E_B"]]; y = np.arange(3)[::-1]
    for yi, (m, r) in zip(y, g.iterrows()):
        ax.hlines(yi, r.MSE_lo, r.MSE_hi, color=COL[m], lw=1.5, alpha=0.6); ax.plot(r.MSE, yi, "o", color=COL[m], ms=9)
    fl = g.floor_E_MSE.iloc[0]; ax.axvline(fl, color=INK2, ls="--", lw=1); ax.text(fl, -0.6, " noise floor", fontsize=8, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([LAB[m] for m in g.index]); ax.set_xlim(left=0); ax.grid(axis="y", visible=False)
    ax.set_title(f"drug-effect MSE per donor — {cs}", loc="left"); ax.set_xlabel("MSE")
plt.tight_layout(); plt.show()
for cs in ["selected", "selected, real effect"]:
    print(cs); display(M[(M.set == cs) & (M.target == "E")].set_index("method")[["MSE", "MSE_lo", "MSE_hi", "MAE", "share_mean_effect_err_removed",
                       "between_R2", "better_than_ref", "p_vs_ref", "floor_E_MSE"]].round(5))
e = M[(M.set == "selected") & (M.target == "E")].set_index("method"); er = M[(M.set == "selected, real effect") & (M.target == "E")].set_index("method")
bm = e.drop(index="E0").MSE.idxmin()
display(Markdown(f'''**Reading.**
- Best strategy for the drug effect: **{LAB[bm]}**, MSE {e.loc[bm,"MSE"]:.5f} vs {e.loc["E0","MSE"]:.5f} for the mean training effect.
  It removes {100*e.loc[bm,"share_mean_effect_err_removed"]:.0f} % of the mean-effect error and is better for {int(e.loc[bm,"better_than_ref"])} of 19 donors (p = {e.loc[bm,"p_vs_ref"]:.2g}).
- On real effects only: {100*er.loc[bm,"share_mean_effect_err_removed"]:.0f} % removed, between-donor R² {er.loc[bm,"between_R2"]:.3f} (E0: {er.loc["E0","between_R2"]:.3f}).
- Noise floor of the drug effect (sampling only): {e.loc["E0","floor_E_MSE"]:.5f}, i.e. {100*e.loc["E0","floor_E_MSE"]/e.loc[bm,"MSE"]:.0f} % of the best model's error.'''))
""")

md(r"""
## 2. By drug and by stim
""")

code(r"""
for col in ["drug", "stim"]:
    t = B[(B.set == "selected") & (B.by == col) & (B.target == "E")].pivot(index="value", columns="method", values="MSE")
    t = t[["E0", "E_A", "E_B"]]; t["B vs E0 %"] = 100 * (t.E_B / t.E0 - 1); t["A vs E0 %"] = 100 * (t.E_A / t.E0 - 1)
    fig, ax = plt.subplots(figsize=(9, 0.35 * len(t) + 1.2)); y = np.arange(len(t))[::-1]
    for m, dy in zip(["E0", "E_A", "E_B"], [0.2, 0, -0.2]): ax.plot(t[m], y + dy, "o", color=COL[m], ms=7, label=LAB[m])
    ax.set_yticks(y); ax.set_yticklabels(t.index); ax.grid(axis="y", visible=False); ax.set_xlabel("drug-effect MSE (selected features)")
    ax.legend(fontsize=8, loc="lower right"); ax.set_title(f"by {col}", loc="left"); plt.tight_layout(); plt.show()
    display(t.round(5))
""")

md(r"""
## 3. Drug + stim quantile, and the control

How well the absolute drug + stim quantile is predicted, for comparison with the stim models. The control part shows how
much the batch-corrected drug-arm data differ from the stim inputs: the plain shift is poor, while TabPFN corrects it.
""")

code(r"""
for cs in ["selected", "selected, real effect"]:
    print(cs); display(M[(M.set == cs) & M.target.isin(["D", "C"])].set_index(["target", "method"])[["MSE", "MSE_lo", "MSE_hi", "MAE",
                       "between_R2", "better_than_ref", "ref", "p_vs_ref"]].round(5))
""")

md(r"""
## 4. With the same information as CellOT: the donor's measured control ("step")

In simco's `drug_map_choice` notebook, the CellOT drug maps were tested in a "step" setting: applied to the donor's
**measured control cells** (stim + the drug's DMSO plate). Here TabPFN gets the same information. For the test donor it
uses the measured control profile instead of a predicted one:
- **E_S**: control deviation at the row's quantile × marker, plus per-marker mean and spread.
- **E_S65**: E_S plus the whole control profile, 13 markers × 5 quantiles.

Compared on the same cases with the mean training effect (E0) and the two-step model with a predicted control (E_B).
dev_R2 = 1 − MSE / MSE(E0).
""")

code(r"""
SS = WORK / "drug_tab" / "step_score"
if (SS / "metrics.csv").exists():
    T = pd.read_csv(SS / "metrics.csv"); PF = pd.read_csv(SS / "per_feature.csv")
    LB = {"E0": "mean training effect (E0)", "E_B": "two-step, predicted control (E_B)", "E_S": "step, measured control (E_S)",
          "E_S65": "step, measured control + full profile (E_S65)"}
    CL = {"E0": REF, "E_B": S1, "E_S": S2, "E_S65": S3}
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.4))
    for ax, cs in zip(axs, ["selected", "selected, real effect"]):
        g = T[T.set == cs].set_index("method"); y = np.arange(len(g))[::-1]
        for yi, (m, r) in zip(y, g.iterrows()):
            ax.hlines(yi, r.MSE_lo, r.MSE_hi, color=CL[m], lw=1.5, alpha=0.6); ax.plot(r.MSE, yi, "o", color=CL[m], ms=9)
        ax.axvline(g.floor_E_MSE.iloc[0], color=INK2, ls="--", lw=1); ax.set_yticks(y); ax.set_yticklabels([LB[m] for m in g.index])
        ax.set_xlim(left=0); ax.grid(axis="y", visible=False); ax.set_title(f"drug-effect MSE per donor — {cs}", loc="left")
    plt.tight_layout(); plt.show()
    for cs in ["selected", "selected, real effect"]:
        print(cs); display(T[T.set == cs].set_index("method")[["MSE", "MSE_lo", "MSE_hi", "MAE", "dev_R2", "between_R2", "better_than_E0", "p_vs_E0"]].round(5))
    print("per-feature dev_R2 on real-effect features (with shuffled-prediction nulls)")
    display(PF.drop(columns=["drug", "stim", "celltype", "q", "marker"]).describe(percentiles=[.5, .9]).round(3))
    gs = T[T.set == "selected"].set_index("method"); best = gs.drop(index="E0").MSE.idxmin()
    display(Markdown(f'''**Reading.** Best with the control known: **{LB[best]}**, MSE {gs.loc[best,"MSE"]:.5f} vs {gs.loc["E0","MSE"]:.5f} for the mean effect
(dev_R2 {gs.loc[best,"dev_R2"]:.3f}), better for {int(gs.loc[best,"better_than_E0"])} of 19 donors (p = {gs.loc[best,"p_vs_E0"]:.2g}).
Between-donor R² {gs.loc[best,"between_R2"]:.3f} (E0 {gs.loc["E0","between_R2"]:.3f}; −0.114 is what a constant leave-one-out mean gives).'''))
else:
    print("step results not available yet:", SS)
""")

md(r"""
## 5. Summary
""")

code(r"""
d = M[(M.set == "selected") & (M.target == "D")].set_index("method"); bd = d.MSE.idxmin()
display(Markdown(f'''- **Drug effect:** {LAB[bm]} is the best predictor (MSE {e.loc[bm,"MSE"]:.5f} vs {e.loc["E0","MSE"]:.5f} for the mean effect, {100*e.loc[bm,"share_mean_effect_err_removed"]:.0f} % of its error removed).
- **Drug + stim quantile:** best {LAB[bd]}, MSE {d.loc[bd,"MSE"]:.4f}, MAE {d.loc[bd,"MAE"]:.4f}.
- Drug effects are small relative to stim effects, so the drug-effect error is close to its noise floor ({e.loc["E0","floor_E_MSE"]:.5f}). Most of the drug + stim error comes from predicting the stim response itself.'''))
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = REPO / "notebooks" / "drug_heldout.ipynb"; nbf.write(nb, out); print(f"[DONE ] -> {out}")
