"""Build notebooks/where_cellot_skill.ipynb: where are the CellOT stim maps good or bad on NEW donors, and why the plateau.

Held-out donors only (learning-curve design: 5 repeats, 4 test donors, maps trained on k donors). Input:
WORK/where_skill/cases.pkl from lc/where_skill.py; population sizes from work/heldout/pheno. Readings and the answer
live in lc/where_skill_readings.md and are merged in at build time.
    python -m lc.where_skill && python -m lc.build_where_skill_notebook && \
    jupyter nbconvert --to notebook --execute --inplace notebooks/where_cellot_skill.ipynb
"""
from pathlib import Path

import nbformat as nbf

cells = []
md = lambda t: cells.append(nbf.v4.new_markdown_cell(t.strip()))
code = lambda t: cells.append(nbf.v4.new_code_cell(t.strip()))

md(r"""
# Where is CellOT good on new donors, and why the plateau? (stim maps, held-out donors only)

**Question (2026-10-06).** The learning curve plateaus at a median skill of ~0.6 on held-out donors, against a
split-half ceiling of ~0.87. Is there any stim, celltype or marker where CellOT predicts a *new* donor clearly better
than the plateau, especially on the **patient-specific** part of the response (the shared part can be fixed by a
delta shift)? Where is it weak, and is that biology, noise, or out-of-distribution donors? What would get past the
plateau, or why is it not possible?

Only held-out donors are scored: every map is trained on k donors and scored on 4 others.
""")
md("<!-- ANSWER -->")
md(r"""
## 1. Definitions

For a test donor, one (stim, celltype, marker, quantile q):
- *u* = her Unstim quantile, *t* = her stimulated quantile (the truth), response = t − u.
- **shift** baseline = training-mean stimulated value + (u − training-mean Unstim value): every donor gets the
  **average response** of the training donors on top of her own Unstim value. No network, no patient-specific response.
- **Donor-specific response** r = t − shift: how much her response differs from the training average.
- Over the 4 test donors of a repeat, r = **shared part** (their mean: how far these 4 donors' average response is
  from the training average) + **patient-specific part** rc (each donor's deviation from that mean).
- A map's own donor-specific prediction is (prediction − shift), split the same way.
- **specific R²** = 1 − Σ(predicted rc − true rc)² / Σ(true rc)²: the share of the real between-donor differences in
  response that the map captures. 0 = no better than giving every donor the same response; negative = worse.
- **ceiling** = 1 − (sampling noise) / Σ rc², with the noise from split halves of her stim and Unstim cells. It is the
  best reachable specific R², so it says whether the between-donor differences are real or noise.
- **corr** = correlation of predicted and true rc (direction); **SD ratio** = how much the map varies between donors
  relative to the truth (amplitude).
- **skill** (the learning-curve readout) = 1 − |prediction − t| / |t − u|, median per repeat, mean over repeats.
Cases: effect |t − u| > 0.0273 (the technical floor) and at least 3 test donors in the group.
""")
code(r"""
import warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 120)
WORK = Path("/scratch/users/bmerwan/atlas_heldout/work")
FIG = WORK / "lc/where_skill"
FLOOR = 0.0273
D = pd.read_pickle(WORK / "lc/where_skill/cases.pkl")
D = D[(D.t - D.u).abs() > FLOOR].copy()
GRP = ["repeat", "stim", "celltype", "k", "q", "marker"]
D["n"] = D.groupby(GRP).t.transform("size"); D = D[D.n >= 3].copy()
D["r"] = D.t - D["shift"]; D["rbar"] = D.groupby(GRP).r.transform("mean"); D["rc"] = D.r - D.rbar
D["nv"] = (D.noise ** 2 + D.noise_u ** 2) / 4 * (D.n - 1) / D.n
METH = ["CellOT", "gauss_ot", "quantile_map"]
for m in METH:
    rh = D[m] - D["shift"]
    D["hb_" + m] = rh.groupby([D[c] for c in GRP]).transform("mean"); D["h_" + m] = rh - D["hb_" + m]
D15 = D[D.k == 15].copy()
print(f"{len(D):,} cases over k = 3/5/8/15; at k = 15: {len(D15):,} cases, {D15.donor.nunique()} distinct test donors, "
      f"{D15.groupby(['stim', 'celltype']).ngroups} (stim, celltype) maps")

def spec(g, m="CellOT"):
    sst = (g.rc ** 2).sum()
    return pd.Series({"specific R²": 1 - ((g["h_" + m] - g.rc) ** 2).sum() / sst, "ceiling": 1 - g.nv.sum() / sst,
                      "corr": np.corrcoef(g["h_" + m], g.rc)[0, 1], "SD ratio": np.sqrt((g["h_" + m] ** 2).sum() / sst),
                      "cases": len(g)})

def skill(p, g): return (1 - (p - g.t).abs() / (g.t - g.u).abs()).groupby(g.repeat).median().mean()

def table(g, by, extra=True):
    T = g.groupby(by).apply(spec)
    if extra:
        T["gauss_ot specific R²"] = g.groupby(by).apply(lambda x: spec(x, "gauss_ot")["specific R²"])
        R = g.groupby(by + ["repeat"] if isinstance(by, list) else [by, "repeat"]).apply(lambda x: spec(x)["specific R²"])
        T["repeats > 0 (of 5)"] = (R > 0).groupby(level=list(range(R.index.nlevels - 1))).sum()
    return T
""")

md(r"""
## 2. The plateau: how big is the patient-specific part, and how much of it is noise?

Error budget at k = 15 (mean squared error per case), and the learning curve split into its two parts.
""")
code(r"""
g = D15
B = pd.DataFrame({
    "shift": {"shared part": (g.rbar ** 2).mean(), "patient-specific part": (g.rc ** 2).mean()},
    "CellOT": {"shared part": ((g.hb_CellOT - g.rbar) ** 2).mean(), "patient-specific part": ((g.h_CellOT - g.rc) ** 2).mean()},
    "gauss_ot": {"shared part": ((g.hb_gauss_ot - g.rbar) ** 2).mean(), "patient-specific part": ((g.h_gauss_ot - g.rc) ** 2).mean()}}).T
B["total"] = B.sum(1)
B.loc["sampling noise (in the truth)"] = [np.nan, g.nv.mean(), g.nv.mean()]
display(B.round(4))
print("overall, k = 15:"); display(pd.DataFrame({m: spec(g, m) for m in METH}).T.round(3))
LC = []
for k, x in D.groupby("k"):
    LC.append({"k": k, "skill shift": skill(x["shift"], x), "skill CellOT": skill(x.CellOT, x),
               "skill gauss_ot": skill(x.gauss_ot, x),
               "rms shared error shift": np.sqrt((x.rbar ** 2).mean()),
               "rms shared error CellOT": np.sqrt(((x.hb_CellOT - x.rbar) ** 2).mean()),
               "specific R² CellOT": spec(x)["specific R²"], "specific R² gauss_ot": spec(x, "gauss_ot")["specific R²"]})
LC = pd.DataFrame(LC).set_index("k"); display(LC.round(3))
""")
md("<!-- READ2 -->")

md(r"""
## 3. Where CellOT captures the patient-specific response (k = 15)

By stim, quantile, marker and celltype. *repeats > 0* = in how many of the 5 repeats the specific R² is positive.
""")
code(r"""
for by in ["stim", "q", "marker", "celltype"]:
    display(table(D15, by).sort_values("specific R²", ascending=False).round(3))
""")
code(r"""
ST = table(D15, ["stim", "celltype"]); SM = table(D15, ["stim", "marker"], extra=False)
fig, axs = plt.subplots(1, 2, figsize=(14, 8.5), gridspec_kw={"width_ratios": [7, 7]})
for ax, T, rows, title in [(axs[0], ST, "celltype", "stim × celltype"), (axs[1], SM, "marker", "stim × marker")]:
    X = T["specific R²"].unstack(0)
    if rows == "celltype": X = X.loc[X.mean(1).sort_values().index]
    else: X = X.loc[X.mean(1).sort_values().index]
    im = ax.imshow(X.values, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(X.shape[1])); ax.set_xticklabels(X.columns)
    ax.set_yticks(range(X.shape[0])); ax.set_yticklabels(X.index, fontsize=8); ax.set_title(title, fontsize=10)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            v = X.values[i, j]
            if np.isfinite(v) and abs(v) >= 0.25: ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                                                          color="white" if abs(v) > 0.4 else "black")
cb = fig.colorbar(im, ax=axs, shrink=0.6)
cb.set_label("specific R² of CellOT on new donors\n(red: captures real between-donor differences; blue: worse than shift)")
fig.suptitle("Patient-specific part of the stim response, held-out donors, k = 15 (blank = pair not in the grid)", fontsize=10)
fig.savefig(FIG / "specific_heatmaps.png", dpi=160, bbox_inches="tight")
""")
code(r"""
print("best (stim, celltype) maps"); display(ST.sort_values("specific R²", ascending=False).head(15).round(3))
print("worst (stim, celltype) maps"); display(ST.sort_values("specific R²").head(15).round(3))
SCM = D15.groupby(["stim", "celltype", "marker"]).apply(spec)
SCM = SCM[SCM.cases >= 40]
print("best (stim, celltype, marker), >= 40 cases"); display(SCM.sort_values("specific R²", ascending=False).head(20).round(3))
""")
md("<!-- READ3 -->")

md(r"""
## 4. Why is it weak where it is weak?

Candidate explanations, each checked against the specific R²:
1. **Noise**: the ceiling (is there anything real to predict?) and the number of cells.
2. **Out of distribution**: |z| = how far the test donor's Unstim value is from the training donors, in training SDs.
3. **Gating**: populations whose size changes under the stim (the "response" partly becomes a change of who is in
   the gate). Population size ratio stim / Unstim from `work/heldout/pheno` (plate controls, median over donors).
4. **Cohort**: pregnant (PV) vs non-pregnant (HV) test donors.
5. **Effect size**: how big the average response is.
""")
code(r"""
x = D15.copy()
x["|z| of test donor Unstim"] = pd.cut(((x.u - x.mu_u) / x.sd_u.replace(0, np.nan)).abs(), [0, 0.5, 1, 2, 3, np.inf])
x["cells (stim well)"] = pd.qcut(x.n_s, 4)
x["cohort"] = np.where(x.donor.str.startswith("PV"), "PV (pregnant)", "HV")
x["mean response"] = pd.cut((x.mu_s - x.mu_u).abs(), [0, 0.05, 0.1, 0.2, 0.5, np.inf])
for by in ["|z| of test donor Unstim", "cells (stim well)", "cohort", "mean response"]:
    T = x.groupby(by).apply(spec)
    T["CellOT closer than shift"] = x.groupby(by).apply(lambda g: ((g.h_CellOT - g.rc).abs() < g.rc.abs()).mean())
    display(T.round(3))
""")
code(r"""
# population size change under each stim (plate-control wells, same donor and plate)
P = pd.concat([pd.read_csv(f, usecols=["celltype", "stim", "treatment", "donor", "n_cells"]).drop_duplicates()
               for f in sorted((WORK / "heldout/pheno").glob("*.csv.gz"))])
P = P[P.treatment.str.startswith("DMSO")]
P["freq"] = P.n_cells / P.groupby(["stim", "treatment", "donor"]).n_cells.transform("sum")
Pu = P[P.stim == "Unstim"].set_index(["celltype", "treatment", "donor"]).freq
Ps = P[P.stim != "Unstim"].copy()
Ps["l2"] = np.log2(Ps.freq.to_numpy() / Pu.reindex(pd.MultiIndex.from_frame(Ps[["celltype", "treatment", "donor"]])).to_numpy())
SZ = Ps.replace([np.inf, -np.inf], np.nan).groupby(["celltype", "stim", "donor"]).l2.mean().groupby(["stim", "celltype"]).median()
ST2 = ST.join(SZ.rename("log2 size ratio")).dropna(subset=["log2 size ratio"])
ST2["size change"] = pd.cut(ST2["log2 size ratio"].abs(), [-0.01, 0.25, 0.5, np.inf],
                            labels=["< 1.2×", "1.2–1.4×", "≥ 1.4×"])
print("(stim, celltype) maps by how much the population changes size under the stim")
display(ST2.groupby("size change").agg(maps=("cases", "size"), median_specific_R2=("specific R²", "median"),
                                       mean_corr=("corr", "mean")).round(3))
display(ST2[ST2["size change"] == "≥ 1.4×"].sort_values("specific R²")[["specific R²", "corr", "log2 size ratio"]].round(2))
""")
code(r"""
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
T = x.groupby("|z| of test donor Unstim").apply(spec)
axs[0].bar(range(len(T)), T["specific R²"], color=["#d62728" if v > 0 else "#1f77b4" for v in T["specific R²"]])
axs[0].set_xticks(range(len(T))); axs[0].set_xticklabels([str(i) for i in T.index], fontsize=8)
axs[0].axhline(0, c="k", lw=.8); axs[0].set_xlabel("|z|: test donor's Unstim value vs training donors (SDs)")
axs[0].set_ylabel("specific R² of CellOT"); axs[0].set_title("Out-of-range donors", fontsize=10); axs[0].grid(axis="y", alpha=.3)
for lab, c in [("< 1.2×", "#2ca02c"), ("1.2–1.4×", "#ff7f0e"), ("≥ 1.4×", "#9467bd")]:
    s = ST2[ST2["size change"] == lab]
    axs[1].scatter(s["log2 size ratio"], s["specific R²"].clip(-1, 1), s=14, c=c, label=f"size change {lab}")
axs[1].axhline(0, c="k", lw=.8); axs[1].set_xlabel("log2 population size, stim / Unstim")
axs[1].set_ylabel("specific R² (clipped at −1)"); axs[1].set_title("One dot per (stim, celltype) map", fontsize=10)
axs[1].legend(fontsize=7, frameon=False); axs[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(FIG / "why_weak.png", dpi=160)
""")
md("<!-- READ4 -->")

md(r"""
## 5. What gets past the plateau now: shift for the average, CellOT only for the donor-specific part

Three ways to combine the shift with CellOT, all built from training information or from the test donors' *inputs*
(never their truth):
- **hybrid** = shift + CellOT's donor-specific prediction (CellOT's own average response is replaced by the
  shift's, i.e. the delta correction; the centring uses CellOT's predictions on the 4 test donors, not their truth);
- **gated** = hybrid only for the (stim, celltype, marker) where CellOT's specific R² was positive on the *other four*
  repeats, shift elsewhere (cross-fitted);
- **scaled** = shift + β × CellOT's donor-specific prediction, β (clipped to [0, 1.5]) fitted on the other four
  repeats per (stim, celltype, marker).
""")
code(r"""
g = D15.copy(); g["key"] = list(zip(g.stim, g.celltype, g.marker))
out = []
for r in sorted(g.repeat.unique()):
    tr, te = g[g.repeat != r], g[g.repeat == r].copy()
    st = tr.groupby("key").apply(lambda z: pd.Series({
        "spec_other": 1 - ((z.h_CellOT - z.rc) ** 2).sum() / (z.rc ** 2).sum(),
        "beta": (z.h_CellOT * z.rc).sum() / max((z.h_CellOT ** 2).sum(), 1e-12)}))
    te = te.join(st, on="key"); out.append(te)
O = pd.concat(out)
cands = {"shift": O["shift"], "CellOT": O.CellOT, "gauss_ot": O.gauss_ot, "hybrid": O["shift"] + O.h_CellOT,
         "gated": O["shift"] + np.where(O.spec_other > 0, O.h_CellOT, 0),
         "scaled": O["shift"] + O.beta.clip(0, 1.5).fillna(0) * O.h_CellOT}
sst = (O.rc ** 2).sum(); rows = {}
for n, p in cands.items():
    hc = (p - O["shift"]) - (p - O["shift"]).groupby([O[c] for c in GRP]).transform("mean")
    e = (p - O.t) ** 2
    pdn = e.groupby([O.repeat, O.donor]).mean()
    rows[n] = {"skill": skill(p, O), "specific R²": 1 - ((hc - O.rc) ** 2).sum() / sst, "MSE": e.mean(),
               "MSE vs CellOT": e.mean() / ((O.CellOT - O.t) ** 2).mean() - 1}
    rows[n]["_pdn"] = pdn
R5 = pd.DataFrame({n: {k: v for k, v in d.items() if k != "_pdn"} for n, d in rows.items()}).T
base = rows["shift"]["_pdn"]
R5["donor-repeats better than shift"] = [f"{(rows[n]['_pdn'] < base).sum()}/{len(base)}" for n in rows]
cot = rows["CellOT"]["_pdn"]
R5["donor-repeats better than CellOT"] = [f"{(rows[n]['_pdn'] < cot).sum()}/{len(cot)}" for n in rows]
display(R5.round(4))
print(f"gated: CellOT's donor-specific part is used for {(O.spec_other > 0).mean():.0%} of cases")
print("specific R² of the gated combination by stim:")
pg = cands["gated"]; O["h_gated"] = (pg - O["shift"]) - (pg - O["shift"]).groupby([O[c] for c in GRP]).transform("mean")
display(O.groupby("stim").apply(lambda z: 1 - ((z.h_gated - z.rc) ** 2).sum() / (z.rc ** 2).sum()).round(3).to_frame("gated specific R²"))
""")
md("<!-- READ5 -->")


md(r"""
## 7. Retraining with per-donor normalisation and input clipping (submitted 2026-10-06)

§3–§4 traced the failures to the pooled-OT stretch: a map between pooled donors sends a cell at a given rank of
pooled Unstim to the same rank of pooled stim, so where the stim cloud is wider for reasons unrelated to baseline
(pSTAT5/6, PI, LPS), it turns a donor's baseline offset into an amplified response. Three paired, context-free
CellOT arms (`lc/cond_train.py --ctx none`, k = 15, same training as the paired control `none`) test the fixes:
- **dnorm**: trained on `cache_dnorm` (each donor's cells minus her own Unstim median, `lc/dnorm_cache.py`);
  prediction + her offset;
- **dnorm_clip**: dnorm, and each test cell clamped per marker to the [0.5, 99.5] percentiles of the map's pooled
  training Unstim cells (`--clip-test 0.5`);
- **clip**: clipping only, raw cells.
Scored by `lc/where_skill.py --ks 15 --extra ...` → `work/lc/where_skill/cases_dnorm.pkl` (slurm/dnorm_score.sbatch).
""")
code(r"""
def compare(F7, arms):
    if not F7.exists():
        print("pending: training jobs not finished (see work/lc/STATUS.md)"); return
    E = pd.read_pickle(F7)
    E = E[(E.t - E.u).abs() > FLOOR].copy()
    ARMS = [m for m in arms if m in E and E[m].notna().any()]
    E = E.dropna(subset=ARMS)
    E["n"] = E.groupby(GRP).t.transform("size"); E = E[E.n >= 3].copy()
    E["r"] = E.t - E["shift"]; E["rbar"] = E.groupby(GRP).r.transform("mean"); E["rc"] = E.r - E.rbar
    E["nv"] = (E.noise ** 2 + E.noise_u ** 2) / 4 * (E.n - 1) / E.n
    for m in ARMS:
        rh = E[m] - E["shift"]; E["hb_" + m] = rh.groupby([E[c] for c in GRP]).transform("mean"); E["h_" + m] = rh - E["hb_" + m]
    G7 = [E[c] for c in ["repeat", "stim", "celltype", "q", "marker"]]
    E["uc"] = E.u - E.u.groupby(G7).transform("mean"); E["tc"] = E.t - E.t.groupby(G7).transform("mean")
    rows = {}
    base = ((E["shift"] - E.t) ** 2).groupby([E.repeat, E.donor]).mean()
    for m in ["shift"] + ARMS:
        e = (E[m] - E.t) ** 2; pdn = e.groupby([E.repeat, E.donor]).mean()
        row = {"skill": skill(E[m], E), "MSE": e.mean(), "donor-repeats better than shift": f"{(pdn < base).sum()}/{len(base)}"}
        if m != "shift":
            row.update(spec(E, m)[["specific R²", "corr", "SD ratio"]].to_dict())
            row["shared MSE"] = ((E["hb_" + m] - E.rbar) ** 2).mean()
            pc = E[m] - E[m].groupby(G7).transform("mean")
            row["slope vs Unstim (true %.2f)" % ((E.tc * E.uc).sum() / (E.uc ** 2).sum())] = (pc * E.uc).sum() / (E.uc ** 2).sum()
        rows[m] = row
    display(pd.DataFrame(rows).T.round(4))
    print("specific R² by stim"); display(pd.DataFrame({m: E.groupby("stim").apply(lambda g: spec(g, m)["specific R²"]) for m in ARMS}).round(3))
    print("specific R² by |z| of the test donor's Unstim")
    E["|z|"] = pd.cut(((E.u - E.mu_u) / E.sd_u.replace(0, np.nan)).abs(), [0, 1, 2, 3, np.inf])
    display(pd.DataFrame({m: E.groupby("|z|").apply(lambda g: spec(g, m)["specific R²"]) for m in ARMS}).round(3))
    for m in ARMS: E["pc_" + m] = E[m] - E[m].groupby(G7).transform("mean")
    return E

def dr_mse(E, m): return ((E[m] - E.t) ** 2).groupby([E.repeat, E.donor]).mean()

def paired_dr(E, arms, ref):
    # paired over test donor-repeats (20): mean squared error per donor-repeat, arm vs ref
    from scipy.stats import wilcoxon
    b = dr_mse(E, ref); out = {}
    for m in arms:
        if m == ref: continue
        a = dr_mse(E, m)
        out[m] = {f"donor-repeats better than {ref}": f"{(a < b).sum()}/{len(b)}",
                  f"median MSE ratio vs {ref}": np.median(a / b), "Wilcoxon p": wilcoxon(a, b).pvalue}
    return pd.DataFrame(out).T

def skill_by(E, by, arms): return pd.DataFrame({m: E.groupby(by).apply(lambda g: skill(g[m], g)) for m in arms})

E7 = compare(WORK / "lc/where_skill/cases_dnorm.pkl", ["CellOT", "paired", "dnorm", "dnorm_clip", "clip", "gauss_ot"])
""")
code(r"""
# §7b: paired tests over donor-repeats, and the pooled-OT stretch feature by feature
A7 = ["CellOT", "paired", "dnorm", "dnorm_clip", "clip"]
print("paired over the 20 test donor-repeats, vs the paired control (same training, raw cells)")
display(paired_dr(E7, A7 + ["shift"], "paired").round(4))
F = ["stim", "celltype", "q", "marker"]
slope = lambda g, c: (g[c] * g.uc).sum() / (g.uc ** 2).sum()
S7 = E7.groupby(F).apply(lambda g: pd.Series({"true": slope(g, "tc"), **{m: slope(g, "pc_" + m) for m in A7}}))
S7["stretched"] = (S7.paired - S7.true) > 0.5
print(f"{len(S7)} features (stim, celltype, quantile, marker); 'stretched' = paired control's slope vs Unstim exceeds "
      f"the true slope by > 0.5: {S7.stretched.mean():.1%}")
print("share of features whose slope exceeds the true one by > 0.5, per arm:")
display(pd.Series({m: ((S7[m] - S7.true) > 0.5).mean() for m in A7}).round(3).to_frame("share"))
print("median slope vs Unstim (across test donors) in stretched / other features")
display(S7.groupby("stretched")[["true"] + A7].median().round(3))
E7 = E7.join(S7.stretched, on=F)
print("specific R² and skill in stretched / other features")
display(pd.concat({"specific R²": pd.DataFrame({m: E7.groupby("stretched").apply(lambda g: spec(g, m)["specific R²"]) for m in A7}),
                   "skill": skill_by(E7, "stretched", A7 + ["shift"])}).round(3))
print("paired tests restricted to the stretched features"); display(paired_dr(E7[E7.stretched], A7 + ["shift"], "paired").round(4))
print("slope vs Unstim by stim"); display(E7.groupby("stim").apply(lambda g: pd.Series({"true": slope(g, "tc"), **{m: slope(g, "pc_" + m) for m in A7}})).round(3))
print("skill by stim"); display(skill_by(E7, "stim", A7 + ["shift"]).round(3))
print("specific R² by quantile and by marker")
display(pd.DataFrame({m: E7.groupby("q").apply(lambda g: spec(g, m)["specific R²"]) for m in A7}).round(3))
display(pd.DataFrame({m: E7.groupby("marker").apply(lambda g: spec(g, m)["specific R²"]) for m in A7}).round(3))
print("by |z| of the test donor's Unstim: cases, MSE, and paired tests vs the paired control")
display(pd.concat([E7.groupby("|z|").size().rename("cases"),
                   pd.DataFrame({m: E7.groupby("|z|").apply(lambda g: ((g[m] - g.t) ** 2).mean()) for m in A7 + ["shift"]})], axis=1).round(4))
for b, g in E7.groupby("|z|"):
    print(f"|z| in {b}"); display(paired_dr(g, ["dnorm", "dnorm_clip", "clip", "CellOT"], "paired").round(4))
""")
code(r"""
COL = {"paired": "#2a78d6", "dnorm": "#eb6834", "dnorm_clip": "#e87ba4", "clip": "#1baf7a", "shift": "#6b6b6b",
       "CellOT": "#2a78d6", "condot_ct": "#eda100", "condot_global": "#eb6834"}
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4))
arms = ["paired", "dnorm", "dnorm_clip", "clip"]
X = S7.groupby("stretched")[["true"] + arms].median()
w = 0.17
for i, m in enumerate(["true"] + arms):
    axs[0].bar(np.arange(2) + (i - 2) * w, X[m], w * 0.9, color="#c3c2b7" if m == "true" else COL[m], label="truth" if m == "true" else m)
axs[0].set_xticks([0, 1]); axs[0].set_xticklabels([f"other features ({1 - S7.stretched.mean():.0%})", f"stretched features ({S7.stretched.mean():.0%})"])
axs[0].axhline(1, c="k", lw=.6, ls=":"); axs[0].set_ylabel("slope of prediction vs donor's Unstim value")
axs[0].set_title("dnorm removes the stretch (slope back to ~1)", fontsize=10); axs[0].legend(fontsize=7, frameon=False)
Z = pd.DataFrame({m: E7.groupby("|z|").apply(lambda g: spec(g, m)["specific R²"]) for m in arms})
for i, m in enumerate(arms):
    axs[1].bar(np.arange(len(Z)) + (i - 1.5) * 0.2, Z[m].clip(-0.8), 0.18, color=COL[m], label=m)
axs[1].set_xticks(range(len(Z))); axs[1].set_xticklabels([f"{i.left:g}–{i.right:g}" for i in Z.index])
axs[1].axhline(0, c="k", lw=.8); axs[1].set_xlabel("|z|: test donor's Unstim value vs training donors (SDs)")
axs[1].set_ylabel("specific R²"); axs[1].set_title("...but loses the donor-specific signal", fontsize=10)
axs[1].legend(fontsize=7, frameon=False)
for ax in axs: ax.grid(axis="y", alpha=.3); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "dnorm_stretch.png", dpi=160)
""")
md("<!-- READ7 -->")

md(r"""
## 8. CondOT-faithful: one map shared across stims, conditioned on the perturbation (submitted 2026-10-06)

CondOT (Bunne, Krause, Cuturi, NeurIPS 2022, github.com/bunnech/condot) conditions the map on a description of the
*perturbation* (dose, drug one-hot, mode of action, combination of genetic perturbations), never on the donor. Its
PICNN (vendored verbatim, MIT, `lc/picnn.py`) lets a context network modulate every layer while staying convex in the
cells. Two arms, trained exactly as CondOT / CellOT (pooled unpaired cells of the 15 training donors, one condition per
iteration, 10 g : 1 f, Adam 1e-4 (0.5, 0.9), uniform(0, 0.1) init), checkpoint = best mean val median error on the
training donors' val cells (`lc/condot_train.py`):
- **condot_ct**: one PICNN per celltype, context = one-hot stim (CondOT's `onehot` experiment), 50k iterations;
- **condot_global**: one PICNN for all 168 pairs, context = one-hot stim + one-hot celltype, 150k iterations.
Compared with the per-pair unpaired CellOT of the learning curve (`CellOT`, same pooling) and the paired control.
The question: does sharing one map across stims (and celltypes) borrow strength, especially for weak or small pairs?
""")
code(r"""
E8 = compare(WORK / "lc/where_skill/cases_condot.pkl", ["CellOT", "paired", "condot_ct", "condot_global", "gauss_ot"])
""")
code(r"""
# §8b: paired tests, weak / small pairs, training validation error
import json, glob
A8 = ["CellOT", "paired", "condot_ct", "condot_global"]
print("paired over the 20 test donor-repeats, vs per-pair unpaired CellOT (same pooling as CondOT)")
display(paired_dr(E8, A8 + ["shift"], "CellOT").round(4))
print("vs the paired control"); display(paired_dr(E8, ["condot_ct", "condot_global"], "paired").round(4))
print("error split: shared part (average response of the 4 test donors) vs patient-specific part, MSE per case")
display(pd.DataFrame({m: {"shared": ((E8["hb_" + m] - E8.rbar) ** 2).mean(), "patient-specific": ((E8["h_" + m] - E8.rc) ** 2).mean()}
                      for m in A8}).T.round(4))
print("skill by stim"); display(skill_by(E8, "stim", A8 + ["shift"]).round(3))
for s in ["IL33", "TNFa", "GMCSF", "PI"]:
    print(f"{s}: paired vs CellOT"); display(paired_dr(E8[E8.stim == s], ["paired", "condot_ct", "condot_global"], "CellOT").round(4))
E8["mean response"] = pd.cut((E8.mu_s - E8.mu_u).abs(), [0, 0.05, 0.1, 0.2, 0.5, np.inf])
print("skill by size of the average response"); display(skill_by(E8, "mean response", A8 + ["shift"]).round(3))
ctn = E8.groupby("celltype").n_u.median().sort_values()
E8["celltype size"] = E8.celltype.map(pd.qcut(ctn, 4, labels=["smallest 7", "2nd", "3rd", "largest 7"]))
print("smallest 7 celltypes:", ", ".join(ctn.index[:7]))
display(pd.concat({"skill": skill_by(E8, "celltype size", A8 + ["shift"]),
                   "specific R²": pd.DataFrame({m: E8.groupby("celltype size").apply(lambda g: spec(g, m)["specific R²"]) for m in A8})}).round(3))
for b, g in E8.groupby("celltype size"):
    print(f"celltype size {b}: vs CellOT"); display(paired_dr(g, ["paired", "condot_ct", "condot_global"], "CellOT").round(4))
PM = E8.groupby(["stim", "celltype"]).apply(lambda g: pd.Series({m: ((g[m] - g.t) ** 2).mean() for m in A8 + ["shift"]}))
PM["CellOT third"] = pd.qcut(PM.CellOT / PM["shift"], 3, labels=["CellOT best vs shift", "middle", "CellOT worst vs shift"])
print("per (stim, celltype) map: share where CondOT has lower error than CellOT, by how CellOT does against the shift")
display(PM.groupby("CellOT third").apply(lambda g: pd.Series({"maps": len(g), "condot_ct better": (g.condot_ct < g.CellOT).mean(),
        "condot_global better": (g.condot_global < g.CellOT).mean(), "median condot_ct / CellOT MSE": (g.condot_ct / g.CellOT).median()})).round(3))
V = []
for f in sorted(glob.glob(str(WORK / "lc/condot/celltype/repeat_*/*/meta.json")) + glob.glob(str(WORK / "lc/condot/global/repeat_*/meta.json"))):
    m = json.load(open(f)); V.append({"arm": "condot_" + ("ct" if m["mode"] == "celltype" else "global"), "best val": m["best_val_median_err"]})
V = pd.DataFrame(V)
cv = []
for d in sorted((WORK / "lc/train").glob("repeat_[0-9]")):
    v = np.load(d / "val_err.npy"); mp = pd.read_csv(d / "maps.csv"); cv += list(np.nanmin(v, 0)[(mp.k == 15).values])
print("validation median error on the TRAINING donors' val cells (lower = better fit; CellOT: best per map, CondOT: one "
      "checkpoint for all its conditions)")
display(pd.concat([V.groupby("arm")["best val"].describe()[["count", "50%", "mean", "max"]],
                   pd.Series(cv).describe()[["count", "50%", "mean", "max"]].to_frame("CellOT k=15 (per map)").T]).round(4))
""")
code(r"""
fig, ax = plt.subplots(figsize=(7.5, 4))
ref = dr_mse(E8, "CellOT")
for i, m in enumerate(["paired", "shift", "condot_ct", "condot_global"]):
    r = dr_mse(E8, m) / ref
    ax.scatter(np.full(len(r), i) + np.random.default_rng(0).uniform(-.12, .12, len(r)), r, s=22, color=COL[m], alpha=.8)
    ax.plot([i - .25, i + .25], [r.median()] * 2, c="k", lw=1.5)
    ax.text(i + .28, r.median(), f"{(r < 1).sum()}/20", fontsize=8, va="center")
ax.axhline(1, c="k", lw=.8, ls=":"); ax.set_xticks(range(4)); ax.set_xticklabels(["paired CellOT", "shift", "condot_ct", "condot_global"])
ax.set_ylabel("MSE / CellOT's MSE, per test donor-repeat"); ax.set_yscale("log")
ax.set_yticks([0.6, 0.8, 1, 1.25, 1.5, 2, 2.5]); ax.set_yticklabels(["0.6", "0.8", "1", "1.25", "1.5", "2", "2.5"]); ax.minorticks_off()
ax.set_title("Below 1 = better than per-pair CellOT on that donor (label: donor-repeats below 1)", fontsize=9)
ax.grid(axis="y", alpha=.3); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "condot_vs_cellot.png", dpi=160)
""")
md("<!-- READ8 -->")

md("<!-- READ6 -->")

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
txt = Path(__file__).with_name("where_skill_readings.md")
if txt.exists():
    parts, cur = {}, None
    for line in txt.read_text().splitlines():
        if line.startswith("<!-- ") and line.endswith(" -->"):
            cur = line[5:-4]; parts[cur] = []
        elif cur: parts[cur].append(line)
    for c in nb.cells:
        if c.cell_type == "markdown" and c.source.startswith("<!-- ") and c.source[5:-4] in parts:
            c.source = "\n".join(parts[c.source[5:-4]]).strip()
nb.cells = [c for c in nb.cells if not (c.cell_type == "markdown" and c.source.startswith("<!-- "))]
out = Path(__file__).resolve().parents[1] / "notebooks/where_cellot_skill.ipynb"
nbf.write(nb, out); print("wrote", out)
