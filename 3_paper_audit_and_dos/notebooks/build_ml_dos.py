"""Generate notebooks/ml_dos_supervised.ipynb: can classic / deep supervised learning on *measured* Unstim
features beat CellOT + Stabl at predicting days before labor (DOS)?

Every table is recomputed from simco_work/onset_test/ml_dos/summary/*.csv (written by ml_dos/summarize.py).
    python notebooks/build_ml_dos.py [--out notebooks/ml_dos_supervised.ipynb]
    jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=cellot <nb>
Readings: notebooks/ml_dos_readings.md (summary + next steps), merged in right after the introduction.
"""
import argparse
from pathlib import Path

import nbformat as nbf

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(Path(__file__).parent / "ml_dos_supervised.ipynb"))
ap.add_argument("--readings", default=str(Path(__file__).parent / "ml_dos_readings.md"))
a = ap.parse_args()

cells = []
md = lambda t: cells.append(nbf.v4.new_markdown_cell(t.strip()))
code = lambda t: cells.append(nbf.v4.new_code_cell(t.strip()))

md(r"""
# Predicting days before labor from measured Unstim cells — supervised learning vs CellOT + Stabl

**Question.** The paper predicts `DOS` (days from blood draw to labor onset, negative) with Stabl + XGBoost on
CellOT-*simulated* stimulation features (Fig 5b). Can ordinary supervised learning on what was actually
*measured* — each woman's unstimulated cells — do as well or better, with no simulation at all?

**Inputs (measured only, no map).** Per visit, from her gated Unstim cells, 28 celltypes:
- `freq` — log10 celltype frequency (28)
- `q10 q25 q50 q75 q90` — quantiles of **all 31 markers** (13 functional + 18 phenotypic) (5 × 868)
- `mean sd` — moments of all 31 markers (2 × 868)
- `corr` — correlations between the 13 functional markers (2184)

Three feature sets ("arms"): `freqmed` = freq + medians (896); `qtl` = freq + all quantiles (4368);
`full` = + mean, sd, corr (8288). All three come from the same OOL Unstim cell files that CellOT + Stabl used — the
paper's own Unstim block is the 13 functional-marker medians of these cells — so **no source of information beyond
what CellOT + Stabl had**. Excluded by design (user, 2026-10-07): the OOL 15-min stimulations and the clinical
variables (Age, BMI, Gravida, Para). They were run in an earlier version and are not reported.

**Never an input:** EGA (gestational age at the draw), the visit (G1/G2/G3), GA at delivery, anything derived from
DOS (`EGA_DOS`, `DOS_diff`, `Delivery`), and the total cell count (acquisition, could proxy the run date). EGA is
used below *only to evaluate* the predictions, never to make them.

**Splits.** Exactly the paper's: `GroupShuffleSplit(100, test_size=0.2, random_state=42)` grouped by woman,
over the paper's feature-table row order. All visits of a woman are on the same side, so the model never sees
another time point of a test woman. `36_G1` / `52_G1` (Unstim tubes actually stimulated) are removed from both
sides of every split. Every hyperparameter is tuned **inside** the training women by `GroupKFold(5)` over women;
feature screening, imputation and scaling are refitted in every fold.

**The reference** is rerun through the same code on the same splits: CellOT table + the paper's own Stabl ALasso
selections + XGBoost `params_43` → **R² 0.415** (= `paper_protocol/aligned_excl`, the paper's number with the
row-order bug fixed and the two visits removed). A second, stronger reference: CellOT + measured features + Stabl
(`comb_cellot`, **0.448**). Both references are slightly favoured: `params_43` was chosen on these very splits (by
preterm AUROC; it ranks 43rd of 97 by aligned R², so the R² optimism is small), and Stabl's selections were made with
the two bad visits in.

**How to read the scores.** `R2 / RMSE / MAE` follow Stabl's convention: per sample, the median of its predictions
over the ~20 splits where it was in test, then one score over the 175 samples. `median split R²` is the typical
single-split score (~35 test visits). Paired tests compare per-split R² with the reference on the same 100 splits
(Wilcoxon; splits overlap, so p-values are optimistic). Holm-adjusted over all models.
""")

r = Path(a.readings)
if r.exists():
    md(r.read_text())

code(r"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import wilcoxon, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 120)
O = Path("/scratch/users/bmerwan/simco_work/onset_test"); M = O / "ml_dos"; S = M / "summary"
sc = pd.read_csv(S / "scores.csv"); ps = pd.read_csv(S / "per_split.csv"); oof = pd.read_csv(S / "oof.csv")
ega = pd.read_csv(S / "ega_context.csv"); ic = pd.read_csv(S / "inner_choice.csv")
REF = ("ref_cellot", "stabl_p43"); REF2 = ("ref_comb_cellot", "stabl_p43")
ARMS = ["freqmed", "qtl", "full"]
FAM_ORDER = ["null", "linear", "kernel", "bagging", "boosting", "neural", "neural (cells)", "ensemble"]
ref_r2 = sc.set_index(["arm", "model"]).loc[REF, "R2"]; ref2_r2 = sc.set_index(["arm", "model"]).loc[REF2, "R2"]
print(f"runs on disk: {len(sc)}  (zoo {(~sc.arm.str.startswith('ref_')).sum()}, references 2)")
print(f"reference CellOT + Stabl: R2 {ref_r2:.3f};  CellOT + measured + Stabl: R2 {ref2_r2:.3f}")
incomplete = sc[sc.n_splits < 100]
sc = sc[sc.n_splits == 100]  # every table below uses complete runs only
print("incomplete runs (fewer than 100 splits):", "none" if incomplete.empty else incomplete[["arm", "model", "n_splits"]].to_string(index=False))
""")

md(r"""
## 1. Context: what a score means here

Two anchors on the same splits: predicting the training mean (no information), and EGA alone (one linear
coefficient). EGA is **not allowed** as an input — it is the ceiling a pure gestational-age clock reaches, and DOS
is to a large extent `7 × (40 − EGA)`. Any immune model is partly an estimate of gestational age.
""")

code(r"""
rows = pd.read_csv(O / "ina_13OG_final_long_allstims_filtered.csv", index_col=0, usecols=[0]).index
y = pd.read_csv(O / "outcome_table_all_pre.csv", index_col=0)["DOS"].loc[rows].astype(float)
clin = pd.read_csv(O / "Preprocessed_OOL_Clinical_new.csv", index_col=0).loc[rows]
g = rows.to_series().str.split("_").str[0].values
ex = np.isin(rows, ["36_G1", "52_G1"])
P = {r: [] for r in rows}
for tr, te in GroupShuffleSplit(100, test_size=0.2, random_state=42).split(np.zeros(len(rows)), groups=g):
    tr, te = tr[~ex[tr]], te[~ex[te]]
    m = LinearRegression().fit(clin.EGA.values[tr, None], y.values[tr])
    for r, v in zip(rows[te], m.predict(clin.EGA.values[te, None])): P[r].append(v)
med = pd.Series({r: np.median(v) for r, v in P.items() if v})
ctx = pd.DataFrame([
    dict(model="constant (training mean)", **sc.query("model == 'mean'")[["R2", "RMSE", "MAE"]].iloc[0].to_dict()),
    dict(model="CellOT + Stabl (paper, reference)", **sc.set_index(["arm", "model"]).loc[REF, ["R2", "RMSE", "MAE"]].to_dict()),
    dict(model="CellOT + measured + Stabl (reference 2)", **sc.set_index(["arm", "model"]).loc[REF2, ["R2", "RMSE", "MAE"]].to_dict()),
    dict(model="EGA alone (NOT allowed — ceiling of a GA clock)", R2=r2_score(y.loc[med.index], med),
         RMSE=mean_squared_error(y.loc[med.index], med) ** .5, MAE=mean_absolute_error(y.loc[med.index], med)),
])
ctx.round(3)
""")

md(r"""
## 2. Every model on every feature set

Pooled R² (Stabl convention). Rows = models grouped by family, columns = feature sets. Dashed line on the bar
chart below = CellOT + Stabl (0.415), dotted = CellOT + measured + Stabl (0.448).

- `stack` = super learner: base models' out-of-fold predictions inside the training women → non-negative blend.
- `auto` = per split, the zoo model with the lowest **inner** CV error (chosen on training women only). It is the
  honest answer to "the best model of the zoo": picking the top row of this table instead would select on the
  same splits that score it.
""")

code(r"""
zoo = sc[sc.arm.isin(ARMS)].copy()
zoo["family"] = pd.Categorical(zoo.family, FAM_ORDER, ordered=True)
cells = zoo.query("arm == 'cells'")
tab = zoo.pivot_table(index=["family", "model"], columns="arm", values="R2", observed=True).reindex(columns=ARMS)
tab["best arm"] = tab[ARMS].idxmax(1)
ds = sc.query("arm == 'cells'")
print("DeepSets on raw cells:", "not run yet" if ds.empty else f"R2 {ds.R2.iloc[0]:.3f}, RMSE {ds.RMSE.iloc[0]:.1f} d")
tab.style.format("{:.3f}", subset=ARMS).background_gradient(cmap="RdYlGn", vmin=0, vmax=0.5, subset=ARMS)
""")

code(r"""
from matplotlib.patches import Patch
fig, ax = plt.subplots(figsize=(13.5, 6))
fu = zoo.query("arm == 'full' and model != 'mean'").set_index("model")
cl = sc.query("arm == 'cells'").set_index("model")
best = pd.concat([fu, cl]).sort_values("R2")
colors = {"linear": "#4C72B0", "kernel": "#8172B2", "bagging": "#55A868", "boosting": "#DD8452",
          "neural": "#C44E52", "neural (cells)": "#8C1C13", "ensemble": "#222"}
yy = np.arange(len(best))
ax.barh(yy, best.R2, color=[colors.get(f, "#999") for f in best.family])
ax.set_yticks(yy, [m + ("  [raw cells]" if m == "deepsets" else "") for m in best.index])
ax.axvline(ref_r2, ls="--", c="k", lw=1); ax.axvline(ref2_r2, ls=":", c="k", lw=1)
h = [Patch(color=c, label=f) for f, c in colors.items()]
h += [plt.Line2D([], [], ls="--", c="k", label=f"CellOT + Stabl ({ref_r2:.3f})"),
      plt.Line2D([], [], ls=":", c="k", label=f"CellOT + measured + Stabl ({ref2_r2:.3f})")]
ax.legend(handles=h, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
ax.set_xlabel("pooled out-of-fold R² (DOS)"); ax.set_xlim(0, .6)
ax.set_title("Bars: measured Unstim only (`full`). Same 100 splits as the paper's CellOT + Stabl")
plt.tight_layout(); plt.show()
""")

md(r"""
**Reading the bar chart:** bars are the `full` feature set (measured Unstim cells only, the richest summary, fixed in
advance as the main arm, so there is no choice of feature set per model). `mean` is omitted. The top bar is still the best of ~25 runs on the
same splits; `auto` and `stack` are the choices made inside the training women.

## 3. Paired against the reference, split by split

Mean Δ = mean over the 100 splits of (model's R² − reference's R²); wins = splits where the model is better;
p = Wilcoxon signed-rank, then Holm over all zoo runs. A model "beats CellOT + Stabl" only if Δ > 0 with Holm p < 0.05.
""")

code(r"""
cols = ["arm", "model", "family", "R2", "RMSE", "MAE", "median_split_R2",
        "vs_cellot_mean_diff", "vs_cellot_wins", "vs_cellot_p", "vs_cellot_p_holm",
        "vs_comb_cellot_mean_diff", "vs_comb_cellot_wins", "vs_comb_cellot_p_holm"]
pt = zoo.sort_values("R2", ascending=False)[cols]
pt.head(30).style.format({c: "{:.3f}" for c in cols if c not in ("arm", "model", "family", "vs_cellot_wins", "vs_comb_cellot_wins")} |
                        {"vs_cellot_p": "{:.2g}", "vs_cellot_p_holm": "{:.2g}", "vs_comb_cellot_p_holm": "{:.2g}"})
""")

code(r"""
sig_better = zoo.query("vs_cellot_mean_diff > 0 and vs_cellot_p_holm < 0.05")
sig_worse = zoo.query("vs_cellot_mean_diff < 0 and vs_cellot_p_holm < 0.05")
above = zoo.query("R2 > @ref_r2")
print(f"zoo runs with pooled R2 above CellOT+Stabl ({ref_r2:.3f}): {len(above)} of {len(zoo)}")
print(f"significantly better per split (Holm < 0.05): {len(sig_better)}  ->", ", ".join(sig_better.arm + "/" + sig_better.model) or "none")
print(f"significantly worse per split (Holm < 0.05):  {len(sig_worse)}")
for arm in ARMS:
    r = zoo.query("arm == @arm and model == 'auto'")
    if len(r):
        r = r.iloc[0]
        print(f"auto [{arm:9s}] R2 {r.R2:.3f}  RMSE {r.RMSE:.1f} d  median split {r.median_split_R2:.3f}  "
              f"vs CellOT {r.vs_cellot_mean_diff:+.3f} ({r.vs_cellot_wins}/100, p {r.vs_cellot_p:.2g})")
""")

md(r"""
## 4. Split-by-split distribution

Per-split R² (each point = one of the 100 test sets of ~13 women / ~35 visits) for the references and the best
run of each family. Spread across splits is large: a single split says little, which is why the comparison is
paired.
""")

code(r"""
fam_best = zoo[(zoo.model != "mean") & (zoo.arm == "full")].loc[lambda d: d.groupby("family", observed=True).R2.idxmax()]
show = [REF, REF2] + list(zip(fam_best.arm, fam_best.model)) + [tuple(x) for x in sc.query("arm == 'cells'")[["arm", "model"]].values]
lab = {REF: "CellOT+Stabl", REF2: "CellOT+meas.+Stabl"}
data, names = [], []
for a_, m_ in show:
    d = ps.query("arm == @a_ and model == @m_").R2.clip(lower=-1)
    if len(d):
        data.append(d.values); names.append(lab.get((a_, m_), f"{m_}\n[{a_}]"))
fig, ax = plt.subplots(figsize=(12, 4.5))
ax.boxplot(data, labels=names, showfliers=False)
for i, d in enumerate(data, 1):
    ax.scatter(np.random.default_rng(i).normal(i, .05, len(d)), d, s=5, alpha=.4)
ax.axhline(0, c="grey", lw=.5); ax.set_ylabel("per-split R² (clipped at −1)")
ax.set_title("Per-split R², references vs the best run of each family on measured Unstim only (`full`)"); plt.xticks(fontsize=8); plt.tight_layout(); plt.show()
""")

md(r"""
## 5. Predicted vs observed, and what is left beyond gestational age

Left: CellOT + Stabl. Right: `auto` on measured Unstim cells only (`full`).
Right-hand table: correlation of each out-of-fold prediction with DOS, with EGA, and the **partial correlation with
DOS given EGA** — what the immune prediction adds beyond knowing when the blood was drawn. (EGA is used here only to
evaluate.)
""")

code(r"""
au = zoo.query("model == 'auto'").sort_values("R2", ascending=False)
fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
for ax, key, t in [(axs[0], REF, "CellOT + Stabl (paper)"), (axs[1], ("full", "auto"), "auto, measured Unstim cells only")]:
    d = oof.query("arm == @key[0] and model == @key[1]")
    pre = d.assign(G=d["sample"].str.split("_").str[1])
    for gg, c in zip(["G1", "G2", "G3"], ["#4C72B0", "#DD8452", "#55A868"]):
        q = pre[pre.G == gg]; ax.scatter(q.y, q.pred, s=12, c=c, label=gg)
    lim = [-125, 10]; ax.plot(lim, lim, c="grey", lw=.6); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(f"{t}\nR² {r2_score(d.y, d.pred):.3f}, RMSE {mean_squared_error(d.y, d.pred) ** .5:.1f} d", fontsize=10)
    ax.set_xlabel("observed DOS (days)")
axs[0].set_ylabel("out-of-fold prediction (median over splits)"); axs[0].legend(title="visit", fontsize=8)
plt.tight_layout(); plt.show()
e = ega.merge(sc[["arm", "model", "R2"]], on=["arm", "model"]).sort_values("R2", ascending=False)
e = e[e.arm.isin(ARMS + ["cells"]) | e.arm.str.startswith("ref_")]
e = e[e.arm.str.startswith("ref_") | (e.model.isin(["auto", "stack"])) | (e.R2 >= e[e.arm.isin(ARMS)].R2.quantile(.9))]
e.round(3)
""")

md(r"""
## 6. What the models use

Importance share (|coefficient| for linear models, impurity / gain for trees), normalised to 1 per split and
averaged over the 100 splits, for the tree ensembles and penalised linear models on the `full` arm. Aggregated by
block (frequency, quantile level, moment, correlation) and by marker type (functional vs phenotypic).
""")

code(r"""
imp = pd.read_csv(S / "importances.csv") if (S / "importances.csv").exists() else pd.DataFrame()
FUNC = [l.strip() for l in open("/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/13features.txt") if l.strip()]
def block(f):
    p = f.split("__")
    return p[-1] if len(p) > 1 else "?"
def mtype(f):
    p = f.split("__")
    if len(p) < 3: return p[-1]
    return "functional" if p[1] in FUNC else ("correlation" if p[-1] == "corr" else "phenotypic")
if len(imp):
    imp["block"] = imp.feature.map(block); imp["marker type"] = imp.feature.map(mtype)
    sub = imp[imp.arm == "full"]
    by_block = sub.pivot_table(index="block", columns="model", values="mean_share", aggfunc="sum").fillna(0)
    by_type = sub.pivot_table(index="marker type", columns="model", values="mean_share", aggfunc="sum").fillna(0)
    display(by_block.round(3)); display(by_type.round(3))
    for m_ in ["rf", "xgb", "enet"]:
        t = sub[sub.model == m_].sort_values("mean_share", ascending=False).head(15)
        if len(t):
            print(f"\n{m_} [full] — top 15 features (share of importance, splits where it appears)")
            print(t[["feature", "mean_share", "n_splits"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
""")

md(r"""
## 7. What the inner CV chose, and cost

The hyperparameters picked inside the training women (most frequent choice over the 100 splits), the super
learner's weights, and the time per split (one core).
""")

code(r"""
ch = ic[ic.arm.isin(ARMS + ["cells"])].copy()
mode = ch.groupby(["arm", "model"]).params.agg(lambda v: v.value_counts().index[0] if len(v) else "")
freq = ch.groupby(["arm", "model"]).params.agg(lambda v: v.value_counts().iloc[0] / len(v) if len(v) else np.nan)
cost = ch.groupby(["arm", "model"]).seconds.median()
t = pd.DataFrame({"most frequent choice": mode, "share of splits": freq, "median s / split": cost})
display(t.query("arm == 'full' or arm == 'cells'").round(2))
w = ch.query("model == 'stack'").copy()
if len(w):
    W = pd.DataFrame([json.loads(p) for p in w.params], index=w.arm.values)
    print("super learner weights (mean over splits):"); display(W.groupby(level=0).mean().round(2))
""")

md(r"""
## 8. Is it the features or the learner? The same zoo on the CellOT table

The references use Stabl (a sparse selection, ~20 features) + XGBoost; the zoo's best models are dense (kernels,
neural networks) and use thousands of weak features. To separate the two, the same zoo models were run on the
paper's CellOT table (`cellot`, 1216 columns = 852 simulated stims + 364 measured Unstim medians) and on measured +
CellOT stims (`full_cellot`). If dense learners lift CellOT as much as measured features, the gain is the learner.
""")

code(r"""
LRN = ["full", "cellot", "full_cellot"]
LRN_M = ["ridge", "gp", "krr_rbf", "svr_rbf", "xgb_p43", "stack", "dnn", "dnn_wide"]
scc = sc[sc.n_splits == 100]
lr_ = scc[scc.arm.isin(LRN) & scc.model.isin(LRN_M)].pivot_table(index="model", columns="arm", values="R2").reindex(index=LRN_M, columns=LRN)
lr_["CellOT+Stabl (ref)"] = ref_r2
display(lr_.style.format("{:.3f}", na_rep="pending").background_gradient(cmap="RdYlGn", vmin=0, vmax=0.6, subset=LRN))
rows_ = []
for m_ in LRN_M:
    a_ = ps.query("arm == 'full' and model == @m_").set_index("split").R2
    for arm_ in ["cellot", "full_cellot"]:
        b_ = ps.query("arm == @arm_ and model == @m_").set_index("split").R2
        if len(a_) == 100 and len(b_) == 100:
            dd = b_ - a_
            rows_.append(dict(model=m_, arm=arm_, mean_diff_vs_measured=dd.mean(), wins=int((dd > 0).sum()), p=wilcoxon(dd).pvalue))
if rows_:
    print("same learner, CellOT-based table minus measured `full` (per split; positive = CellOT table better):")
    display(pd.DataFrame(rows_).round(3))
""")

md(r"""
## 9. Preterm detection from the same predictions (nothing refit)

The models were trained on DOS only. A fixed rule turns each out-of-fold DOS prediction into a predicted pregnancy
length, **L = 7 × EGA − predicted DOS** (days), and a visit is called preterm if **L < 259 d (37 weeks)**, i.e. a
predicted delivery more than 21 days before the due date. The AUROC uses L itself (shorter = more preterm), so no
threshold is chosen. 14 preterm visits from **5 women**, all late preterm (34–35 weeks); 161 term visits.

**Read the "detrended" and "G1" columns, not "all visits".** Preterm women are sampled earlier (their late visits do
not happen), so even a model with no information (constant DOS) gets AUROC 0.71 from `7 × EGA` alone. Two fixes:
- *detrended*: remove from L its linear trend on EGA (fitted over all visits, preterm labels unused). Any score that
  only reflects when the blood was drawn gives ~0.5 here (the constant model's L becomes exactly 0, so its value is
  meaningless).
- *G1*: first visit only, one per woman, timing roughly matched (EGA 27.0 vs 28.3 weeks); the constant model still
  gets 0.65 there.

p = permutation of the preterm label across women (10 000); CI = bootstrap over women. With 5 preterm women both are
rough. `sens_259` / `spec_259` = share of preterm / term visits correctly called by the 259-day rule;
`women_flagged_259` = preterm women flagged at ≥ 1 visit (of 5).
""")

code(r"""
pa = pd.read_csv(S / "preterm_auc.csv")
cols_ = ["model", "AUROC_all_detrended", "p_all_detrended", "AUROC_G1", "p_G1", "AUROC_G1_detrended", "p_G1_detrended",
         "AUROC_all", "CI_lo_all", "CI_hi_all", "sens_259", "spec_259", "women_flagged_259"]
pa[cols_].style.format({c: "{:.3f}" for c in cols_ if c not in ("model", "women_flagged_259")} | {c: "{:.3f}" for c in cols_ if c.startswith("p_")})
""")

code(r"""
psc = pd.read_csv(S / "preterm_scores.csv")
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, m_ in zip(axs, ["CellOT + Stabl (paper)", "wide deep network"]):
    d = psc[psc.model == m_]
    for lab_, c_, mk in [(0, "#9BB3D4", "o"), (1, "#C44E52", "D")]:
        q = d[d.preterm == lab_]
        ax.scatter(q.EGA, q.pred_length / 7, s=22 if lab_ else 12, c=c_, marker=mk,
                   label="preterm (5 women)" if lab_ else "term")
    ax.axhline(259 / 7, ls="--", c="k", lw=1); ax.axhline(40, ls=":", c="grey", lw=1)
    r_ = pa.set_index("model").loc[m_]
    ax.set_title(f"{m_}\nAUROC detrended {r_.AUROC_all_detrended:.2f}, G1 {r_.AUROC_G1:.2f}", fontsize=10)
    ax.set_xlabel("EGA at the blood draw (weeks)")
axs[0].set_ylabel("predicted pregnancy length (weeks)"); axs[0].legend(fontsize=8, loc="lower right")
axs[1].text(axs[1].get_xlim()[1], 259 / 7, "37 wk ", ha="right", va="bottom", fontsize=8)
plt.tight_layout(); plt.show()
""")

md(r"""
## 10. Limits

- 175 visits / 62 women; each test set ~13 women. Per-split R² is noisy and splits overlap (each woman is in test
  ~20 times), so paired p-values are optimistic; Holm handles the number of models, not the overlap.
- The references are slightly favoured: `params_43` was chosen on these splits (rank 1 of 95 by preterm AUROC; 43rd of
  97 by aligned R², so its R² optimism is small), and Stabl selected with the two bad visits in. The zoo's grids were
  fixed before any result and every choice is made on training women.
- The zoo has its own, larger selection effect: 100+ runs scored on the same splits. The top row of any table is an
  argmin; quote `auto` / `stack` (choices made inside the training women) or a pre-specified model, not the maximum.
- DOS is mostly gestational age: any model here is largely a GA clock built from immune features (section 5).
- The OOL Unstim cells are systematically offset from the atlas (CLAUDE.md, *atlas_vs_ool*); that does not affect this
  benchmark (no atlas, no map) but it does affect the CellOT reference's inputs.
- No LightGBM / CatBoost (not installed in the shared venvs); sklearn's `HistGradientBoosting` is the same
  histogram-boosting algorithm as LightGBM.
""")

md(r"""
## Commands

```bash
# features (once, ~2 min with xargs -P 6): ml_dos/build_features.py one <sample> / assemble -> ml_dos/measured_unstim_full.csv
bash ml_dos/launch.sh                 # snapshot code to simco_work/onset_test/ml_dos/code, submit array + summary
# resume a task: sbatch --array=<i> simco_work/onset_test/ml_dos/code/task.sbatch   (tasks.txt line i+1)
python ml_dos/summarize.py            # -> simco_work/onset_test/ml_dos/summary/
python notebooks/build_ml_dos.py && jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=cellot notebooks/ml_dos_supervised.ipynb
```
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "CellOT (py3.9)", "language": "python", "name": "cellot"}
nbf.write(nb, a.out)
print(f"wrote {a.out} ({len(cells)} cells)")
