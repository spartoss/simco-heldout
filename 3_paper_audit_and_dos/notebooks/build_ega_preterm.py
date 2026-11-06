"""Generate notebooks/ega_preterm_nullmaps.ipynb -- provenance record for the 2026-09-22/23 work:
gestational age (EGA) vs DOS, the preterm AUROC, and the null-map controls.

Same convention as stabl_results.ipynb: every table is recomputed from the source CSVs
(nothing pasted), and the notebook is executed in place.
    python notebooks/build_ega_preterm.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/ega_preterm_nullmaps.ipynb
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(src):
    cells.append(nbf.v4.new_code_cell(src.strip()))


md(r"""
# Fig 5b under the microscope — gestational age, preterm AUROC, and null maps

Companion to `stabl_results.ipynb`. Questions answered here, in order:

1. **How the R² is computed**, and why our 20-split runs give ~0.29–0.30 while the paper says 0.415.
2. **What DOS and EGA are**, and how much of DOS gestational age alone explains (R² 0.878).
3. **When preterm and term women were sampled** (EGA per visit).
4. **Why term and preterm RMSE look equal** — and why that is not evidence of preterm signal.
5. **The preterm AUROC** (paper 0.73): all visits, and G1 only where sampling time is matched.
6. **Null-map controls**: does CellOT's gain come from the modelled perturbation, or from any
   nonlinear map of the Unstim cells?
7. **The gestational-age-free target `DOS_offset`** — does anything predict labor timing beyond
   the due date?
8. **Every feature set on the paper's own protocol** (100 splits, `params_43`, and the same
   97-config XGBoost search the lab ran for CellOT, second stage re-run with the row-order bug
   fixed) — CellOT vs `gauss_ot`, measured features, null maps, combined sets, with paired tests.
9. **The drug arm**: how `params_43` was chosen, checks a–d, and whether the drug ranking survives
   a change of DOS model.

Nothing is retrained here; everything reads CSVs under `$SCRATCH` and the read-only group home.
""")

code(r"""
import warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata, pearsonr
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
pd.set_option("display.width", 180, "display.max_columns", 40, "display.max_rows", 120)

O   = Path("/scratch/users/bmerwan/simco_work/onset_test")
GH  = Path("/home/groups/gbrice/ptb-drugscreen/ool_stabl/onset_test")      # read-only
RNG = np.random.default_rng(0)

clin  = pd.read_csv(O / "Preprocessed_OOL_Clinical_new.csv", index_col=0)
order = pd.read_csv(O / "ina_13OG_final_long_allstims_filtered.csv", index_col=0, usecols=[0]).index

def rescore_preds(arm):
    # paper-configuration OOF predictions: col '0' = prediction, 'DOS' = truth (don't median across both)
    return pd.read_csv(O / f"rescore_params43/{arm}/Training CV/STABL ALasso/STABL ALasso predictions.csv",
                       index_col=0).dropna()

def r2_of(summary_dir, model="STABL ALasso"):
    s = pd.read_csv(Path(summary_dir) / "Summary/Scores training CV.csv", index_col=0)
    return s.loc[model]

def first(x):   # "0.415 [0.28, 0.52]" -> 0.415
    return float(str(x).split()[0])

def auc(y, s):  # rank AUROC, fast enough for permutation tests
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def perm_p(y, s, n=20000):
    a = auc(y, s); null = np.array([auc(RNG.permutation(y), s) for _ in range(n)])
    return a, (np.sum(np.abs(null - .5) >= abs(a - .5)) + 1) / (n + 1)

def oof(y, X, n_splits=100):
    # pooled out-of-fold prediction exactly like Stabl's: same GroupShuffleSplit, median per sample
    rows = order.intersection(y.index); yy = y.loc[rows]; g = rows.str.split("_").str[0]
    P = {r: [] for r in rows}
    for tr, te in GroupShuffleSplit(n_splits, test_size=0.2, random_state=42).split(np.zeros(len(rows)), yy, g):
        if X is None:
            pr = np.full(len(te), yy.iloc[tr].mean())
        else:
            Xr = X.loc[rows]; pr = LinearRegression().fit(Xr.iloc[tr], yy.iloc[tr]).predict(Xr.iloc[te])
        for r, v in zip(rows[te], pr): P[r].append(v)
    return pd.Series({r: np.median(v) for r, v in P.items() if v})

paper = rescore_preds("both")
idx   = paper.index
DOS   = paper["DOS"].astype(float)
EGA   = clin.loc[idx, "EGA"]
PRE   = (clin.loc[idx, "Delivery"].str.strip() == "preterm").astype(int)
VISIT = clin.loc[idx, "Timepoint"]
print(len(idx), "samples,", idx.str.split("_").str[0].nunique(), "women;",
      PRE.sum(), "preterm samples from", idx[PRE == 1].str.split("_").str[0].nunique(), "women")
""")

md(r"""
## 1. How R² is computed, and 0.30 vs 0.415

Per split (`GroupShuffleSplit(n_splits, test_size=0.2, random_state=42)`, grouped by woman):
Stabl selects ~20 features on the training women, XGBoost is fit on them and predicts the
held-out women. Each sample is in test in ~20 % of the splits; its predictions are **medianed**,
giving one out-of-fold prediction per sample. Then **one** R² over the 177 samples:
R² = 1 − Σ(y − ŷ)² / Σ(y − ȳ)². 0 = no better than predicting the average; < 0 = worse.

The paper's 0.415 vs our ~0.29 is two settings, not a different model:
""")

code(r"""
rows = [
 ("ours: 20 splits, default XGBoost (Unstim + sim stims)", O / "cellot_both_n20/both"),
 ("lab: 100 splits, default XGBoost (…_GSS run)",          GH / "results_ina_13OG_final_long_allstims_filtered_xgboost_knockoff_GSS"),
 ("paper Fig 5b: 100 splits + params_43.json",             GH / "GS_cellot_longallstims_foldinfo"),
 ("our rescore of the paper configuration",                O / "rescore_params43/both"),
]
pd.DataFrame([{"run": n, "R2 (ALasso)": r2_of(d)["R2"], "RMSE": r2_of(d)["RMSE"]} for n, d in rows])
""")

md(r"""
More splits → each sample's prediction is a median of ~20 instead of ~4 → higher aggregate R².
`params_43.json` is the best of 97 XGBoost configs **scored on the same 100 splits** (an argmin
leak). Comparisons *between* our arms are fair because they all share settings.

## 2. DOS, EGA, and how much of DOS gestational age explains

`DOS` = days from blood draw to labor onset (negative). `EGA` = weeks pregnant at the draw.
Woman 100, three visits — the 40-week due date already predicts DOS to a few days:
""")

code(r"""
w = clin[clin.Patient == 100][["Timepoint", "EGA", "DOS"]].copy()
w["due-date prediction 7·(EGA−40)"] = (7 * (w.EGA - 40)).round()
w["offset = DOS − due-date pred."]   = w.DOS - w["due-date prediction 7·(EGA−40)"]
w
""")

code(r"""
# Same 100 splits, same pooled-OOF scoring as Stabl
ega_only = oof(DOS, EGA.to_frame()).loc[idx]
mean_only = oof(DOS, None).loc[idx]
tab = pd.DataFrame({
    "model": ["average DOS for everyone", "EGA alone (1 linear coef.)", "paper Fig 5b (immune only)"],
    "R2":   [r2_score(DOS, mean_only), r2_score(DOS, ega_only), r2_score(DOS, paper["0"])],
    "RMSE": [np.sqrt(((DOS - p) ** 2).mean()) for p in (mean_only, ega_only, paper["0"])],
}).round(3)
tab
""")

code(r"""
# How much of the paper's prediction is a gestational-age estimate?
import statsmodels.api as sm
res_y = sm.OLS(DOS, sm.add_constant(EGA)).fit().resid
res_p = sm.OLS(paper["0"], sm.add_constant(EGA)).fit().resid
print("corr(prediction, DOS) = %.3f   corr(prediction, EGA) = %.3f   partial r(prediction, DOS | EGA) = %.3f"
      % (pearsonr(paper["0"], DOS)[0], pearsonr(paper["0"], EGA)[0], pearsonr(res_p, res_y)[0]))

# The lab also ran the model WITH EGA as an input (use_ega):
lab = [("immune (Unstim + sim stims) + EGA", GH / "GS_cellot_longallstims_EGA_foldinfo"),
       ("immune, no EGA (= Fig 5b)",          GH / "GS_cellot_longallstims_foldinfo"),
       ("sim stims only + EGA",               GH / "GS_ina_no_unstim_42_foldinfo"),
       ("sim stims only, no EGA",             GH / "GS_ina_no_unstim_42_foldinfo_noega")]
pd.DataFrame([{"lab run (100 splits, params_43)": n, "R2": r2_of(d)["R2"], "RMSE": r2_of(d)["RMSE"]} for n, d in lab])
""")

md(r"""
EGA alone ≈ 0.88; the lab's immune + EGA run is 0.890 — immune features add at most ~0.01 R²
over gestational age (and the stim-only + EGA run is *below* EGA alone). The paper's prediction
correlates almost as much with EGA as with DOS: it is mostly a noisy gestational-age estimate.

## 3. When were preterm and term women sampled?
""")

code(r"""
d = clin.loc[idx].assign(group=PRE.map({1: "preterm", 0: "term"}), delivery_wk=EGA - DOS / 7)
t = d.groupby(["Timepoint", "group"]).agg(n=("EGA", "size"), EGA_mean=("EGA", "mean"), EGA_min=("EGA", "min"),
                                          EGA_max=("EGA", "max"), delivery_week=("delivery_wk", "mean")).round(1)
t
""")

md(r"""
Preterm women (5, all late-preterm, 34–35 wk) deliver before the late visits, so their G2/G3
samples are ~4 weeks earlier than term. At G1 the groups are nearly matched (27.0 vs 28.3).

## 4. Why term and preterm RMSE look equal

Paper: RMSE 24 (term) vs 25 (preterm). The same pattern appears for a model that knows nothing:
""")

code(r"""
e = lambda p: DOS - p
pd.DataFrame([{
    "model": n, "R2": r2_score(DOS, p),
    "RMSE term": np.sqrt((e(p)[PRE == 0] ** 2).mean()), "RMSE preterm": np.sqrt((e(p)[PRE == 1] ** 2).mean()),
    "mean pred. preterm": p[PRE == 1].mean(), "true mean preterm": DOS[PRE == 1].mean(),
} for n, p in [("average DOS for everyone", mean_only), ("EGA-only model", ega_only), ("SIMCO (paper)", paper["0"])]]).round(
    {"R2": 3, "RMSE term": 1, "RMSE preterm": 1, "mean pred. preterm": 1, "true mean preterm": 1})
""")

md(r"""
Preterm samples are drawn early (DOS more negative) **and** deliver early (DOS less negative);
the two cancel, so their DOS sits near the population average (−32 vs −43, smaller spread). Any
prediction pulled toward the average looks good on them by coincidence. The EGA-only model — the
one that knows gestational age — shows the expected pattern (7 vs 31 days). SIMCO predicts
preterm samples at −41 on average, like the know-nothing model: it does not predict "labor soon".

## 5. Preterm AUROC

Paper score = expected pregnancy length `7·EGA − predicted DOS`
(`stablVMax/stabl/cross_validation_drug_vs_dmso.py:165`). EGA enters *after* the model, so the
fair baseline is the same formula with a constant prediction (= ranking by EGA).
""")

code(r"""
arms = {a: rescore_preds(a)["0"].reindex(idx) for a in ["both", "no_Unstim", "unstim_only"]}
scores = {
    "SIMCO paper (Unstim + sim stims)": -(7 * EGA - arms["both"]),
    "sim stims only":                   -(7 * EGA - arms["no_Unstim"]),
    "measured Unstim only":             -(7 * EGA - arms["unstim_only"]),
    "EGA alone (constant prediction)":  -EGA,
    "EGA-only model":                   -(7 * EGA - ega_only),
}
L = (7 * EGA - arms["both"]) / 7
g1 = (VISIT == "G1").values
resid_g1 = (L[g1] - np.polyval(np.polyfit(EGA[g1], L[g1], 1), EGA[g1])).values
out = []
for n, s in scores.items():
    a_all = auc(PRE.values, s.values)
    a_g1, p_g1 = perm_p(PRE.values[g1], s.values[g1])
    out.append({"score": n, "AUROC all visits": a_all, "AUROC G1": a_g1, "perm p (G1)": p_g1})
a, p = perm_p(PRE.values[g1], -resid_g1)
out.append({"score": "SIMCO, EGA trend removed", "AUROC all visits": np.nan, "AUROC G1": a, "perm p (G1)": p})
print("G1:", PRE[g1].sum(), "preterm vs", (1 - PRE[g1]).sum(), "term samples (one per woman)")
pd.DataFrame(out).round(3)
""")

code(r"""
# G1 samples drawn before 26 weeks: predicted pregnancy length (weeks), shortest first
early = pd.DataFrame({"EGA": EGA, "pred DOS": arms["both"], "pred length (wk)": L,
                      "true length (wk)": EGA - DOS / 7, "preterm": PRE})[g1 & (EGA <= 26).values]
early.sort_values("pred length (wk)").round(1)
""")

md(r"""
All visits: SIMCO 0.75 vs **0.71 from EGA alone** — most of the paper's AUROC is sampling time.
**G1 only** (timing matched): SIMCO ~0.81, p ≈ 0.02, and ~0.80 with the EGA trend removed, while
EGA alone is not significant. A real-looking early-pregnancy lead — but a post hoc subgroup of
**5 women**, and in tension with §7.
""")

md(r"""
### 5b. Why the "no information" baseline is 0.71 and not 0.5, and what PR-AUC changes (2026-09-24)

There are two different "no information" scores:
- **Due-date rule.** Everyone delivers at 40 weeks, i.e. predicted DOS = 7·(EGA − 40). Then the predicted length
  7·EGA − predicted DOS is 280 days for everyone, every woman gets the same score, and the AUROC is **exactly 0.5**.
- **The paper's formula with a DOS model that knows nothing** (predicted DOS = the cohort mean). Then the predicted
  length is 7·EGA + a constant, so women sampled early *look* like short pregnancies. Preterm women are sampled earlier
  (§3), because they have already delivered before the late visits, so this score reaches **0.71**. That is the
  sampling effect: it comes from the date of the draw, not from the blood.

The due-date rule's own error, (true DOS − 7·(EGA − 40)), uses the real DOS. It is the outcome itself (§7's
`DOS_offset`, AUROC 1.0 as an oracle), not a prediction.

A perfect gestational-age clock (predicted DOS = 7·(EGA − 40)) would cancel the EGA term and fall back to 0.5. SIMCO's
predicted DOS rises only ~2.9 days per week of EGA instead of 7, so a leftover EGA term stays in the paper's score.
The "EGA trend removed" rows fit that trend within each subset and remove it.

**PR-AUC** (average precision): chance level is the preterm prevalence (~7.9 %), not 0.5. It changes the scale, not
the confound.
""")

code(r"""
from sklearn.metrics import roc_auc_score, average_precision_score
predL = 7 * EGA - arms["both"]                                   # SIMCO predicted pregnancy length (days)
cands = {
    "due-date rule (everyone at 40 wk)":                  lambda m: -(7 * EGA[m] - 7 * (EGA[m] - 40)),
    "paper formula, DOS model that knows nothing":        lambda m: -(7 * EGA[m] - DOS.mean()),
    "paper formula, SIMCO prediction":                    lambda m: -predL[m],
    "SIMCO, EGA trend removed (fit within subset)":       lambda m: -(predL[m] - np.polyval(np.polyfit(EGA[m], predL[m], 1), EGA[m])),
    "true length (oracle, uses the real DOS)":            lambda m: -(7 * EGA[m] - DOS[m]),
}
rows = []
for sub, m in [("all visits", np.ones(len(idx), bool)), ("G1 only", g1)]:
    y = PRE.values[m]
    for n, f in cands.items():
        sc = f(m).values
        rows.append({"subset": sub, "score": n, "preterm / total": f"{y.sum()}/{len(y)}",
                     "AUROC": roc_auc_score(y, sc), "PR-AUC": average_precision_score(y, sc), "chance PR-AUC": y.mean()})
print("slope of SIMCO predicted DOS on EGA (days/week):", round(np.polyfit(EGA, arms["both"], 1)[0], 2), "- a perfect clock = 7")
pd.DataFrame(rows).round(3)
""")

md(r"""
Reading:
- **All visits.** The due-date rule gives 0.5. The paper's formula with an uninformative model already gives 0.71
  (PR-AUC 0.16, 2× chance). SIMCO gives 0.75 (PR-AUC 0.29), but **0.68 once its leftover EGA trend is removed**, which
  is below the uninformative formula. So the paper's all-visit preterm AUROC is mostly sampling timing.
- **G1.** Sampling times barely differ here, so the uninformative formula only reaches 0.64. SIMCO gets **0.81 AUROC /
  0.48 PR-AUC** (6× chance) and ~0.80 AUROC with the trend removed.
- The G1 result is a lead, not a finding. There are only 5 preterm women, one woman moves the AUROC by ~0.1, and the
  subgroup was chosen after looking at the data.
""")

md(r"""
## 6. Null-map controls — is it the perturbation or the nonlinearity?

Each Fig 5b feature = median marker of a woman's Unstim cells pushed through a trained CellOT map.
Controls replace the map, same 20 splits, same schema (`baselines/null_map_features.py`):
`rraw` untrained ICNN; `rscaled` untrained ICNN scaled to the trained map's displacement;
`ctswap` same stim, another celltype's map; `crossswap` another stim **and** celltype's map.
`permuted` (stims swapped *within* a celltype) is **retracted**: 356/852 of its columns are exact
copies of real columns, so it barely changes the information.
""")

code(r"""
refs = {"Unstim only": O / "cmp_unstim_vs_simstim_n20/unstim_only",
        "gauss_ot":   O / "gaussot_stabl_n20/{t}", "CellOT": O / "cmp_unstim_vs_simstim_n20/no_Unstim"}
rows = []
for t in ["no_Unstim", "both"]:
    ds = {"Unstim only": refs["Unstim only"], "gauss_ot": O / f"gaussot_stabl_n20/{t}",
          "CellOT": O / ("cmp_unstim_vs_simstim_n20/no_Unstim" if t == "no_Unstim" else "cellot_both_n20/both")}
    for d in sorted((O / "null_maps/stabl_n20").glob(f"*/{t}")):
        ds[d.parent.name] = d
    for arm, d in ds.items():
        f = Path(d) / "Summary/Scores training CV.csv"
        if not f.exists(): continue
        s = pd.read_csv(f, index_col=0)
        for m in s.index:
            rows.append({"table": t, "arm": arm, "family": arm.split("_s")[0], "model": m, "R2": first(s.loc[m, "R2"])})
nm = pd.DataFrame(rows)
nm.pivot_table(index=["table", "model"], columns="family", values="R2", aggfunc="mean").round(3)
""")

code(r"""
# paired per-split R² vs CellOT (positive = CellOT better), written by null_maps_compare.py
pp = pd.read_csv(O / "null_maps/null_maps_paired.csv")
pp["family"] = pp.arm.str.split("_s").str[0]
pp.groupby(["family", "table", "model"]).agg(seeds=("arm", "nunique"),
    mean_diff=("mean_diff_cellot_minus_arm", "mean"), n_p_below_05=("wilcoxon_p", lambda p: int((p < .05).sum()))).round(3)
""")

md(r"""
The untrained, scale-matched ICNN recovers ~80 % of CellOT's gain over measured Unstim; CellOT's
edge over it is consistent only for Lasso on the stim-only table. **The swap controls (`ctswap`,
`crossswap`, the valid wrong-perturbation test) match CellOT**: over their 24 paired comparisons
CellOT is never significantly better, and the one p < 0.05 is a cross-swap draw *beating* it.
Which stimulation a map was trained on does not matter for predicting DOS. Caveat: the "wrong"
maps are still trained on real immune data, and stims share pathways, so they may move cells in
similar directions.

## 7. The gestational-age-free target `DOS_offset`

`DOS_offset = DOS − 7·(EGA − 40)` = days labor started before (+) / after (−) the 40-week due
date. No fitted parameter; EGA and clinical covariates predict nothing on it (R² ≈ 0).
""")

code(r"""
rows = []
for d in sorted((O / "night2/stabl_n20").glob("*")):
    f = d / "Summary/Scores training CV.csv"
    if d.name.endswith("failed_v1") or not f.exists(): continue
    s = pd.read_csv(f, index_col=0)
    for m in s.index:
        rows.append({"run": d.name, "target": "DOS_offset" if d.name.startswith("o_") else "DOS",
                     "model": m, "R2": first(s.loc[m, "R2"]), "RMSE": first(s.loc[m, "RMSE"])})
n2 = pd.DataFrame(rows)
display(n2.pivot_table(index=["target", "run"], columns="model", values="R2").round(3))
clinical_offset = pd.read_csv(O / "night2/clinical_offset_n20.csv").round(3)
clinical_offset
""")

md(r"""
**Measured Unstim, summarised properly, nearly matches CellOT on DOS with no map at all**:
`d_freqmed` (log cell frequencies + medians of all 31 markers) and `d_qtl` (+ quantiles of the 13
functional markers) reach ALasso 0.271 / 0.265 vs CellOT 0.287 on the same splits, against −0.17
for the paper's measured-Unstim block (13 functional medians only). Frequencies alone 0.09,
marker correlations alone ~0. So the information CellOT "unlocks" is largely present in the
measured Unstim data once phenotypic markers and frequencies are included.

On `DOS_offset` every feature set is below 0 (CellOT −0.03 to −0.11), at the level of predicting
the average. At 20 splits and 5 preterm women this does not exclude a small signal, but nothing
predicts labor timing beyond the due date across all visits.

## 8. Every feature set on the paper's own protocol

Same protocol as the paper's Fig 5b headline, for every feature set: Stabl on **100 splits**
(`stabl_launch/paper_protocol_arm.sh`, selections in `paper_protocol/stabl_n100/`; CellOT reuses the
lab's own run `foldfeats_GSS`), then the XGBoost second stage rescored with (a) **`params_43.json`**,
the paper's setting, and (b) **all 97 settings**, keeping the best by RMSE, the selection the lab ran.

**The second stage is refit with rows and patient groups aligned** (`stabl_launch/aligned_rescore.py`).
The lab's rescore script takes `y` in the outcome-table order and the patient groups in the
feature-table order: 94 % of its test samples have the same woman in training, and each split's
features were selected on a training set holding 80 % of its test samples. That bug leaves CellOT
unchanged (0.415 → 0.418) but inflated the other feature sets by 0.02–0.04, so only aligned numbers
are shown here.
""")

code(r"""
AL = O / "paper_protocol/aligned"
labels = {"cellot": "CellOT (paper maps)", "gaussot_both": "gauss_ot (13×13 affine map)",
          "d_freqmed": "measured: freq + 31-marker medians (no map)",
          "d_qtl": "measured: + functional-marker quantiles (no map)",
          "rscaled_s0_both": "untrained ICNN, scale-matched", "crossswap_s0_both": "cross-swap (wrong stim & celltype)",
          "comb_cellot": "CellOT stims + measured (combined)", "comb_gaussot": "gauss_ot stims + measured (combined)"}
sweeps, folds, rows = {}, {}, []
for arm, lab in labels.items():
    if not (AL / arm / "sweep.csv").exists():
        rows.append({"features": lab, "status": "pending"}); continue
    t = pd.read_csv(AL / arm / "sweep.csv"); sweeps[arm] = t
    folds[arm] = pd.read_csv(AL / arm / "folds.csv")
    rows.append({"features": lab, "status": f"{len(t)} configs",
                 "R2 params_43": t.set_index("params").loc["params_43", "R2"],
                 "R2 best of 97": t.R2.max(), "best config": t.params.iloc[0],
                 "R2 median config": t.R2.median(), "R2 worst config": t.R2.min()})
rows.append({"features": "EGA alone (reference, §2)", "status": "", "R2 params_43": r2_score(DOS, ega_only)})
pd.DataFrame(rows).round(3)
""")

code(r"""
# Paired per-split R² against CellOT on the 100 identical aligned splits; positive = better than CellOT
from scipy.stats import wilcoxon
out = []
for arm in sweeps:
    if arm == "cellot": continue
    for mode, pa, pc in [("params_43", "params_43", "params_43"),
                         ("each its best of 97", sweeps[arm].params.iloc[0], sweeps["cellot"].params.iloc[0])]:
        d = folds[arm][pa] - folds["cellot"][pc]
        out.append({"features": labels[arm], "setting": mode, "mean Δ R² per split": d.mean(),
                    "wins vs CellOT": f"{int((d > 0).sum())}/{len(d)}", "Wilcoxon p": wilcoxon(d).pvalue})
pd.DataFrame(out).round(4)
""")

code(r"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 4))
names = list(sweeps)
ax.boxplot([sweeps[a].R2 for a in names], vert=False, labels=[labels[a] for a in names], widths=.5)
for i, a in enumerate(names, 1):
    ax.plot(sweeps[a].set_index("params").loc["params_43", "R2"], i, "o", color="C3", label="params_43" if i == 1 else None)
ax.axvline(r2_score(DOS, ega_only), color="grey", ls="--", label="EGA alone")
ax.set_xlabel("pooled out-of-fold R² (100 aligned splits, STABL ALasso selections)"); ax.legend(loc="lower right", fontsize=8)
ax.set_title("Each box = the 97 XGBoost settings; red = the paper's params_43")
plt.tight_layout()
""")

md(r"""
**Reading.** With the splits aligned, CellOT is modestly but **significantly** ahead of every
alternative on the paper's protocol with `params_43`: the pooled gap is 0.02–0.09 R² and every
paired p-value is below 0.05 (the 100 splits overlap, so these p-values are somewhat optimistic).
So the maps do carry some signal beyond a good summary of the unstimulated cells and beyond maps
for the wrong perturbation. But that edge (≈0.02 over measured + quantiles) is a fraction of what
the XGBoost setting alone moves (the width of each box), and everything sits far below gestational
age alone (0.878).

## 9. The drug arm (Fig 5d/5e/5g)

The paper's drug run is `comp_9drugs_long_unstimpred_treated/run_43`. Its grid-search summary shows
that **`params_43` is the setting with the best preterm AUROC** (0.729, the maximum; only 5th by
RMSE), and that **the number of drugs with a positive predicted shift ranges from 3 to 9 across
settings**. Checks a–d (`stabl_launch/drug_arm_checks.py`, an aligned re-implementation of
`cv_drug_vs_dmso`): (a) share of drug-treated values outside the training range; (b) share of
drug-treated women farther from every untreated woman than the untreated 95th percentile;
(c) shift in units of a size-matched random-sign perturbation; (d) share of the 100 fold models
agreeing in sign. The drug perturbation is held fixed (the paper's CellOT drug maps) while the DOS
model changes (`baselines/drug_features.py`, `drug_models/`).
""")

code(r"""
GS = pd.read_csv(GH / "comp_9drugs_long_unstimpred_treated/gridsearch_summary_treated_unstim.csv")
display(GS[GS.Run == "run_43"])
print("rank of run_43 by AUROC:", int(GS.AUROC.rank(ascending=False)[GS.Run == "run_43"].iloc[0]),
      "| by RMSE:", int(GS["RMSE global"].rank()[GS.Run == "run_43"].iloc[0]), "of", len(GS))
GS.Nb_Positive_Deltas.value_counts().sort_index().rename("configs").to_frame().T
""")

code(r"""
cmp = pd.read_csv(O / "drug_models/compare_by_drug.csv")
order = ["CellOT (paper tables)", "gauss_ot", "measured + quantiles", "measured freq + medians"]
print("mean predicted shift (days; negative = labor later)")
display(cmp.pivot_table(index="drug", columns="model", values="mean shift (d)")[order].round(1).sort_values(order[0]))
print("share of the 100 fold models agreeing in sign")
display(cmp.pivot_table(index="drug", columns="model", values="folds same sign")[order].round(2))
print("shift in units of the size-matched random-sign null SD")
display(cmp.pivot_table(index="drug", columns="model", values="shift / null SD")[order].round(1))
cmp.groupby("model")[["untreated R2", "frac implausible (b)", "frac out of range (a)"]].mean().loc[order].round(3)
""")

md(r"""
**Reading.** With CellOT features, three drugs (THF, SALPZ, MAP) have a consistent predicted effect
(labor 6–9 days later, same sign in 85–95 % of the fold models, beyond a size-matched random
perturbation); the other six are coin flips. The ranking survives a change of *stim model*
(`gauss_ot`, Spearman +0.75, which adds RIF) but **not** a DOS model built on unstimulated features
(Spearman +0.17), where SA, RIF and CHT take over. That comparison is not like-for-like: the
measured model only sees the drug through the Unstim→drug maps, the others through the
stim→drug maps, so THF/SALPZ/MAP's effect lives in the *drug-under-stimulation* features. But in
the CellOT model 55 % of drug-treated women are implausible (vs 7–9 % for the measured model): the
predicted effects are extrapolations of a model dominated by gestational age. The rebuilt CellOT
drug tables match the lab's to ≤1e-4 except SALPZ (22 columns, up to 0.64) and RIF (4 columns),
unexplained; SALPZ's shift moves −6.8 → −7.8 with them, ranking unchanged.

## 10. Commands


```bash
python stabl_launch/clinical_baselines.py                    # EGA-only / clinical, 100 splits
python -m baselines.null_map_features --map-kind <kind> ...  # null-map feature tables
sbatch --array=... stabl_launch/null_maps.sbatch             # null-map Stabl runs (arms.txt)
python -m baselines.distrib_features --out night2/distrib_all.csv
sbatch --array=... stabl_launch/night2.sbatch                # DOS / DOS_offset runs (night2/arms.txt)
sbatch --array=... stabl_launch/paper_protocol.sbatch        # 100-split Stabl + lab rescore (bug-affected, see §8)
sbatch --array=... stabl_launch/aligned_rescore.sbatch       # aligned 97-config rescore (paper_protocol/aligned/arms.txt)
python stabl_launch/drug_arm_checks.py                       # drug-arm checks a-d (CellOT)
sbatch --array=0-8 stabl_launch/drug_models_build.sbatch     # drug-treated tables per DOS model, then drug_models_checks.sbatch
python notebooks/build_ega_preterm.py && jupyter nbconvert --to notebook --execute --inplace notebooks/ega_preterm_nullmaps.ipynb
```
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9"},
}
out = "/scratch/users/bmerwan/simco/notebooks/ega_preterm_nullmaps.ipynb"
nbf.write(nb, out)
print("wrote", out, f"({len(cells)} cells)")
