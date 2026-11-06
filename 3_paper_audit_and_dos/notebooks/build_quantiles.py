"""Generate notebooks/quantiles.ipynb: everything about quantile features in one place, measured and simulated.

Every table is recomputed from result files (same convention as the other notebooks).
    python notebooks/build_quantiles.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/quantiles.ipynb
Needs simco_work/diagnosis/shap_by_feature{,_simq}.csv (stabl_launch/diag_shap.py [--simq]).
"""
import nbformat as nbf

cells = []
md = lambda t: cells.append(nbf.v4.new_markdown_cell(t.strip()))
code = lambda t: cells.append(nbf.v4.new_code_cell(t.strip()))

md(r"""
# Quantile features — measured and simulated, in one place

**Two kinds of quantile features** (10th, 25th, 75th, 90th percentile of a marker in a celltype, per woman):
- **measured** quantiles of her unstimulated cells: `<celltype>_<marker>_q<k>_qtl`, 13 functional markers.
  Their median counterpart is `<celltype>_<marker>_med`, which also exists for the 18 phenotypic markers.
- **simulated** quantiles of her cells after a simulated stimulation (CellOT or gauss_ot map):
  `<celltype>_<marker>_q<k>_<stim>q`. The median counterpart is the paper's feature, `<celltype>_<marker>_<stim>`.

**Models** (Stabl ALasso, 100 splits, then XGBoost `params_43` with aligned rows):

| name | features |
|---|---|
| `d_qtl` | measured only: frequencies + 31-marker medians + measured quantiles |
| `comb_cellot` / `comb_gaussot` | simulated stim medians + the measured set above |
| `comb_cellot_simq` / `comb_gaussot_simq` | the same + the simulated quantiles |

The questions, in order:
1. Does a stimulation move the tails like the median?
2. How well is each quantile's shift predicted?
3. How correlated is each quantile feature with its median?
4. How correlated is it with DOS, and beyond its median?
5. Does Stabl select it, as an addition or a replacement?
6. How reliable is that selection?
7. Does XGBoost use it?
8. Does it change R²?

`36_G1` and `52_G1` (the two mishandled visits) are excluded from every OOL-cohort analysis.
""")

code(r"""
import ast, re, json, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200, "display.max_columns", 40, "display.max_rows", 60)
W = Path("/scratch/users/bmerwan/simco_work"); O = W / "onset_test"; P = O / "paper_protocol"; S100 = P / "stabl_n100"
EXCL = ["36_G1", "52_G1"]; FLOOR = 0.0273
STIMS = ["TNFa", "LPS", "IL246", "IFNa", "GMCSF", "PI", "IL33"]
FUNC = [l.strip() for l in open("/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/13features.txt") if l.strip()]
QK = [10, 25, 75, 90]

def family(f):
    if f.endswith("_freq"): return "measured frequency"
    if f.endswith("_qtl"): return "measured quantile"
    if f.endswith("_med"): return "measured median (functional)" if f.split("_")[-2] in FUNC else "measured median (phenotypic)"
    if re.search(r"_q\d+_[A-Za-z0-9]+q$", f): return "simulated quantile"
    if f.endswith("_Unstim"): return "measured median (functional)"
    if f.split("_")[-1] in STIMS: return "simulated median"
    return "other"

def own_median(f):
    # the median feature of the same (celltype, marker[, stim]) as a quantile feature
    m = re.match(r"(.+)_q\d+_qtl$", f)
    if m: return m.group(1) + "_med"
    m = re.match(r"(.+)_q\d+_([A-Za-z0-9]+)q$", f)
    return f"{m.group(1)}_{m.group(2)}" if m else None

def qlevel(f):
    m = re.search(r"_q(\d+)_", f); return int(m.group(1)) if m else None

X = pd.read_csv(P / "tables/comb_cellot_simq.csv", index_col=0)
XG = pd.read_csv(P / "tables/comb_gaussot_simq.csv", index_col=0)
y = pd.read_csv(O / "outcome_table_all_pre.csv", index_col=0)["DOS"]
keep = X.index.intersection(y.index).drop(EXCL, errors="ignore")
X, XG, y = X.loc[keep], XG.reindex(keep), y.loc[keep]
fam = pd.Series({c: family(c) for c in X.columns})
print(f"{len(keep)} samples; features per family (CellOT table):"); print(fam.value_counts().to_string())
""")

md(r"""
## 1. Does a stimulation move the tails like the median? (atlas, real cells)

This is the reason quantiles could carry *new* perturbation information. The data are the per-donor cells of the drug
screen (DMSO wells, 13 functional markers; the per-donor cache of the learning-curve project, read-only). For each
(stim, celltype, marker) whose median shift exceeds the noise floor, and for each donor, shift_q = q(stimulated) −
q(unstimulated) is computed at q = 10/25/50/75/90. Two views:
- **Population level**: across all (stim, celltype, marker), how does the average q10 / q90 shift compare with the
  average median shift?
- **Donor level**: across the 19 donors, are the donors whose median shifts more also the ones whose tails shift
  more? If the tails move in lockstep with the median, a quantile adds little that the median does not already say.
""")
code(r"""
CACHE = Path("/scratch/users/bmerwan/cellot_lc_work/cache")
pairs = [tuple(l.split(",")[:2]) for l in open("/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/valid_jobs.txt") if l.strip()]
QS5 = [10, 25, 50, 75, 90]
rows = []
for ct in sorted({c for _, c in pairs}):
    z = np.load(CACHE / f"{ct}.npz")
    for st in [s for s, c in pairs if c == ct]:
        for d in sorted({k.split("|")[1] for k in z.files}):
            if f"Unstim|{d}" not in z.files or f"{st}|{d}" not in z.files: continue
            u, s = z[f"Unstim|{d}"], z[f"{st}|{d}"]
            if len(u) < 50 or len(s) < 50: continue
            sh = np.percentile(s, QS5, 0) - np.percentile(u, QS5, 0)
            for j, mk in enumerate(FUNC):
                rows.append({"stim": st, "celltype": ct, "marker": mk, "donor": d, **{f"q{q}": sh[i, j] for i, q in enumerate(QS5)}})
SH = pd.DataFrame(rows)
pop = SH.groupby(["stim", "celltype", "marker"])[[f"q{q}" for q in QS5]].mean()
pop = pop[pop.q50.abs() > FLOOR]
print(f"{len(pop)} (stim, celltype, marker) with a median shift above the floor")
t1 = pd.DataFrame({f"q{q}": {"corr with the median shift (across features)": np.corrcoef(pop[f"q{q}"], pop.q50)[0, 1],
                              "median |shift_q| / |shift_median|": (pop[f"q{q}"].abs() / pop.q50.abs()).median(),
                              "same sign as the median shift": (np.sign(pop[f"q{q}"]) == np.sign(pop.q50)).mean()}
                   for q in [10, 25, 75, 90]}).T
display(t1.round(2).style.set_caption("Population level: average shifts across donors"))
don = []
for key, g in SH.groupby(["stim", "celltype", "marker"]):
    if key not in pop.index or len(g) < 10: continue
    don.append({f"q{q}": spearmanr(g[f"q{q}"], g.q50)[0] for q in [10, 25, 75, 90]})
don = pd.DataFrame(don)
display(don.describe(percentiles=[.25, .5, .75]).loc[["count", "25%", "50%", "75%"]].round(2)
        .style.set_caption("Donor level: Spearman, across donors, of shift_q with the median shift (per feature)"))
""")
code(r"""
# same question on the SIMULATED responses of the OOL women: simulated q minus measured unstimulated q, per woman
MQ = {q: X[[c for c in X.columns if c.endswith(f"_q{q}_qtl")]] for q in QK}
rows = []
for c in [c for c in X.columns if fam[c] == "simulated median"]:
    ct_m, st = c.rsplit("_", 1)
    base_med = f"{ct_m}_med"
    if base_med not in X or f"{ct_m}_q90_{st}q" not in X: continue
    d50 = X[c] - X[base_med]
    r = {"feature": c}
    for q in QK:
        sq, mq = f"{ct_m}_q{q}_{st}q", f"{ct_m}_q{q}_qtl"
        if sq in X and mq in X:
            r[f"q{q}"] = spearmanr(X[sq] - X[mq], d50, nan_policy="omit")[0]
    rows.append(r)
SIMSH = pd.DataFrame(rows).set_index("feature")
display(SIMSH.describe(percentiles=[.25, .5, .75]).loc[["count", "25%", "50%", "75%"]].round(2)
        .style.set_caption("OOL women, CellOT-simulated response: Spearman, across women, of the simulated shift at q with the shift at the median"))
""")

md(r"""
## 2. How well is each quantile's shift predicted? (atlas, pooled cells)

The same computation as `diagnosis.ipynb` Part 2, summarised: skill = 1 − error / |true shift|, median over the
(pair, marker) cases whose true shift exceeds the floor.
""")
code(r"""
M = {"cellot_insample": "CellOT", "gauss_ot": "gauss_ot", "median_shift": "median_shift", "quantile_map": "quantile_map", "identity": "identity"}
QL = ["q10", "q25", "median", "q75", "q90"]
def long_q(raw, keys):
    raw = raw[raw.weighting.isin(["cell", "invariant"]) & raw.method.isin(M)]
    parts = []
    for q in QL:
        pre = "median_error__" if q == "median" else f"{q}_error__"
        mk = [c[len(pre):] for c in raw.columns if c.startswith(pre)]
        m = raw.melt(id_vars=keys + ["method"], value_vars=[pre + x for x in mk], var_name="marker", value_name="err")
        m["marker"] = m.marker.str[len(pre):]; m["q"] = q; parts.append(m)
    return pd.concat(parts)
out = {}
for arm, d, keys in [("stims", "pooled_q", ["celltype", "stim"]), ("drugs", "pooled_drug_q", ["celltype", "stim", "drug"])]:
    raw = pd.concat(pd.read_csv(f) for f in sorted((W / "baselines" / d).glob("*.csv")))
    w = long_q(raw, keys).pivot_table(index=keys + ["marker", "q"], columns="method", values="err").rename(columns=M)
    w = w[w.identity > FLOOR]
    out[arm] = pd.DataFrame({q: {m: (1 - w.xs(q, level="q")[m] / w.xs(q, level="q").identity).median()
                                 for m in ["CellOT", "gauss_ot", "quantile_map", "median_shift"]} for q in QL})
display(pd.concat(out).round(2).style.set_caption("Median skill per quantile, effects above the noise floor"))
""")

md("## 3. How correlated is a quantile feature with its own median? (OOL women, as features)")
code(r"""
rows = []
for c in [c for c in X.columns if fam[c] in ("measured quantile", "simulated quantile")]:
    m = own_median(c)
    if m in X:
        rows.append({"kind": fam[c], "q": qlevel(c), "r": X[c].corr(X[m])})
R3 = pd.DataFrame(rows)
display(R3.groupby(["kind", "q"]).r.describe(percentiles=[.1, .5, .9])[["count", "10%", "50%", "90%"]].round(2)
        .style.set_caption("Pearson r of each quantile feature with its own median, across women"))
""")

md(r"""
## 4. Correlation with DOS — raw, and beyond the feature's own median

For each quantile feature: Spearman ρ with DOS, next to its own median's ρ. Then the **partial** correlation of the
quantile with DOS once its median is regressed out of both: what the quantile says about DOS that its median does not.
The p-values treat the 175 visits as independent although women have up to 3 visits, so they are optimistic.
""")
code(r"""
from scipy.stats import rankdata, pearsonr
def partial_spearman(a, b, ctrl):
    ra, rb, rc = (rankdata(v) for v in (a, b, ctrl))
    res = lambda v: v - np.polyval(np.polyfit(rc, v, 1), rc)
    return pearsonr(res(ra), res(rb))
rows = []
for c in [c for c in X.columns if fam[c] in ("measured quantile", "simulated quantile")]:
    m = own_median(c)
    if m not in X: continue
    ok = X[[c, m]].notna().all(1)
    rq, pq = spearmanr(X.loc[ok, c], y[ok]); rm, _ = spearmanr(X.loc[ok, m], y[ok])
    pr, pp = partial_spearman(X.loc[ok, c].values, y[ok].values, X.loc[ok, m].values)
    rows.append({"kind": fam[c], "q": qlevel(c), "feature": c, "rho_quantile": rq, "rho_own_median": rm,
                 "partial_rho_given_median": pr, "p_partial": pp})
R4 = pd.DataFrame(rows)
R4.to_csv(W / "diagnosis/quantiles_dos_correlation.csv", index=False)
summ = R4.groupby("kind").agg(features=("feature", "size"),
        median_abs_rho_quantile=("rho_quantile", lambda v: v.abs().median()),
        median_abs_rho_own_median=("rho_own_median", lambda v: v.abs().median()),
        share_quantile_stronger=("rho_quantile", lambda v: np.nan),
        median_abs_partial=("partial_rho_given_median", lambda v: v.abs().median()),
        share_partial_p_lt_001=("p_partial", lambda v: (v < 0.001).mean()))
summ["share_quantile_stronger"] = R4.assign(s=R4.rho_quantile.abs() > R4.rho_own_median.abs()).groupby("kind").s.mean()
display(summ.round(3))
display(R4.groupby(["kind", "q"]).agg(median_abs_rho=("rho_quantile", lambda v: v.abs().median()),
        median_abs_partial=("partial_rho_given_median", lambda v: v.abs().median())).round(3).unstack("kind"))
print("largest partial correlations (the quantiles that say most beyond their median):")
display(R4.reindex(R4.partial_rho_given_median.abs().sort_values(ascending=False).index).head(12).round(3))
""")

md(r"""
## 5. What Stabl selects (100 splits), and are the quantiles additions or replacements?

Selections per split, by feature family, for each model. Then, for the simulated-quantile models, whether the
non-quantile selection is identical to the model without quantiles: Stabl selects block by block, so identical means
the quantiles are pure additions. Finally the most selected quantiles, with their own median's correlation and
selection frequency.
""")
code(r"""
def sels(run):
    s = pd.read_csv(S100 / run / "Training CV/Selected Features STABL ALasso.csv", index_col=0)["Fold selected features"].apply(ast.literal_eval)
    return list(s)
RUNS = {"measured only (d_qtl)": "d_qtl", "CellOT + measured": "comb_cellot", "gauss_ot + measured": "comb_gaussot",
        "CellOT + measured + sim. q": "comb_cellot_simq", "gauss_ot + measured + sim. q": "comb_gaussot_simq"}
SEL = {n: sels(r) for n, r in RUNS.items()}
rows = []
for n, L in SEL.items():
    c = pd.Series([family(f) for l in L for f in l]).value_counts() / len(L)
    rows.append(c.rename(n))
display(pd.DataFrame(rows).fillna(0).round(1).style.set_caption("Features selected per split, by family (mean over 100 splits)"))
for a, b in [("CellOT + measured + sim. q", "CellOT + measured"), ("gauss_ot + measured + sim. q", "gauss_ot + measured")]:
    same = sum(set(f for f in la if family(f) != "simulated quantile") == set(lb) for la, lb in zip(SEL[a], SEL[b]))
    print(f"{a}: non-quantile selection identical to '{b}' in {same}/100 splits")
qlev = []
for n, L in SEL.items():
    q = pd.Series([qlevel(f) for l in L for f in l if family(f) in ("measured quantile", "simulated quantile")])
    if len(q): qlev.append((q.value_counts(normalize=True).sort_index() * 100).rename(n))
display(pd.DataFrame(qlev).round(0).style.set_caption("Which quantile level is selected (% of that model's quantile selections)"))
""")
code(r"""
for n in ["CellOT + measured + sim. q", "measured only (d_qtl)"]:
    L = SEL[n]; vc = pd.Series([f for l in L for f in l]).value_counts()
    Xs = X if "gauss" not in n else XG
    nonq = [f for f in vc.index if family(f) not in ("measured quantile", "simulated quantile") and f in Xs]
    rows = []
    for f in [f for f in vc.index if family(f) in ("measured quantile", "simulated quantile")][:12]:
        m = own_median(f)
        rows.append({"quantile feature": f, "splits selected": vc[f], "r with own median": Xs[f].corr(Xs[m]) if m in Xs else np.nan,
                     "own median selected (splits)": int(vc.get(m, 0)),
                     "max |r| with a selected non-quantile feature": Xs[nonq].corrwith(Xs[f]).abs().max()})
    display(pd.DataFrame(rows).set_index("quantile feature").round(2).style.set_caption(f"{n}: most selected quantiles"))
""")

md(r"""
## 6. How reliable is the selection? (Stabl's own precision estimate per block)

Per block and split, Stabl picks the threshold minimising FDP+ = (1 + knockoffs above threshold) / (real features
above threshold). 1 − median FDP+ is its estimate of the share of true discoveries in that block. 0 means that
everything selected could be noise.
""")
code(r"""
rows = []
for n, r in RUNS.items():
    for f in sorted((S100 / r / "Training CV/Stabl features STABL ALasso").glob("Stabl features STABL ALasso *.csv")):
        blk = f.stem.replace("Stabl features STABL ALasso ", "")
        t = pd.read_csv(f, index_col=0)
        if "min FDP+" not in t: continue
        kind = "simulated quantiles" if blk.endswith("q") and blk[:-1] in STIMS else ("measured quantiles" if blk == "qtl" else
               "simulated medians" if blk in STIMS else {"med": "measured medians", "freq": "frequencies"}.get(blk, blk))
        rows.append({"model": n, "block": blk, "kind": kind, "precision": 1 - t["min FDP+"].median()})
PR = pd.DataFrame(rows)
display(PR.pivot_table(index="kind", columns="model", values="precision", aggfunc="median").round(2)
        .style.set_caption("Estimated precision (1 − median FDP+), median over the blocks of each kind"))
display(PR[PR.kind == "simulated quantiles"].pivot_table(index="block", columns="model", values="precision").round(2))
""")

md("## 7. Does XGBoost use them? (|TreeSHAP| share by family, 100 splits, `params_43`)")
code(r"""
sh = pd.concat([pd.read_csv(W / "diagnosis/shap_by_feature.csv"), pd.read_csv(W / "diagnosis/shap_by_feature_simq.csv")])
sh["family"] = sh.feature.map(family)
share = sh.groupby(["model", "family"]).abs_shap.sum().unstack("family").fillna(0)
display((100 * share.div(share.sum(1), axis=0)).round(1).style.set_caption("% of total |SHAP| by feature family"))
""")

md(r"""
## 8. Does adding them change the prediction? (R², aligned rows, `params_43`)

The 20-split and 100-split numbers are **not comparable with each other**: the pooled R² depends on the number of
splits. Comparisons are made within one split count.
""")
code(r"""
rows = []
for tag, lab in [("aligned_n20", "20 splits, all 177"), ("aligned_n20_excl", "20 splits, without 36_G1 / 52_G1"),
                 ("aligned", "100 splits, all 177"), ("aligned_excl", "100 splits, without 36_G1 / 52_G1")]:
    for a, ref in [("comb_cellot_simq", "comb_cellot"), ("comb_gaussot_simq", "comb_gaussot"), ("d_qtl", "d_freqmed")]:
        fa, fr = P / tag / a / "sweep.csv", P / tag / ref / "sweep.csv"
        if not (fa.exists() and fr.exists()): continue
        sa, sr = pd.read_csv(fa).set_index("params"), pd.read_csv(fr).set_index("params")
        da = pd.read_csv(P / tag / a / "folds.csv")["params_43"] - pd.read_csv(P / tag / ref / "folds.csv")["params_43"]
        r = {"setting": lab, "with quantiles": a, "R² with": sa.loc["params_43", "R2"], "R² without": sr.loc["params_43", "R2"],
             "per-split Δ": da.mean(), "p": wilcoxon(da.dropna()).pvalue, "splits better": f"{(da > 0).sum()}/{da.notna().sum()}"}
        if len(sa) == 97 and len(sr) == 97:
            r["best of 97 with / without"] = f"{sa.R2.max():.3f} / {sr.R2.max():.3f}"
        rows.append(r)
display(pd.DataFrame(rows).round(3))
""")

code(r"""
# measured medians stuck at the floor: for these (celltype, marker) only the quantiles vary between women
flat = [c for c in X.columns if fam[c] == "measured median (functional)" and c.endswith("_med") and X[c].nunique() <= 2]
print(f"{len(flat)} measured functional medians take <= 2 values across the 175 visits (at 0):",
      pd.Series([c.split('_')[-2] for c in flat]).value_counts().to_dict())
""")

md(r"""
## 9. Reading (2026-09-28)

**1. Stimulations mostly move the tails with the median, but not entirely.**
- At population level the q25/q75 shifts track the median shift almost exactly (r 0.95–0.97, same sign 98–100 %).
  The q90 shift is larger (1.24× the median shift) and the q10 shift smaller (0.42×): stims stretch the upper tail.
- Between donors, a donor whose median shifts more also has tails that shift more: Spearman 0.89–0.92 at q25/q75,
  0.82 at q90, 0.72 at q10.
- So quantiles carry a little donor-specific response information beyond the median, mostly in the extreme tails.
  The CellOT-simulated responses of the OOL women show the same structure (0.86 at q25, 0.67 at q90).

**2. Quantile shifts are well predicted, but best by the per-marker quantile map** (stims: 0.83–0.96 against
CellOT's 0.81–0.89). median_shift fails at the tails by construction.

**3. As features, quantiles are near-copies of their median**, except for the measured extreme tails.
- Simulated: r 0.92 at q25/q75, 0.79 at q10/q90.
- Measured: r 0.86–0.88 at q25/q75, lower at q10 (0.59) and q90 (0.72).
- **68 measured medians are stuck at 0** (at most 2 distinct values over the 175 visits; pSTAT1, pp38, pSTAT6 at baseline). For those, only the upper quantiles
  vary, so the quantile is the only measured information about that marker.

**4. Correlation with DOS: the quantiles do carry some information beyond their median.**
- Raw |ρ| is similar to the median's (~0.1).
- The partial ρ given the median is ~0.08 for simulated quantiles and ~0.05 for measured ones.
- 6.5 % of simulated and 2.5 % of measured quantiles have a partial correlation with p < 0.001 (optimistic,
  repeated visits).
- The strongest are **pSTAT1 upper tails** (q75/q90), measured or simulated under IFNa, in naive / memory CD4 T and NK
  cells (partial ρ 0.41–0.49), exactly where the pSTAT1 median is stuck at 0. That is a biologically coherent axis
  (type-I interferon signalling) that the median-only features cannot see.

**5–7. Stabl selects them, but as redundant additions.**
- Simulated quantiles: ~28 per split for CellOT (25 for gauss_ot), i.e. 41 % of the selections, and **~30 % of
  XGBoost's |SHAP|**, taken from the other families.
- The non-quantile selection is **identical in 100/100 splits**, so they are pure additions.
- The most selected ones correlate 0.90–0.99 with their own median, which is usually selected too.
- Stabl's own precision estimate for the quantile blocks is **0** (IFNa, IL246, LPS, GMCSF) to 0.5–0.67 (PI,
  TNFa, IL33), against 0.88 for the measured medians. Measured quantiles: ~1.8 per split, mostly q90, precision 0.
- The pSTAT1-tail features from point 4 are **not** among the most selected.

**8. R² does not move.**
- At 100 splits, aligned: CellOT + measured 0.452 with vs 0.449 without (p 0.54), 0.449 vs 0.448 without the two
  visits. gauss_ot + measured: 0.447 vs 0.441, and 0.445 vs 0.446.
- Measured quantiles on top of measured frequencies + medians: 0.397 vs 0.384 (p 0.29).

**Overall.** Quantiles are mostly the median in another form. The one place where they carry distinct information
is the upper tail of markers whose median sits at the floor, notably pSTAT1 under IFNa. That signal correlates with
DOS beyond the median, but the pipeline neither selects it reliably nor turns it into better prediction. A targeted
test (a pSTAT1-tail block, or tail features only for floor-median markers) would be the way to check whether it is
real or a correlation with gestational age.
""")


md(r"""
## 10. Held-out re-test: quantile feature sets at 100 splits (2026-10-02)

Computed in the separate project `atlas_heldout` (`heldout/quantile_arms.py`, `heldout/selq_corr.py`,
`heldout/analyze_quantile_arms.py`, `heldout/quantile_selection_report.py`; state in `atlas_heldout/work/STATUS.md`),
read here. Differences with §5–8: **retrained** CellOT stim maps (all 19 donors, all 196 pairs, last iterate),
36_G1 / 52_G1 removed **before** Stabl, and a new inclusion rule. Same Stabl (ALasso, 100 splits) and aligned
XGBoost rescore (`params_43` and the 97 settings).

**Inclusion rule.** A (stim, celltype, marker) gets a simulated feature if *any* quantile q10–q90 moves by > 0.1
(mean over the 19 atlas donors of |per-donor shift|), with **no IL33 exception** → 1,085 medians (vs the paper's 852:
97 IL33 features dropped, 331 added, mostly upper tails under PI / LPS / IL-2/4/6).

| set | content |
|---|---|
| `ref852` | the paper's 852 simulated medians + measured frequencies, 31-marker medians, functional quantiles |
| `rule_med` | the 1,085 rule medians + the same measured features |
| `rule_rawq` | `rule_med` + all simulated q10/q25/q75/q90 of those features |
| `rule_selq` | only the simulated (feature, quantile) whose own shift passes 0.1 |
| `rule_shape` | median + spreads (q90 − q50, q50 − q10), simulated and measured |
| `rule_rep` | one quantile per feature: the most variable across women |
| `rule_corr` | `rule_rawq` with near-duplicates removed: columns visited medians → frequencies → quantiles, most variable first; kept unless \|r\| ≥ 0.9 with a kept column |
| `selq_corr` | the same filter on `rule_selq` |
| `selq_corr_eff` | same, but inside a group of near-copies the column kept is the one the stim moves most; measured medians and frequencies first |
""")
code(r"""
HO = Path("/scratch/users/bmerwan/atlas_heldout/work/heldout/quantile_arms")
sets = pd.read_csv(HO / "summary/sets.csv", index_col=0)
display(sets.T.style.set_caption("R² (aligned, 100 splits), selected features and what they are made of"))
pr = pd.read_csv(HO / "summary/paired.csv")
display(pr[pr.vs == "ref852"].round(4).style.hide(axis="index").set_caption("Per-split paired tests against ref852"))
""")
md(r"""
### 10.1 Which quantiles are selected
Share of all selections (100 splits) by kind and quantile, and simulated-quantile selections by stim.
""")
code(r"""
SR = HO / "selection_report"
byq = pd.read_csv(SR / "by_q.csv", index_col=[0, 1]); byq.index = byq.index.set_names(["kind", "q"])
display((byq * 100).round(1).style.background_gradient(cmap="Blues", axis=None).set_caption("% of selections by kind and quantile (q = -1: frequency)"))
display((pd.read_csv(SR / "by_stim.csv", index_col=0) * 100).round(1).style.background_gradient(cmap="Blues", axis=None)
        .set_caption("% of selections that are simulated quantiles, by stim"))
""")
md(r"""
### 10.2 What the most selected quantiles stand for, and how much they move
For each simulated quantile selected in ≥ 50 % of splits: `ratio_vs_median` = the atlas stim shift at that quantile
divided by the shift at the median (> 1: the tail moves more than the median); `r_own_median` = correlation across the
175 women with its own simulated median (≈ 1: it is the median again); `group_size` / `group_members` = the
near-duplicates (|r| ≥ 0.9) the correlation filter merged into it; `implausible` = extrapolated values (atlas_vs_ool).
""")
code(r"""
cols = ["column", "sel_freq", "q", "shift_q", "shift_q50", "ratio_vs_median", "r_own_median", "group_size", "group_r_median", "group_members", "implausible"]
for s in ["rule_corr", "selq_corr", "selq_corr_eff"]:
    F = pd.read_csv(SR / f"features_{s}.csv")
    t = F[(F.kind == "sim quantile") & (F.sel_freq >= 0.5)].sort_values("sel_freq", ascending=False)
    display(t[[c for c in cols if c in t]].round(2).style.hide(axis="index").set_caption(f"{s}: simulated quantiles selected in ≥ 50 % of splits"))
""")
md(r"""
### 10.3 Are the selected quantiles the ones that move more than their median, and are they copies of it?
Selected simulated quantiles (weighted by how often they are selected) against all simulated quantiles offered to Stabl.
""")
code(r"""
import re
inc = pd.read_csv("/scratch/users/bmerwan/atlas_heldout/work/heldout/features/inclusion.csv")
shift = {(r.celltype, r.marker, r.stim, q): getattr(r, f"shift_q{q}") for r in inc.itertuples() for q in (10, 25, 50, 75, 90)}
simq = pd.read_csv("/scratch/users/bmerwan/atlas_heldout/work/heldout/features/ool_cellot_allq.csv", index_col=0).drop(["36_G1", "52_G1"], errors="ignore")
def cand(s):
    cols = pd.read_csv(HO / f"tables/{s}.csv", nrows=0, index_col=0).columns
    out = []
    for c in cols:
        m = re.match(r"(.*)_([^_]+)_q(\d+)_(\w+)q$", c)
        if not m: continue
        ct, mk, q, st = m.groups(); q = int(q); med = f"{ct}_{mk}_{st}"
        r50 = shift.get((ct, mk, st, 50)); rq = shift.get((ct, mk, st, q))
        rr = simq[c].corr(simq[med]) if med in simq and c in simq else np.nan
        out.append({"column": c, "q": q, "ratio_vs_median": rq / r50 if r50 else np.nan, "r_own_median": rr})
    return pd.DataFrame(out)
rows = []
for s in ["rule_rawq", "rule_selq", "rule_corr", "selq_corr", "selq_corr_eff"]:
    C = cand(s); F = pd.read_csv(SR / f"features_{s}.csv"); F = F[F.kind == "sim quantile"]
    w = F.sel_freq / F.sel_freq.sum()
    wmed = lambda x: np.interp(0.5, np.cumsum(w.to_numpy()[np.argsort(x.to_numpy())]), np.sort(x.to_numpy()))
    rows.append({"set": s, "offered": len(C), "ever selected": len(F),
                 "ratio_vs_median: offered (median)": C.ratio_vs_median.median(),
                 "ratio_vs_median: selected (weighted median)": wmed(F.ratio_vs_median.fillna(1)),
                 "share offered moving more than median": (C.ratio_vs_median > 1).mean(),
                 "share of selections moving more than median": (w * (F.ratio_vs_median > 1)).sum(),
                 "r_own_median: offered (median)": C.r_own_median.median(),
                 "r_own_median: selected (weighted median)": wmed(F.r_own_median.fillna(1)),
                 "share of selections with r_own_median < 0.9": (w * (F.r_own_median < 0.9)).sum(),
                 "share of selections implausible": (w * F.implausible).sum()})
display(pd.DataFrame(rows).set_index("set").T.round(2).style.set_caption("Selected vs offered simulated quantiles"))
""")


md(r"""
### 10.4 Reading (2026-10-02)

**Prediction.** Every set with simulated quantiles beats `ref852` per split, by +0.043 to +0.052 R²
(65–72 / 100 splits, p < 1e-4), except `rule_shape` (spreads instead of raw quantiles: nothing) and `rule_rep` (+0.027).
Best: `rule_corr` 0.493 (`params_43`) / 0.524 (best of 97), against 0.450 / 0.484 for `ref852`. In days that is
23.1 → 22.2 RMSE, against 10.9 for gestational age alone. This **contradicts §8** (no gain from simulated quantiles,
paper's maps, two visits removed only after Stabl); which change explains it is still open.

**Which quantiles.** In every set about **half of all selections are simulated quantiles**, and they are mostly
**upper tails**: q90 is 27–33 % of all selections, q75 7–14 %, q10/q25 4–9 % each; simulated medians 22–30 %.
Measured quantiles are ~7 % of selections, almost all q90. By stim, IL-2/4/6 and PI lead (8–13 % of selections each).

**They are the tails that move more than their median.** Weighted by how often they are selected, the selected
quantiles move **1.3–1.6× as much as their own median** under the stim, against 0.8–1.1× for all quantiles offered;
68–82 % of selections are quantiles that move more than their median, against 37–60 % of those offered. Extreme cases:
`CD4Teff_pSTAT1_q90_PIq` (the median does not move, the tail moves 11× more), `CD4Tregs_pSTAT5_q75_PIq` (4.9×). So Stabl
does pick up tail-specific responses, the very thing a median feature cannot represent.

**But they are not independent of the median.** The selected ones correlate 0.78–0.87 with their own simulated median
across women, about as much as the quantiles offered: they share ~70 % of their variance with it.

**What the most selected ones stand for** (groups merged by the correlation filter, \|r\| ≥ 0.9):
- `CD4Tnaive_pSTAT1_q90_LPSq` (100 / 100 splits in every set) stands for 7–10 columns: the **pSTAT1 upper tail** in naive /
  memory CD4 and CD8 T cells under IFNa or LPS. This is the type-I interferon axis of §4 (partial ρ with DOS 0.41–0.49).
- `CD4Tem_pSTAT3_q90_IL246q` / `CD4Tnaive_pSTAT3_q90_PIq` stand for 12–14 **pSTAT3 upper tails** in CD4 / CD8 T cells under
  IL-2/4/6, PI and TNFa: the naive-CD4 pSTAT3 axis that also carries the drug arm (THF / SALPZ).
- `CD4Tregs_pSTAT5_*` (pSTAT5 in Tregs under PI / IFNa) and `CD4Tcm_CCR2pos_pSTAT3_q90_GMCSFq` (stands for nothing else).

**Caveats.** 1–5 % of selections are columns with impossible values (map extrapolation, atlas–OOL input shift), e.g.
`CD8Tcm_CCR2pos_pNFkB_q10_TNFaq` (90 % of splits in `selq_corr_eff`) and CD56hi NK HLADR low tails. And none of these
sets has been tested *beyond gestational age* yet (§4 of `ega_preterm_nullmaps.ipynb`-style partial correlation).
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python", "version": "3.9"}}
out = "/scratch/users/bmerwan/simco/notebooks/quantiles.ipynb"
nbf.write(nb, out); print("wrote", out, f"({len(cells)} cells)")
