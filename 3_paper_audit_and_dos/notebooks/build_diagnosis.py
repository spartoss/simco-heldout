"""Generate notebooks/diagnosis.ipynb -- (1) which added features are useful in the combined models,
and how reliable Stabl's selections are; (2) how well CellOT / gauss_ot / baselines predict the
QUANTILES of each marker after stim or drug simulation on the atlas.

Every table recomputed from result files (same convention as the other notebooks).
    python notebooks/build_diagnosis.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/diagnosis.ipynb
"""
import nbformat as nbf

cells = []
md = lambda t: cells.append(nbf.v4.new_markdown_cell(t.strip()))
code = lambda t: cells.append(nbf.v4.new_code_cell(t.strip()))

md(r"""
# Diagnosis — added features, selection reliability, and quantile prediction

**Part 1.** In the models that combine simulated stims with measured features (and in the measured-only
model), which feature families does Stabl keep, how reliably, and which ones actually drive the XGBoost
predictions? Three views: *selection* (how often a feature is picked across the 100 splits), *estimated
precision* (Stabl's own false-discovery estimate per block), and *use* (TreeSHAP contribution in the
refitted prediction models).

**Part 2.** On the atlas (19 donors, held-out test cells), how well are the 10th/25th/50th/75th/90th
percentiles of each marker predicted after a simulated stimulation or drug — the quantities a
"simulated quantile" feature would be made of.
""")

code(r"""
import ast, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200, "display.max_columns", 40, "display.max_rows", 80)
W = Path("/scratch/users/bmerwan/simco_work"); O = W / "onset_test"; B = W / "baselines"
S = O / "paper_protocol/stabl_n100"
FUNC = Path("/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/"
            "13features.txt").read_text().split()
STIMS = ["TNFa", "LPS", "IL246", "IFNa", "GMCSF", "PI", "IL33"]
ARMS = {  # name -> (feature table, Stabl run dir)
    "measured only (freq + med + qtl)": (O / "night2/distrib_all_v2.csv", S / "d_qtl"),
    "CellOT stims + measured": (O / "paper_protocol/tables/comb_cellot.csv", S / "comb_cellot"),
    "gauss_ot stims + measured": (O / "paper_protocol/tables/comb_gaussot.csv", S / "comb_gaussot"),
}

def family(f):
    suf = f.rsplit("_", 1)[1]
    if suf == "freq": return "cell frequency"
    if suf == "qtl": return f"quantile {f.rsplit('_', 2)[1]}"
    if suf == "med": return "functional median (measured)" if f.rsplit("_", 2)[1] in FUNC else "phenotypic median (measured)"
    if suf == "Unstim": return "functional median (measured)"
    if suf in STIMS: return "simulated stim"
    return "other"

def fam_group(fam):
    return "quantile (measured)" if fam.startswith("quantile") else fam

def selections(run):
    s = pd.read_csv(run / "Training CV/Selected Features STABL ALasso.csv", index_col=0)
    return s["Fold selected features"].apply(ast.literal_eval).tolist()
""")

md(r"""
## Part 1a — what Stabl selects

Share of all selections (100 splits) per feature family, and the mean number kept per split.
""")

code(r"""
rows, sel_all = [], {}
for name, (_, run) in ARMS.items():
    sel = selections(run); sel_all[name] = sel
    fams = pd.Series([fam_group(family(f)) for l in sel for f in l])
    per = fams.value_counts()
    for fam, n in per.items():
        rows.append({"model": name, "family": fam, "share of selections": n / len(fams),
                     "mean kept per split": n / len(sel)})
fam_tab = pd.DataFrame(rows).pivot_table(index="family", columns="model", values="share of selections").fillna(0)
display((fam_tab * 100).round(1).astype(str) + " %")
pd.DataFrame(rows).pivot_table(index="family", columns="model", values="mean kept per split").fillna(0).round(1)
""")

code(r"""
# Which quantile levels, and which phenotypic markers / cell frequencies, get picked
for name, sel in sel_all.items():
    f = pd.Series([x for l in sel for x in l])
    q = f[f.str.endswith("_qtl")].str.rsplit("_", n=2).str[1].value_counts()
    ph = f[f.str.endswith("_med") & ~f.str.rsplit("_", n=2).str[1].isin(FUNC)].str.rsplit("_", n=2).str[1].value_counts()
    fr = f[f.str.endswith("_freq")].str[:-5].value_counts()
    print(f"=== {name}")
    print("  quantile levels:", q.to_dict())
    print("  phenotypic markers (median):", ph.head(8).to_dict())
    print("  cell frequencies:", fr.head(8).to_dict())
""")

md(r"""
**Stable core** — features selected in at least half of the 100 splits (selection frequency).
""")

code(r"""
for name, sel in sel_all.items():
    freq = pd.Series([x for l in sel for x in set(l)]).value_counts() / len(sel)
    core = freq[freq >= 0.5]
    print(f"=== {name}: {len(core)} features in ≥ 50 % of splits")
    display(pd.DataFrame({"selected in": (core * 100).round(0).astype(int).astype(str) + " %",
                          "family": [family(x) for x in core.index]}))
""")

md(r"""
## Part 1b — estimated precision of the selection, per block

Stabl picks, per block and per split, the stability threshold that minimises FDP+ = (1 + # knockoffs
above threshold) / # real features above threshold (`Stabl features STABL ALasso <block>.csv`).
1 − FDP+ is Stabl's own estimate of the share of *true* discoveries. The "+1" makes it conservative
for blocks that keep few features.
""")

code(r"""
rows = []
for name, (_, run) in ARMS.items():
    sel = sel_all[name]
    for f in sorted((run / "Training CV/Stabl features STABL ALasso").glob("Stabl features STABL ALasso *.csv")):
        blk = f.stem.replace("Stabl features STABL ALasso ", "")
        t = pd.read_csv(f, index_col=0)
        kept = [sum(1 for x in l if (x.endswith("_" + blk))) for l in sel]
        rows.append({"model": name, "block": blk, "median FDP+": t["min FDP+"].median(),
                     "est. precision (1 − FDP+)": 1 - t["min FDP+"].median(),
                     "splits with ≥1 kept": f"{sum(k > 0 for k in kept)}/100", "mean kept": np.mean(kept)})
prec = pd.DataFrame(rows)
prec.pivot_table(index="block", columns="model", values="est. precision (1 − FDP+)").round(2)
""")

code(r"""
# Selection stability across splits (CVS = median Jaccard between the split-wise selections) and size
rows = []
for name, (_, run) in {**ARMS, "CellOT alone (paper)": (None, O / "foldfeats_GSS"),
                       "gauss_ot alone": (None, S / "gaussot_both")}.items():
    f = run / "Summary/Scores training CV.csv"
    sel = selections(run)
    n = pd.Series([len(l) for l in sel])
    row = {"model": name, "kept per split (median)": n.median(), "IQR": f"{n.quantile(.25):.0f}–{n.quantile(.75):.0f}"}
    if f.exists():
        sc = pd.read_csv(f, index_col=0).loc["STABL ALasso"]; row["CVS (stability)"] = sc["CVS"]
    rows.append(row)
pd.DataFrame(rows)
""")

md(r"""
## Part 1c — which families the prediction actually uses (TreeSHAP)

The XGBoost second stage refitted exactly as the paper does (`params_43`, median impute + scaling on the
training women), **rows and patient groups aligned** (see CLAUDE.md, row-order bug). For every test woman,
TreeSHAP splits the prediction exactly into per-feature contributions; the share of mean |contribution|
per family says what the model leans on — not just what Stabl let through.
""")

code(r"""
# computed by stabl_launch/diag_shap.py (venv-stabl: needs xgboost) -> simco_work/diagnosis/shap_by_feature.csv
shap = pd.read_csv(W / "diagnosis/shap_by_feature.csv")
shap["family"] = shap.feature.map(lambda f: fam_group(family(f)))
shap["share"] = shap.abs_shap / shap.groupby("model").abs_shap.transform("sum")
display((shap.pivot_table(index="family", columns="model", values="share", aggfunc="sum").fillna(0) * 100)
        .round(1).astype(str) + " %")
for name in ARMS:
    t = shap[shap.model == name].sort_values("share", ascending=False).head(10)
    print(f"=== top-10 features by share of |SHAP|, {name}")
    display(t[["feature", "family", "share"]].assign(share=lambda d: (d.share * 100).round(1)))
""")

md(r"""
### Reading, Part 1 (2026-09-24)

- **The measured blocks are selected identically in all three models.** Stabl works block by block
  with fixed splits and seeds, so the frequency / median / quantile selections are the same whether or
  not simulated stims are present; the combined models are the measured selections *plus* the stim
  selections (~20 CellOT / ~17 gauss_ot features per split).
- **What is useful among the added features: phenotypic medians and cell frequencies.** In the
  combined CellOT model, simulated stims carry 54 % of the prediction (|SHAP|), phenotypic medians
  21 %, cell frequencies 19 %, measured functional medians 4 %, quantiles 3 %. Phenotypic markers most
  used: CD45RA, CD14, CCR2, CD62L, CD123, CD11b (e.g. pDC CCR2, intMC CD45RA, cMC CD14, B-cell CD62L);
  frequencies: cMCs, B cells, CD4−CD8− T cells, intMCs, ncMCs.
- **Quantiles contribute little and are not reliably selected**: ~1.8 per split, mostly the 90th
  percentile (137 of 179 selections, e.g. naive CD4 T pSTAT3 q90), 3 % of |SHAP|, and Stabl's own
  estimated precision for that block is ~0 (median FDP+ = 1). The median block is the most reliable
  (estimated precision 0.88), frequencies 0.50.
- **Simulated stim blocks: CellOT's are more reliable than gauss_ot's** (estimated precision PI 0.80 vs
  0.67, GMCSF 0.67 vs 0, IFNa / IL246 0.50 vs 0), consistent with CellOT's better selection stability
  (CVS 0.42 vs 0.38 combined). The stable core of both combined models is dominated by naive CD4 T
  pSTAT3 / pSTAT5 and granulocyte pS6 under several stims — the same axis that carries THF / SALPZ.
""")

md(r"""
## Part 2 — how well are marker quantiles predicted after simulation? (atlas)

The atlas benchmark re-run with per-marker quantile errors |q_pred − q_true| at the 10th, 25th, 75th and
90th percentiles (plus the median), on the held-out test cells of CellOT's own split, cell-weighted arm
(`baselines/pooled_q/` for stims, `baselines/pooled_drug_q/` for drugs; the pre-existing metrics are
unchanged to 6e-8). *Skill* = 1 − error / identity's error: 1 = perfect, 0 = no better than assuming the
perturbation does nothing. Stims: the 127 pairs with a real effect (identity MMD ≥ 0.005). Drugs: the
(pair, marker) cases whose effect at that quantile exceeds the 0.0273 noise floor (two identical control
wells), and separately those above 0.1.
""")

code(r"""
QL = ["q10", "q25", "median", "q75", "q90"]
METH = {"cellot_insample": "CellOT", "gauss_ot": "gauss_ot", "median_shift": "median shift",
        "quantile_map": "quantile map", "identity": "identity"}
def load(d):
    fs = sorted((B / d).glob("*.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None

def long_q(raw, keys):
    raw = raw[raw.weighting.isin(["cell", "invariant"]) & raw.method.isin(METH)]
    mk = [c[len("median_error__"):] for c in raw.columns if c.startswith("median_error__")]
    parts = []
    for q in QL:
        pre = "median_error__" if q == "median" else f"{q}_error__"
        m = raw.melt(id_vars=keys + ["method"], value_vars=[pre + x for x in mk], var_name="marker", value_name="err")
        m["marker"] = m.marker.str[len(pre):]; m["q"] = q; parts.append(m)
    return pd.concat(parts)

def skill_table(L, keys, mask=None):
    w = L.pivot_table(index=keys + ["marker", "q"], columns="method", values="err")
    if mask is not None: w = w[mask(w)]
    out = []
    for q in QL:
        s = w.xs(q, level="q")
        for m in METH:
            if m == "identity": continue
            sk = 1 - s[m] / s["identity"]
            out.append({"quantile": q, "method": METH[m], "n": len(s), "median error": s[m].median(),
                        "median skill": sk.median(),
                        "beats CellOT": np.nan if m == "cellot_insample" else (s[m] < s["cellot_insample"]).mean()})
    return pd.DataFrame(out)

stim = load("pooled_q"); drug = load("pooled_drug_q")
print("stim celltypes done:", None if stim is None else stim.celltype.nunique(), "/ 28 | drug:",
      None if drug is None else drug.celltype.nunique(), "/ 28")
""")

code(r"""
if stim is not None:
    idm = stim[stim.method == "identity"].set_index(["celltype", "stim"]).mmd
    live = idm[idm >= 0.005].index
    st = stim.set_index(["celltype", "stim"]).loc[stim.set_index(["celltype", "stim"]).index.isin(live)].reset_index()
    L = long_q(st, ["celltype", "stim"])
    t0 = skill_table(L, ["celltype", "stim"])
    print("all cases, 10th percentile median skill:", t0[t0["quantile"] == "q10"].set_index("method")["median skill"].round(2).to_dict(),
          "<- dominated by markers whose low tail does not move (identity error ~0 blows the ratio up)")
    t = skill_table(L, ["celltype", "stim"], mask=lambda w: w["identity"] > 0.0273)
    display(t.pivot_table(index="quantile", columns="method", values="median skill").loc[QL].round(2)
            .style.set_caption("Stims, effects above the noise floor — median skill (1 = perfect, 0 = no better than 'no effect')"))
    display(t.pivot_table(index="quantile", columns="method", values="median error").loc[QL].round(4)
            .style.set_caption("Stims — median absolute error"))
    display(t[t.method != "CellOT"].pivot_table(index="quantile", columns="method", values="beats CellOT").loc[QL].round(2)
            .style.set_caption("Stims — share of (pair, marker) cases where the method beats CellOT"))
""")

code(r"""
if stim is not None:
    # per marker: CellOT skill at the tails vs the median
    w = L.pivot_table(index=["celltype", "stim", "marker", "q"], columns="method", values="err")
    w = w[w["identity"] > 0.0273]
    sk = (1 - w["cellot_insample"] / w["identity"]).rename("skill").reset_index()
    display(sk.pivot_table(index="marker", columns="q", values="skill", aggfunc="median")[QL].round(2)
            .style.set_caption("Stims — CellOT median skill per marker and quantile"))
""")

code(r"""
if drug is not None:
    Ld = long_q(drug, ["celltype", "stim", "drug"])
    for thr, lab in [(0.0273, "effect above the noise floor"), (0.1, "effect above 0.1")]:
        t = skill_table(Ld, ["celltype", "stim", "drug"], mask=lambda w: w["identity"] > thr)
        display(t.pivot_table(index="quantile", columns="method", values="median skill").loc[QL].round(2)
                .style.set_caption(f"Drugs, {lab} — median skill"))
        display(t[t.method != "CellOT"].pivot_table(index="quantile", columns="method", values="beats CellOT").loc[QL].round(2)
                .style.set_caption(f"Drugs, {lab} — share of cases beating CellOT"))
        print(f"{lab}: n per quantile =", t[t.method == 'CellOT'].set_index('quantile').n.to_dict())
""")

md(r"""
### Reading (2026-09-24, all 28 celltypes, both arms)

Skill = 1 − error / error of "no effect" (identity), per (pair, marker, quantile), median over cases.
The unfiltered 10th-percentile skill is negative for every map except quantile_map (e.g. CellOT −0.57).
That is an artefact: many low tails barely move, so the identity error in the denominator is close to 0.
Only cases whose observed effect exceeds the technical noise floor (0.0273, DMSO2 vs DMSO3) are read.

**Stims: the simulated quantiles are predicted well.**
- CellOT recovers 82–90 % of the observed shift at every quantile, and gauss_ot is essentially tied with it (0.80–0.91).
- median_shift is excellent at the median (0.97) but poor at the tails (0.53 at q10, 0.60 at q90). A constant shift cannot
  change the spread, and the stims do change it.
- **quantile_map is the best method at every quantile except the median** (0.84–0.97). It beats CellOT in 59–78 % of cases,
  and its median absolute error is 2–3× smaller than CellOT's at q75/q90 (0.006–0.009 vs 0.021–0.024).
- So the tails of each marker are *more* predictable per marker than CellOT predicts them. A per-marker 1-D OT fitted
  on the atlas does better, which is expected: it fits each marginal exactly.
- CellOT's weak spots are the low tails of pSTAT1 (q10 −1.07) and pSTAT6 (−0.41), plus CD25 and HLADR (≈ 0.6 everywhere,
  HLADR q90 0.33).

**Drugs: about half of the observed shift is recovered, at every quantile.**
- Above the noise floor (5–9 k cases per quantile), CellOT's skill is 0.41–0.48 and gauss_ot's 0.43–0.52. All methods sit
  between 0.2 and 0.6, and quantile_map again leads (0.50–0.58) while beating CellOT in only 56–59 % of cases.
- On real effects (> 0.1, 0.5–1.8 k cases) CellOT's skill rises to 0.47–0.57. It is then ahead of gauss_ot at every quantile
  (gauss_ot beats it in only 36–46 %) and far ahead of median_shift at the tails (0.06–0.07 at q10/q90).
  quantile_map remains a coin flip against it (49–55 %).
- Drug effects are small (CLAUDE.md: 72 % below the technical floor), so a skill of about 0.5 means half of a small shift is
  predicted. That is the ceiling on the precision of any simulated-drug feature, quantile or median.

**Consequence for the simulated-quantile Stabl features (Part 3).**
- For the stim arm, the simulated quantiles are well predicted on the atlas, so adding them to Stabl is not feeding it noise.
- They are equally well predicted by gauss_ot, and better by quantile_map. So if they help DOS prediction, that alone
  will not show CellOT is needed.
""")

md(r"""
## Part 2b — drugs above the noise floor: quantile_map vs CellOT vs gauss_ot, where CellOT loses, and whether Stabl uses those features

Scope: the atlas pooled drug benchmark (cell-weighted). Keep the cases (celltype, stim, drug, marker, quantile) whose observed
effect exceeds the 0.0273 technical floor: 32 691 cases, 1508 pairs. Paired Wilcoxon two ways:
- **per case**, which is optimistic because the cases within a pair are not independent;
- **per pair**, on the mean over each (celltype, stim, drug) pair. This is the one to quote.

`gap` = CellOT error − quantile_map error, so positive means quantile_map is better. Feature = `<celltype>_<marker>_<stim>`.
Its selection frequency is taken over the 100 folds of the Fig 5b ALasso run (`foldfeats_GSS`) and of CellOT + measured.
Computed by `stabl_launch/diag_drug_qmap.py`, run below, which writes to `simco_work/diagnosis/drug_qmap/`.
""")

code(r"""
%run /scratch/users/bmerwan/simco/stabl_launch/diag_drug_qmap.py
""")

code(r"""
T = pd.read_csv(OUT / "paired_table.csv")
fmt = {"p (cases)": "{:.1e}", "p (pairs)": "{:.1e}"}
display(T.style.format(fmt | {c: "{:.4f}" for c in T.columns if "err" in c or "diff" in c}
                       | {c: "{:.0f}" for c in T.columns if "%" in c}).hide(axis="index")
        .set_caption("quantile_map vs CellOT / gauss_ot, drug effects above the noise floor"))
for g in ["stim", "celltype", "marker", "drug", "q"]:
    display(pd.read_csv(OUT / f"gap_by_{g}.csv", index_col=0).head(10).round(4)
            .style.set_caption(f"Where CellOT loses to quantile_map — by {g} (share = % of CellOT's total excess error)"))
F = pd.read_csv(OUT / "gap_by_feature.csv", index_col=0)
display(F.head(15).round(3).style.set_caption("Top 15 features by CellOT excess error, with Stabl selection frequency"))
top = set(F[F.sel_fig5b >= 0.2].index)
display(F.loc[sorted(top, key=lambda k: -F.at[k, "sel_fig5b"])].round(3)
        .style.set_caption("The 24 features selected in >= 20 % of the Fig 5b folds"))
w["feature"] = w.celltype + "_" + w.marker + "_" + w.stim
s = w[w.feature.isin(top)]
rows = []
for lab, ss in [("selected >= 20 %", s), ("selected, effect > 0.1", s[s.identity > 0.1]),
                ("selected, Granulocytes", s[s.celltype == "Granulocytes"]), ("selected, CD4Tnaive", s[s.celltype == "CD4Tnaive"])]:
    r = {"subset": lab, "n": len(ss)}
    for o in ["CellOT", "gauss_ot"]:
        d = ss[o] - ss.quantile_map
        r[f"qmap beats {o} %"] = 100 * (d > 0).mean(); r[f"p vs {o}"] = wilcoxon(d).pvalue
    r["CellOT beats gauss_ot %"] = 100 * (ss.gauss_ot > ss.CellOT).mean()
    rows.append(r)
display(pd.DataFrame(rows).style.format({"p vs CellOT": "{:.1e}", "p vs gauss_ot": "{:.1e}"}, precision=0).hide(axis="index")
        .set_caption("On the features that drive DOS"))
""")

md(r"""
### Reading (2026-09-24)

- **Above the noise floor, quantile_map is the best drug-quantile predictor at every quantile.**
  - vs CellOT: it wins in 71 % of pairs (p 5e-76). Median error 0.0234 against CellOT's 0.0284.
  - vs gauss_ot: it wins in 65 % of pairs (p 2e-42).
  - CellOT has the largest error of the three.
- **Where CellOT loses to quantile_map:**
  - by stim: PI carries 33 % of CellOT's excess error;
  - by celltype: cMCs 15 %; in Granulocytes quantile_map wins 93 % of cases; CD4Tnaive, CD4 Tcm/Tem and B cells follow;
  - by marker: CD25 18 %, HLADR 11 %;
  - by drug: LPZ 25 % (the drug referenced to DMSO1);
  - by quantile: q90 39 %.
  - intMCs and ncMCs are ties.
- **Selection by Lasso.** Across all features, CellOT's excess error is unrelated to how often Stabl selects the feature
  (Spearman 0.01). The largest single gaps are myeloid CD25/HLADR features that are almost never selected.
- **But the 24 features that drive DOS are where CellOT is weak.**
  - These are the features selected in at least 20 % of Fig 5b folds: Granulocytes pS6 and CD4Tnaive pSTAT3/5, i.e. the
    features carrying THF's predicted shift.
  - On them, quantile_map beats CellOT in 71 % of cases, and CellOT beats gauss_ot in only 41 %.
  - The drug effects there are small: only 59 of 474 cases exceed 0.1. On those 59, CellOT and gauss_ot tie.
- **What this means.** The earlier "CellOT is the best non-diagonal drug map on effects > 0.1" does not carry over to the
  features that matter for DOS.
- Limit: these are cell-level quantiles pooled over the 19 atlas donors, not per-woman accuracy.
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python", "version": "3.9"}}
out = "/scratch/users/bmerwan/simco/notebooks/diagnosis.ipynb"
nbf.write(nb, out); print("wrote", out, f"({len(cells)} cells)")
