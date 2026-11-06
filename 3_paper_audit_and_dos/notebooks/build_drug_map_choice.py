"""Generate notebooks/drug_map_choice.ipynb: which drug map for the virtual twins — judged on HELD-OUT donors only.

Question: is a per-marker quantile map (quantile_map) a better drug map than a trained CellOT drug map, above all on
each patient's own drug effect, and where do they differ? Every accuracy number here comes from donors the maps never
saw (atlas_heldout: 5 repeats, maps trained on 15 donors, scored on the other 4). No in-sample comparison is used;
the earlier in-sample version is archived in notebooks/archive/drug_map_choice_insample_2026-10-06.ipynb.
The OOL sections (how the women are treated, what XGBoost sees) have no ground truth; they are deployment checks on
women outside the atlas.
    python notebooks/build_drug_map_choice.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/drug_map_choice.ipynb
"""
from pathlib import Path

import nbformat as nbf

cells = []
md = lambda t: cells.append(nbf.v4.new_markdown_cell(t.strip()))
code = lambda t: cells.append(nbf.v4.new_code_cell(t.strip()))

md(r"""
# Which drug map for the virtual twins? CellOT vs quantile_map — held-out donors only

**The setting.** In the drug arm each OOL woman's features (simulated stim responses + measured Unstim) are passed
through a *drug map* (DMSO → drug, learned on the 19-donor drug screen), and the DOS model (Stabl + XGBoost, Fig 5b)
predicts how much her labour timing would shift. The paper's drug maps are CellOT networks. The alternatives:
- **quantile_map**: per marker, moves each quantile of the control distribution onto the same quantile of the drug
  distribution. Monotone, so a treated *feature* is that 1-D function applied to the untreated feature value.
- **median_shift**: the same pooled median change for every woman (no patient-specificity at all).
- **gauss_ot** (reference): affine 13×13 map fitted on means and covariances.

**Rule of this notebook (2026-10-06): only held-out donors.** A map is only scored on donors whose cells it never
saw. In-sample scores (the paper's maps on the 19 donors they were trained on, or an 80/20 split of *cells* of the
same donors) reward memorising those donors and are not used anywhere below. The previous, in-sample version of this
notebook is archived (`notebooks/archive/drug_map_choice_insample_2026-10-06.ipynb`); §9 lists what changed.
""")
md("<!-- ANSWER -->")

md(r"""
## 1. Design: how the held-out test works

Source: `/scratch/users/bmerwan/atlas_heldout` (`heldout/score_drugs.py`), scores in `work/heldout/drugs/scores/`.

- **5 repeats.** In each, 15 donors train every map and **4 test donors** are scored: 20 donor-repeats, 14 distinct
  donors (HV04, HV09–HV11, PV01–PV10). In every repeat *all* maps are refit on the same 15 donors: new CellOT drug
  maps (paper architecture 128-128-64-32, 20k iterations, last iterate — no checkpoint chosen on test data), the
  closed-form maps, and the learning-curve stim maps (CellOT and gauss_ot) used by the pipelines.
- **Truth** for a test donor = Q(her drug-well cells) − Q(her control-well cells), Q = quantile 10/25/50/75/90 of
  one marker (13 markers), batch-corrected wells, ≤ 5k cells per well. 196 (stim, celltype) pairs + Unstim × 9 drugs.
- **Noise** = |truth on one random half of her cells − truth on the other half|. A case is **real** only if
  |truth| > 0.0273 (the technical floor: two identical control wells differ that much) **and** |truth| > 2 × noise.
  Everything below is on real cases.
- **Two ways to use a drug map:**
  - **step** = drug map applied to the donor's *real* control cells: tests the drug map alone.
  - **pipeline**, starting like the OOL women from her *unstimulated* cells:
    - `chain:CellOT>CellOT` = **the paper**: Unstim → CellOT stim map → CellOT drug map on cells → quantiles;
    - `feature:CellOT>quantile_map` = **the proposal**: Unstim → CellOT stim map → quantiles → quantile_map on the
      feature values;
    - `feature:CellOT>median_shift`, `chain:gauss_ot>gauss_ot`, `feature:gauss_ot>quantile_map` for reference.
- **Skill** = 1 − |predicted − true| / |true|: 1 perfect, 0 = as bad as "the drug does nothing".
- **DOS features** = the (stim, celltype, marker) medians selected in ≥ 5 % of the Fig 5b Stabl folds (90 features,
  94 % of all selections), at q = 50. This is what the DOS model actually reads.
- **Honest tests.** Cases share donors and maps, so case-level p-values are far too small. Each comparison below is
  tested on independent-ish units: **drug maps** (repeat × drug × stim × celltype: mean error difference over its
  cases, Wilcoxon) and **donor-repeats** (20 units: median error per donor, sign count + Wilcoxon).
- **DMSO1 plate.** LPZ and SA are referenced to DMSO1, whose PI wells are an outlier; LPZ/SA under PI is a plate
  artefact, not a drug effect. Flagged everywhere and excluded from the headline numbers.
""")
code(r"""
import warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, binomtest, spearmanr
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200, "display.max_columns", 40, "display.max_rows", 100)
W = Path("/scratch/users/bmerwan/simco_work"); O = W / "onset_test"; DC = W / "drug_choice"
HA = Path("/scratch/users/bmerwan/atlas_heldout/work/heldout/drugs")
PRES = Path("/scratch/users/bmerwan/simco/presentation"); PRES.mkdir(exist_ok=True)
FLOOR = 0.0273
COL = {"CellOT": "#d62728", "quantile_map": "#1f77b4", "median_shift": "#2ca02c", "gauss_ot": "#9467bd"}
STEP = {"step:CellOT": "CellOT", "step:quantile_map": "quantile_map", "step:median_shift": "median_shift",
        "step:gauss_ot": "gauss_ot"}
PIPE = {"chain:CellOT>CellOT": "paper: CellOT stim → CellOT drug (cells)",
        "feature:CellOT>quantile_map": "CellOT stim → quantile_map on features",
        "feature:CellOT>median_shift": "CellOT stim → median_shift on features",
        "chain:gauss_ot>gauss_ot": "gauss_ot stim → gauss_ot drug (cells)",
        "feature:gauss_ot>quantile_map": "gauss_ot stim → quantile_map on features"}
F = pd.read_csv(DC / "selected_features_fig5b.csv"); USED = F[F.sel_freq >= 0.05]
UK = set(map(tuple, USED[["stim", "celltype", "marker"]].to_numpy()))

cols = ["repeat", "donor", "celltype", "stim", "drug", "q", "marker", "true", "noise"] + list(STEP) + list(PIPE)
A = pd.concat([pd.read_csv(f, usecols=cols) for f in sorted((HA / "scores").glob("r*.csv.gz"))], ignore_index=True)
A["used"] = np.array([(s, c, m) in UK for s, c, m in zip(A.stim, A.celltype, A.marker)]) & (A.q == 50).to_numpy()
A["dmso1_PI"] = A.drug.isin(["LPZ", "SA"]) & (A.stim == "PI")
H = A[(A.true.abs() > FLOOR) & (A.true.abs() > 2 * A.noise)].copy()
H["cohort"] = np.where(H.donor.str.startswith("HV"), "HV (not pregnant)", "PV (pregnant)")
H["effect"] = pd.cut(H.true.abs(), [FLOOR, 0.05, 0.1, 0.2, np.inf], labels=["0.03–0.05", "0.05–0.1", "0.1–0.2", "> 0.2"])
H["feature kind"] = np.where(H.used, "DOS feature (q50)", np.where(H.q == 50, "other medians (q50)", "other quantiles"))
for c in list(STEP) + list(PIPE): H["e|" + c] = (H[c] - H.true).abs()
Hc = H[~H.dmso1_PI]                                     # headline set: DMSO1 plate artefact removed
print(f"{len(A):,} held-out cases; real: {len(H):,} ({len(H) / len(A):.0%}); without LPZ/SA under PI: {len(Hc):,}")
print(f"donor-repeats {H.groupby(['repeat', 'donor']).ngroups}, distinct donors {H.donor.nunique()}; "
      f"DOS-feature cases (q50): {H.used.sum():,} ({Hc.used.sum():,} without the DMSO1 cases)")
print("real cases by quantile:", H.q.value_counts().sort_index().to_dict())
""")

md("## 2. Compute cost")
code(r"""
t = pd.read_csv(W / "timing/cellot_train_minutes.csv")
used_pairs = set(map(tuple, USED[["stim", "celltype"]].to_numpy()))
drug_t = t[t.kind.isin(["drug", "drug_unstim"])]
used_t = drug_t[[(s, c) in used_pairs for s, c in zip(drug_t.stim, drug_t.celltype)]]
cost = pd.DataFrame([
    {"drug maps": "CellOT, all (1512 stim×drug + 252 Unstim×drug)", "maps": len(drug_t), "CPU-hours": drug_t.minutes.sum() / 60},
    {"drug maps": "CellOT, only those the DOS model uses", "maps": len(used_t), "CPU-hours": used_t.minutes.sum() / 60},
    {"drug maps": "quantile_map (+ median_shift), all 1764", "maps": 1764, "CPU-hours": 14.7}])
display(cost.round(1))
print(f"ratio CellOT / quantile_map: {cost['CPU-hours'][0] / 14.7:,.0f}x")
""")
md(r"""
**Reading §2.** CellOT drug maps cost ~12,300 CPU-hours (the paper's own training times); quantile_map ~15 core-hours
for all 1,764 maps, nearly all of it reading the data — about 800× less. Cost is not an accuracy question, so it is
the same in- or out-of-sample.
""")

md(r"""
## 3. Accuracy on new donors: the drug map alone (step)

Each drug map is applied to the test donor's *real* control cells. Median skill per map, on all real cases and on
subsets. Then paired comparisons against CellOT with the two honest units.
""")
code(r"""
def fmtp(T):
    T = T.copy()
    for c in T.columns:
        if str(c).startswith("p ("): T[c] = T[c].map(lambda v: "" if pd.isna(v) else f"{v:.1e}")
    return T

def skill(s, c): return (1 - s["e|" + c] / s.true.abs()).median()

def overview(S, names):
    subsets = {"all real": S, "effect > 0.1": S[S.true.abs() > 0.1], "medians (q50)": S[S.q == 50],
               "DOS features (q50)": S[S.used]}
    T = pd.DataFrame({k: {names[c]: skill(s, c) for c in names} for k, s in subsets.items()})
    T.loc["cases"] = [len(s) for s in subsets.values()]
    return T

def paired(S, a, b):
    # a positive difference = b better (smaller error)
    d = S["e|" + a] - S["e|" + b]
    pm = d.groupby([S.repeat, S.drug, S.stim, S.celltype]).mean()
    pd_ = S.groupby(["repeat", "donor"]).apply(lambda g: g["e|" + a].median() - g["e|" + b].median())
    return {"cases": len(S), "B better (cases)": (d > 0).mean(),
            "median err A": S["e|" + a].median(), "median err B": S["e|" + b].median(),
            "drug maps: B better": (pm > 0).mean(), "p (drug maps)": wilcoxon(pm).pvalue if len(pm) > 5 else np.nan,
            "donors: B better": f"{(pd_ > 0).sum()}/{len(pd_)}",
            "p (donors)": wilcoxon(pd_).pvalue if len(pd_) > 5 else np.nan}

def paired_table(S, pairs, names):
    subsets = {"all real": S, "effect > 0.1": S[S.true.abs() > 0.1], "medians (q50)": S[S.q == 50],
               "DOS features (q50)": S[S.used]}
    rows = []
    for k, s in subsets.items():
        for a, b in pairs:
            rows.append({"subset": k, "A": names[a], "B": names[b], **paired(s, a, b)})
    return pd.DataFrame(rows)

print("Median skill, step, without LPZ/SA under PI")
display(overview(Hc, STEP).round(3))
print("Same, LPZ/SA under PI only (the DMSO1 plate artefact)")
display(overview(H[H.dmso1_PI], STEP).round(3))
P3 = paired_table(Hc, [("step:CellOT", "step:quantile_map"), ("step:CellOT", "step:median_shift"),
                       ("step:CellOT", "step:gauss_ot")], STEP)
display(fmtp(P3).round(4))
""")
md("<!-- READ3 -->")

md(r"""
## 4. Accuracy on new donors: the whole pipeline (what an OOL woman actually goes through)

Starting from the test donor's unstimulated cells, as for an OOL woman. The comparison that decides the drug arm is
the paper's chain against the same CellOT stim map followed by quantile_map on the features.
""")
code(r"""
print("Median skill, pipelines, without LPZ/SA under PI")
display(overview(Hc, PIPE).round(3))
P4 = paired_table(Hc, [("chain:CellOT>CellOT", "feature:CellOT>quantile_map"),
                       ("chain:CellOT>CellOT", "feature:CellOT>median_shift"),
                       ("feature:CellOT>quantile_map", "feature:CellOT>median_shift"),
                       ("chain:CellOT>CellOT", "chain:gauss_ot>gauss_ot")], PIPE)
display(fmtp(P4).round(4))
""")
code(r"""
# figure: median skill per map on new donors, step and pipeline, all real cases vs DOS features
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
O3, O4 = overview(Hc, STEP), overview(Hc, PIPE)
for ax, T, title in [(axs[0], O3, "Drug map alone (on real control cells)"),
                     (axs[1], O4, "Full pipeline (from unstimulated cells)")]:
    T = T.drop("cases").drop(columns=["effect > 0.1"])
    x = np.arange(T.shape[1]); n = T.shape[0]; wd = 0.8 / n
    pal = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b"]
    for i, (m, row) in enumerate(T.iterrows()):
        ax.bar(x + (i - (n - 1) / 2) * wd, row.values, wd, color=pal[i], label=m)
    ax.set_xticks(x); ax.set_xticklabels(T.columns); ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=.3)
    ax.legend(fontsize=7, loc="upper left", ncol=1, framealpha=.95); ax.set_ylim(0, 0.72)
axs[0].set_ylabel("median skill on new donors\n(1 = perfect, 0 = 'no effect')")
fig.suptitle("Held-out donors only, LPZ/SA under PI excluded", fontsize=10)
fig.tight_layout(); fig.savefig(PRES / "drug_choice_heldout_skill.png", dpi=160)
""")
md("<!-- READ4 -->")

md(r"""
## 5. The per-patient question: shared part vs patient-specific part, on new donors

A test donor's true effect on one feature = the **shared part** (the average effect over the test donors of that
repeat) + her **patient-specific part** (her deviation from it). A drug map can be wrong on either.
- **Shared-part skill** = 1 − |mean predicted − mean true| / |mean true| (groups with |mean true| above the floor).
- **Specific error ratio** = Σ (true dev − predicted dev)² / Σ (true dev)². 1 = no better than giving everyone the
  same effect (median_shift is exactly 1); < 1 = the map's donor-to-donor variation is real; > 1 = it adds error.
- Part of each true deviation is sampling noise (from the split-half noise), which nothing can predict. **Real
  differences captured** = (1 − ratio) / (1 − noise share): 0 % none, 100 % all of the non-noise part.
- **Tracking r** = correlation across donors between predicted and true deviations; **SD ratio** = how much the map
  varies between donors relative to the truth.

One group = (repeat, drug, stim, celltype, marker, quantile) with its 4 test donors
(`baselines/drug_patient_decomp.py` → `per_patient/heldout_groups.csv.gz`, all cases, not only the real ones).
Groups are restricted to those where donors really differ: **reliability** = share of the between-donor variance that
is not noise > 0.5, estimated per (drug, stim, celltype, marker, quantile) **pooled over the 5 repeats** (so 20
donor-repeats rather than 4 — the per-repeat estimate with 3 degrees of freedom is too noisy to select on). The
selection uses the truth only, so it is the same for every map.
""")
code(r"""
GH = pd.read_csv(DC / "per_patient/heldout_groups.csv.gz")
GH = GH[GH.n == 4].copy()
key = ["drug", "stim", "celltype", "marker", "q"]
pool = GH.groupby(key).agg(ss=("ss_true", "sum"), nv=("noise_var", "mean"), k=("n", "size"))
pool["reliability"] = 1 - pool.nv / (pool.ss / (3 * pool.k))
GH = GH.join(pool.reliability, on=key)
GH["used"] = np.array([(s, c, m) in UK for s, c, m in zip(GH.stim, GH.celltype, GH.marker)]) & (GH.q == 50).to_numpy()
GH["dmso1_PI"] = GH.drug.isin(["LPZ", "SA"]) & (GH.stim == "PI")
GH = GH[~GH.dmso1_PI]
ALLM = {**STEP, **PIPE}

def decomp_table(G, names):
    rows = {}
    for m, lab in names.items():
        ok = G[G[f"{m}|ss_spec"].notna()]
        ratio = ok[f"{m}|ss_spec"].sum() / ok.ss_true.sum()
        noise = (ok.noise_var * (ok.n - 1)).sum() / ok.ss_true.sum()
        sh = ok[ok.mean_true.abs() > FLOOR]
        r = ok[f"{m}|cross"] / np.sqrt(ok.ss_true * ok[f"{m}|ss_pred"])
        b2 = (ok.n * ok[f"{m}|bias"] ** 2).sum(); sp = ok[f"{m}|ss_spec"].sum()
        rows[lab] = {"groups": len(ok), "shared-part skill": (1 - sh[f"{m}|bias"].abs() / sh.mean_true.abs()).median(),
                     "specific error ratio": ratio, "noise share": noise,
                     "real differences captured": (1 - ratio) / (1 - noise), "tracking r": r.median(),
                     "SD ratio": np.sqrt(ok[f"{m}|ss_pred"] / ok.ss_true).median(),
                     "error from shared part": b2 / (b2 + sp)}
    return pd.DataFrame(rows).T

def decomp_paired(G, a, b):
    ok = G[G[f"{a}|ss_spec"].notna() & G[f"{b}|ss_spec"].notna()].copy()
    ok = ok[ok.ss_true > 1e-12]
    ok["d_spec"] = (ok[f"{a}|ss_spec"] - ok[f"{b}|ss_spec"]) / ok.ss_true
    sh = ok[ok.mean_true.abs() > FLOOR].copy(); sh["d_sh"] = sh[f"{a}|bias"].abs() - sh[f"{b}|bias"].abs()
    mk = ["repeat", "drug", "stim", "celltype"]
    ps, pb = ok.groupby(mk).d_spec.mean(), sh.groupby(mk).d_sh.mean()
    return {"drug maps": len(ps),
            "specific part: B better": f"{(ps > 0).mean():.0%} (p {wilcoxon(ps).pvalue:.1e})" if len(ps) > 5 else "–",
            "shared part: B better": f"{(pb > 0).mean():.0%} (p {wilcoxon(pb).pvalue:.1e})" if len(pb) > 5 else "–"}

REL = GH[GH.reliability > 0.5]
print(f"groups: {len(GH):,}; with real between-donor differences (pooled reliability > 0.5): {len(REL):,} "
      f"({len(REL) / len(GH):.0%}); DOS features among them: {REL.used.sum():,}")
for title, s in [("all groups (no reliability filter)", GH),
                 ("groups with real between-donor differences", REL),
                 ("... DOS features only (q50)", REL[REL.used])]:
    print(title); display(decomp_table(s, ALLM).round(3))
display(pd.DataFrame({f"{ALLM[a]} (A) vs {ALLM[b]} (B)": decomp_paired(REL, a, b) for a, b in
                      [("step:CellOT", "step:quantile_map"), ("step:CellOT", "step:gauss_ot"),
                       ("chain:CellOT>CellOT", "feature:CellOT>quantile_map")]}).T)
print("same, DOS features only")
display(pd.DataFrame({f"{ALLM[a]} (A) vs {ALLM[b]} (B)": decomp_paired(REL[REL.used], a, b) for a, b in
                      [("step:CellOT", "step:quantile_map"),
                       ("chain:CellOT>CellOT", "feature:CellOT>quantile_map")]}).T)
""")
code(r"""
print("patient-specific part by stim (groups with real differences): % of real differences captured")
bys = {}
for s, g in REL.groupby("stim"):
    t = decomp_table(g, {"step:CellOT": "CellOT", "step:quantile_map": "quantile_map", "step:gauss_ot": "gauss_ot",
                         "chain:CellOT>CellOT": "paper chain", "feature:CellOT>quantile_map": "CellOT→qmap"})
    bys[s] = 100 * t["real differences captured"]; bys[s]["groups"] = len(g)
display(pd.DataFrame(bys).T.round(1))
print("by quantile")
byq = {}
for q, g in REL.groupby("q"):
    t = decomp_table(g, {"step:CellOT": "CellOT", "step:quantile_map": "quantile_map", "step:gauss_ot": "gauss_ot"})
    byq[q] = 100 * t["real differences captured"]
display(pd.DataFrame(byq).T.round(1))
""")
code(r"""
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
ms = [("step:CellOT", "CellOT"), ("step:quantile_map", "quantile_map"), ("step:gauss_ot", "gauss_ot"),
      ("step:median_shift", "median_shift")]
sets = {"all reliable groups": REL, "DOS features (q50)": REL[REL.used]}
x = np.arange(len(sets)); wd = 0.2
for i, (m, lab) in enumerate(ms):
    T = [decomp_table(s, {m: lab}).iloc[0] for s in sets.values()]
    axs[0].bar(x + (i - 1.5) * wd, [t["shared-part skill"] for t in T], wd, color=COL[lab], label=lab)
    axs[1].bar(x + (i - 1.5) * wd, [100 * t["real differences captured"] for t in T], wd, color=COL[lab], label=lab)
for ax, t, yl in [(axs[0], "Shared part (average over the test donors)", "median skill"),
                  (axs[1], "Patient-specific part", "% of real donor differences captured")]:
    ax.set_xticks(x); ax.set_xticklabels(list(sets)); ax.set_title(t, fontsize=10); ax.set_ylabel(yl)
    ax.grid(axis="y", alpha=.3); ax.axhline(0, c="k", lw=.8)
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
fig.suptitle("Drug map alone on new donors (held-out), groups where donors really differ, DMSO1-PI excluded", fontsize=10)
fig.tight_layout(rect=[0, 0.08, 1, 1]); fig.savefig(PRES / "drug_choice_shared_specific_heldout.png", dpi=160)
""")
md("<!-- READ5 -->")

md(r"""
## 6. Where do CellOT and quantile_map differ on new donors?

Drug map alone (step), real cases, DMSO1-PI excluded. *qmap better* = share of cases where quantile_map's error is
smaller (> 0.5: quantile_map wins). p-values on drug maps (repeat × drug × stim × celltype).
""")
code(r"""
def where(s, by, a="step:CellOT", b="step:quantile_map"):
    out = []
    for k, g in s.groupby(by, observed=True):
        d = g["e|" + a] - g["e|" + b]
        pm = d.groupby([g.repeat, g.drug, g.stim, g.celltype]).mean()
        out.append({by: k, "cases": len(g), "skill CellOT": skill(g, a), "skill quantile_map": skill(g, b),
                    "qmap better": (d > 0).mean(), "p (drug maps)": wilcoxon(pm).pvalue if len(pm) > 5 else np.nan})
    return pd.DataFrame(out).set_index(by)
for by in ["feature kind", "q", "effect", "stim", "drug", "cohort"]:
    display(fmtp(where(Hc, by)).round(3))
d = Hc["e|step:CellOT"] - Hc["e|step:quantile_map"]
print("% of cases where quantile_map is better, quantile × effect size")
display((100 * (d > 0).groupby([Hc.q, Hc.effect], observed=True).mean().unstack()).round(0))
print("DMSO1 plate cases (LPZ/SA under PI), for reference")
display(fmtp(where(H[H.dmso1_PI], "q")).round(3))
""")
code(r"""
def heat(ax, s, rows, cols, title, mincases=20):
    dd = s["e|step:CellOT"] - s["e|step:quantile_map"]
    w = (dd > 0).groupby([s[rows], s[cols]]).agg(["mean", "size"])
    w = w["mean"].where(w["size"] >= mincases).unstack()
    w = w.loc[w.notna().sum(1) > 0]
    im = ax.imshow(100 * w.values, cmap="RdBu_r", vmin=20, vmax=80, aspect="auto")
    ax.set_xticks(range(w.shape[1])); ax.set_xticklabels(w.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(w.shape[0])); ax.set_yticklabels(w.index, fontsize=7); ax.set_title(title, fontsize=9)
    return im, w
s50 = Hc[Hc.q == 50]
fig, axs = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={"width_ratios": [13, 9]})
im, HCM = heat(axs[0], s50, "celltype", "marker", "celltype × marker (medians, all stims and drugs)")
im, HSD = heat(axs[1], s50, "stim", "drug", "stim × drug (medians)")
cb = fig.colorbar(im, ax=axs, shrink=.6)
cb.set_label("% of new-donor cases where quantile_map beats CellOT\n(red: quantile_map better, blue: CellOT better)")
fig.suptitle("Where the drug maps differ on held-out donors (q = 50, < 20 cases blank)", fontsize=10)
fig.savefig(PRES / "drug_choice_where_heatmap.png", dpi=160, bbox_inches="tight")
# the low tails (q10, q25), where CellOT does better
lt = Hc[Hc.q.isin([10, 25])]
fig, axs = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={"width_ratios": [13, 9]})
im, _ = heat(axs[0], lt, "celltype", "marker", "celltype × marker (q10 and q25)")
im, _ = heat(axs[1], lt, "stim", "drug", "stim × drug (q10 and q25)")
cb = fig.colorbar(im, ax=axs, shrink=.6)
cb.set_label("% of new-donor cases where quantile_map beats CellOT\n(red: quantile_map better, blue: CellOT better)")
fig.suptitle("Low tails on held-out donors (q = 10 and 25, < 20 cases blank)", fontsize=10)
fig.savefig(PRES / "drug_choice_where_heatmap_lowtail.png", dpi=160, bbox_inches="tight")
print("low tails (q10, q25), by stim and by drug")
display(fmtp(where(lt, "stim")).round(3)); display(fmtp(where(lt, "drug")).round(3))
print("celltypes, medians: CellOT's best and quantile_map's best")
wc = where(s50, "celltype").sort_values("qmap better")
display(fmtp(pd.concat([wc.head(5), wc.tail(5)])).round(3))
print("markers, medians"); display(fmtp(where(s50, "marker").sort_values("qmap better")).round(3))
""")
md("<!-- READ6 -->")

md(r"""
## 7. Per donor, and the features the DOS model uses

One row per test donor-repeat (DOS features at q = 50, DMSO1-PI excluded): which map is closer more often, for the
drug map alone and for the whole pipelines. Then per DOS feature, weighted by how often Stabl selects it.
""")
code(r"""
s = Hc[Hc.used]
def per_donor(s, a, b):
    g = s.groupby(["repeat", "donor"])
    return pd.DataFrame({"cases": g.size(), "median err A": g["e|" + a].median(), "median err B": g["e|" + b].median(),
                         "B better (cases)": g.apply(lambda x: (x["e|" + a] > x["e|" + b]).mean())})
for a, b, t in [("step:CellOT", "step:quantile_map", "drug map alone: CellOT (A) vs quantile_map (B)"),
                ("chain:CellOT>CellOT", "feature:CellOT>quantile_map", "pipelines: paper chain (A) vs CellOT→quantile_map (B)")]:
    pdn = per_donor(s, a, b)
    print(f"{t}: B has the smaller median error in {(pdn['median err B'] < pdn['median err A']).sum()} of {len(pdn)} "
          f"donor-repeats (sign test p {binomtest(int((pdn['median err B'] < pdn['median err A']).sum()), len(pdn)).pvalue:.3f}); "
          f"B closer in > 50 % of cases for {(pdn['B better (cases)'] > .5).sum()} of {len(pdn)}")
    display(pdn.round(4).T)
""")
code(r"""
s = s.assign(feature=s.celltype + "_" + s.marker + "_" + s.stim)
pf = where(s, "feature").join(USED.set_index("feature").sel_freq)
print(f"{len(pf)} of the 90 DOS features have real held-out cases; quantile_map closer more often on "
      f"{(pf['qmap better'] > .5).sum()}, CellOT on {(pf['qmap better'] < .5).sum()}; "
      f"selection-weighted share of cases quantile_map better: {np.average(pf['qmap better'], weights=pf.sel_freq):.0%}")
print("most-selected DOS features (selected in >= 30 % of folds)")
display(fmtp(pf[pf.sel_freq >= .3].sort_values("sel_freq", ascending=False)).round(3))
print("DOS features where CellOT is closer most often (>= 15 cases)")
display(fmtp(pf[pf.cases >= 15].sort_values("qmap better").head(8)).round(3))
""")
md("<!-- READ7 -->")

md(r"""
## 8. Deployment on the OOL women (no ground truth)

The OOL women are outside the atlas, so applying any drug map to them is out of sample by construction — but there
is no measured truth, so these sections are **checks on what the DOS model is fed**, not accuracy. Maps here are the
deployed ones (fitted on all 19 atlas donors); the DOS model is the paper's (CellOT stim features, 100-fold ALasso
selections, `params_43`, aligned orders, 36_G1 / 52_G1 excluded). Only the drug map changes.

### 8a. How differently are the women treated?
Per used feature and drug, D = treated − untreated per woman. *het* = SD(D) / SD(untreated); *rank ρ* = whether the
women keep their order; *own R²* = share of var(D) explained by the woman's own value on that feature (what a
per-feature map can express). Weighted by selection frequency (`stabl_launch/drug_heterogeneity_ool.py`).
""")
code(r"""
Hh = pd.read_csv(DC / "heterogeneity_ool.csv")
wavg = lambda g, c: np.average(g[c].dropna(), weights=g.loc[g[c].notna(), "sel_freq"]) if g[c].notna().any() else np.nan
display(Hh.groupby("method").apply(lambda g: pd.Series({"het": wavg(g, "het"), "rank ρ": wavg(g, "rank_rho"),
                                                        "own R²": wavg(g, "own_R2")})).round(3))
""")
md(r"""
### 8b. What XGBoost is asked about, and the drug shifts
*implausible* = share of treated women farther from every training woman than 95 % of real untreated test women are
(XGBoost extrapolates); *out of range* = share of treated feature values outside the training range; *shift* = mean
predicted change in days (negative = labour later); *agree* = share of the 100 fold models with that sign.
""")
code(r"""
E = O / "drug_models/excl"
C = {"CellOT drug maps (paper)": "cellot_cellotdrug", "quantile_map": "cellot_qmapdrug", "median_shift": "cellot_mshiftdrug"}
S8 = {}
for n, dd in C.items():
    tt = pd.read_csv(E / dd / "summary_by_drug.csv", index_col=0)
    S8[n] = pd.DataFrame({"shift (d)": tt["mean"], "agree": tt[["frac_folds_negative", "frac_folds_positive"]].max(1),
                          "z vs random": tt["mean"] / tt["null_sd_of_mean_shift"],
                          "implausible": tt["frac_treated_beyond_p95_untreated"],
                          "out of range": tt["frac_values_outside_train_range"]})
S8 = pd.concat(S8, axis=1)
display(S8.xs("shift (d)", axis=1, level=1).join(S8.xs("agree", axis=1, level=1), rsuffix=" | agree").round(2))
display(pd.DataFrame({n: {"implausible treated women": S8[n]["implausible"].mean(),
                          "values out of training range": S8[n]["out of range"].mean(),
                          "drugs beating a random perturbation (|z| > 3)": int((S8[n]["z vs random"].abs() > 3).sum())}
                      for n in C}).T.round(3))
""")
md("<!-- READ8 -->")

md("<!-- READ9 -->")

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}

# fill readings / answer from the companion markdown file (sections separated by "<!-- KEY -->" lines)
txt = Path(__file__).with_name("drug_map_choice_readings.md")
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
out = Path(__file__).with_name("drug_map_choice.ipynb")
nbf.write(nb, out); print("wrote", out)
