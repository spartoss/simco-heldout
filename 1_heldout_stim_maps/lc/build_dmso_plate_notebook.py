"""Build notebooks/dmso_plate.ipynb: which plate control (DMSO1/2/3) feeds a stim map, what that costs, and whether
using one plate only (DMSO1, the user's request; DMSO2/3 as controls) reduces the noise. Held-out donors only.

Part 1 from WORK/dmso/facts (lc/dmso_facts.py, per-well quantiles of well_ceiling). Part 2 from WORK/dmso/cases.pkl
(lc/dmso_score.py); sections print "pending" until the jobs have written it. Readings in lc/dmso_plate_readings.md,
merged at build time.
    python -m lc.build_dmso_plate_notebook && jupyter nbconvert --to notebook --execute --inplace notebooks/dmso_plate.ipynb
"""
from pathlib import Path

import nbformat as nbf

cells = []
md = lambda t: cells.append(nbf.v4.new_markdown_cell(t.strip()))
code = lambda t: cells.append(nbf.v4.new_code_cell(t.strip()))

md(r"""
# Which DMSO plate does a stim map use, and does one plate only reduce the noise?

**Question (user, 2026-10-06).** "When we work on a stim, which DMSO do we use, and what is the consequence? I want
to use only DMSO1 for everything — from training to test, for all methods — and know whether that reduces the noise."

Every (donor, stim) of the atlas was measured in three wells, one per plate, each next to its own plate control:
`DMSO1`, `DMSO2`, `DMSO3` (column `drug` of `obs` in `ptb_concatenated_batchcorrected/<celltype>_HVPV.h5ad`).
Unstim has the same three wells, so a plate-matched Unstim → stim map is possible.
Part 1 states what the code does today and measures how much plate-to-plate difference that lets in. Part 2 is the
experiment: the same held-out design (5 repeats × 4 test donors, k = 15 training donors) with five data arms.
""")
md("<!-- ANSWER -->")
md(r"""
## Part 1 — facts

### 1.1 Which wells each pipeline uses (code)

| pipeline | Unstim (source) | stim (target) | plate matched? |
|---|---|---|---|
| paper stim maps `cross_validation_original_13_HVPV` (168 configs, all `drug_used: DMSO`, no `batch_correction`) | all `DMSO*` wells pooled | all `DMSO*` wells pooled | **no** |
| learning curve `lc/cache.py` → `work/lc/cache` (and `cache_full`, `cache_dnorm`) | pooled, ≤ 10k random cells | pooled, ≤ 10k random cells | **no** |
| `heldout/cache_stim_all` (all 196 pairs, same code with `--all-stims`) | pooled | pooled | **no** |
| scoring truth `lc/score.py:truth_tables` (and `where_skill`, `cond_*`) | test donor's pooled cache | test donor's pooled cache | **no** |

- simco `cellot/data/cell.py:81`: with `drug_used: DMSO` the loader keeps `data.obs['drug'].str.startswith("DMSO")`,
  i.e. DMSO1 + DMSO2 + DMSO3. The source/target label comes only from `stim` (`:214`), the split only from
  `split`/`transport` (`:325`, the 80/20 cut grouped by stim, `:394`), so the plate label is dropped. Training
  draws source and target batches independently (`cellot/train/train.py:202-203`): an Unstim cell from plate 1
  can be transported towards stim cells of plate 3. The runtime batch correction (`cell.py:116-119`, which loops over
  the three DMSO treatments) is not used by the paper's stim run (no `batch_correction` key in its configs).
- The drug arm differs: `cell.py:69-77` keeps one control, the plate's `use_DMSO` from `batchcorrection_2.csv`
  (LPZ and SA → DMSO1).
- `lc/cache.py` (same filter, `startswith("DMSO")`, key = (stim, donor), one random permutation per key): the 10k
  cells of a block are a random mix of the three plates in proportion to their cell counts. `lc/cond_train.py`
  pairs source and target by donor, not by plate.
- Test donors: input = her pooled Unstim, truth = her pooled stim quantiles. The split-half noise
  (`lc/score.py:truth_tables`, `x[:h]` vs `x[h:]` of the shuffled block) compares two halves that hold the same mix
  of plates, so it **cannot see a plate effect**.
""")
code(r"""
import warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 120)
W = Path("/scratch/users/bmerwan/atlas_heldout/work/lc/dmso")
FIG = W / "fig"; FIG.mkdir(parents=True, exist_ok=True)
FLOOR = 0.0273
F = W / "facts"
S = pd.read_csv(F / "shares.csv")
for x in ["stim", "Unstim"]:
    t = S[[f"{x}_DMSO{i}" for i in (1, 2, 3)]]
    S[[f"{x} share DMSO{i}" for i in (1, 2, 3)]] = t.div(t.sum(1), axis=0).values
print("Share of each plate in the pooled 10k-cell blocks (mean over celltype x donor):")
display(S.groupby("stim")[[c for c in S if "share" in c]].mean().round(3))
""")
md(r"""
### 1.2 How large is the plate effect? (per-well quantiles, ≤ 30k cells per well)

Per (stim, celltype, donor, q10–q90, marker), from well_ceiling's per-well quantiles (`lc/dmso_facts.py`):
- **R** = pooled stim − pooled Unstim (the response the learning curve scores); between-donor SD = SD of R over the 19 donors;
- **r_p** = stim − Unstim on plate p (plate-matched response); its SD across the 3 plates = **plate SD**;
- the same on **pseudo-wells** (the cells re-dealt at random into groups of the well sizes) = sampling part; plate-only = the rest;
- *matched / unmatched plate-only var*: plate variance of r_p (Unstim from the same plate) over that of the stim level
  alone (Unstim from all plates). < 1 means the plate offset is partly shared by Unstim and stim of a plate, so
  matching removes it.
Live features: |mean R| > 0.0273. SDs are root-mean-square over cases (arcsinh units).
""")
code(r"""
T = pd.read_csv(F / "by_stim.csv", index_col=0)
cols = ["features", "response |R| median", "between-donor SD of R", "plate-only SD of r", "sampling SD of r (pseudo-wells)",
        "plate-only SD of Unstim level", "plate-only var / between-donor var", "plate var in pooled truth / between var",
        "matched / unmatched plate-only var", "median plate-only SD of r", "median between-donor SD of R"]
display(T[cols].round(3))
P = pd.read_csv(F / "by_stim_plate.csv")
print("How far each plate's response sits from the other two (dev_p = r_p - mean of the other two plates):")
display(P.pivot(index="stim", columns="plate", values="RMS systematic (mean over donors)").round(3)
        .add_prefix("systematic RMS ").join(P.pivot(index="stim", columns="plate", values="mean signed dev x sign(R)").round(3)
        .add_prefix("signed (along R) ")).join(P.pivot(index="stim", columns="plate",
        values="share of cases where this plate is the odd one").round(2).add_prefix("odd one out ")))
fig, ax = plt.subplots(figsize=(8, 3.6))
st = [s for s in T.index if s != "all"]; x = np.arange(len(st)); w = 0.27
for i, (c, lab) in enumerate([("between-donor SD of R", "between donors"), ("plate-only SD of r", "between plates (same donor)"),
                              ("sampling SD of r (pseudo-wells)", "sampling (one well)")]):
    ax.bar(x + (i - 1) * w, T.loc[st, c], w, label=lab)
ax.set_xticks(x, st); ax.set_ylabel("SD of the response (arcsinh)"); ax.legend(frameon=False, fontsize=8)
ax.set_title("Plate-to-plate vs donor-to-donor differences in the stim response", fontsize=10); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(FIG / "plate_vs_donor.png", dpi=150)
""")
md("<!-- READ1 -->")

md(r"""
## Part 2 — one plate only, everywhere (held-out donors)

Five arms, each its own cache (Unstim and stim from the same wells, ≤ 10k cells per donor × condition), each used
for training, for the test donor's input and for her truth:

| arm | wells | why |
|---|---|---|
| `pool` | DMSO1+2+3 (today's cache) | reference |
| `poolN1` | pooled, cut to the DMSO1 cell count of each block | same cells count as DMSO1, plates still mixed: isolates "fewer cells" |
| `DMSO1` | plate 1 only, Unstim → stim plate-matched | **the requested arm** (plate 1 = the PI outlier plate) |
| `DMSO2`, `DMSO3` | plate 2 / 3 only | controls: plate 1 specifically, or plate matching in general? |

Methods: CellOT (`lc/cond_train --ctx none`, paired by donor, 20k iterations, retrained for every arm including
`pool`), gauss_ot, quantile_map, median_shift (`lc/baselines`), feature-level shift; identity. k = 15. Scored by
`lc/dmso_score.py` → `work/lc/dmso/cases.pkl`. Tests are on the 20 donor-repeats, never on cases.

Three readouts, because each arm's own truth is a different measurement:
1. **own truth** — each arm scored against its own wells (what a DMSO1-only pipeline would report);
2. **common pooled truth** — every arm's prediction scored against the test donor's pooled three-well response (the
   best estimate of her real response): does training/inputting one plate predict the donor better?
3. **common DMSO1 truth** — the target is one well; adds the *replicate* reference (her own response on plate 2/3,
   plate-matched, transferred to plate 1), i.e. the best any model can do against one well.
""")
code(r"""
import os; CF = Path(os.environ.get("DMSO_CASES", W / "cases.pkl"))   # env override = smoke tests only
READY = CF.exists()
if not READY:
    print("pending: work/lc/dmso/cases.pkl not written yet (jobs in work/lc/STATUS.md)")
else:
    D = pd.read_pickle(CF)
    for c in ["arm", "stim", "celltype", "donor", "marker"]: D[c] = D[c].astype(str)
    METH = ["CellOT", "gauss_ot", "quantile_map", "median_shift", "shift"]
    ARMS = [a for a in ["pool", "poolN1", "DMSO1", "DMSO2", "DMSO3"] if a in set(D.arm)]
    avail = D.groupby("arm")[[m for m in METH if m != "shift"]].apply(lambda x: x.notna().mean()).round(2)
    print("fraction of rows with a prediction:"); display(avail)
    KEY = ["repeat", "stim", "celltype", "donor", "q", "marker"]
    D["effect"] = (D.t - D.u).abs()
""")
md(r"""
### 2.1 Own truth: what a one-plate pipeline would report

skill = 1 − |pred − t| / |t − u| (median per repeat, mean over the 5 repeats), effects > 0.0273 on the arm's own
truth; *ceiling* = split-half ceiling of that truth (blind to plate effects); *n* = cases per repeat.
""")
code(r"""
def skill_tab(X, pred, t, u, by=None):
    sk = 1 - (X[pred] - X[t]).abs() / (X[t] - X[u]).abs()
    g = [X.repeat] + ([X[b] for b in by] if by else [])
    m = sk.groupby(g).median()
    return m.groupby(level=list(range(1, m.index.nlevels))).mean() if by else m.mean()

if READY:
    rows = {}
    for a in ARMS:
        X = D[(D.arm == a) & (D.effect > FLOOR)]
        r = {m: skill_tab(X.dropna(subset=[m]), m, "t", "u") if X[m].notna().any() else np.nan for m in METH}
        r["ceiling"] = (1 - X.noise / X.effect).groupby(X.repeat).median().mean()
        r["n"] = len(X) / X.repeat.nunique(); r["median cells (stim)"] = X.n_s.median()
        rows[a] = r
    display(pd.DataFrame(rows).T.round(3))
    print("CellOT and gauss_ot by stim (own truth):")
    out = {}
    for a in ARMS:
        X = D[(D.arm == a) & (D.effect > FLOOR)]
        for m in ["CellOT", "gauss_ot"]:
            if X[m].notna().any(): out[(m, a)] = skill_tab(X.dropna(subset=[m]), m, "t", "u", ["stim"])
    display(pd.DataFrame(out).round(3))
""")
code(r"""
# Paired tests on own truth: per donor-repeat median skill (20 donor-repeats); sanity check against the learning curve.
def dr_skill(X, m, t="t", u="u"):
    s = 1 - (X[m] - X[t]).abs() / (X[t] - X[u]).abs()
    return s.groupby([X.repeat, X.donor]).median()

def paired(A, B, higher_better=True):
    d = (A - B).dropna(); w = (d > 0) if higher_better else (d < 0)
    return {"mean diff": d.mean(), "A better (x/20)": f"{w.sum()}/{len(d)}", "p (Wilcoxon)": wilcoxon(d).pvalue}

if READY:
    Xp = D[(D.arm == "pool") & (D.effect > FLOOR)]
    print(f"sanity: pool arm CellOT (retrained) {skill_tab(Xp, 'CellOT', 't', 'u'):.3f} vs the learning curve's paired "
          f"CellOT none on the same cases {skill_tab(Xp.dropna(subset=['CellOT_lcnone']), 'CellOT_lcnone', 't', 'u'):.3f}")
    OWN = {(a, m): dr_skill(D[(D.arm == a) & (D.effect > FLOOR)], m) for a in ARMS for m in ["CellOT", "gauss_ot", "shift"]}
    rows = []
    for m in ["CellOT", "gauss_ot", "shift"]:
        for A, B in [("poolN1", "pool"), ("DMSO1", "pool"), ("DMSO1", "poolN1"), ("DMSO2", "poolN1"), ("DMSO3", "poolN1"),
                     ("DMSO1", "DMSO2"), ("DMSO1", "DMSO3")]:
            rows.append({"method": m, "A vs B": f"{A} vs {B}", **paired(OWN[(A, m)], OWN[(B, m)])})
    print("own truth, paired skill per donor-repeat (A better = higher skill):")
    display(pd.DataFrame(rows).set_index(["method", "A vs B"]).round(4))
    print("own truth without PI (CellOT, gauss_ot):")
    display(pd.DataFrame({a: {m: skill_tab(D[(D.arm == a) & (D.effect > FLOOR) & (D.stim != "PI")], m, "t", "u")
                              for m in ["CellOT", "gauss_ot"]} for a in ARMS}).T.round(3))
""")
md("<!-- READ21 -->")
md(r"""
### 2.2 Common target: the donor's pooled three-plate response

Every arm's prediction against the `pool` arm's truth (t0, u0); cases with |t0 − u0| > 0.0273 present in every arm.
*better than pool in x/20* = donor-repeats where the arm's median |error| is below the `pool` arm's (same method);
Wilcoxon over the 20 donor-repeats.
""")
code(r"""
if READY:
    P0 = D[D.arm == "pool"][KEY + ["t", "u"]].rename(columns={"t": "t0", "u": "u0"})
    C = D.merge(P0, on=KEY)
    C = C[(C.t0 - C.u0).abs() > FLOOR]
    res = []
    for m in METH:
        if C[m].isna().all(): continue
        am = [a for a in ARMS if C.loc[C.arm == a, m].notna().any()]      # arms that have this method
        if "pool" not in am: continue
        Cm = C[C.arm.isin(am)].dropna(subset=[m]); Cm = Cm[Cm.groupby(KEY).arm.transform("nunique") == len(am)]
        e = (Cm[m] - Cm.t0).abs(); dr = e.groupby([Cm.arm, Cm.repeat, Cm.donor]).median().unstack(0)
        for a in am:
            Xa = Cm[Cm.arm == a]
            row = {"method": m, "arm": a, "skill vs pooled truth": skill_tab(Xa, m, "t0", "u0"),
                   "median |err|": e[Cm.arm == a].median()}
            if a != "pool" and a in dr:
                d = dr[a] - dr["pool"]
                row["better than pool (donor-repeats)"] = f"{(d < 0).sum()}/{d.notna().sum()}"
                row["p (Wilcoxon)"] = wilcoxon(d.dropna()).pvalue if d.notna().sum() > 5 else np.nan
            res.append(row)
    R22 = pd.DataFrame(res).set_index(["method", "arm"]); display(R22.round(4))
    print("skill vs pooled truth, by stim:")
    BS = {}
    for m in ["CellOT", "gauss_ot", "quantile_map"]:
        Cm = C.dropna(subset=[m])
        if not len(Cm): continue
        for a in ARMS: BS[(m, a)] = skill_tab(Cm[Cm.arm == a], m, "t0", "u0", ["stim"])
    BS = pd.DataFrame(BS); display(BS.round(3))
    m0 = "CellOT" if ("CellOT", "pool") in BS else "gauss_ot"
    fig, ax = plt.subplots(figsize=(9, 3.6)); st = list(BS.index); x = np.arange(len(st)); w = 0.8 / len(ARMS)
    for i, a in enumerate(ARMS):
        if (m0, a) in BS: ax.bar(x + (i - (len(ARMS) - 1) / 2) * w, BS[(m0, a)], w, label=a)
    ax.set_xticks(x, st); ax.set_ylabel(f"{m0} skill vs the donor's\npooled 3-plate response"); ax.axhline(0, color="k", lw=.6)
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.set_title("Held-out donors, k = 15: which plate wells to train on and input (higher = better)", fontsize=10)
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "skill_by_stim_pooled_truth.png", dpi=150)
""")
code(r"""
# CellOT vs pooled truth, PI and the other stims separately; paired median |error| per donor-repeat (A better = lower).
if READY:
    rows = []
    for sub, Cs in [("other stims", C[C.stim != "PI"]), ("PI", C[C.stim == "PI"])]:
        e = (Cs.CellOT - Cs.t0).abs().groupby([Cs.arm, Cs.repeat, Cs.donor]).median().unstack(0)
        for A, B in [("poolN1", "pool"), ("DMSO1", "pool"), ("DMSO2", "pool"), ("DMSO3", "pool"), ("DMSO1", "poolN1"),
                     ("DMSO2", "poolN1"), ("DMSO3", "poolN1"), ("DMSO1", "DMSO2")]:
            rows.append({"stims": sub, "A vs B": f"{A} vs {B}", "skill A": skill_tab(Cs[Cs.arm == A], "CellOT", "t0", "u0"),
                         "skill B": skill_tab(Cs[Cs.arm == B], "CellOT", "t0", "u0"), **paired(e[A], e[B], higher_better=False)})
    display(pd.DataFrame(rows).set_index(["stims", "A vs B"]).round(4))
""")
md("<!-- READ22 -->")
md(r"""
### 2.3 Common target: one well (DMSO1), and the replicate reference

Truth = the test donor's DMSO1 well (t1, u1). Predictions: each arm's map as is, plus two **replicates** (no model):
her own plate-matched response on another plate moved onto her DMSO1 Unstim, u1 + (t_p − u_p), p = 2, 3, and their
mean. Repeated with DMSO2 as the target so plate 1 is not the only one judged.
""")
code(r"""
def one_well(target):
    others = [p for p in ["DMSO1", "DMSO2", "DMSO3"] if p != target]
    if not {target, *others} <= set(ARMS): return None
    base = D[D.arm == target][KEY + ["t", "u"]].rename(columns={"t": "tt", "u": "ut"})
    for p in others:
        base = base.merge(D[D.arm == p][KEY + ["t", "u"]].rename(columns={"t": "t_" + p, "u": "u_" + p}), on=KEY)
        base["rep_" + p] = base.ut + base["t_" + p] - base["u_" + p]
    base["rep_mean"] = base.ut + np.mean([base["t_" + p] - base["u_" + p] for p in others], 0)
    base = base[(base.tt - base.ut).abs() > FLOOR]
    out = {}
    for p in others + ["mean"]: out[f"replicate ({p})"] = skill_tab(base, "rep_" + p, "tt", "ut")
    for m in ["CellOT", "gauss_ot", "quantile_map", "shift"]:
        for a in ARMS:
            X = base.merge(D[D.arm == a][KEY + [m]], on=KEY).dropna(subset=[m])
            if len(X): out[f"{m} [{a}]"] = skill_tab(X, m, "tt", "ut")
    rs = {p: skill_tab(base, "rep_" + p, "tt", "ut", ["stim"]) for p in others}
    return pd.Series(out), pd.DataFrame(rs)

if READY:
    for tg in ["DMSO1", "DMSO2"]:
        r = one_well(tg)
        if r is None: print(f"pending: {tg} target needs all three single-plate arms"); continue
        print(f"target = {tg} well:"); display(r[0].round(3).to_frame("skill"))
        print(f"replicate skill by stim (target {tg}):"); display(r[1].round(3))
""")
code(r"""
# One-well targets, CellOT: the plate-matched arm vs pool / poolN1 / the donor's own replicate wells (mean of the other two
# plates), paired median |error| per donor-repeat (A better = lower), PI and the other stims separately.
if READY and {"DMSO1", "DMSO2", "DMSO3"} <= set(ARMS):
    rows = []
    for tg in ["DMSO1", "DMSO2", "DMSO3"]:
        oth = [p for p in ["DMSO1", "DMSO2", "DMSO3"] if p != tg]
        B = D[D.arm == tg][KEY + ["t", "u"]].rename(columns={"t": "tt", "u": "ut"})
        for p in oth: B = B.merge(D[D.arm == p][KEY + ["t", "u"]].rename(columns={"t": "t_" + p, "u": "u_" + p}), on=KEY)
        B["replicate"] = B.ut + np.mean([B["t_" + p] - B["u_" + p] for p in oth], 0)
        B = B[(B.tt - B.ut).abs() > FLOOR]
        for a in ["pool", "poolN1", tg]: B = B.merge(D[D.arm == a][KEY + ["CellOT"]].rename(columns={"CellOT": a}), on=KEY)
        for sub, Bs in [("all", B), ("other stims", B[B.stim != "PI"]), ("PI", B[B.stim == "PI"])]:
            e = {k: (Bs[k] - Bs.tt).abs().groupby([Bs.repeat, Bs.donor]).median() for k in ["pool", "poolN1", tg, "replicate"]}
            r = {"target": tg, "stims": sub, **{f"skill {'matched' if k == tg else k}": skill_tab(Bs, k, "tt", "ut") for k in ["pool", "poolN1", tg, "replicate"]}}
            for ref in ["pool", "poolN1", "replicate"]:
                pr = paired(e[tg], e[ref], higher_better=False)
                r[f"matched better than {ref}"] = pr["A better (x/20)"]; r[f"p vs {ref}"] = pr["p (Wilcoxon)"]
            rows.append(r)
    display(pd.DataFrame(rows).set_index(["target", "stims"]).round(4))
""")
md("<!-- READ23 -->")
md(r"""
### 2.4 Noise in each truth: split-half vs plate-to-plate

Per stim, median over live cases: effect |t − u| (own truth), split-half noise of t, and the plate-to-plate
difference of the plate-matched response |(t_p − u_p) − (t_p' − u_p')| between single-plate arms.
""")
code(r"""
if READY:
    N = D[D.effect > FLOOR].groupby(["stim", "arm"]).agg(effect=("effect", "median"), split_half=("noise", "median"),
                                                          cells=("n_s", "median")).unstack("arm")
    display(N.round(3))
    pl = [p for p in ["DMSO1", "DMSO2", "DMSO3"] if p in ARMS]
    if len(pl) == 3:
        M = D[D.arm.isin(pl)].assign(r=lambda x: x.t - x.u).pivot_table(index=KEY, columns="arm", values="r")
        M = M.join(P0.set_index(KEY)); M = M[(M.t0 - M.u0).abs() > FLOOR]
        PP = pd.DataFrame({f"{a} vs {b}": (M[a] - M[b]).abs().groupby(level="stim").median()
                           for a, b in [("DMSO1", "DMSO2"), ("DMSO1", "DMSO3"), ("DMSO2", "DMSO3")]})
        PP["pooled effect"] = (M.t0 - M.u0).abs().groupby(level="stim").median()
        print("plate-to-plate |difference of plate-matched responses| (held-out test donors, <= 10k cells per well):")
        display(PP.round(3))
""")
md(r"""
### 2.5 Is the donor-specific part better captured? (specific R²)

As in `where_cellot_skill.ipynb`: r = t − shift, rc = r centred over the repeat's test donors; specific R² =
1 − Σ(predicted rc − rc)² / Σ rc². Computed against each arm's own truth and against the pooled truth (shift and
centring from the `pool` arm there).
""")
code(r"""
GRP = ["repeat", "stim", "celltype", "q", "marker"]
def spec(X, m, t, sh):
    X = X.dropna(subset=[m]).copy()
    X["n"] = X.groupby(GRP)[t].transform("size"); X = X[X.n >= 3]
    r = X[t] - X[sh]; rc = r - r.groupby([X[c] for c in GRP]).transform("mean")
    h = X[m] - X[sh]; hc = h - h.groupby([X[c] for c in GRP]).transform("mean")
    return 1 - ((hc - rc) ** 2).sum() / (rc ** 2).sum()

if READY:
    S0 = D[D.arm == "pool"][KEY + ["shift"]].rename(columns={"shift": "shift0"})
    rows = {}
    for a in ARMS:
        X = D[(D.arm == a) & (D.effect > FLOOR)]
        Y = C[C.arm == a].merge(S0, on=KEY)
        rows[a] = {f"{m} own": spec(X, m, "t", "shift") for m in ["CellOT", "gauss_ot"] if X[m].notna().any()}
        rows[a].update({f"{m} vs pooled": spec(Y, m, "t0", "shift0") for m in ["CellOT", "gauss_ot"] if Y[m].notna().any()})
    display(pd.DataFrame(rows).T.round(3))
    print("CellOT specific R2 on own truth, per repeat (5 repeats x 4 test donors):")
    display(pd.DataFrame({a: [spec(D[(D.arm == a) & (D.effect > FLOOR) & (D.repeat == r)], "CellOT", "t", "shift")
                              for r in sorted(D.repeat.unique())] for a in ARMS}).round(3).rename_axis("repeat"))
""")
md("<!-- READ25 -->")

md(r"""
## Part 3 — can the plate be recognised from the cells? And how big is the shared well effect? (2026-10-07)

**Design fact (user, 2026-10-07): each well holds all donors (barcoded), one (stim or Unstim, plate) per well.** So
whatever is common to a well is common to every donor in it: that is the only plate signature a map trained on other
donors could ever learn. Checked first: the single-plate caches are built per cell from its demultiplexed `drug` label
(`lc/cache.py:68`) and partition the pooled cache exactly (pDCs: 182 + 159 + 129 = 470 cells, 100 % found in the pool);
each arm trains, feeds the test donor and scores her on its own plate's cells (`meta.json` `cache_dir`).

`lc/plate_classify.py`: per (condition, celltype), PI excluded, Unstim included, 13 functional markers, **held-out
donors** (GroupKFold over donors), chance 1/3:
- *cell*: one cell → plate (gradient boosting);
- *bag*: a batch of 100 cells (5 quantiles × 13 markers) → plate (logistic regression);
- *bag_dc*: the same, each donor's batches centred on her own mean batch (no plate label used): "given a donor's
  batches, which one is plate k?".

`lc/plate_magnitude.py`: per donor, the 5 × 13 quantiles of each plate's well, centred on her 3-plate mean. The
**shared well offset** = the part of that deviation common to all donors (learnable); **donor × well** = the rest
(new for every donor and well, not learnable); both for the level (one well) and for the response (stim well −
Unstim well of the same plate, i.e. what a plate-matched map sees). Compared with the between-donor SD (of the
3-plate means) and the response size.
""")
code(r"""
CL = W / "classify"
A = pd.read_csv(CL / "accuracy.csv")
print("accuracy on held-out donors (chance 0.33): median over celltypes")
display(A.pivot_table(index="condition", columns="readout", values="accuracy", aggfunc="median").round(3))
print("bag_dc: recall per plate (median over celltypes)")
display(A[A.readout == "bag_dc"].groupby("condition")[["recall_DMSO1", "recall_DMSO2", "recall_DMSO3"]].median().round(3))
M = pd.read_csv(CL / "plate_magnitude.csv")
print("plate effect size (arcsinh, median over celltypes)")
display(M.groupby(["kind", "condition"])[["shared_well_rms", "donor_x_well_rms", "between_donor_sd", "response_size"]]
        .median().round(3))
R = M[(M.kind == "response") & (M.condition != "PI")].copy()
R["true donor SD (noise removed)"] = np.sqrt((R.between_donor_sd ** 2 - R.donor_x_well_rms ** 2 / 3).clip(lower=0))
print("response, PI excluded: between-donor SD of the 3-plate mean vs the donor x well noise it still contains")
display(R.groupby("condition")[["between_donor_sd", "donor_x_well_rms", "true donor SD (noise removed)"]].median().round(3))
""")

md(r"""
### 3b. The same decomposition on one footing (variance components, all SDs)

Supersedes the magnitude table above, which mixed statistics (an uncorrected RMS of plate offsets, a corrected SD, a
mean of SDs and a mean absolute response). `lc/plate_variance.py`: for one feature, donor d, plate p,
x[d, p] = mu + a[d] + w[p] + e[d, p] (every well holds all donors). SD_donor = real differences between donors,
SD_well = well effect shared by all donors (learnable), SD_dxw = donor × well + cell sampling (not learnable); each
estimated from the two-way table with the noise contributions removed, as an SD per feature, then RMS over the 65
features, median over celltypes. *mean_response* = RMS of the average response, for scale. Last columns: the share of a
donor's apparent deviation that is real, for one well (a[d] + e) and for the 3-well average (a[d] + e / 3).
""")
code(r"""
V = pd.read_csv(CL / "plate_variance.csv")
T = V.groupby(["kind", "condition"])[["mean_response", "SD_donor", "SD_well", "SD_dxw"]].median()
T["real share, one well"] = T.SD_donor ** 2 / (T.SD_donor ** 2 + T.SD_dxw ** 2)
T["real share, 3 wells"] = T.SD_donor ** 2 / (T.SD_donor ** 2 + T.SD_dxw ** 2 / 3)
display(T.round(3))
""")

md("<!-- READ3 -->")

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
txt = Path(__file__).with_name("dmso_plate_readings.md")
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
out = Path(__file__).resolve().parents[1] / "notebooks/dmso_plate.ipynb"
nbf.write(nb, out); print("wrote", out)
