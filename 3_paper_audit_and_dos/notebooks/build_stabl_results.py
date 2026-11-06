"""Generate notebooks/stabl_results.ipynb -- the provenance record for the Stabl half.

Same convention as baselines_results.ipynb: every table is recomputed in the notebook
from the source CSVs rather than pasted, and the notebook is executed in place so the
outputs are readable without re-running anything.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C, M = [], []
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(src):
    cells.append(nbf.v4.new_code_cell(src.strip()))


md(r"""
# Stabl half — reading the paper's Fig 5b, and what it does and does not establish

Companion to `baselines_results.ipynb` (which covers the cell-level CellOT-vs-baselines
benchmark). This one covers the **downstream** question: the Stabl model that turns
CellOT-simulated stimulation responses into a predicted time to onset of spontaneous
labor (DOS), i.e. Figure 5b/5c.

Four things are established here, in order:

1. **How to read `Scores training CV.csv`** — the leading number is one aggregate R² over
   pooled out-of-fold predictions, the brackets are a bootstrap CI, and neither is a mean
   over splits.
2. **Which stored run is the paper's Fig 5b**, and that it is a *rescore* whose entire
   advantage comes from the downstream XGBoost config `params_43.json`, itself chosen by
   an argmin over the very splits it is reported on.
3. **Measured Unstim predicts nothing; the simulated responses carry the whole signal** —
   now shown at the paper's own configuration, not a weaker one.
4. **A shared 13×13 affine map (`gauss_ot`) reproduces CellOT's downstream result.** So
   Fig 5b does not establish that CellOT is needed.

Everything below reads from `$SCRATCH` and the read-only group home. Nothing is retrained.
Dates: analyses run 2026-09-22.
""")

code(r"""
import ast, glob, os, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160, "display.max_columns", 50)

O    = Path("/scratch/users/bmerwan/simco_work/onset_test")
GH   = Path("/home/groups/gbrice/ptb-drugscreen/ool_stabl/onset_test")   # read-only
RNG  = np.random.default_rng(0)

y = pd.read_csv(O / "outcome_table_all_pre.csv", index_col=0)["DOS"].astype(float)
print(f"DOS: n={len(y)}  patients={len({i.rsplit('_',1)[0] for i in y.index})}  "
      f"mean={y.mean():.1f}  sd={y.std():.1f}  range=[{y.min():.0f}, {y.max():.0f}]")
print(f"constant-mean predictor:  RMSE={np.sqrt(((y-y.mean())**2).mean()):.1f} d   "
      f"MAE={np.abs(y-y.mean()).mean():.1f} d")
""")

md(r"""
## §1 — What the numbers in `Scores training CV.csv` actually are

`R2`, `RMSE`, `MAE`: **one** score over the 177 pooled out-of-fold predictions. Per sample,
the prediction is the median across the splits in which it was in test
(`stablVMax/stabl/single_omic.py:295`), then `r2_score(y, predictions)` (`:329`). The
brackets are a **1000-draw bootstrap 95 % CI over rows** (`stablVMax/stabl/utils.py:344`),
unseeded. Only `N features` and `CVS` use median [IQR] across folds (`:316-326`).

Two consequences people get wrong, both checked below:
""")

code(r"""
# (a) the aggregate R2 is NOT the mean of the per-split R2 -- the per-split value is much
# lower, because each split tests ~13 patients against that subset's own variance, and the
# median-over-splits prediction is an implicit ensemble.
rows = []
for arm, d in [("unstim_only", O/"cmp_unstim_vs_simstim_n20/unstim_only"),
               ("no_Unstim",   O/"cmp_unstim_vs_simstim_n20/no_Unstim")]:
    fm  = pd.read_csv(d/"Summary/fold_metrics.csv")
    agg = pd.read_csv(d/"Summary/Scores training CV.csv", index_col=0)
    for m, g in fm.groupby("model"):
        rows.append({"arm": arm, "model": m,
                     "R2 per-split median": round(g.r2.median(), 3),
                     "R2 per-split mean":   round(g.r2.mean(), 3),
                     "R2 aggregate":        float(str(agg.loc[m, "R2"]).split(" ")[0])})
print("CellOT, 20 splits\n")
print(pd.DataFrame(rows).to_string(index=False))
""")

code(r"""
# (b) the bracket is a row bootstrap. Reproduce it, check its run-to-run wobble, and check
# the worry that resampling rows rather than patients makes it too narrow.
p = pd.read_csv(O/"cmp_unstim_vs_simstim_n20/no_Unstim/Training CV/STABL Lasso/STABL Lasso predictions.csv",
                index_col=0).iloc[:, 0].astype(float)
idx = p.index.intersection(y.index)
yt, yp = y.loc[idx].values, p.loc[idx].values
print(f"point estimate R2 = {r2_score(yt, yp):.3f}   (score table says 0.298)")

def boot(idx_fn, n=1000):
    s = [r2_score(yt[k], yp[k]) for k in (idx_fn() for _ in range(n))]
    return np.percentile(s, [2.5, 97.5])

n = len(yt)
by_row = lambda: RNG.integers(0, n, n)
pat  = np.array([i.rsplit("_", 1)[0] for i in idx]); upat = np.unique(pat)
loc  = {u: np.where(pat == u)[0] for u in upat}
by_patient = lambda: np.concatenate([loc[u] for u in RNG.choice(upat, len(upat))])

w = {}
for label, fn in [("rows (what the code does)", by_row), ("patients (cluster bootstrap)", by_patient)]:
    lo, hi = boot(fn); w[label] = hi - lo
    print(f"  {label:30} [{lo:.3f}, {hi:.3f}]   width {hi-lo:.3f}")
ratio = w["patients (cluster bootstrap)"] / w["rows (what the code does)"]
print(f"\n-> resampling patients instead of rows widens the CI by {100*(ratio-1):+.0f}% "
      f"(re-run to run this moves by a few %, both bootstraps being unseeded).")
print("   So the repeated-measures worry does not materially bite -- plausibly because DOS")
print("   varies a lot between G1, G2 and G3, so a patient's 3 rows are not redundant.")
print("   rows left out of a single draw:", n - len(set(by_row())), f"/ {n}")
""")

md(r"""
## §2 — Which stored run is the paper's Fig 5b

The preprint reports, for Fig 5b: **Pearson R = 0.64, RMSE = 24.1 days, 21 features
(range 18–30), 177 samples / 63 patients**, and AUROC 0.73 [0.60–0.85] for Fig 5c.

Scan every `Scores training CV.csv` under the group-home Stabl tree on RMSE.
""")

code(r"""
hits = []
for f in glob.glob(f"{GH}/*/Summary/Scores training CV.csv") + \
         glob.glob(f"{GH}/*/*/Summary/Scores training CV.csv"):
    try:    d = pd.read_csv(f, index_col=0)
    except Exception: continue
    if "RMSE" not in d.columns: continue
    for m, row in d.iterrows():
        try:  rmse = float(str(row["RMSE"]).split(" ")[0])
        except Exception: continue
        if 23.9 <= rmse <= 24.3:
            hits.append({"dir": os.path.relpath(f, GH).split("/Summary")[0], "model": m,
                         "RMSE": row["RMSE"], "R2": row["R2"], "N features": row["N features"]})
h = pd.DataFrame(hits)
print(f"{len(glob.glob(f'{GH}/*/Summary/Scores training CV.csv'))} + "
      f"{len(glob.glob(f'{GH}/*/*/Summary/Scores training CV.csv'))} score tables scanned\n")
print(h[h["N features"].str.startswith("21.000")].to_string(index=False))
""")

code(r"""
# Pearson recomputed on its own out-of-fold predictions -> the preprint's 0.64.
pp = pd.read_csv(GH/"GS_cellot_longallstims_foldinfo/Training CV/STABL ALasso/STABL ALasso predictions.csv",
                 index_col=0).iloc[:, 0].astype(float)
i  = pp.index.intersection(y.index)
print(f"GS_cellot_longallstims_foldinfo / STABL ALasso, n={len(i)}")
print(f"  Pearson r = {pearsonr(y.loc[i], pp.loc[i])[0]:.3f}   (paper: 0.64)")
print(f"  R2        = {r2_score(y.loc[i], pp.loc[i]):.3f}")
print(f"  RMSE      = {np.sqrt(mean_squared_error(y.loc[i], pp.loc[i])):.1f} d   (paper: 24.1)")
print(f"  MAE       = {mean_absolute_error(y.loc[i], pp.loc[i]):.1f} d")
""")

md(r"""
**Beware the near-miss.** `results_ina_13OG_final_long_allstims_filtered_xgboost_knockoff_GSS`
is the same comparison under the obvious-looking name and scores R² 0.319 / RMSE 26.0 — it
is *not* the paper's run.

### It is a rescore, not an independent fit

The two runs select **identical features in 100/100 folds**. Stabl did the same job in both;
the whole 0.319 → 0.415 gap is the downstream XGBoost regressor.
""")

code(r"""
a = pd.read_csv(GH/"GS_cellot_longallstims_foldinfo/Training CV/Selected Features STABL ALasso.csv", index_col=0)
b = pd.read_csv(GH/"results_ina_13OG_final_long_allstims_filtered_xgboost_knockoff_GSS/Training_CV/Selected_Features_STABL_ALasso.csv", index_col=0)
n = min(len(a), len(b))
same = sum(set(ast.literal_eval(a.iloc[i, 0])) == set(ast.literal_eval(b.iloc[i, 0])) for i in range(n))
print(f"folds with identical selected-feature lists: {same}/{n}")

# and the paper's own run is overwhelmingly built on simulated, not measured, features
from collections import Counter
cnt, tot, unst = Counter(), 0, 0
for s in a["Fold selected features"]:
    fs = ast.literal_eval(s); tot += len(fs)
    for x in fs:
        cnt[x.rsplit("_", 1)[-1]] += 1
        unst += x.endswith("_Unstim")
print(f"\n{len(a)} splits, {tot} selections, {tot/len(a):.1f} features per split")
print(f"measured-Unstim share: {100*unst/tot:.1f}%  ({unst}/{tot})")
print("by block:", dict(cnt.most_common()))
""")

md(r"""
### The headline is an argmin over the splits it is reported on

`grid_search_model.py:109` builds the *same* `GroupShuffleSplit(n_splits=100, test_size=0.2,
random_state=42)`, and `:268-272` keeps the parameter set minimising RMSE on it. So 24.1 d is
the best of 97 `param_grid_files/params_*.json` candidates evaluated on the reported CV — the
same class of selection leak as CellOT's `model.pt` checkpoint, one level up.

**`params_43.json` is the winning config**, verified: rescoring the near-miss run's fold
features with it reproduces the paper to three decimals, in 51 seconds.
""")

code(r"""
ref  = pd.read_csv(GH/"GS_cellot_longallstims_foldinfo/Summary/Scores training CV.csv", index_col=0)
mine = pd.read_csv(O/"rescore_probe/params_43/Summary/Scores training CV.csv", index_col=0)
print("paper's stored run:"); print(ref.to_string())
print("\nrescore with params_43.json (51 s, stabl_launch/rescore_arms_params43.sh):")
print(mine.to_string())
print("\n-> R2, RMSE, MAE and N features match; only the unseeded bootstrap bounds differ.")
""")

md(r"""
## §3 — Measured Unstim vs simulated responses, at the paper's own configuration

Because the rescore is a minute, all three feature blocks can be put on the paper's footing:
same `params_43.json`, same ALasso, same 100 splits, each reusing its own stored Stabl
selection. 95 s for the three.
""")

code(r"""
rows = []
for arm, lab in [("unstim_only", "measured Unstim only"),
                 ("no_Unstim",   "simulated stims only"),
                 ("both",        "Unstim + simulated  (= paper Fig 5b)")]:
    d = pd.read_csv(O/f"rescore_params43/{arm}/Summary/Scores training CV.csv", index_col=0)
    r = d.loc["STABL ALasso"]
    rows.append({"features": lab, "R2": r["R2"], "RMSE": r["RMSE"].split(" ")[0] + " d",
                 "MAE": r["MAE"].split(" ")[0] + " d", "N features": r["N features"].split(" ")[0]})
print("100 splits, STABL ALasso, params_43  (constant-mean predictor: RMSE 31.5 d)\n")
print(pd.DataFrame(rows).to_string(index=False))
""")

md(r"""
Measured Unstim alone is worth **essentially nothing** — its CI straddles 0 and Stabl keeps 2
features. The simulated responses carry the entire result, and adding measured Unstim back on
top buys +0.015 R², far inside the CI. The same holds inside the paper's own run: 9.2 % of its
2602 feature selections are measured Unstim (§2).

The scatter of predicted vs observed days is `notebooks/plot_pred_vs_obs.py` →
`simco_work/onset_test/pred_vs_obs_DOS.png`: note the strong regression to the mean — when the
observed DOS is −110 days the model rarely predicts below −80.

## §4 — Which baselines can test whether CellOT is needed

A Stabl feature is the per-marker **median** of the transported cells. That disqualifies every
**diagonal** baseline *by construction*, before any compute:

| baseline | map | transported median | verdict |
|---|---|---|---|
| `median_shift` | `x + d` | `median(x) + d` | same column after standardisation |
| `quantile_map` | per-marker 1-D OT | monotone function of one Unstim feature | no new information |
| `linear_median*` | `x + (a + Bu − u)` | `a + B·u` | affine in that celltype's 13 Unstim medians |
| **`gauss_ot`** | `mu_t + A(x − mu_s)`, full 13×13 `A` | **mixes markers → nonlinear functional of the patient's joint cell distribution** | **the real test** |

`gauss_ot` is the cheapest map that has the property CellOT's has. `baselines/simco_features.py`
builds its feature table: `fit` (168 maps on the train side of CellOT's own cell split, 10 min)
then `apply` (transport all 177 OOL patients' Unstim cells, 10 min).
""")

code(r"""
g = pd.read_csv(O/"gaussot_13OG_full.csv", index_col=0)
c = pd.read_csv("/scratch/users/bmerwan/simco_work/cellot_results/ina_13OG_final_long_allstims.csv").set_index("Individual")
print(f"gauss_ot table {g.shape} | CellOT raw {c.shape} | same row order: {list(g.index)==list(c.index)}")

u = [x for x in g.columns if x.endswith("_Unstim") and x in c.columns]
print(f"\nsanity 1 -- the {len(u)} measured-Unstim columns are identical to CellOT's: "
      f"{np.isclose(g[u], c.loc[g.index, u]).mean():.3f}")
print("            (independent proof the pipeline sees the same cells, markers and celltype map)")

s = [x for x in g.columns if x in c.columns and not x.endswith("_Unstim")]
print(f"sanity 2 -- the {len(s)} stim columns do differ: mean |gaussOT - cellOT| = "
      f"{(g[s]-c.loc[g.index, s]).abs().values.mean():.3f}, against a feature sd of {c[s].std().mean():.3f}")

for view in ["no_Unstim", "both"]:
    r = pd.read_csv(O/f"ina_13OG_final_long_allstims_filtered{'_no_Unstim' if view=='no_Unstim' else ''}.csv", index_col=0)
    out = g.reindex(index=r.index, columns=r.columns)
    print(f"sanity 3 -- view '{view}': {out.shape}, missing columns "
          f"{len([x for x in r.columns if x not in g.columns])}, NaN {int(out.isna().sum().sum())}")
""")

md(r"""
## §5 — `gauss_ot` reproduces CellOT's downstream result

Same script (`run_regression_cv.py`), same `random_state=42` splits, same column schema, 20
splits. The CellOT `both` arm was rerun on 2026-09-22 because the original was killed mid-fold.
""")

code(r"""
runs = {("CellOT",  "no_Unstim"): O/"cmp_unstim_vs_simstim_n20/no_Unstim",
        ("CellOT",  "both"):      O/"cellot_both_n20/both",
        ("gauss_ot","no_Unstim"): O/"gaussot_stabl_n20/no_Unstim",
        ("gauss_ot","both"):      O/"gaussot_stabl_n20/both",
        ("measured","unstim"):    O/"cmp_unstim_vs_simstim_n20/unstim_only"}
tab = {}
for (src, view), d in runs.items():
    s = pd.read_csv(d/"Summary/Scores training CV.csv", index_col=0)
    tab[f"{src} / {view}"] = {m: float(str(s.loc[m, "R2"]).split(" ")[0]) for m in s.index}
print("aggregate R2, 20 splits\n")
print(pd.DataFrame(tab).T[["STABL Lasso", "STABL ALasso", "STABL ElasticNet"]].to_string())
""")

code(r"""
# The sharp test: the 20 splits are literally identical (random_state=42), so compare
# per-split R2 pairwise. This removes the split-to-split noise the CIs above are full of.
def fm(p): return pd.read_csv(Path(p)/"Summary/fold_metrics.csv")

out = []
for view in ["both", "no_Unstim"]:
    for m in ["STABL Lasso", "STABL ALasso", "STABL ElasticNet"]:
        a = fm(runs[("CellOT", view)]).query("model == @m").sort_values("fold").r2.values
        b = fm(runs[("gauss_ot", view)]).query("model == @m").sort_values("fold").r2.values
        n = min(len(a), len(b)); a, b = a[:n], b[:n]; d = a - b
        lo, hi = np.percentile([np.mean(RNG.choice(d, n)) for _ in range(4000)], [2.5, 97.5])
        out.append({"table": view, "model": m.replace("STABL ", ""),
                    "CellOT": round(a.mean(), 3), "gauss_ot": round(b.mean(), 3),
                    "mean delta": round(d.mean(), 3), "CI95": f"[{lo:+.3f}, {hi:+.3f}]",
                    "wilcoxon p": round(wilcoxon(a, b).pvalue, 3),
                    "CellOT wins": f"{int((d>0).sum())}/{n}"})
print("paired per-split R2, CellOT minus gauss_ot\n")
print(pd.DataFrame(out).to_string(index=False))
""")

md(r"""
**Six cells, six times indistinguishable from zero.** On the paper's own `both` table with
Lasso the gap is 0.002 and CellOT wins only 8/20 splits.

So **Fig 5b does not establish that CellOT is needed**: the downstream signal survives a map
with no neural network in it, fitted from a mean and a covariance per (stim, celltype) in 20
minutes, against 168 models trained for ~30 h each. This is consistent with the cell-level
benchmark in `baselines_results.ipynb`, where `gauss_ot` already beat CellOT on `mean_error`
in 96.9 % of live pairs and on `corr_error` in ~95 %.

What CellOT does keep is **selection stability** (`CVS`, the Jaccard overlap between folds):
""")

code(r"""
cvs = {}
for (src, view), d in runs.items():
    if view == "unstim": continue
    s = pd.read_csv(d/"Summary/Scores training CV.csv", index_col=0)
    cvs[f"{src} / {view}"] = {m: s.loc[m, "CVS"].split(" ")[0] for m in s.index}
print(pd.DataFrame(cvs).T[["STABL Lasso", "STABL ALasso", "STABL ElasticNet"]].to_string())
""")

md(r"""
## §6 — Does adding the measured-Unstim block hurt?

At 20 splits, `both` scores *below* `no_Unstim` for CellOT + Lasso, which looks like Stabl
failing to ignore useless features. It is not: five of the six cells show no effect, the sixth
is borderline, and **the sign reverses at 100 splits** (stored runs: 0.328 → 0.333; at
`params_43`: 0.400 → 0.415).
""")

code(r"""
out = []
for src in ["CellOT", "gauss_ot"]:
    for m in ["STABL Lasso", "STABL ALasso", "STABL ElasticNet"]:
        x = fm(runs[(src, "no_Unstim")]).query("model == @m").sort_values("fold").r2.values
        z = fm(runs[(src, "both")]).query("model == @m").sort_values("fold").r2.values
        n = min(len(x), len(z)); d = z[:n] - x[:n]
        lo, hi = np.percentile([np.mean(RNG.choice(d, n)) for _ in range(4000)], [2.5, 97.5])
        out.append({"source": src, "model": m.replace("STABL ", ""),
                    "both - no_Unstim": round(d.mean(), 3), "CI95": f"[{lo:+.3f}, {hi:+.3f}]",
                    "wilcoxon p": round(wilcoxon(x[:n], z[:n]).pvalue, 2),
                    "both wins": f"{int((d>0).sum())}/{n}"})
print("paired per-split R2, adding the measured-Unstim block\n")
print(pd.DataFrame(out).to_string(index=False))
""")

md(r"""
Why it *can* happen at all, even when Stabl works — the expectation "more features can never
hurt" is false for any method with a data-driven selection threshold at finite n:

1. **The artificial block grows with the real one.** `artificial_proportion=1.0`, so going
   852 → 1216 real features also adds 364 knockoffs — and the FDR threshold is estimated *from*
   those. The null distribution changes; this is not "add candidates, keep the good ones".
   This mechanism is certain, it is in the code.
2. **The lambda grid is fixed** (`np.logspace(0, 2, 10)`) and is not rescaled with `p`, so the
   same alphas over a wider design matrix land at a different effective sparsity.
3. **The added block is collinearity, not information** — the simulated features are a
   deterministic function of the same patient's Unstim cells. Lasso is precisely the method
   that is unstable under correlated predictors, which fits the effect showing up on Lasso and
   not on ALasso or ElasticNet. Caveat cutting the other way: `CVS` does *not* degrade on
   `both` (§5), so this mechanism is plausible here but not demonstrated.

## §7 — Limits to state with any of the above

* §5 and §6 are **20 splits, not the paper's 100**, so the absolute numbers are the weaker
  configuration and the CIs are wide; the comparison is internally exact (same splits, same
  schema, same script) but the paired tests rest on 20 points.
* Putting `gauss_ot` on the paper's exact footing means a 100-split run (~8 h at 5 cores, ~3 h
  at 16). The rescore shortcut does **not** apply: `get_foldinfo_on_existing_run.py:139`
  hardcodes `n_splits=100` and would not match 20-fold feature lists.
* `DOS` as "days to onset of spontaneous labor" is inferred from the data, not confirmed
  against the paper.
* The AUROC 0.73 of Fig 5c was not reproduced here; only the regression arm was.

## §8 — Commands

```bash
# the paper's Fig 5b, and the three arms on its footing        (~95 s)
bash stabl_launch/rescore_arms_params43.sh

# measured vs simulated from scratch, any n_splits             (1.3 h / 8 h / 10 h at 100)
bash stabl_launch/run_unstim_vs_simstim.sh 20

# the gauss_ot feature table                                   (10 min + 10 min)
cat /scratch/users/bmerwan/simco_work/baselines/gaussot_celltypes.txt \
  | xargs -P 4 -I{} python -m baselines.simco_features fit --celltype {} \
      --outdir $SCRATCH/simco_work/baselines/gaussot_maps
python -m baselines.simco_features apply \
  --maps $SCRATCH/simco_work/baselines/gaussot_maps \
  --out  $SCRATCH/simco_work/onset_test/gaussot_13OG_full.csv

# Stabl on it                                                  (1.6 h / 1.7 h at 20 splits)
python stabl_launch/run_regression_cv.py --features_path .../gaussot_13OG_no_Unstim.csv \
  --results_dir .../gaussot_stabl_n20/no_Unstim --artificial_type knockoff \
  --model_chosen xgboost --n_splits 20

# the scatter
python notebooks/plot_pred_vs_obs.py
```

Run inside the existing interactive allocation, not via `sbatch`. Stabl parallelises internally
(`n_jobs=-1`, joblib over the 1000 bootstrap fits) and saturates whatever cores it is given;
memory peaks around 1 GB, so `bigmem` buys nothing and a GPU is useless here.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9"},
}
out = "/scratch/users/bmerwan/simco/notebooks/stabl_results.ipynb"
nbf.write(nb, out)
print("wrote", out, f"({len(cells)} cells)")
