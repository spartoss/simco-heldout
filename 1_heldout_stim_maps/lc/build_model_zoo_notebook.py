"""Build notebooks/model_zoo.ipynb: every held-out-donor stim predictor tried so far, how each was built and tuned, and
its result on the same cases.
Input: WORK/model_zoo/*.csv from lc/model_zoo.py; side studies from WORK/fm/ (ImmuneFM sanity, fine-tunes, residual
prediction, closed-form phenotype stratification) and mfm_heldout/work/runs (MFM training logs).
    python -m lc.model_zoo && python -m lc.build_model_zoo_notebook && \
    jupyter nbconvert --to notebook --execute --inplace notebooks/model_zoo.ipynb          (env_cpu.sh)
"""
from pathlib import Path

import nbformat as nbf

cells = []
md = lambda t: cells.append(nbf.v4.new_markdown_cell(t.strip()))
code = lambda t: cells.append(nbf.v4.new_code_cell(t.strip()))

md(r"""
# Model zoo — every way we tried to predict a NEW donor's stim response, and how fairly each was tested

**Task.** For a held-out donor, predict the stimulated distribution of each (stim, celltype) from that donor's
Unstim cells only (13 functional markers, arcsinh/5, batch-corrected atlas, DMSO wells). Learning-curve design:
5 repeats × 4 test donors, maps trained on k = 3, 5, 8 or 15 donors. Built 2026-10-07 from all runs to date
(`atlas_heldout`, `mfm_heldout`).

**Why this notebook.** The classic CellOT was debugged and tuned over weeks (parity with the paper's model,
checkpoint selection, training length); most newer variants were run once, with defaults. §2 says, for every
method, what it is, its preprocessing and hyperparameters **and how much tuning it got**, so a negative result can
be read as "does not work" or "not tested properly".

**Contents.** §1 definitions · §2 inventory (what / preprocessing / hyperparameters / tuning effort) · §3 results on
common cases (learning curve, k = 15 ranking, paired tests, by stim) · §4 side studies that are not maps (ImmuneFM,
donor covariates, closed-form stratification, flow-matching training) · §5 what is under-tested and a fair tuning
protocol · §6 files.
""")

md(r"""
## 1. Definitions

A **case** = (repeat, stim, celltype, k, test donor, quantile q10…q90, marker). Truth = the test donor's stim
quantile; effect = |truth − Unstim|; only cases with effect > 0.0273 (atlas technical floor) are scored.
- **skill** = 1 − |pred − truth| / effect: 1 = perfect, 0 = no better than "no change". Median per repeat, mean over
  repeats.
- **dev_R2** = 1 − Σ(pred − truth)² / Σ(truth − mean of the training donors' stim)². How much of the test donor's
  deviation from the average training donor is predicted.
- **between_R2**: predictions and truth both centred across the repeat's 4 test donors (removes everything common to
  the donors): how well the *differences between donors* are predicted. The "identity" row = what the donors'
  Unstim differences alone explain (0.68).
- **% closer** (paired): share of cases where method A's error is below method B's; p = Wilcoxon over
  (repeat, map, donor) median errors.

**Common cases.** Every table compares methods on exactly the same cases (a case is kept only if every method of
the frame predicts it). Two frames: **lc** = methods run at k = 3/5/8/15; **k15** = k = 15 only, adding the arms run
at k = 15 only. **Repeats 1–4** (repeat 0 was used to choose the context weight decay).
""")

code(r"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
W = Path('/scratch/users/bmerwan/atlas_heldout/work/lc'); Z = W / 'model_zoo'
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 100)
C = pd.read_csv(Z / 'lc_curve.csv'); C15 = pd.read_csv(Z / 'k15_curve.csv')
P = pd.read_csv(Z / 'lc_paired.csv'); P15 = pd.read_csv(Z / 'k15_paired.csv')
print('cases: lc', C.n_cases.max(), '| k15', C15.n_cases.max())
""")

md(r"""
## 2. Inventory: what each method is, and how much it was tuned

Tuning effort: **extensive** = debugged, checked against a reference, several settings tried; **some** = one
targeted sweep or inner cross-validation; **none** = one run with defaults (ours or the paper's).

### 2.1 Maps without any donor information

| method | what it is | preprocessing / data | key hyperparameters | tuning effort | code |
|---|---|---|---|---|---|
| `CellOT` | per-(stim, celltype) ICNN OT map (Bunne et al. 2023), source = pooled Unstim cells of the k training donors, target = pooled stim cells, drawn independently | 13 markers, ≤ 10k cells per donor × condition, 80 % train / 20 % val per donor | ICNN 4 × 64, batch 256, 10 g : 1 f, Adam 1e-4 (0.5, 0.9), 50k it., best g on val median error | **extensive** (parity with the paper's `model.pt`, iterations, caps, checkpoint) | `lc/train.py` |
| `CellOT_paired` | same network, but each batch row draws its source and target cell from the **same** training donor (donor-matched batches; not a donor-specific map) | same | same, 20k it. | **extensive** (inherits CellOT) | `lc/cond_train.py --ctx none` |
| `gauss_ot`, `quantile_map`, `median_shift` | closed-form maps fitted on the pooled training cells | same | none | n/a | `lc/baselines.py` |
| `shift` | **feature level, no cells**: training stim mean + (test Unstim − training Unstim mean), per quantile × marker | quantiles | none | n/a | in the scorers |
| `clip` (k15) | CellOT_paired, test cells clamped to the training Unstim [0.5, 99.5] percentiles | same | clip 0.5 | none | `cond_train --clip-test` |
| `dnorm`, `dnorm_clip` (k15) | CellOT_paired trained on donor-normalised cells (each donor minus her own Unstim median), offset added back | `cache_dnorm` | as paired | none | `lc/dnorm_cache.py` |
| `condot_ct` (k15) | CondOT PICNN (Bunne, Krause, Cuturi 2022), one map per celltype shared across stims, context = stim one-hot | pooled, unpaired | 50k it. (~8k per stim), CondOT defaults | none; **under-fit** (val err 0.042 vs 0.021 for CellOT) | `lc/condot_train.py` |
| `condot_global` (k15) | one PICNN for all 168 pairs, context = stim + celltype one-hot | same | 150k it. | none; **under-fit** (val err 0.09) | same |
| `fm` | flow matching (MFM paper's FM baseline): one global velocity field for all pairs, conditioned on stim + celltype one-hot | all 38 markers kept, flow on 13; source→target, independent coupling | MLP 7 × 512, Adam 1e-4, 60k steps, batch 8 pairs × 256 cells, best of 12 checkpoints on val | **none** (paper defaults, made for 43-d organoid data); val err ~0.08 vs ~0.035 for CellOT → **under-fit** | `mfm_heldout/mfm/train.py --arm fm` |

### 2.2 Maps that use the test donor's own Unstim profile beyond its cells

| method | what the map is conditioned on | how it enters | key hyperparameters | tuning effort | code |
|---|---|---|---|---|---|
| `gauss_ot_icpt` / `_knn` / `_cond` | 3 PCs of the donor's same-celltype 13 Unstim medians | closed form: training donors weighted by similarity (knn) / ridge correction of the target moments (cond) / intercept only (icpt) | ridge α, bandwidth | **some** (inner leave-one-donor-out on training donors) | `lc/cond_baselines.py` |
| `ctx_pc` | same 3 PCs | ICNN bias shift + positive gate, linear in the context (CondOT-style, convex in the cells) | no regularisation | none → memorises the training donors | `cond_train --ctx pc` |
| `ctx_pc_wd1` | same 3 PCs | same, L2 penalty λ = 1 on the context weights | λ = 1 | **some** (λ sweep: 6 values, 4 celltypes, repeat 0, on the `ft` context) | `--ctx pc --ctx-wd 1` |
| `ctx_pcblood` | own 13 medians + the 13 medians of the 4 dominant celltypes (Granulocytes, CD4Tnaive, cMCs, CD8Tnaive) + their 4 frequencies, each block weighted equally, → 3 PCs | same, λ = 1 | λ, n PCs = 3 | none of its own (λ from the sweep) | `--ctx pcblood` |
| `ctx_ft_immunefm` / `ctx_ft_scratch` | 3 PCs of the donor's mean Unstim embedding from a transformer fine-tuned on the atlas (masked markers, 64 quantile bins, 38 markers), started from ImmuneFM's weights / from random | same, λ = 1 | encoder: 6 layers, dim 128, 30k steps; map as above | encoder: one recipe; map: λ only | `fm/finetune.py`, `--ctx ft` |
| `ctx_cluster` | the **cell's** phenotype subcluster (k-means K = 4 on phenotypic markers stable under that stim), source and target paired within (donor, subcluster) | one-hot per cell | K = 4, stability 0.2 | none | `--ctx cluster` |
| `mfm` | Meta Flow Matching (Atanackovic et al. 2025): a GCN embeds the donor's Unstim population (38 markers), the velocity field is conditioned on it | learned end to end across all (donor, celltype) populations | GCN 2 × 128, kNN 100; flow as `fm` | **none** (paper defaults), under-fit | `--arm mfm` |
| `mfm_blood` | + GCN embeddings of the 4 dominant celltypes' Unstim populations | concatenated | as mfm | none | `--arm mfm_blood` |
| `mfm_lastiter` | `mfm` without checkpoint selection (first wave) | | | none | |

### 2.3 Ensembles (no fitting)

| method | definition |
|---|---|
| `ens` | 0.5 × `CellOT_paired` + 0.5 × `shift`, weight fixed in advance (no tuning, no leakage) |
| `ens_orig` | 0.5 × `CellOT` + 0.5 × `shift` |
""")

md(r"""
## 3. Results on common cases (repeats 1–4)

### 3.1 Learning curve — the main methods
""")

code(r"""
def piv(D, col):
    return D.pivot(index='method', columns='k', values=col)
fam = C.drop_duplicates('method').set_index('method').family
T = pd.concat({c: piv(C, c) for c in ('skill', 'dev_R2', 'between_R2')}, axis=1).round(3)
T.insert(0, 'family', fam)
T.sort_values(('skill', 15), ascending=False)
""")

code(r"""
# Skill vs k for the reference methods and the best of each family. Fixed categorical order (validated palette:
# adjacent CVD dE >= 9.1); direct labels at the line ends because three slots are below 3:1 contrast.
SHOW = [('ens', '#2a78d6', 'o'), ('shift', '#eb6834', 's'), ('CellOT_paired', '#1baf7a', 'D'),
        ('CellOT', '#eda100', '^'), ('ctx_pcblood', '#e87ba4', 'v')]
OTHER = ['mfm', 'fm']
fig, ax = plt.subplots(figsize=(8.5, 4.8))
ks = [3, 5, 8, 15]; S = piv(C, 'skill')
labels = sorted([(S.loc[m, 15], m, '#0b0b0b', 9) for m, _, _ in SHOW] +
                [(S.loc[m, 15], m, '#8a8880', 8) for m in OTHER], reverse=True)
ypos = []                                    # end-of-line labels, pushed apart to >= 0.0055 skill units
for y, m, col, fs in labels:
    y_ = min(y, ypos[-1] - 0.0055) if ypos else y; ypos.append(y_)
    ax.annotate(m, (15, y), xytext=(16.0, y_), textcoords='data', va='center', fontsize=fs, color=col,
                arrowprops=dict(arrowstyle='-', color='#c3c2b7', lw=0.6, shrinkA=0, shrinkB=3))
for m in OTHER:
    ax.plot(ks, S.loc[m, ks], color='#b8b6ad', lw=1.2, zorder=1)
for m, c, mk in SHOW:
    ax.plot(ks, S.loc[m, ks], color=c, lw=2, marker=mk, ms=7, mec='#fcfcfb', mew=1.5, label=m, zorder=3)
ax.set_xticks(ks); ax.set_xlabel('training donors (k)'); ax.set_ylabel('median skill on held-out donors')
ax.set_xlim(2.5, 19.5); ax.grid(axis='y', color='#e6e5e0', lw=0.8); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc='lower right', fontsize=9)
ax.set_title('Skill vs number of training donors (grey: other families)', loc='left', fontsize=11)
plt.tight_layout(); plt.show()
""")

md(r"""
### 3.2 k = 15: every method, including the k = 15-only arms
""")

code(r"""
T15 = C15.set_index('method')[['family', 'skill', 'dev_R2', 'between_R2']].sort_values('skill', ascending=False)
T15 = T15.drop(index='identity').round(3)
fig, axes = plt.subplots(1, 3, figsize=(12, 7.5), sharey=True)
order = T15.index[::-1]
for ax, col, ref in zip(axes, ['skill', 'dev_R2', 'between_R2'], [None, None, 0.679]):
    v = T15.loc[order, col].clip(lower=0 if col != 'skill' else 0.4)
    ax.hlines(range(len(order)), v.min() - 0.02, v, color='#e6e5e0', lw=1)
    ax.plot(v, range(len(order)), 'o', color='#2a78d6', ms=7, mec='#fcfcfb', mew=1.5)
    for rm, ls in (('CellOT_paired', '--'), ('shift', ':')):
        ax.axvline(T15.loc[rm, col], color='#52514e', lw=1, ls=ls)
    ax.set_title(col + ('  (clipped at 0)' if col != 'skill' else ''), loc='left', fontsize=10)
    ax.grid(axis='x', color='#e6e5e0', lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels(order, fontsize=9)
fig.suptitle('k = 15, held-out donors (dashed: CellOT_paired, dotted: shift)', x=0.01, ha='left', fontsize=11)
plt.tight_layout(); plt.show()
T15
""")

md(r"""
### 3.3 Paired tests against the references (k = 15, and k = 3)
% of cases where the row method is closer to the truth than the reference; 50 = tie.
""")

code(r"""
def ptab(Pd, k):
    x = Pd[Pd.k == k].pivot(index='a', columns='b', values='pct_a_closer').round(1)
    p = Pd[Pd.k == k].pivot(index='a', columns='b', values='p_wilcoxon')
    return x.astype(str) + p.map(lambda v: '' if pd.isna(v) else (' ***' if v < 1e-3 else ' *' if v < 0.05 else ''))
print('k = 15'); display(ptab(P15, 15))
print('k = 3');  display(ptab(P, 3))
""")

md(r"""
### 3.4 By stim (k = 15, median skill)
""")

code(r"""
B = pd.read_csv(Z / 'k15_by_stim.csv', index_col=0).drop(index='identity')
B.loc[B.mean(1).sort_values(ascending=False).index].round(3)
""")

md(r"""
**Reading §3.**
- **Best overall: `ens`** (CellOT_paired + shift, fixed 50/50): first on skill, dev_R2 and between_R2 at every k,
  closer than CellOT_paired in 55–59 % of cases and than shift in 51–54 % (p < 0.02 at k = 3, < 1e-4 above).
- **Only one donor-informed map beats the context-free paired CellOT, and only at k = 15: `ctx_pcblood`**
  (0.628 vs 0.620, closer in 51.2 % of cases, p 8e-4; between_R2 0.695, tied with `ctx_cluster` as the best single
  model, above shift 0.679). Every donor context costs skill at k ≤ 8.
- **Preprocessing fixes (k15):** clipping = paired (50.0 %); donor normalisation is worse (47.5 %, p < 1e-4): it
  loses the donor-specific signal.
- **Methods that share one network across stims/celltypes (CondOT, FM/MFM) are worse**, and all of them are
  documented as under-fit (validation error 2–4 × CellOT's): these verdicts are about *this* training, not the
  method.
""")

md(r"""
## 4. Side studies (representations and closed forms, not maps)

### 4.1 ImmuneFM frozen (phase 0): can the published model read the atlas?
""")

code(r"""
print((W / 'fm/sanity/summary.txt').read_text())
""")

md(r"""
### 4.2 Fine-tuning a transformer on the atlas: from ImmuneFM's weights vs from scratch
Masked-marker accuracy / cross-entropy on held-out cells (val side of training donors, and the 4 test donors),
64 quantile tokens; marginal = per-marker token frequencies.
""")

code(r"""
rows = []
for r in range(5):
    for init in ('pretrained', 'random'):
        L = pd.read_csv(W / f'fm/finetune/repeat_{r}/{init}/log.csv')
        last = L[(L.arm == init) & (L.step == L.step.max())].set_index('split')
        mg = L[L.arm == 'marginal'].set_index('split')
        rows.append({'repeat': r, 'init': init, 'test_acc': last.loc['test', 'acc'], 'test_ce': last.loc['test', 'ce'],
                     'marginal_acc': mg.loc['test', 'acc'], 'marginal_ce': mg.loc['test', 'ce']})
FT = pd.DataFrame(rows); FT.groupby('init')[['test_acc', 'test_ce', 'marginal_acc', 'marginal_ce']].mean().round(3)
""")

md(r"""
### 4.3 Donor covariates for the residual beyond the shift (closed form, ridge)
Residual r = (S − mean S_train) − (U − mean U_train) per feature; res_R2 = share of r predicted (0 = shift).
Frozen embeddings: lodo (19 folds) and lc15; fine-tuned: lc15 fold of their own repeat only.
""")

code(r"""
S1 = pd.read_csv(W / 'fm/residual_predict/summary.csv').pivot(index='method', columns='proto', values='res_R2')
ft = []
for r in range(5):
    for init in ('pretrained', 'random'):
        x = pd.read_pickle(W / f'fm/residual_predict_ft/r{r}_{init}/cases.pkl')
        x = x[x.method.str.startswith('fm_')].assign(method=lambda d: d.method + '_' + ('immunefm' if init == 'pretrained' else 'scratch'))
        ft.append(x)
F = pd.concat(ft).groupby('method').apply(lambda g: 1 - g.err2.sum() / g.r2.sum(), include_groups=False).rename('lc15')
pd.concat([S1, F.to_frame()]).sort_values('lc15', ascending=False).round(3)
""")

md(r"""
### 4.4 Phenotype stratification in closed form (one gauss_ot / median shift per phenotype subcluster, K = 4)
""")

code(r"""
for t in ('raw_K4', 'raw_K4_stable'):
    print((W / f'fm/stratified/{t}/summary.txt').read_text().split('by stim')[0])
""")

md(r"""
### 4.5 Flow-matching training: validation error over steps (checkpoint selection runs)
CellOT's val median error on the same kind of cases is ~0.02–0.035; a flow stuck near 0.08 is under-fit.
""")

code(r"""
R = Path('/scratch/users/bmerwan/mfm_heldout/work/runs'); rows = []
for arm in ('fm_sel', 'mfm_sel', 'mfm_blood_sel'):
    for f in sorted(R.glob(f'{arm}/repeat_*/k*/log.csv')):
        L = pd.read_csv(f)
        if 'val_err' not in L: continue
        v = L.dropna(subset=['val_err'])
        rows.append({'arm': arm, 'run': f.parent.parent.name + '/' + f.parent.name, 'best': v.val_err.min(),
                     'best_step': int(v.loc[v.val_err.idxmin(), 'step']), 'last': v.val_err.iloc[-1],
                     'spread': v.val_err.max() - v.val_err.min()})
V = pd.DataFrame(rows); V.groupby('arm')[['best', 'last', 'spread', 'best_step']].median().round(3)
""")

md(r"""
## 5. What is under-tested, and a fair way to finish the comparison

| method | why its verdict is not final | what a fair run would change |
|---|---|---|
| `fm` / `mfm` / `mfm_blood` | paper hyperparameters for other data; validation error ~0.08 (2–4 × CellOT); best checkpoints late and noisy | learning rate, network width / depth, steps (×3–5), per-marker standardisation, EMA of weights, OT (minibatch) coupling, smaller kNN for small populations |
| `condot_ct` / `condot_global` | under-fit (val 0.042 / 0.09); ~8k it. per stim vs 20–50k for a CellOT map | matched budget (~300k it. per celltype), identity initialisation |
| `ctx_pcblood` (the one positive) | λ and n PCs not tuned for it; block weights fixed | λ per k, n PCs 2–8, context as an extra input instead of bias/gate, more dominant celltypes |
| `ctx_ft_*` | one encoder recipe; encoder never probed after fine-tuning | probes on fine-tuned embeddings; embedding PCs vs raw PCs at equal λ |
| `ctx_cluster` | K = 4 only; pSTAT4 / pPLCg counted as phenotypic | K 2/8, soft (distance-to-centroid) context, phenotype-only markers |
| `ens` | weight fixed at 0.5 | weight per stim or per k chosen by inner cross-validation; ensembles with `ctx_pcblood` |

**Fair protocol.** Same budget for each candidate (e.g. 8–12 configurations), each configuration chosen by inner
validation on the **training donors only** (hold out 3 of the 15 training donors, train on 12, select; then retrain
on 15), test donors never used for any choice, then the winners compared head-to-head on §3's common cases.
Priority by expected value: (1) ensemble weights, (2) `ctx_pcblood`, (3) FM (if FM cannot reach CellOT's fit, MFM
cannot either), (4) CondOT at matched budget.

## 6. Files
- Scoring: `lc/model_zoo.py` → `work/lc/model_zoo/{lc,k15}_{curve,paired,by_stim}.csv`, `cases.pkl`.
- This notebook: `lc/build_model_zoo_notebook.py`.
- Run logs and dates: `work/lc/STATUS.md`, `/scratch/users/bmerwan/mfm_heldout/work/STATUS.md`.
- Related notebooks: `where_cellot_skill.ipynb` (where CellOT works, dnorm / clip, CondOT), `dmso_plate.ipynb`
  (plate / well noise: how much of a donor's apparent response is real), `learning_curve.ipynb`.
""")

nb = nbf.v4.new_notebook(); nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parents[1] / "notebooks/model_zoo.ipynb"
nbf.write(nb, out); print("->", out)
