"""Generate notebooks/summary_all_results.ipynb -- one place for every result on CellOT vs the alternatives.

HELD-OUT ONLY (user decision 2026-10-06): a map is scored only on donors whose cells it never saw. In-sample scores (the
paper's 80/20 split of cells of the same 19 donors, CellOT checkpoints scored on their training donors, per-patient
scores on the drug-screen patients the maps were trained on) are removed; the section "What was removed and why" lists
them. The previous in-sample version is archived in notebooks/archive/*_insample_2026-10-06.*.

Part A: modelling the perturbation on the atlas, held-out donors only (atlas_heldout), plus data facts (not model scores).
Part B: predicting labour (DOS) on the OOL cohort (women held out by the CV).

Same convention as the other builders: every table is recomputed from the source CSVs under
/scratch/users/bmerwan/{simco_work, atlas_heldout/work, ool_longitudinal/work} (and read-only group-home score files);
numbers that exist only in CLAUDE.md / STATUS files / other notebooks are cited as such. Nothing is retrained.
    python notebooks/build_summary_all_results.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/summary_all_results.ipynb
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(src):
    cells.append(nbf.v4.new_code_cell(src.strip()))


# ---- prose: executive summary and the 'Reading' under each table (numbers checked against the executed outputs) ----
TEXT = {}
TEXT["title"] = r"""
# SIMCO — every result on CellOT vs the alternatives, held-out only (2026-10-06)

**Rule for this notebook (user decision, 2026-10-06): no in-sample comparison.** A map is scored only on donors whose
cells it never saw, and a DOS model only on women outside its training fold. In-sample scores reward memorisation and
have been removed; they are listed, with what changed, in **"What was removed and why"** near the end. The previous
version is archived as `notebooks/archive/summary_all_results_insample_2026-10-06.ipynb`.

Pure aggregation: each table is recomputed from the CSVs written by earlier runs (`simco_work/`, `atlas_heldout/work/`,
`ool_longitudinal/work/`, and read-only score files in the lab's group home). Nothing is retrained and Stabl is not
rerun. Numbers that exist only in `CLAUDE.md`, a STATUS file or another notebook are labelled **cited**. Builder:
`notebooks/build_summary_all_results.py`.

**Conventions.**
- *Skill* = 1 − error / error of "no effect" (identity): 1 = perfect, 0 = no better than assuming nothing happens.
- *Aligned* = DOS second stage refitted with the row-order bug fixed. The lab's rescore script put the same woman in
  training and test for 94 % of test samples, so its numbers are partly in-sample; they appear only to document the bug.
- *Selected on the test splits*: `params_43` and every "best of 97 configs" were chosen on the same 100 CV splits they
  are reported on. They are flagged wherever they appear.
- 20-split and 100-split numbers are never in the same comparison column.
"""

TEXT["exec"] = r"""
## Executive summary

**Part A — modelling the perturbation on new donors (atlas, 19 donors; 4 held out per repeat, 5 repeats)**

1. **Stim maps on new donors: CellOT ties a closed-form Gaussian map.** With 15 training donors, median skill over
   q10–q90 is CellOT 0.600, `gauss_ot` 0.595, `quantile_map` 0.538 and `median_shift` 0.512. The split-half ceiling
   is 0.87, so about a third of the achievable skill is missing for every map. Three training donors already give
   90 % of the 15-donor skill.
2. **Pregnant test donors need pregnant training donors.** HV-trained maps lose 0.04–0.08 of skill on PV donors.
   Typical donors are the good training donors and the easy test donors.
3. **A patient covariate inside the map does not help.** No donor context (Unstim PCs, a fine-tuned cytometry
   foundation model, phenotype clusters) beats context-free CellOT on new donors (all ≈ 0.62 at k = 15, worse at
   k ≤ 8). A **feature shift with no network** scores 0.615 and tracks between-donor deviations best (dev R² 0.75).
   Only 5–8 % of what remains after that shift can be predicted from Unstim.
4. **Drug maps on new donors recover only 11–19 % of the drug shift**, and **no map predicts how one patient's drug
   response differs from another's**: on donor groups with real differences the donor-specific part captured is
   CellOT −1 %, `quantile_map` −7 %, `gauss_ot` +1 %, paper chain −4 % (0 = giving everyone the same effect). So
   per-patient accuracy is accuracy on the *shared* (average) part, where `quantile_map` beats CellOT: 0.36 vs 0.32,
   and **0.64 vs 0.47 on the features the DOS model reads**.
5. **Which drug map, on new donors.** Drug map alone: on medians `quantile_map` beats CellOT in **20 of 20**
   donor-repeats; over all quantiles they **tie** (12 of 20); on effects > 0.1 CellOT wins 20 of 20 but by 2 % of
   error; CellOT wins the low tails (q10/q25). Whole pipeline: **CellOT stim → `quantile_map` beats the paper's chain
   in 19 of 20 donors on the DOS features**, and the paper's chain loses even to CellOT stim → constant shift in 20 of
   20 donors on all real cases.
6. **Data facts, not model comparisons.** Drugs move cells ~48× less than stims, and 72–76 % of drug pairs move less
   than two identical control wells differ. IL33 is below its own plate-to-plate floor. Under PI the DMSO1 control
   wells are an outlier against every drug and every other control. OOL Unstim cells sit 2–5 atlas donor-SDs below
   the atlas on several phospho-markers, so every simulated DOS feature is a map evaluated off its training range.

**Part B — predicting labour (DOS) in the OOL cohort (177 visits, 63 women; never in the atlas; CV holds out women)**

7. **Fig 5b reproduces** (R² 0.415, RMSE 24.1 d, 21 features) through the lab's rescore script; aligned it is
   **0.418**. Its XGBoost config `params_43` was **selected on the same 100 splits**, by preterm AUROC (rank 1 of 95
   in the lab's grid search), not by RMSE (5th of 97 in the lab script, 43rd aligned).
8. **On the aligned protocol (100 splits), CellOT leads every single-source alternative by a modest margin:**
   CellOT 0.418 (24.1 d), `gauss_ot` 0.399, measured with quantiles 0.397, measured medians + frequencies 0.384,
   wrong-stim/celltype maps 0.343, untrained ICNN 0.329 (per split p 0.03 to < 1e-4). That is 0.4–1.8 d of RMSE.
   **CellOT + measured features is the one improvement that holds:** 0.449 (23.4 d), +0.041 per split, p 0.014. At 20
   splits none of these are separable.
9. **Simulated quantiles: two analyses disagree** (no gain in simco at 100 splits, +0.04–0.05 in atlas_heldout's
   rerun with retrained maps and the two mishandled visits removed before Stabl). Unexplained; not pairable.
10. **Gestational age alone beats every immune model**: EGA R² **0.878** (11.0 d) vs 0.45 (23.4 d). Beyond EGA the
    aligned Fig 5b prediction has partial r **0.17 (p 0.20)**. On the gestational-age-free target every feature set
    has R² < 0, and one prediction per woman does not beat the constant 39.0 weeks.
11. **The preterm AUROC (0.73) is mostly sampling time** (a no-information DOS model already gives 0.71). The G1
    "early lead" is **0.745, p 0.07** on aligned predictions (0.814, p 0.017 on the bug-affected ones): not
    significant, 5 preterm women.
12. **Drug effects on DOS:** THF (7–11 d later labour, ≥ 92 % of folds) and SALPZ (3–6 d) hold in all five drug-map
    chains; MAP's effect exists only with CellOT drug maps. The paper's chain makes 48–81 % of treated women
    implausible for XGBoost; `quantile_map` drug maps 6–10 % (LPZ/SA 20–38 %).
13. **Cost:** CellOT training took **12,557 CPU-h**; the closed-form drug maps take 12–15 core-h (~1000× less).

**Bottom line.** On donors it never saw, CellOT is as good as a closed-form Gaussian stim map and no better than a
quantile map as a drug map — on the DOS features it is worse, and no map captures patient-specific drug response.
On DOS, CellOT's advantage over the alternatives is real but about 1 day of RMSE, against the 13 days that gestational
age alone gains over CellOT. THF and SALPZ do not depend on CellOT; MAP does.
"""

TEXT["partA"] = r"""
# Part A — modelling the perturbation (atlas), held-out donors only

Methods compared with CellOT:
- **floor:** `identity` ("nothing happens");
- **diagonal:** `median_shift` (x + d), `quantile_map` (1-D OT per marker);
- **marker-mixing:** `gauss_ot` (affine Bures–Wasserstein map), plus conditional variants (`_cond`, `_icpt`, `_knn`);
- **conditional CellOT** with donor contexts, and a no-network **feature shift**.

Protocol (atlas_heldout): 5 repeats × 4 test donors; maps trained on k = 1…15 of the other donors (CellOT retrained on
GPU, exactness-checked against the paper's `model.pt`). The CellOT checkpoint is either the last iterate or the
"paper-style best" chosen on held-out cells of the *training* donors, so it never sees a test donor. Truth = each test
donor's own stimulated q10–q90; cases count when the effect exceeds 0.0273.

A0 comes first because it is **not a model comparison**: it measures the data (effect sizes, noise floors, plate
artefacts) and conditions how every model result should be read.
"""

TEXT["A0"] = r"""
## A0. Data facts (measurements, not a model comparison)

Everything in this section uses only the identity map, i.e. it compares measured cells with measured cells. No fitted
map is scored, so the in-sample rule does not apply. The effect sizes are the identity rows of the pooled benchmark
files (observed stim/drug vs control on the test-side cells); the floors compare control wells with each other.
"""

TEXT["A0_r"] = r"""
**Reading — data, not models.**
- **Effect sizes.** Drugs move cells ~**48×** less than stims (median identity MMD 0.00081 vs 0.039).
- **Noise floors.** Sampling floor 0.00061, technical floor (DMSO2 vs DMSO3) 0.00130.
  - **IL33 is 0.66× its own technical floor**: two identical wells differ more than IL33-stimulated cells differ from
    unstimulated ones. TNFa is 3× (marginal); every other stim 21–57×.
  - **76 % of drug pairs move less than their own plate-to-plate floor** (this aggregation; `CLAUDE.md` quotes 72.2 %
    from `baselines_results.ipynb` §10.7 — same conclusion).
- **DMSO1 plate artefact.** Under PI every drug sits far from DMSO1 (raw MMD 0.04–0.09) and close to DMSO2/DMSO3
  (0.001–0.011), including CHT, which is referenced to DMSO2. So under PI a drug's apparent effect is set by the
  control it was assigned to; LPZ/SA under PI (the two DMSO1 drugs) are excluded from the held-out drug headlines in A4.
"""

TEXT["A1_r"] = r"""
**Reading.** At k = 15 training donors:
- CellOT 0.600 (last iterate) / 0.596 (paper-style checkpoint) and `gauss_ot` 0.595 are **tied at every k**;
- `quantile_map` 0.538 and `median_shift` 0.512 trail;
- the split-half ceiling is 0.872, so about a third of the achievable skill is still missing for every map.

The curve plateaus early: 90 % of the 15-donor skill at k ≈ 3 (2–5), 95 % at k ≈ 5. The uncapped-cell, 20k-iteration
run reproduces the table to within 0.004. Note: the ceiling applies to this per-donor truth only.
"""

TEXT["A2_r"] = r"""
**Reading.**
- **PV test donors need PV training donors.** HV-trained maps lose 0.04–0.08 of skill on PV donors (p ≤ 0.03 for every
  method at k = 6). PV-trained maps also lose on HV donors at k = 6 (CellOT −0.038, p 0.04). A mix of both is safest.
- **Good training donors and easy test donors are typical ones.** In the pairwise design CellOT's skill correlates
  ρ = −0.83 with the baseline distance between the two donors. The test donor explains 34–41 % of the variance and the
  specific pair 36–50 %.
- **In small celltypes more donors beat more cells.** Doubling donors gains CellOT +0.29 of skill, against +0.19 for
  doubling cells per donor.
"""

TEXT["A3_r"] = r"""
**Reading** (clean repeats 1–4, k = 15 unless stated; all on held-out donors).

**Closed form (5 repeats).** A covariate-conditioned `gauss_ot` gains a little (cond 0.611 vs pool 0.595), mostly
through a donor-independent intercept (`icpt` 0.604). The donor-specific part is +0.006.

**Between donors** (predictions and truth centred across each repeat's 4 test donors). The identity map — each donor
keeps her Unstim difference — already reproduces 75 % of the true between-donor variance. CellOT adds about 1 point
(0.765); the `gauss_ot` variants add nothing; `quantile_map` makes it worse (0.67).

**Conditional CellOT.** No donor context beats context-free paired CellOT (`none`, 0.621): regularised PCs 0.623,
ImmuneFM fine-tuned 0.622, same encoder from random init 0.619, phenotype clusters 0.620. Each is 50.0–50.7 % "closer"
than `none` at k = 15 and worse at k ≤ 8. Unregularised PCs memorise the training donors (0.20 at k = 3). The
**no-network feature shift** has the best dev R² (0.749); `none` is closer than it on only 49.3 % of cases.

**Residual beyond the shift.** Only 5–8 % of it is predictable from Unstim (whole-blood ridge 0.052 at k = 15 / 0.082
leave-one-donor-out; shuffled null −0.06). The donor-ID "oracle" that used to sit here was scored on its training
donors and is removed (see "What was removed").

**Cited, not recomputed** (`atlas_heldout/work/lc/STATUS.md`): a fine-tuned ImmuneFM covariate reaches res_R² 0.11–0.13
but does not help inside the map.
"""

TEXT["A4"] = r"""
## A4. Drug maps on held-out donors (replaces the old in-sample drug-map choice)

Two sources, both held-out:
- `atlas_heldout/work/heldout/drugs/analysis/overall.csv` (recomputed here): all 353k real cases, every step / chain /
  feature-level pipeline.
- **`drug_map_choice.ipynb`, rebuilt 2026-10-06 on held-out donors only** (cited, not recomputed: it reads 2.3M scored
  cases). Retrained CellOT drug maps (paper architecture, last iterate) on 20 donor-repeats (14 distinct donors).
  LPZ/SA under PI (the DMSO1 artefact, A0) are excluded from its headline numbers. Its answer section is
  `notebooks/drug_map_choice_readings.md`.
"""

TEXT["A4_r1"] = r"""
**Reading.** Every drug pipeline recovers only **11–19 %** of the drug shift on new donors (all real cases). Overall,
the drug map alone: CellOT 0.19 ≈ `gauss_ot` 0.19 > `quantile_map` 0.17 > `median_shift` 0.11. On the q50 features the
DOS model reads (2056 cases) the order flips: `quantile_map` 0.46, `median_shift` 0.45, `gauss_ot` 0.43, CellOT 0.35,
paper chain 0.34. The best-predicted block is LPZ/SA under PI (A0's plate artefact), because it is shared by all donors.
"""

TEXT["A4_r2"] = r"""
**Reading — the drug map alone** (applied to each donor's real control cells; cited from `drug_map_choice.ipynb` §3).
Quote the donor counts (x of 20), not the case-level p-values.
- **Medians: `quantile_map`.** Smaller median error in **20 of 20** donor-repeats (0.030 vs 0.035, 14 % smaller); DOS
  features 18 of 20 (0.030 vs 0.036). `median_shift` and `gauss_ot` also beat CellOT on medians in 16–20 of 20.
- **All quantiles: a tie.** 12 of 20 donors, p 0.25 (median error 0.0519 vs 0.0517).
- **Effects > 0.1: CellOT, narrowly.** All 20 donors, but by 2 % of error (0.132 vs 0.135); it ties `gauss_ot` there.
- **Where CellOT wins: the low tails.** `quantile_map` is closer in only 44 % of q10 and 48.5 % of q25 cases, in every
  stim and drug (pSTAT5 in lymphocytes, CD25, CD56hi NK; Unstim, TNFa, RIF, SA).
"""

TEXT["A4_r3"] = r"""
**Reading — the whole pipeline**, as an OOL woman goes through it: Unstim → stim map → drug map (cited, §4).
- **CellOT stim → `quantile_map` on features beats the paper's chain** (CellOT stim → CellOT drug on cells) in **19 of
  20** donors on the DOS features (error 0.031 vs 0.036; skill 0.44 vs 0.34), 20 of 20 on all real cases and on
  medians, 18 of 20 on effects > 0.1 (tiny margin).
- **The paper's chain loses even to a constant shift** (CellOT stim → `median_shift`): 20 of 20 donors on all real
  cases, 16 of 20 on the DOS features. CellOT's large-effect edge from the drug map alone does not survive the chain.
- **On the DOS features `quantile_map` = `median_shift`** (11 of 20 donors, p 0.96).
"""

TEXT["A4_r4"] = r"""
**Reading — per patient: shared part vs patient-specific part** (cited, §5; donor groups whose between-donor
differences are real, 34 % of groups — across all groups 84 % of the between-donor spread is sampling noise).
- **Patient-specific part: no map captures it.** CellOT −1.2 %, `quantile_map` −7.4 %, `gauss_ot` +1.2 %, paper chain
  −3.9 % of the real donor-to-donor differences (0 = same effect for everyone). On the DOS features 0.8–6.0 % for every
  map, none distinguishable from a constant. CellOT's donor variation points weakly the right way (r 0.17) but is too
  small and noisy to reduce error; `quantile_map`'s adds error, mostly in the low tail.
- **Shared part (the average effect): `quantile_map` 0.36 vs CellOT 0.32**; on the DOS features **0.64 vs 0.47**
  (`median_shift` 0.61, `gauss_ot` 0.62). Pipelines: CellOT → `quantile_map` 0.36 / 0.65 vs paper chain 0.30 / 0.47.
- So per patient, accuracy *is* accuracy on the shared part: on the DOS features it is 46 % of CellOT's per-patient
  error against 32 % for `quantile_map`.
"""

TEXT["A5_r"] = r"""
**Reading — data fact.** OOL Unstim cells are systematically lower than the atlas Unstim cells the maps were trained
on: pS6 −1.34, pNFkB −1.30, pMK2 −0.83, pCREB −0.74 arcsinh (2–5 atlas donor-SDs); pSTAT6 is higher (+0.45). Every
simulated DOS feature, the paper's included, is a map evaluated outside its training range. With the retrained maps
113 of 12,740 simulated OOL columns take impossible values, mostly low tails in CD56hi NK cells. The likely cause is
protocol (incubation of the atlas Unstim wells); not pregnancy (the gap is nearly the same against the atlas PV donors,
cited from `atlas_vs_ool.ipynb`). To confirm with the lab.
"""

TEXT["fig_atlas"] = r"""
**Reading the figure** (held-out donors only).
- **Left:** stim maps, median skill on 4 new donors against the number of training donors (mean of 5 repeats). CellOT
  and `gauss_ot` overlap at every k; the diagonal maps sit below; all plateau far under the split-half ceiling.
- **Right:** drug map alone on new donors (cited, `drug_map_choice.ipynb` §3, LPZ/SA under PI excluded), median skill
  per subset of real cases. CellOT is ahead on all real cases and on effects > 0.1 in skill terms; `quantile_map`,
  `median_shift` and `gauss_ot` are ahead on the medians and on the DOS features.
  In absolute error per donor the all-real comparison is a tie (12 of 20) — skill weighs the small low-tail effects,
  where CellOT is relatively strongest.
"""

TEXT["partB"] = r"""
# Part B — predicting labour (DOS) on the OOL cohort

DOS = days from blood draw to labour onset (negative). 177 samples / 63 women (175 without the two mishandled visits
36_G1 and 52_G1, whose "Unstim" tubes were stimulated). The paper's protocol: `GroupShuffleSplit(100, test_size 0.2,
random_state 42)` by woman; Stabl ALasso selection; median-impute + scale; XGBoost `params_43`; one pooled R² over the
per-sample median of out-of-fold predictions.

**Why this part stays.** The OOL women are never in the atlas, so the maps never saw them; Stabl and XGBoost are fitted
on training women and scored on held-out women. Two in-sample elements remain and are flagged where they appear:
1. the lab's rescore script (row-order bug: the same woman in training and test for 94 % of test samples) — only aligned
   numbers are used for conclusions;
2. configuration selection on the reported splits: `params_43` (picked by preterm AUROC on these splits) and every
   "best of 97" column (picked by R² on these splits). Read `params_43` and "median config" columns; treat "best of 97"
   as an optimistic bound.
"""

TEXT["B1"] = r"""
**Reading.** The Fig 5b artifact (`GS_cellot_longallstims_foldinfo`) is a **rescore** of the Stabl selections of the
0.319 run: its 100 per-fold feature lists are identical to that run's (`stabl_results.ipynb` §2), only the XGBoost config
changed. Rescoring with `params_43` reproduces 0.415 / 24.120 / 18.947 exactly.

Two qualifications, both in-sample elements:
1. **The rescore script carries the row-order bug** (B2). For CellOT it happens not to matter: aligned, 0.418.
2. **`params_43` was selected on the same 100 splits** — by preterm AUROC (rank 1 of 95 in the lab's drug grid search,
   AUROC 0.729), not by RMSE. By RMSE on the Fig 5b selections it ranks 5th of 97 in the lab script (0.415 vs best
   0.427) and 43rd of 97 aligned (0.418 vs 0.441, about the median config). So its R² optimism is small, but it is not
   a pre-specified configuration.
"""

TEXT["B2"] = r"""
**Reading.** The lab's rescore script reads `y` in outcome-table order but `groups` in feature-table order, so 94 % of
test samples have the same woman in training: those numbers are partly **in-sample** and are not used anywhere below.
The table only sizes the bug: CellOT is unchanged (0.415 → 0.418, its selections came from a correct Stabl stage);
single feature sets are inflated by +0.014 to +0.041; **the combined sets by +0.080** (lab 0.529 / 0.521 vs aligned
0.449 / 0.441).
"""

TEXT["B3"] = r"""
**Reading** (aligned, 100 identical splits, all 177 samples; paired tests per split at `params_43`).

| comparison | result |
|---|---|
| CellOT alone | 0.418 (RMSE 24.1 d) |
| `gauss_ot` | 0.399, −0.082 per split, p 0.002 |
| measured features, with quantiles | 0.397, p 0.03 |
| measured features, medians + frequencies | 0.384, p 0.006 |
| wrong-stim/celltype CellOT maps | 0.343, p < 1e-4 |
| untrained ICNN | 0.329, p < 1e-4 |
| CellOT + measured | **0.449 (23.4 d), +0.041, p 0.014** |
| `gauss_ot` + measured | 0.441, p 0.25 vs CellOT |
| CellOT + measured + simulated quantiles | 0.452, ≈ CellOT + measured |

- Every single-source alternative trails CellOT; the **median-config** column (least affected by selection) gives the
  same order (CellOT 0.416 vs 0.33–0.38).
- **Combining CellOT with measured features is the only improvement that survives.** The best-vs-best comparisons
  (+0.045, p 0.005) agree, but they select the config on these test splits — optimistic, not used for conclusions.
- Per-split medians are much lower than pooled R² (CellOT 0.31 vs 0.42), because the pooled R² medians each sample over
  about 20 test splits.
- In days, the spread from the best immune set to the worst null is 23.4 → 25.8 d. Gestational age alone is 11.0 d.
"""

TEXT["B3b"] = r"""
**Reading.** Dropping 36_G1 and 52_G1 from training and scoring:
- CellOT barely moves (0.418 → 0.415);
- the no-CellOT arms lose 0.02–0.04: `gauss_ot` 0.364, measured 0.367–0.372;
- CellOT's lead therefore grows: `gauss_ot` −0.091, p 0.001;
- the combined sets hold: CellOT + measured 0.448, +0.043, p 0.003.

The column "outliers kept in training" shows that the two visits mostly *helped* the other arms. The Stabl selection
itself still saw them; only atlas_heldout's quantile arms (B3d) removed them before Stabl.

Stabl model (ALasso / Lasso / ElasticNet): every arm stays within ~0.03 across the three, no consistent winner. Lasso is
slightly best for the combined sets (0.455–0.457). Not tested pairwise.
"""

TEXT["B3c"] = r"""
**Reading — 20 splits only, not comparable with 100-split columns.**

*Aligned rescore on the first 20 splits* (a strict prefix of the 100). CellOT scores 0.410 and no arm differs from it
significantly. Simulated quantiles: −0.014 (p 0.57); at 100 splits without 36_G1/52_G1 still −0.006 (p 0.50). The
100-split separation in B3 needs all 100 splits.

*Stabl's own pipeline at 20 splits* (default XGBoost, no config selection, no row-order bug — the cleanest numbers
here, but low-powered):
- `gauss_ot` matches CellOT on the `both` table (0.267–0.287 vs 0.267–0.298; paired p 0.73–0.87);
- the valid wrong-perturbation controls `ctswap` / `crossswap` match it too (0.24–0.32); the only significant paired
  result is a crossswap seed *beating* CellOT (p 0.04);
- an untrained scale-matched ICNN (`rscaled`) reaches ~0.21, ~80 % of the way from measured Unstim (−0.17) to CellOT;
- a huge random displacement (`rraw`) destroys the signal.

`permuted` is **retracted** as a control (356 of its 852 columns are copies of real columns). Measured distributional
features with no map reach 0.27 (`d_freqmed`, `d_qtl`) against 0.287 for CellOT (paired p 0.14–0.20).
"""

TEXT["B3d"] = r"""
**Reading — a separate experiment; do not pair it with B3.** 175 rows (so different GroupShuffleSplit draws), retrained
CellOT maps (all 19 donors, all 196 pairs), and an any-quantile inclusion rule (1085 features, no IL33 exception).

`ref852` (the paper's 852 columns + measured) scores 0.450, consistent with `comb_cellot` excl at 0.448. Adding simulated
quantiles gives **+0.043 to +0.052 per split at `params_43`** (`rule_rawq`, `rule_selq`, `rule_corr`, `selq_corr`;
65–72 / 100 wins, p < 1e-4), up to 0.493 (`rule_corr`). The "best of sweep" values (up to 0.524) are selected on these
splits and not used. Spreads instead of raw quantiles (`rule_shape`) give nothing.

**This contradicts simco's 100-split simq result** (B3c: −0.006). Unexplained; candidates: the two visits removed before
Stabl here, retrained maps, the new inclusion rule. Not pairable (different splits).
"""

TEXT["fig_dos"] = r"""
**Reading the figure.** Left: pooled R² at the paper's config `params_43` (filled) and at the best of 97 configs
(open — chosen on these same splits, so an optimistic bound), with lines for CellOT and gestational age alone. Right:
paired per-split difference from CellOT at `params_43`, 95 % bootstrap CI over the 100 splits.
- The two combined sets with CellOT are the only arms whose CI lies entirely above 0.
- Every single-source alternative's CI lies below 0.
- The whole range of immune feature sets (0.33–0.45) sits far from gestational age alone (0.88).
"""

TEXT["B4"] = r"""
**Reading.** On the paper's 100 splits, EGA with one linear coefficient gives R² **0.878** (RMSE 11.0 d), against
0.418 / 24.1 d for CellOT. Age, BMI, gravida and para add nothing. The lab's runs with EGA as an input (0.890 / 0.869)
go through the bug-affected rescore script, so they are upper bounds; even so they add ≤ 0.012 over EGA alone.

Beyond EGA, the aligned out-of-fold Fig 5b prediction correlates 0.64 with EGA and has a **partial r of 0.17** with DOS
given EGA; its woman-clustered p in `DOS ~ EGA + pred` is **0.20**. The bug-affected predictions gave 0.20 / p 0.064
(the figures `CLAUDE.md` quoted until today). The in-sample R² of that OLS (0.884 → 0.887) is no longer shown.
Either way, the immune model adds at most a marginal, non-significant amount beyond gestational age.
"""

TEXT["B5"] = r"""
**Reading.** `DOS_offset` = days relative to the due date. On it, EGA and clinical covariates give R² ≈ 0 (RMSE ~11 d),
and **every immune feature set is below 0** (20 splits): CellOT −0.03 to −0.11, `gauss_ot` −0.14 to −0.21, measured
−0.26 to −0.46. Nothing here predicts when labour starts relative to the due date. The R² 0.42 of Fig 5b is a noisy
gestational-age clock.
"""

TEXT["B6"] = r"""
**Reading.** Preterm cases are 14 samples from **5 women**, all late-preterm.

The paper scores preterm risk as 7·EGA − predicted DOS, so EGA enters after the model:
- the due-date rule gives 0.5;
- a DOS model with no information gives **0.706**, purely because preterm women are sampled earlier;
- the aligned CellOT predictions give 0.722, and **0.623 once their leftover EGA trend is removed** — below the
  no-information score.

At G1, where sampling times are matched: **aligned 0.745 (p 0.070), 0.707 (p 0.133) with the trend removed.** The
bug-affected rows (0.814, p 0.019; 0.797, p 0.029) are shown only because `CLAUDE.md` quoted them; they are not used.
Treat the G1 lead as unestablished: a post hoc subgroup of 5 women, not significant on correct predictions.
"""

TEXT["B7"] = r"""
**Reading.** Target: gestational age at delivery, one value per woman, leave-one-woman-out, 63 women. Nothing beats
the constant 39.0 weeks (RMSE 10.95 d for 51 women with three visits, 11.38 d for 59 with two), with measured features
(`freqmed`), the paper's 852 simulated features (`sim`), or both. The ElasticNet keeps 0–1 features per fold, so every
model predicts the training mean. Per `ool_longitudinal/README.md`, trajectories and leak-free landmarks are the same or
worse. Consistent with B5.
"""

TEXT["B8"] = r"""
**Reading.** Five chains, all with 36_G1/52_G1 excluded; shifts are predictions on held-out women from 100 fold
models. The **paper chain** (CellOT stim maps + CellOT drug maps) predicts THF −8.0 d (92 % of folds), SALPZ −6.3 d,
MAP −4.7 d and CHT +3.0 d. Only THF, SALPZ and MAP exceed a size-matched random perturbation (−5.0 / −4.1 / −3.5 null
SDs).

Across chains:
- **THF is robust**: −7.3 to −10.8 d, 92–100 % of folds, including the no-neural-network chain.
- **SALPZ is robust**: −3.1 to −6.3 d, 83–98 %.
- **MAP disappears** without CellOT drug maps: −0.3 to −1.8 d, 55–79 %.
- **CHT** is consistently +2 to +4 d (labour earlier).
- RIF, SA and PRA change sign with the drug map.

Rank agreement with the paper chain: Spearman 0.85 (CellOT + `quantile_map`), 0.75 (`gauss_ot` + `gauss_ot`),
0.60 / 0.53 for the others.

**Plausibility.** In the paper chain 48–81 % of treated women fall outside the untreated 95th percentile; with diagonal
drug maps 6–10 %, except LPZ/SA (20–38 %, the DMSO1 drugs). So most of the paper chain's drug effects are XGBoost
extrapolating off the data. A4 adds the held-out reason to prefer the diagonal drug maps: on new donors they predict
the DOS features' drug effect better than CellOT drug maps, and no drug map predicts patient-specific response.
"""

TEXT["B9"] = r"""
**Reading.** CellOT training time, measured from the paper's own files on 1 core:
- 168 stim maps: 274 CPU-h (median 55 min each);
- 1512 stim×drug maps: 10,672 CPU-h (median 4.1 h);
- 252 Unstim×drug maps: 1,612 CPU-h.

That is **12,557 CPU-h** in total. The 30 h / 25 h in the launchers are time limits, not run times. The closed-form
alternatives cost about 1000× less: 1764 `gauss_ot` drug maps in 12 core-h, `quantile_map` + `median_shift` drug maps
in 14.7 core-h (cited, `drug_map_choice.ipynb` §2), the stim maps in ~10 minutes (cited). The GPU re-implementation in
atlas_heldout trains a CellOT map in 13–18 GPU-seconds (cited, `work/lc/STATUS.md`).
"""

TEXT["removed"] = r"""
## What was removed and why (2026-10-06)

**Rule:** a map is scored only on donors whose cells it never saw. Each item below scored a map (or a model) on the
data it was fitted or selected on, which rewards memorisation. The archived copy keeps them:
`notebooks/archive/summary_all_results_insample_2026-10-06.ipynb` (builder alongside).

| removed (in-sample) | why in-sample | held-out counterpart | how the conclusion changed |
|---|---|---|---|
| **Pooled 80/20 stim benchmark** (old A1.1): win rates on 127 live pairs — CellOT best on MMD; `median_shift` beats it on the marker median in 98 %; `gauss_ot` on mean/correlation in 95–97 % | 80/20 split of *cells*; all 19 donors on both sides | learning curve, A1 | Skill falls from ~0.87 to ~0.60 for the trained maps. CellOT's in-sample MMD edge does not become an edge on new donors: it **ties `gauss_ot`** (0.600 vs 0.595) and both beat the diagonal maps; `median_shift`'s in-sample win on the median does not carry over (last, 0.512). |
| **Pooled quantile skill** (old A1.2): CellOT 0.87, `gauss_ot` 0.87, `quantile_map` 0.94, `median_shift` 0.75 | same cell split | A1 (same skill definition, per-donor truth) | `quantile_map` drops **from first to third** (0.94 → 0.54): a per-marker map memorises the pooled marginals. Ranking on new donors: CellOT ≈ `gauss_ot` > `quantile_map` > `median_shift`. |
| **Pooled drug arm** (old A1.3): 48 MMD-live pairs, the 301-pair per-marker reselection (CellOT tied with `quantile_map`/`median_shift`, 50.3 %), pooled drug quantile skill (`quantile_map` 0.54, `gauss_ot` 0.49, CellOT 0.45; CellOT ahead of `gauss_ot` on effects > 0.1) | cell split, same drug-screen donors | A4 | Levels fall to 11–19 %. In-sample "`quantile_map` best overall" becomes, on new donors: **tie over all quantiles** (12 of 20 donors), `quantile_map` on the **medians** (20 of 20) and DOS features (18 of 20), CellOT on the **low tails** and narrowly on effects > 0.1 (2 % of error). The *effect-size* facts (48× smaller, plate floor) are measurements and stay in A0. |
| **Selection leak** (old A1.5): `model.pt` vs `last.pt` on the test cells that chose `model.pt` | checkpoint selected on the cells it is scored on | A1: "paper-style best" is chosen on held-out cells of *training* donors | On new donors last iterate and paper-style checkpoint coincide (0.600 vs 0.596). |
| **CellOT's live-pair median error "equals the technical floor"** (0.027 vs 0.0273) | CellOT scored on its training donors | — | Dropped; the floor itself stays as a data fact (A0). |
| **In-sample drug-map choice** (old `drug_map_choice` numbers): per-patient skill `quantile_map` 0.53 vs CellOT 0.44 on the 19 drug-screen patients, `quantile_map` better for 18 of 19 patients, "CellOT captures 13 % of real between-patient differences" | per-patient scores on the patients the maps were trained on | A4 (rebuilt `drug_map_choice.ipynb`) | **"13 % captured" was memorisation: on new donors −1 %**, and no map captures the patient-specific part. The `quantile_map` advantage survives **only on the medians / shared part** (0.36 vs 0.32; DOS features 0.64 vs 0.47); over all quantiles it is a tie. New: the paper's chain loses to CellOT → `quantile_map` in 19 of 20 donors on the DOS features. |
| **Donor-ID oracle** (old A2.4): between-donor R² 0.79 → 0.89 when the donor is known | scored on the donors' own validation cells | none possible (a donor ID cannot generalise) | The claim "donor-specific response exists" now rests only on the noise analysis in A4 (16 % of the between-donor spread is real); whether it is *predictable* is answered by A3/A4: no. |
| **In-sample OLS R²** in B4 (DOS ~ EGA + pred, 0.884 → 0.887) | fitted and scored on the same rows | partial r and clustered p on out-of-fold predictions (kept) | No change: partial r 0.17, p 0.20. |
| **Lab rescore numbers as results** (lab-script R² of non-CellOT arms, lab best-of-97, bug-affected G1 AUROC / partial r) | same woman in training and test for 94 % of test samples | aligned rescore (B3, B4, B6) | Combined sets 0.529 → 0.449; G1 lead 0.814 (p 0.017) → 0.745 (p 0.07); partial r 0.20 (p 0.064) → 0.17 (p 0.20). Kept only in B2/B4/B6 to size the bug, labelled NOT USED. |

**Flagged, not removed:** `params_43` and every "best of 97 / best of sweep" are configurations chosen on the reported
splits (B1, B3, B3d). They are not training-on-test, but they are selection on test; read `params_43` and the
median-config column, and treat best-of-97 as optimistic.
"""

TEXT["contradictions"] = r"""
## Contradictions and corrections

Items 1, 2 and 4 were applied to `CLAUDE.md` on 2026-10-06 (with the verification).

1. **`params_43` is not the RMSE argmin.** On the Fig 5b selections it ranks **5th of 97** by RMSE in the lab script
   (0.415 vs 0.427) and **43rd** aligned; it is **rank 1 by preterm AUROC** (5th by RMSE) in the lab's drug grid
   search. Selection on the same splits still holds, via AUROC; its R² optimism is small.
2. **The G1 preterm lead and the partial correlation beyond EGA** were computed on bug-affected predictions. Aligned:
   **0.745 (p 0.07)** and **0.17 (p 0.20)** instead of 0.814 (p 0.017) and 0.20 (p 0.064).
3. **Simulated quantiles.** simco: no gain (100 splits, −0.006, p 0.50). atlas_heldout: +0.043–0.052 (p < 1e-4).
   Unexplained; not pairable.
4. **The row-order bug inflated the combined sets by +0.08**, not the "0.02–0.04" that holds for single sets.
5. **Noise-floor aggregation.** This notebook: 76.3 % / 12.9 % of drug pairs below their own technical / sampling floor,
   IL33 at 0.66×; `CLAUDE.md`: 72.2 % / 11.9 % and 0.70×. Same conclusion, different aggregation.
6. **CellOT vs `gauss_ot` on DOS depends on power.** 20 splits, Stabl's own scoring: dead heat (p 0.73–0.87). Aligned
   100 splits: CellOT ahead (−0.082, p 0.002). The 20-split design lacks power.
7. **Two different "CellOT" numbers on held-out donors.** Learning-curve CellOT (0.600 at k = 15, 5 repeats) vs
   context-free conditional CellOT `none` (0.621, repeats 1–4): `none` uses paired-donor sampling. Compare each only
   within its own table.
8. **Held-out drug-skill tables use different case sets.** `overall.csv` (A4.1) includes LPZ/SA under PI; the
   `drug_map_choice` tables (A4.2–A4.4) exclude them. Rankings agree; levels differ by up to 0.02.

**Gaps (not run, or not aggregated here).**
- A matched-n noise floor and a batch-corrected floor for the drug arm.
- A full Stabl rerun without 36_G1/52_G1 for the simco arms (selection still saw them).
- A 100-split null-map rerun with ctswap and several seeds.
- Stabl with EGA as an unpenalised covariate, or Stabl on the EGA residual.
- A DOS model whose `XGBoost` configuration is chosen inside the CV (nested), to remove the `params_43` selection.
- A held-out-patient DOS evaluation of retrained drug maps; the "direct" scenario is deferred in atlas_heldout.
- A drug-arm re-analysis of LPZ/SA against a clean control.
"""

TEXT["sources"] = r"""
## Sources and how to rebuild

```
module load python/3.9.0 && source /scratch/users/bmerwan/venv-cellot/bin/activate
python notebooks/build_summary_all_results.py
jupyter nbconvert --to notebook --execute --inplace notebooks/summary_all_results.ipynb
```

Figures are written to `notebooks/figures_summary/`.

Source notebooks for detail:
- `atlas_heldout/notebooks/learning_curve.ipynb` (A1–A3);
- `drug_map_choice.ipynb` + `drug_map_choice_readings.md` (A4, held-out only since 2026-10-06);
- `atlas_heldout/notebooks/atlas_vs_ool.ipynb` (A5);
- `baselines_results.ipynb` §10.7 (noise floors; its model comparisons are in-sample);
- `stabl_results.ipynb`, `ega_preterm_nullmaps.ipynb` §8 (aligned protocol);
- `cellot_vs_gaussot.ipynb`, `quantiles.ipynb` (simq, quantile arms);
- `ool_longitudinal/README.md`.
"""


md(TEXT["title"])
md(TEXT["exec"])

code(r'''
import ast, glob, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr, rankdata
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 200)

W   = Path("/scratch/users/bmerwan/simco_work")
B   = W / "baselines"
O   = W / "onset_test"
PP  = O / "paper_protocol"
AH  = Path("/scratch/users/bmerwan/atlas_heldout/work")
LC  = AH / "lc"
HO  = AH / "heldout"
LGT = Path("/scratch/users/bmerwan/ool_longitudinal/work")
GH  = Path("/home/groups/gbrice/ptb-drugscreen/ool_stabl/onset_test")      # read-only
FIG = Path("/scratch/users/bmerwan/simco/notebooks/figures_summary"); FIG.mkdir(exist_ok=True)
FLOOR = 0.0273          # technical floor: DMSO2 vs DMSO3 median_error, n ~ 20.8k
LIVE = 0.005            # identity-MMD live/null cut
EXCL = ["36_G1", "52_G1"]
RNG = np.random.default_rng(0)
COL = {"CellOT": "#2a78d6", "gauss_ot": "#eb6834", "quantile_map": "#1baf7a", "median_shift": "#eda100",
       "linear_median_ridge": "#e87ba4", "EGA": "#52514e"}

def first(x):                       # "0.415 [0.28, 0.52]" -> 0.415
    return float(str(x).split()[0])

def boot_ci(d, n=5000):
    d = np.asarray(d, float); m = RNG.choice(d, (n, len(d))).mean(1)
    return np.percentile(m, [2.5, 97.5])

def show(df, caption, fmt=3):
    print(f"\n=== {caption}")
    with pd.option_context("display.float_format", lambda v: f"{v:.{fmt}f}"):
        display(df)
''')

# =====================================================================================================================
md(TEXT["partA"])
md(TEXT["A0"])
code(r'''
# identity rows only = observed effect (stim/drug vs control, test-side cells); no fitted map is read here
def identity_rows(d):
    return pd.concat([pd.read_csv(f).query("method == 'identity'") for f in sorted(d.glob("*.csv"))], ignore_index=True)
idm = identity_rows(B / "pooled").set_index(["celltype", "stim"]).mmd
didm = identity_rows(B / "pooled_drug").set_index(["celltype", "stim", "drug"]).mmd
print(f"stim pairs {len(idm)}, median identity MMD {idm.median():.5f}; drug pairs {len(didm)}, median {didm.median():.5f} "
      f"({idm.median() / didm.median():.0f}x smaller)")
nf_all = pd.concat([pd.read_csv(f) for f in sorted((B / "noise_floor").glob("*.csv"))])
pairs168 = set(pd.read_csv(B / "pairs_168.txt", header=None)[0])
nf = nf_all[(nf_all.scope == "pooled") & (nf_all.stim + "/" + nf_all.celltype).isin(pairs168)]
samp = nf[nf.pair_type == "split_half"].mmd.median()
tech = nf[(nf.source == "DMSO2") & (nf.target == "DMSO3")]
print(f"MMD floors over the 168 pairs (median): sampling (split half) {samp:.5f}; technical DMSO2 v DMSO3 {tech.mmd.median():.5f}")
eff = idm.reset_index().assign(pair=lambda x: x.stim + "/" + x.celltype).set_index("pair").mmd
fl = tech.assign(pair=lambda x: x.stim + "/" + x.celltype).set_index("pair").mmd
st = pd.DataFrame({"effect": eff, "floor": fl}).assign(stim=lambda x: x.index.str.split("/").str[0])
st["ratio"] = st.effect / st.floor
show(st.groupby("stim").median().rename(columns={"effect": "median effect", "floor": "median floor",
                                                 "ratio": "median per-pair effect / floor"}),
     "DATA FACT: stim effect vs technical floor (identity MMD, pooled; source baselines/noise_floor/, identity rows of baselines/pooled/)", 5)
tech_all = nf_all.query("scope == 'pooled' and source == 'DMSO2' and target == 'DMSO3'")
own_t = tech_all.set_index(["celltype", "stim"]).mmd
own_s = nf_all.query("scope == 'pooled' and pair_type == 'split_half'").groupby(["celltype", "stim"]).mmd.mean()
dd_ = didm.reset_index(level="drug", drop=True)
print(f"DATA FACT: drug pairs moving less than their own (celltype, stim) technical floor: "
      f"{100 * (dd_.values < own_t.reindex(dd_.index).values).mean():.1f} %; below their own sampling floor: "
      f"{100 * (dd_.values < own_s.reindex(dd_.index).values).mean():.1f} % (CLAUDE.md quotes 72.2 % / 11.9 %)")
cs = pd.read_csv(B / "control_swap_PI.csv")
show(cs.groupby(["cells", "drug", "control"]).mmd.agg(["min", "max"]).unstack("control"),
     "DATA FACT: control swap under PI, identity MMD drug vs each control, range over 4 celltypes (source baselines/control_swap_PI.csv)", 4)
''')
md(TEXT["A0_r"])

md("## A1. Stim maps: learning curve on held-out donors (5 repeats × 4 test donors, k = 1…15 training donors)")
code(r'''
lc = pd.read_csv(LC / "summary/curve.csv")
t = lc[lc.k.isin([1, 3, 5, 15])].pivot_table(index="method", columns="k", values="skill")
t["ceiling (split-half)"] = lc.groupby("method").ceiling.first()
show(t, "median skill on held-out donors, effect > 0.0273, capped cache (source atlas_heldout/work/lc/summary/curve.csv)", 3)
lcf = pd.read_csv(LC / "summary_full/curve/curve.csv")
show(lcf[lcf.k.isin([3, 15])].pivot_table(index="method", columns="k", values="skill"),
     "same, uncapped cells, 20k iterations (summary_full/curve/curve.csv)", 3)
sat = pd.read_csv(LC / "summary/saturation.csv")
show(sat, "k (training donors) needed for 90 % / 95 % of the 15-donor skill (column fraction)", 2)
''')
md(TEXT["A1_r"])

md("## A2. Donor experiments: HV vs PV, good / hard donors, cells vs donors (held-out)")
code(r'''
D = LC / "donors_summary"
show(pd.read_csv(D / "hvpv/table.csv").pivot_table(index=["method", "k"], columns="train_group", values=["HV", "PV"]),
     "skill by training group (columns) and test group (HV / PV) (source donors_summary/hvpv/table.csv)", 3)
pt = pd.read_csv(D / "hvpv/paired_tests.csv")
show(pt[pt.comparison == "HV - PV training"].set_index(["method", "k", "test_group"])[["mean_diff", "n_test_donor_repeats", "p"]],
     "HV-trained minus PV-trained skill, by test group", 3)
show(pd.read_csv(D / "small/doubling_effects.csv").set_index("method"),
     "8 smallest celltypes: gain in skill from doubling donors vs doubling cells per donor", 3)
show(pd.read_csv(D / "pairwise/summary.csv", index_col=0).iloc[[0, 1, 2, 4, 5, 6]],
     "pairwise (one training donor -> one test donor): variance shares and correlates", 2)
''')
md(TEXT["A2_r"])

md("## A3. A patient covariate in the map (held-out)")
code(r'''
cs1 = pd.read_csv(LC / "cond_summary/med_z/curve.csv")
show(cs1[cs1.k.isin([3, 15])].pivot_table(index="method", columns="k", values="mean"),
     "step 1, closed form: skill k = 3 / 15, 5 repeats (source lc/cond_summary/med_z/curve.csv)", 3)
cb = pd.read_csv(LC / "cond_between/med_z/between.csv")
show(cb[(cb.k == 15)].pivot_table(index="method", columns="subset", values="between_R2"),
     "share of the true between-donor variance reproduced at k = 15, test donors (after centring; source lc/cond_between/med_z/between.csv)", 3)
cf = pd.read_csv(LC / "ctx_score/final_r1234/curve.csv")
show(cf[cf.k.isin([3, 15])].pivot_table(index="method", columns="k", values=["skill", "dev_R2"]),
     "steps 2-3, conditional CellOT, clean repeats 1-4 (source lc/ctx_score/final_r1234/curve.csv)", 3)
cp = pd.read_csv(LC / "ctx_score/final_r1234/paired.csv")
show(cp[(cp.k == 15) & ((cp.b == "none") | (cp.a == "none"))], "paired at k = 15: % of cases where a is closer than b", 3)
show(pd.read_csv(LC / "residual_predict/summary.csv").pivot_table(index="method", columns="proto", values="res_R2"),
     "share of the residual beyond the feature shift predicted from Unstim (res_R2; 0 = shift; lc15 = k 15 held-out, "
     "lodo = leave one donor out; source lc/residual_predict/summary.csv)", 3)
''')
md(TEXT["A3_r"])

md(TEXT["A4"])
md("### A4.1 Every drug pipeline on held-out donors (recomputed)")
code(r'''
ov = pd.read_csv(HO / "drugs/analysis/overall.csv", index_col=0)
show(ov, "held-out donors, 353k real cases (effect > split-half noise); skill = 1 - err/effect "
     "(source atlas_heldout/work/heldout/drugs/analysis/overall.csv)", 3)
''')
md(TEXT["A4_r1"])
md("### A4.2 The drug map alone: CellOT vs quantile_map / median_shift / gauss_ot (cited)")
code(r'''
SRC = "cited from drug_map_choice.ipynb (held-out donors only, executed 2026-10-06), LPZ/SA under PI excluded"
SUBS = ["all real", "effect > 0.1", "medians (q50)", "DOS features (q50)"]
step_skill = pd.DataFrame([[0.168, 0.132, 0.377, 0.359], [0.141, 0.101, 0.445, 0.451],
                           [0.118, 0.071, 0.428, 0.436], [0.168, 0.110, 0.426, 0.443]],
                          index=["CellOT", "quantile_map", "median_shift", "gauss_ot"], columns=SUBS)
step_skill.loc["cases"] = [326116, 80273, 32097, 1888]
show(step_skill, f"§3 median skill of the drug map alone ({SRC})", 3)
step_pair = pd.DataFrame([
    ["all real", "quantile_map", 0.0519, 0.0517, "12/20", 0.25],
    ["all real", "median_shift", 0.0519, 0.0510, "17/20", 8.2e-5],
    ["all real", "gauss_ot", 0.0519, 0.0492, "20/20", 1.9e-6],
    ["effect > 0.1", "quantile_map", 0.1324, 0.1347, "0/20", 1.9e-6],
    ["effect > 0.1", "gauss_ot", 0.1324, 0.1325, "12/20", 0.70],
    ["medians (q50)", "quantile_map", 0.0348, 0.0299, "20/20", 1.9e-6],
    ["medians (q50)", "median_shift", 0.0348, 0.0306, "20/20", 1.9e-6],
    ["medians (q50)", "gauss_ot", 0.0348, 0.0312, "20/20", 1.9e-6],
    ["DOS features (q50)", "quantile_map", 0.0356, 0.0301, "18/20", 3.6e-5],
    ["DOS features (q50)", "median_shift", 0.0356, 0.0313, "16/20", 8.5e-4],
    ["DOS features (q50)", "gauss_ot", 0.0356, 0.0303, "17/20", 1.3e-4]],
    columns=["subset", "B (vs CellOT = A)", "median err CellOT", "median err B", "donors: B better", "p (donors)"])
show(step_pair, f"§3 paired per donor-repeat: in how many of 20 donor-repeats B has the smaller median error ({SRC})", 4)
show(pd.DataFrame({"q10": [44.0], "q25": [48.5], "q50": [58.0], "q75": [54.0], "q90": [52.0]}, index=["% cases quantile_map closer"]),
     f"§6 by quantile: CellOT wins the low tails ({SRC})", 1)
''')
md(TEXT["A4_r2"])
md("### A4.3 The whole pipeline (stim map then drug map) on held-out donors (cited)")
code(r'''
pipe_skill = pd.DataFrame([[0.147, 0.113, 0.367, 0.342], [0.161, 0.104, 0.438, 0.435], [0.118, 0.071, 0.428, 0.436],
                           [0.155, 0.100, 0.419, 0.432], [0.159, 0.103, 0.440, 0.444]],
                          index=["paper: CellOT stim -> CellOT drug (cells)", "CellOT stim -> quantile_map on features",
                                 "CellOT stim -> median_shift on features", "gauss_ot stim -> gauss_ot drug (cells)",
                                 "gauss_ot stim -> quantile_map on features"], columns=SUBS)
show(pipe_skill, f"§4 median skill, pipelines ({SRC})", 3)
pipe_pair = pd.DataFrame([
    ["all real", "CellOT stim -> quantile_map", 0.0529, 0.0499, "20/20"],
    ["all real", "CellOT stim -> median_shift", 0.0529, 0.0510, "20/20"],
    ["all real", "gauss_ot -> gauss_ot", 0.0529, 0.0499, "20/20"],
    ["effect > 0.1", "CellOT stim -> quantile_map", 0.1348, 0.1333, "18/20"],
    ["medians (q50)", "CellOT stim -> quantile_map", 0.0357, 0.0305, "20/20"],
    ["DOS features (q50)", "CellOT stim -> quantile_map", 0.0360, 0.0307, "19/20"],
    ["DOS features (q50)", "CellOT stim -> median_shift", 0.0360, 0.0313, "16/20"]],
    columns=["subset", "B (vs paper chain = A)", "median err paper chain", "median err B", "donors: B better"])
show(pipe_pair, f"§4 / §7 paired per donor-repeat against the paper's chain ({SRC})", 4)
''')
md(TEXT["A4_r3"])
md("### A4.4 Per patient: shared part vs patient-specific part (cited)")
code(r'''
pp = pd.DataFrame({
    "shared-part skill (real-difference groups)": [0.318, 0.362, 0.306, 0.368, 0.301, 0.360],
    "patient-specific part captured, % (real-difference groups)": [-1.2, -7.4, 0.0, 1.2, -3.9, -7.0],
    "tracking r": [0.168, -0.105, -0.013, 0.173, 0.093, 0.066],
    "shared-part skill, DOS features": [0.474, 0.641, 0.614, 0.620, 0.467, 0.646],
    "patient-specific captured, %, DOS features": [3.5, 4.6, 0.0, 6.0, 2.1, 0.8],
    "share of per-patient error from the shared part, DOS features": [0.461, 0.322, 0.327, 0.318, 0.470, 0.334]},
    index=["CellOT", "quantile_map", "median_shift (constant by construction)", "gauss_ot",
           "paper: CellOT stim -> CellOT drug", "CellOT stim -> quantile_map"])
show(pp, f"§5, 184,299 donor groups with real between-donor differences (34 %); DOS features 790 groups ({SRC})", 3)
''')
md(TEXT["A4_r4"])

md("## A5. Data fact: the maps are applied to OOL inputs that sit off the atlas scale")
code(r'''
sv = pd.read_csv(HO / "atlas_vs_ool/shift_by_celltype_marker.csv")
t = sv.groupby("marker").agg(**{"median shift q50 (OOL - atlas)": ("shift_q50", "median"), "median z (atlas donor SD)": ("z", "median")})
show(t.sort_values("median shift q50 (OOL - atlas)").round(2), "DATA FACT: OOL Unstim vs atlas Unstim, median over celltypes (source heldout/atlas_vs_ool/)", 2)
sc = pd.read_csv(HO / "atlas_vs_ool/simulated_columns.csv")
print(f"retrained-CellOT simulated OOL columns with impossible values: {int(sc.implausible.sum())} / {len(sc)}")
''')
md(TEXT["A5_r"])

md("## Overview figure for Part A (held-out donors only)")
code(r'''
fig, (b1, b2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
LCM = {"CellOT (last)": "CellOT", "gauss_ot": "gauss_ot", "quantile_map": "quantile_map", "median_shift": "median_shift"}
for m, lab in LCM.items():
    x = lc[lc.method == m].sort_values("k")
    b1.plot(x.k, x.skill, color=COL[lab], lw=2, marker="o", ms=5)
    off = {"CellOT": 0.012, "gauss_ot": -0.012}.get(lab, 0)
    b1.text(15.4, x.skill.iloc[-1] + off, f"{lab} {x.skill.iloc[-1]:.2f}", va="center", fontsize=9, color="#0b0b0b")
ceil = lc.ceiling.iloc[0]
b1.axhline(ceil, color="#52514e", ls=":", lw=1); b1.text(1, ceil + 0.012, f"split-half ceiling {ceil:.2f}", fontsize=8, color="#52514e")
b1.set_xlim(0.5, 19.5); b1.set_ylim(0.35, 0.92); b1.set_xticks([1, 3, 5, 8, 10, 12, 15])
b1.set_xlabel("training donors (k)"); b1.set_ylabel("median skill on 4 held-out donors\n(1 − err / err of 'no effect')")
b1.set_title("stim maps on new donors (5 repeats)", fontsize=11, loc="left")
meths = ["CellOT", "gauss_ot", "quantile_map", "median_shift"]
yy = np.arange(len(SUBS))[::-1]
for j, m in enumerate(meths):
    b2.scatter(step_skill.loc[m, SUBS].astype(float), yy + (1.5 - j) * 0.13, s=60, color=COL[m], label=m, zorder=3,
               edgecolor="white", linewidth=1)
b2.set_yticks(yy); b2.set_yticklabels(["all real cases", "effects > 0.1", "medians (q50)", "DOS features (q50)"], fontsize=9)
b2.set_xlim(0, 0.5); b2.set_xlabel("median skill on new donors")
b2.set_title("drug map alone on new donors (cited, LPZ/SA·PI excluded)", fontsize=11, loc="left")
b2.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0.02, 0.38))
for ax in (b1, b2):
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x" if ax is b2 else "y", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(FIG / "fig_atlas.png", dpi=150); plt.show()
''')
md(TEXT["fig_atlas"])

# =====================================================================================================================
md(TEXT["partB"])
md("### B1. Reproducing Fig 5b, and what `params_43` is")
code(r'''
def scores(p):
    s = pd.read_csv(p, index_col=0); return s
rows = []
for lab, p in [("paper Fig 5b run GS_cellot_longallstims_foldinfo (lab, group home)", GH / "GS_cellot_longallstims_foldinfo"),
               ("near-miss ..._xgboost_knockoff_GSS (same Stabl selections, default XGBoost)", GH / "results_ina_13OG_final_long_allstims_filtered_xgboost_knockoff_GSS"),
               ("our rescore of the GSS selections with params_43 (lab script)", O / "rescore_params43/both")]:
    s = scores(p / "Summary/Scores training CV.csv").loc["STABL ALasso"]
    rows.append({"run": lab, "R2": s.R2, "RMSE (d)": s.RMSE, "MAE (d)": s.MAE, "N features": s["N features"]})
show(pd.DataFrame(rows), "Fig 5b: R = 0.64, RMSE 24.1 d, 21 features (18-30), 177 samples / 63 women (preprint). All three rows go through the lab's rescore script (row-order bug, partly in-sample) and params_43 was selected on these splits; the aligned value is 0.418 (B3)")
lab = pd.read_csv(PP / "rescore/cellot_both/sweep.csv")
al = pd.read_csv(PP / "aligned/cellot/sweep.csv")
gs = pd.read_csv(GH / "comp_9drugs_long_unstimpred_treated/gridsearch_summary_treated_unstim.csv")
r = gs.set_index("Run")
print(f"params_43 on the Fig 5b selections, 97 configs: lab-script rank by RMSE {int(lab.RMSE.rank().loc[lab.params == 'params_43'].iloc[0])} "
      f"(R2 {lab.set_index('params').R2['params_43']:.3f} vs best {lab.R2.max():.3f}); aligned rank "
      f"{int(al.RMSE.rank().loc[al.params == 'params_43'].iloc[0])} (R2 {al.set_index('params').R2['params_43']:.3f} vs best {al.R2.max():.3f})")
print(f"lab drug grid search ({len(gs)} configs): run_43 rank by preterm AUROC {int(r.AUROC.rank(ascending=False)['run_43'])} "
      f"(AUROC {r.AUROC['run_43']:.3f}), by RMSE {int(r['RMSE global'].rank()['run_43'])}")
''')
md(TEXT["B1"])

md("### B2. The row-order bug: lab-script rescore (superseded) vs aligned")
code(r'''
ARMS = {"cellot": ("cellot_both", "CellOT stims + measured Unstim (paper)"),
        "gaussot_both": ("gaussot_both", "gauss_ot stims + measured Unstim"),
        "d_freqmed": ("d_freqmed", "measured freq + 31-marker medians"),
        "d_qtl": ("d_qtl", "measured freq + medians + quantiles"),
        "comb_cellot": ("comb_cellot", "combined: CellOT stims + measured"),
        "comb_gaussot": ("comb_gaussot", "combined: gauss_ot stims + measured"),
        "comb_cellot_simq": (None, "combined CellOT + simulated quantiles"),
        "comb_gaussot_simq": (None, "combined gauss_ot + simulated quantiles"),
        "crossswap_s0_both": ("crossswap_s0_both", "null: wrong stim AND celltype map"),
        "rscaled_s0_both": ("rscaled_s0_both", "null: untrained ICNN, scale-matched")}
rows = []
for a, (labdir, lab_name) in ARMS.items():
    s = pd.read_csv(PP / f"aligned/{a}/sweep.csv").set_index("params")
    r = {"arm": a, "aligned p43": s.R2["params_43"]}
    if labdir:
        ls_ = pd.read_csv(PP / f"rescore/{labdir}/sweep.csv").set_index("params")
        r.update({"lab-script p43 (bug: same woman in train and test; NOT USED)": ls_.R2["params_43"]})
    rows.append(r)
bug = pd.DataFrame(rows).set_index("arm"); bug["inflation p43"] = bug.iloc[:, 1] - bug["aligned p43"]
show(bug.dropna(), "100 splits, params_43, ALasso selections: size of the row-order bug (sources paper_protocol/rescore/*/sweep.csv vs paper_protocol/aligned/*/sweep.csv)")
''')
md(TEXT["B2"])

md("### B3. Every feature set on the aligned paper protocol (100 splits, 177 samples)")
code(r'''
X0 = pd.read_csv(O / "ina_13OG_final_long_allstims_filtered.csv", index_col=0, usecols=[0])
y0 = pd.read_csv(O / "outcome_table_all_pre.csv", index_col=0)["DOS"]
clin = pd.read_csv(O / "Preprocessed_OOL_Clinical_new.csv", index_col=0)
rows_ = X0.index.intersection(y0.index); y = y0.loc[rows_]; g = rows_.str.split("_").str[0].values
splits = list(GroupShuffleSplit(100, test_size=0.2, random_state=42).split(np.zeros(len(rows_)), y, g))
ega = clin.loc[rows_, ["EGA"]].values
P = {r: [] for r in rows_}; ega_fold = []
for tr, te in splits:
    pr = LinearRegression().fit(ega[tr], y.iloc[tr]).predict(ega[te]); ega_fold.append(r2_score(y.iloc[te], pr))
    for r, v in zip(rows_[te], pr): P[r].append(v)
ega_oof = pd.Series({r: np.median(v) for r, v in P.items() if v})
EGA_R2 = r2_score(y.loc[ega_oof.index], ega_oof)
print(f"EGA alone, one linear coefficient, same 100 aligned splits: pooled R2 {EGA_R2:.3f}, "
      f"RMSE {np.sqrt(((y.loc[ega_oof.index] - ega_oof) ** 2).mean()):.1f} d, median split R2 {np.median(ega_fold):.3f}")

def aligned_table(base, ref="cellot", arms=None):
    out = []; F = {}
    arms = arms or [a for a in ARMS if (base / a / "sweep.csv").exists()]
    for a in [ref] + [x for x in arms if x != ref]:
        s = pd.read_csv(base / a / "sweep.csv").set_index("params"); f = pd.read_csv(base / a / "folds.csv"); F[a] = (s, f)
    s0, f0 = F[ref]; b0 = s0.R2.idxmax()
    for a, (s, f) in F.items():
        r = {"arm": a, "n splits": len(f), "R2 p43": s.R2["params_43"], "RMSE p43 (d)": s.RMSE["params_43"],
             "median split R2 p43": float(np.median(f["params_43"]))}
        if len(s) > 1:
            r.update({"R2 best of 97": s.R2.max(), "R2 median config": s.R2.median()})
        if "R2_with" in s: r["R2 p43, outliers kept in training"] = s.R2_with["params_43"]
        if a != ref:
            d = f["params_43"] - f0["params_43"]; lo, hi = boot_ci(d)
            r.update({"Δ vs CellOT per split (p43)": d.mean(), "95% CI lo": lo, "95% CI hi": hi,
                      "wins /n": int((d > 0).sum()), "Wilcoxon p": wilcoxon(d).pvalue})
            if len(s) > 1:
                db = f[s.R2.idxmax()] - f0[b0]
                r.update({"Δ best-vs-best": db.mean(), "p best-vs-best": wilcoxon(db).pvalue})
        out.append(r)
    return pd.DataFrame(out).set_index("arm")

main = aligned_table(PP / "aligned")
main.loc["EGA alone (linear)", ["n splits", "R2 p43", "median split R2 p43"]] = [100, EGA_R2, np.median(ega_fold)]
show(main, "ALIGNED, 100 splits, ALasso selections, all 177 samples (source paper_protocol/aligned/<arm>/{sweep,folds}.csv; "
     "EGA row recomputed here). FLAG: 'R2 best of 97' and 'best-vs-best' pick the config on these same test splits (optimistic); "
     "params_43 was itself chosen on these splits (by preterm AUROC); 'R2 median config' is the least selected summary")
''')
md(TEXT["B3"])
code(r'''
ex = aligned_table(PP / "aligned_excl")
show(ex, "ALIGNED, 100 splits, params_43, 36_G1 / 52_G1 dropped from XGBoost training and scoring (Stabl selection still saw them; "
     "source paper_protocol/aligned_excl/)")
mods = []
for model, base_all, base_ex in [("ALasso", PP / "aligned", PP / "aligned_excl"),
                                  ("Lasso", PP / "aligned_models/Lasso", PP / "aligned_models_excl/Lasso"),
                                  ("ElasticNet", PP / "aligned_models/ElasticNet", PP / "aligned_models_excl/ElasticNet")]:
    for a in ["cellot", "gaussot_both", "d_qtl", "comb_cellot", "comb_gaussot"]:
        sa = pd.read_csv(base_all / a / "sweep.csv").set_index("params"); se = pd.read_csv(base_ex / a / "sweep.csv").set_index("params")
        mods.append({"model": model, "arm": a, "R2 all": sa.R2["params_43"], "R2 without 36_G1/52_G1": se.R2["params_43"]})
show(pd.DataFrame(mods).pivot_table(index="arm", columns="model", values=["R2 all", "R2 without 36_G1/52_G1"]),
     "Stabl model choice, 100 splits, aligned params_43 (source paper_protocol/aligned_models{,_excl}/)")
''')
md(TEXT["B3b"])

md("#### B3c. 20-split results — never compare with the 100-split columns above")
code(r'''
n20 = aligned_table(PP / "aligned_n20", arms=["cellot", "gaussot_both", "d_qtl", "comb_cellot", "comb_gaussot", "comb_cellot_simq", "comb_gaussot_simq"])
show(n20, "ALIGNED, first 20 of the 100 splits, all 177 samples (source paper_protocol/aligned_n20/)")
d = (pd.read_csv(PP / "aligned_n20/comb_cellot_simq/folds.csv")["params_43"] - pd.read_csv(PP / "aligned_n20/comb_cellot/folds.csv")["params_43"])
print(f"simq vs no simq (CellOT combined), 20 splits: Δ {d.mean():+.3f}, p {wilcoxon(d).pvalue:.2f}, {int((d > 0).sum())}/20")
for a, b in [("comb_cellot_simq", "comb_cellot"), ("comb_gaussot_simq", "comb_gaussot")]:
    d = pd.read_csv(PP / f"aligned_excl/{a}/folds.csv")["params_43"] - pd.read_csv(PP / f"aligned_excl/{b}/folds.csv")["params_43"]
    print(f"100 splits, without 36_G1/52_G1: {a} - {b}: Δ {d.mean():+.3f}, p {wilcoxon(d).pvalue:.2f}, {int((d > 0).sum())}/100")
''')
code(r'''
nm = pd.read_csv(O / "null_maps/null_maps_scores.csv"); nm["R2v"] = nm.R2.map(first)
nm["family"] = nm.arm.str.replace(r"_s\d$", "", regex=True)
t = nm.pivot_table(index=["table", "family"], columns="model", values="R2v", aggfunc="mean")
n2 = pd.read_csv(O / "night2/night2_scores.csv"); n2["R2v"] = n2.R2.map(first)
t2 = n2[n2.target == "DOS"].pivot_table(index="run", columns="model", values="R2v")
show(t, "20 splits, Stabl's own pooled scoring (run_regression_cv, default XGBoost, no row-order bug), mean over seeds "
     "(source onset_test/null_maps/null_maps_scores.csv). `permuted` is RETRACTED as a control (copies of real columns)")
show(t2, "20 splits, measured distributional features vs CellOT / gauss_ot (source onset_test/night2/night2_scores.csv)")
pn = pd.read_csv(O / "null_maps/null_maps_paired.csv")
show(pn[pn.model == "STABL ALasso"].set_index(["arm", "table"])[["mean_diff_cellot_minus_arm", "ci_lo", "ci_hi", "wilcoxon_p", "cellot_wins"]],
     "20 splits, ALasso, CellOT minus arm per split (source null_maps_paired.csv)")
''')
md(TEXT["B3c"])

md("#### B3d. atlas_heldout quantile arms — retrained maps, 36_G1/52_G1 removed BEFORE Stabl (175 rows, so different splits)")
code(r'''
qs = pd.read_csv(HO / "quantile_arms/summary/sets.csv").set_index("set")
show(qs.iloc[:, :8], "100 splits, aligned rescore, 175 samples (source atlas_heldout/work/heldout/quantile_arms/summary/sets.csv). "
     "FLAG: 'R2 best of sweep' and 'best vs best' below pick the config on the reported test splits")
qp = pd.read_csv(HO / "quantile_arms/summary/paired.csv")
show(qp[qp.vs == "ref852"].set_index(["set", "config"]), "paired per split vs ref852 (the retrained-map analogue of comb_cellot)")
''')
md(TEXT["B3d"])

md("### B3e. Overview figure: every feature set against CellOT and against gestational age")
code(r'''
order_ = ["comb_cellot", "comb_cellot_simq", "comb_gaussot", "comb_gaussot_simq", "cellot", "gaussot_both", "d_qtl", "d_freqmed",
          "crossswap_s0_both", "rscaled_s0_both"]
NAMES = {"cellot": "CellOT stims + Unstim (paper Fig 5b)", "gaussot_both": "gauss_ot stims + Unstim", "d_qtl": "measured: freq + medians + quantiles",
         "d_freqmed": "measured: freq + 31-marker medians", "comb_cellot": "combined: CellOT + measured", "comb_gaussot": "combined: gauss_ot + measured",
         "comb_cellot_simq": "combined CellOT + simulated quantiles", "comb_gaussot_simq": "combined gauss_ot + simulated quantiles",
         "crossswap_s0_both": "null: wrong-stim/celltype CellOT maps", "rscaled_s0_both": "null: untrained ICNN (scale-matched)"}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True, gridspec_kw={"width_ratios": [1.15, 1]})
yy = np.arange(len(order_))[::-1]
for yi, a in zip(yy, order_):
    r = main.loc[a]
    c = COL["CellOT"] if a == "cellot" else ("#52514e" if a.startswith(("cross", "rscaled")) else COL["gauss_ot"] if "gauss" in a else
        COL["quantile_map"] if a.startswith("d_") else "#4a3aa7")
    a1.plot([r["R2 p43"], r["R2 best of 97"]], [yi, yi], color=c, lw=2, zorder=1)
    a1.scatter(r["R2 p43"], yi, s=60, color=c, zorder=2, label="params_43" if yi == yy[0] else None)
    a1.scatter(r["R2 best of 97"], yi, s=60, facecolor="white", edgecolor=c, lw=2, zorder=2, label=None)
    if a != "cellot":
        a2.errorbar(r["Δ vs CellOT per split (p43)"], yi, xerr=[[r["Δ vs CellOT per split (p43)"] - r["95% CI lo"]], [r["95% CI hi"] - r["Δ vs CellOT per split (p43)"]]],
                    fmt="o", ms=7, color=c, capsize=3, lw=2)
a1.axvline(main.loc["cellot", "R2 p43"], color=COL["CellOT"], ls="--", lw=1)
a1.axvline(EGA_R2, color=COL["EGA"], ls=":", lw=1.5); a1.text(EGA_R2 - 0.01, yy[0] + 0.55, f"gestational age alone {EGA_R2:.2f}", ha="right", fontsize=9, color="#52514e")
a1.text(main.loc["cellot", "R2 p43"] + 0.005, yy[0] + 0.55, "CellOT 0.418", fontsize=9, color="#52514e")
a1.set_yticks(yy); a1.set_yticklabels([NAMES[a] for a in order_], fontsize=9)
a1.set_xlim(0.2, 0.92); a1.set_ylim(-0.8, len(order_) - 0.2); a1.set_xlabel("pooled out-of-fold R² (177 samples)")
a1.set_title("DOS prediction, aligned paper protocol (100 splits)", fontsize=11, loc="left")
from matplotlib.lines import Line2D
a1.legend(handles=[Line2D([], [], marker="o", ls="", color="#52514e", ms=7, label="params_43 (paper config)"),
                   Line2D([], [], marker="o", ls="", mfc="white", mec="#52514e", mew=2, ms=7, label="best of 97 configs (chosen on these splits)")],
          loc="center", bbox_to_anchor=(0.68, 0.5), frameon=False, fontsize=9)
a2.axvline(0, color=COL["CellOT"], ls="--", lw=1)
a2.set_xlabel("per-split R² minus CellOT (mean, 95 % bootstrap CI; params_43)")
a2.set_title("paired against CellOT on the same 100 splits", fontsize=11, loc="left")
for ax in (a1, a2):
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(FIG / "fig_dos_forest.png", dpi=150); plt.show()
''')
md(TEXT["fig_dos"])

md("### B4. Gestational age (EGA): the reference every DOS number needs")
code(r'''
cb100 = pd.read_csv(O / "clinical_baselines.csv").set_index("model")
show(cb100, "clinical baselines on Fig 5b's 100 splits (source onset_test/clinical_baselines.csv)")
rows = []
for lab, p in [("lab: CellOT + Unstim + EGA input (GS_cellot_longallstims_EGA_foldinfo)", GH / "GS_cellot_longallstims_EGA_foldinfo"),
               ("lab: CellOT stims only + EGA (GS_ina_no_unstim_42_foldinfo)", GH / "GS_ina_no_unstim_42_foldinfo")]:
    s = pd.read_csv(p / "Summary/Scores training CV.csv", index_col=0).loc["STABL ALasso"]; rows.append({"run": lab, "R2": s.R2, "RMSE": s.RMSE})
show(pd.DataFrame(rows), "the lab's own runs with EGA as an input (group home, read-only). FLAG: *_foldinfo runs go through the lab's rescore "
     "script (row-order bug, partly in-sample), so these are upper bounds; shown only as context for the EGA-alone row above")
# does the Fig 5b prediction add anything beyond EGA? aligned OOF predictions (drug_checks/median_predictions.csv, No_treat)
import statsmodels.formula.api as smf
pa = pd.read_csv(O / "drug_checks/median_predictions.csv", index_col=0)["No_treat"]
pl = pd.read_csv(O / "rescore_params43/both/Training CV/STABL ALasso/STABL ALasso predictions.csv", index_col=0)["0"]
out = []
for lab, pr in [("aligned (row-order fixed)", pa), ("lab-script rescore, bug-affected (what CLAUDE.md quoted until 2026-10-06; NOT USED)", pl)]:
    dd = pd.DataFrame({"DOS": y0, "EGA": clin.EGA, "pred": pr}).dropna(); dd["woman"] = dd.index.str.split("_").str[0]
    rx = smf.ols("pred ~ EGA", dd).fit().resid; ry = smf.ols("DOS ~ EGA", dd).fit().resid
    f = smf.ols("DOS ~ EGA + pred", dd).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd.woman)[0]})
    out.append({"predictions": lab, "n": len(dd), "R2 vs DOS": r2_score(dd.DOS, dd.pred), "r(pred, DOS)": np.corrcoef(dd.pred, dd.DOS)[0, 1],
                "r(pred, EGA)": np.corrcoef(dd.pred, dd.EGA)[0, 1], "partial r given EGA": np.corrcoef(rx, ry)[0, 1],
                "pred p in DOS ~ EGA + pred (woman-clustered)": f.pvalues["pred"]})
show(pd.DataFrame(out).set_index("predictions"), "beyond EGA, on out-of-fold predictions: partial correlation and the clustered test of "
     "pred in DOS ~ EGA + pred (the in-sample R2 of that OLS is not shown)")
''')
md(TEXT["B4"])

md("### B5. The gestational-age-free target `DOS_offset` = DOS − 7·(EGA − 40)")
code(r'''
o = n2[n2.target == "DOS_offset"].copy(); o["RMSEv"] = o.RMSE.map(first)
show(o.pivot_table(index="run", columns="model", values=["R2v", "RMSEv"]), "20 splits, Stabl pooled scoring (source night2/night2_scores.csv)")
show(pd.read_csv(O / "night2/clinical_offset_n20.csv").set_index("model"), "clinical baselines on DOS_offset, 20 splits")
''')
md(TEXT["B5"])

md("### B6. Preterm AUROC (paper 0.73)")
code(r'''
def auc(yv, s):
    r = rankdata(s); n1 = yv.sum(); n0 = len(yv) - n1
    return (r[yv == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def perm_p(yv, s, n=20000):
    a = auc(yv, s); null = np.array([auc(RNG.permutation(yv), s) for _ in range(n)])
    return (np.sum(np.abs(null - .5) >= abs(a - .5)) + 1) / (n + 1)
idx = pa.index
E = clin.loc[idx, "EGA"]; PRE = (clin.loc[idx, "Delivery"].str.strip() == "preterm").astype(int).values
G1 = (clin.loc[idx, "Timepoint"] == "G1").values; Dv = y0.loc[idx]
print(f"{PRE.sum()} preterm samples from {idx[PRE == 1].str.split('_').str[0].nunique()} women; G1: {PRE[G1].sum()} vs {(1 - PRE[G1]).sum()}")
rows = []
predA = 7 * E - pa; predB = 7 * E - pl.reindex(idx)
for sub, m in [("all visits", np.ones(len(idx), bool)), ("G1 only", G1)]:
    yv = PRE[m]
    cands = {"due-date rule (everyone at 40 wk)": np.zeros(m.sum()),
             "paper formula, DOS model that knows nothing": -(7 * E[m] - Dv.mean()).values}
    for lab, predL in [("ALIGNED", predA), ("lab-script, bug-affected, NOT USED", predB)]:
        cands[f"paper formula, CellOT Fig 5b prediction ({lab})"] = -predL[m].values
        cands[f"  same, EGA trend removed within subset ({lab})"] = -(predL[m] - np.polyval(np.polyfit(E[m], predL[m], 1), E[m])).values
    for n_, s in cands.items():
        rows.append({"subset": sub, "score": n_, "preterm/total": f"{yv.sum()}/{len(yv)}",
                     "AUROC": roc_auc_score(yv, s) if np.ptp(s) > 0 else 0.5,
                     "PR-AUC": average_precision_score(yv, s) if np.ptp(s) > 0 else yv.mean(), "chance PR-AUC": yv.mean(),
                     "perm p": perm_p(yv, s, 20000) if (sub == "G1 only" and np.ptp(s) > 0) else np.nan})
show(pd.DataFrame(rows), "preterm AUROC on the aligned CellOT out-of-fold predictions (n = 177; 14 preterm samples / 5 women)")
''')
md(TEXT["B6"])

md("### B7. Per-woman longitudinal prediction (`ool_longitudinal`, leave-one-woman-out)")
code(r'''
rows = []
for fs in ["freqmed", "sim", "freqmed_sim"]:
    s = pd.read_csv(LGT / f"m012_{fs}/scores.csv")
    for st_ in ["women with G1+G2+G3", "women with G1+G2"]:
        x = s[s.set == st_]
        c = x[x.model.str.startswith("constant")].iloc[0]; mdl = x[~x.model.str.startswith(("constant", "clinical"))]
        b = mdl.loc[mdl.RMSE.idxmin()]
        rows.append({"features": fs, "women": st_, "n": int(c.women), "constant RMSE (d)": c.RMSE, "clinical RMSE": x[x.model == "clinical"].RMSE.iloc[0],
                     "best model RMSE": b.RMSE, "best model": f"{b.model} / {b.visits}", "p vs constant": b.p})
show(pd.DataFrame(rows), "target = gestational age at delivery per woman (source ool_longitudinal/work/m012_*/scores.csv)", 2)
''')
md(TEXT["B7"])

md("### B8. The drug arm on DOS: predicted shift per drug, by drug-map chain (36_G1 / 52_G1 excluded)")
code(r'''
CH = {"cellot_cellotdrug": "CellOT stims + CellOT drug maps (paper chain)", "cellot_qmapdrug": "CellOT stims + quantile_map drugs",
      "cellot_mshiftdrug": "CellOT stims + median_shift drugs", "gauss_gaussdrug": "gauss_ot stims + gauss_ot drugs (no NN)",
      "gauss_qmapdrug": "gauss_ot stims + quantile_map drugs"}
T = {}
for k, lab in CH.items():
    s = pd.read_csv(O / f"drug_models/excl/{k}/summary_by_drug.csv").set_index("drug")
    T[lab] = s
sh = pd.DataFrame({lab: s.drug_mean_shift for lab, s in T.items()})
agree = pd.DataFrame({lab: s[["frac_folds_negative", "frac_folds_positive"]].max(1) * 100 for lab, s in T.items()})
impl = pd.DataFrame({lab: s.frac_treated_beyond_p95_untreated * 100 for lab, s in T.items()})
show(sh.sort_values(sh.columns[0]), "mean predicted shift (days; negative = labour predicted later) (source onset_test/drug_models/excl/*/summary_by_drug.csv)", 1)
show(agree.loc[sh.sort_values(sh.columns[0]).index], "% of the 100 fold models agreeing in sign", 0)
show(impl.loc[sh.sort_values(sh.columns[0]).index], "% of treated women beyond the untreated 95th percentile (implausible)", 0)
ref = sh.iloc[:, 0]
print("Spearman of the drug ranking vs the paper chain:", {c: round(spearmanr(ref, sh[c]).correlation, 2) for c in sh.columns[1:]})
nul = T[list(CH.values())[0]][["drug_mean_shift", "null_sd_of_mean_shift"]].assign(**{"shift / null SD": lambda x: x.drug_mean_shift / x.null_sd_of_mean_shift})
show(nul.sort_values("shift / null SD"), "paper chain: shift vs a size-matched random perturbation (check c)", 2)
''')
md(TEXT["B8"])

md("### B9. Compute cost")
code(r'''
tm = pd.read_csv(W / "timing/cellot_train_minutes.csv")
t = tm.groupby("kind").minutes.agg(["count", "median", "sum"]); t["CPU-h"] = t["sum"] / 60
show(t, "CellOT training time from the paper's files (config.yaml mtime -> cache/last.pt mtime, 1 core; source timing/cellot_train_minutes.csv)", 1)
gd = pd.read_csv(W / "timing/gaussot_drug_fit_seconds.csv")
gh = (gd.elapsed_seconds * gd.cpus).sum() / 3600
print(f"CellOT total {t['CPU-h'].sum():,.0f} CPU-h vs gauss_ot drug maps (1764) {gh:.1f} core-h "
      f"({t['CPU-h'].sum() / gh:,.0f}x); gauss_ot stim maps ~10 min fit (CLAUDE.md, not timed in a CSV)")
''')
md(TEXT["B9"])

# =====================================================================================================================
md(TEXT["removed"])
md(TEXT["contradictions"])
md(TEXT["sources"])

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}
nbf.write(nb, HERE / "summary_all_results.ipynb")
print("wrote", HERE / "summary_all_results.ipynb", len(cells), "cells")
