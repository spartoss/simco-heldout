# 1. Held-out-donor work on the atlas perturbation maps

> **Scale note.** This is the project's working record, written on the skill scale (1 − |error| / |effect|, median per repeat). The summary in the top-level README and in `figures/` uses MSE / RMSE throughout.

**Final result (leave-one-donor-out, 19 donors, 5 stims, selected features q50–q90):** the best predictor of an unseen
donor's stimulated quantiles is **0.4 × CellOT conditioned on phenotype subclusters + 0.6 × (average response + TabPFN
residual)**, MSE 0.0305 vs 0.0398 for the paper's CellOT (−23 %) and 0.0363 for paired CellOT + shift (18/19 donors
better). The TabPFN table per (stim, cell type) uses the donor's raw own-cell-type resting profile, cell-type
frequencies and the same marker × quantile measured in the donor's 27 other cell types; a 72-column version ties it.
Code: `lc/tab_v2.py`, `lc/tab_ens.py`, `lc/final_compare.py`; notebooks `tab_models.ipynb`, `final_comparison.ipynb`.
Drug arm on held-out donors: `lc/drug_tab.py`, `lc/drug_step.py`, `notebooks/drug_heldout.ipynb`.

(Project folder originally named `atlas_heldout`.)

A separate project from `simco/` (it reads simco's data and Stabl scripts read-only and never writes there). The old
paths `/scratch/users/bmerwan/cellot_lc` and `cellot_lc_work` are symlinks to this directory and `work/lc`.
It holds its own copy of the ICNN (`lc/reference_icnn.py`, from simco `cellot/networks/icnns.py`) and a batched GPU
trainer (`lc/batched_icnn.py`, `lc/train.py`) that trains many CellOT maps at once, exactly (`lc/test_equivalence.py`).
Closed-form baselines (`lc/baselines.py`: gauss_ot, quantile_map, median_shift) are vendored from simco.

- `env.sh`: GPU training (cluster module PyTorch 2.4 / CUDA 12.6). `env_cpu.sh`: caches, baselines, scoring, notebooks
  (simco `venv-cellot`, read-only). Stabl runs use simco's `venv-stabl`.
- Run state and job IDs: `work/lc/STATUS.md` (learning curve, donors) and `work/STATUS.md` (heldout/).

## Contents
| part | code | results / notebook |
|---|---|---|
| learning curve: how many donors / cells a stim map needs for a new donor | `lc/` (design, cache, train, baselines, score) | `notebooks/learning_curve.ipynb` |
| donor experiments: HV vs PV, pairwise good/hard donors, cells vs donors in small celltypes, pDC/mDC budgets | `lc/design_donors.py`, `lc/analyze_donors.py`, `lc/hvpv_features.py`, `lc/small_budget.py` | same notebook, `work/lc/donors_summary/` |
| drugs on held-out donors (step / chain / feature-level) | `heldout/drug_cache.py`, `heldout/score_drugs.py`, `heldout/analyze_drugs.py` | `work/heldout/drugs/analysis/` |
| quantile feature sets for Stabl (9 sets, 100 splits, aligned rescore) | `heldout/inclusion.py`, `quantile_arms.py`, `selq_corr.py`, `analyze_quantile_arms.py`, `quantile_selection_report.py` | `work/heldout/quantile_arms/summary/`, simco `notebooks/quantiles.ipynb` §10 |
| phenotypic markers / gate crossing under stims and drugs | `heldout/pheno_summary.py`, `build_phenotype_notebook.py` | `notebooks/phenotype_shifts.ipynb` |
| atlas vs OOL input shift (maps applied off their training range) | `heldout/atlas_vs_ool.py`, `build_atlas_vs_ool_notebook.py` | `notebooks/atlas_vs_ool.ipynb` |
| where CellOT is good / weak on new donors, shared vs patient-specific part, why the plateau, gated hybrid | `lc/where_skill.py`, `lc/build_where_skill_notebook.py`, `lc/where_skill_readings.md` | `notebooks/where_cellot_skill.ipynb`, `work/lc/where_skill/` |
| patient covariate in the map: closed-form, conditional CellOT (`--ctx pc / ft / cluster`, `--ctx-wd`), residual beyond the shift | `lc/cond_baselines.py`, `cond_icnn.py`, `cond_train.py`, `ctx_score.py`, `residual_predict.py` | `work/lc/cond_*`, `work/lc/ctx_score/`, `work/lc/STATUS.md` |
| ImmuneFM (cytometry foundation model) as an encoder: sanity, donor embeddings, fine-tuning on the atlas, phenotype subclusters | `fm/` (see below; weights not included) | `work/lc/fm/` |
| tabular residual models beyond the average response (TabPFN v2, LightGBM): feature sets, lineage features, ablations, column reduction, leave-one-donor-out | `lc/tab_features.py`, `lc/tab_v2.py`, `lc/tab_ens.py`, `lc/tab_compare.py` | `notebooks/tab_models.ipynb` |
| final comparison of every method on leave-one-donor-out, noise floor | `lc/final_compare.py`, `lc/noise_floor_lodo.py` | `notebooks/final_comparison.ipynb` |
| drug effects on held-out donors (TabPFN, step setting with the measured control) | `lc/drug_tables.py`, `lc/drug_tab.py`, `lc/drug_step.py` | `notebooks/drug_heldout.ipynb` |

## ImmuneFM (`fm/`, 2026-10-05)
Vendored from github.com/buttelab/ImmuneFM (Ding, Bhattacharya, Butte, bioRxiv 2025.07.09.664020). **The repo has no
licence: `fm/ckpt/` is for internal research only, do not commit or redistribute it.** `fm/immunefm.py` (model +
`encode`), `fm/tokenize.py` (published per-cell tokens, and per-marker quantile tokens), `meta/fm_marker_map.tsv`
(33/38 panel markers), `fm/sanity.py` (phase-0 gate), `fm/counts.py`, `fm/embed_donors.py`, `fm/finetune.py`,
`fm/stratified.py`; 38-marker cache = `lc.cache --all-markers` -> `work/lc/fm/cache_full/` (same cells as `cache/`).
Conclusion: frozen ImmuneFM is unusable here (its tokens miss 63 % of stim responses; it reconstructs masked markers
worse than per-marker frequencies; embeddings <= raw markers). A small transformer trained on the atlas (masked
markers, 64 quantile bins per marker) gives a better donor covariate than raw quantiles, and starting it from the
ImmuneFM weights adds almost nothing over random init: use the random-init ("from scratch") encoder, which needs no
third-party weights (`fm/finetune.py --init random`).
Follow-up (2026-10-06, `lc/ctx_score.py`, clean repeats 1-4): used as a regularised donor context in conditional
CellOT (`--ctx ft --ctx-wd 1`), neither encoder beats the context-free paired CellOT (k15 skill 0.622 / 0.619 vs
0.621); phenotype subclusters (`--ctx cluster`) tie as well. The shift baseline keeps the best dev_R2.

## Patient covariate in the map (design note, 2026-10-05; carried out, results above and in `lc/cond_*`)
Question: can a map that knows *which kind of donor* a cell comes from predict a new donor better? (Learning curve:
plateau ~0.6 vs split-half ceiling ~0.87; pairwise experiment: donors with similar baselines predict each other better.)
Still one map per (stim, celltype). Covariate choices:
- **Donor ID embedding**: cannot generalise to a new donor → use only as an *oracle diagnostic* in the pooled protocol
  (how much of the response is donor-specific at all).
- **Donor Unstim summary** (generalisable): (1) same-celltype 13-marker medians / quantiles → 2–5 PCs; (2) whole-blood
  profile (frequencies + medians of all celltypes); (3) a learned population embedding (Meta Flow Matching); (4) clinical
  (HV/PV, gestational age).
Plan, cheapest first, on the learning-curve protocol (5 repeats × 4 held-out donors, skill vs plateau/ceiling, plus
whether predicted between-donor differences follow the real ones, and HV/PV transfer):
1. CPU baselines: donor-kNN gauss_ot (training donors weighted by Unstim similarity to the test donor) and conditional
   gauss_ot (shift and 13×13 matrix ridge-linear in the covariate). Note: simco's `linear_median` (patient-adaptive
   median shift) gained nothing.
2. Conditional CellOT, CondOT-style (partially input-convex ICNN with a context branch) added to `lc/batched_icnn.py`,
   same GPU training (20k iterations).
3. Donor-ID oracle in the pooled protocol.
4. Meta Flow Matching only if 1–2 show a covariate helps.
Risks: 15 training donors → memorisation; keep the covariate low-dimensional and regularised.
Candidate covariate (2026-10-08, `notebooks/donor_covariate.ipynb`): 3 regularised-LDA axes of the 28 Unstim log
frequencies, built to maximise between-donor / within-donor variance; recognises held-out donors from one Unstim well
(100 % main stims, 81 % PI with 3 axes) and stays with OOL women across visits (ICC 0.92–0.95). Scores in
`work/heldout/donor_covariate/final_freq_g0.01_{atlas,ool}_scores.csv`.
References: CondOT (Bunne, Krause, Cuturi, NeurIPS 2022, github.com/bunnech/condot); Meta Flow Matching (Atanackovic et
al., ICLR 2025, github.com/lazaratan/meta-flow-matching, embeds the starting population, built for unseen patients);
CellFlow (Theis lab 2025, github.com/theislab/CellFlow, JAX, donor-specific cytokine responses in PBMCs).
