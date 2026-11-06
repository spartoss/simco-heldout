# well_ceiling — is the 0.60 learning-curve plateau a model limit or a data limit?

> **Scale note.** This is the project's working record, written on the skill scale (1 − |error| / |effect|, median per repeat). The summary in the top-level README and in `figures/` uses MSE / RMSE throughout.

Self-contained project (2026-10-06). Reads the atlas (group home) and atlas_heldout's learning-curve outputs read-only;
writes only here. Results and readings: **`notebooks/ceiling.ipynb`** (built by `notebooks/build_ceiling.py`).

## Question
Unstim → stim maps evaluated on held-out donors (atlas_heldout, 5 repeats × 4 test donors) plateau at skill ≈ 0.60 at
k = 15 (CellOT 0.600, gauss_ot 0.595, quantile_map 0.538, median_shift 0.512). The quoted ceiling, 0.87, splits one
pooled measurement into two random halves: cell-sampling noise only. Each (donor, stim) was measured in three
plate-control wells (DMSO1/2/3). How well does one well predict another?

## Method
- Skill exactly as atlas_heldout `lc/score.py` (reimplemented): case = (test donor, stim, celltype, q10–q90, 13
  functional markers); effect = |stim quantile − Unstim quantile| on the pooled wells (≤ 10k cells); skill =
  1 − err / effect; cases with effect > 0.0273; median per repeat, mean over the 5 repeats.
- `wc/extract.py` re-reads `ptb_concatenated_batchcorrected/<ct>_HVPV.h5ad` and takes **exactly** the learning-curve
  cells (same seed and order; `wc/validate_cache.py`: equal to 2e-7), keeping the well label; plus up to 30k
  cells per well. atlas_heldout had no cross-well ceiling and its cache pools the wells without labels.
- `wc/ceiling.py`: split-half (validation), well → well (all ordered pairs; PI without DMSO1, reported separately),
  pseudo-wells (same sizes, random cells: sampling only), source well changed too, plate-matched delta transfer.
- `wc/compare.py`: scores the k = 15 model predictions (atlas_heldout `work/lc/where_skill/cases.pkl`) and the
  replicates **on the same targets** (pooled, or one single well), plus an ideal-model estimate under independent
  equal-variance well noise (error = |wA − wB|/√2 against one well, /√6 against the pool of three), checked by
  the mean-of-two-wells prediction (predicted 0.619, measured 0.608).

## Result
| | skill |
|---|---|
| split-half ceiling: atlas_heldout / this code | 0.872 / 0.872 |
| **replicate-well ceiling** (one well → another, same donor and stim) | **0.559** (pseudo-wells of the same size 0.825; 30k cells/well 0.561) |
| models, pooled target: CellOT / gauss_ot / quantile_map | 0.596 / 0.595 / 0.538 |
| same models against one single well | 0.443 / 0.443 / 0.344 |
| mean of the two other wells → one well | 0.608 |
| ideal model (estimate): one-well target / **pooled target** | 0.689 / **0.733** |

Per stim (replicate well; CellOT pooled; ideal pooled): GMCSF 0.55 / 0.57 / 0.76, IFNa 0.69 / 0.70 / 0.83,
IL246 0.60 / 0.59 / 0.77, IL33 −0.07 / 0.11 / 0.36, LPS 0.62 / 0.60 / 0.80, PI (DMSO2/3) 0.65 / 0.71 / 0.70,
TNFa 0.32 / 0.42 / 0.62; PI with DMSO1 0.44.

The literal replicate ceiling (0.56) is below the plateau only because a replicate is scored against one noisy well
while the models are scored against the pool of three wells. On a common target the models are clearly worse than a
replicate (0.44 vs 0.56). The realistic ceiling for the learning curve's pooled target is ~0.73: about half the
0.60 → 0.87 gap is plate/well noise no model can recover, and the other half (0.60 → 0.73) is room the models leave
(0.07–0.20 per celltype, median 0.14; ~0 for PI on the pooled target, small for TNFa and IL33).

## Decision
**Not at the data limit → the "clearly > 0.6" branch applies.** Next test: **cross-stim transfer** (the donor's
measured response to one stim as input to predict another), aiming at ~0.73 on the pooled score. Deprioritise IL33 and
TNFa (mostly plate noise) and handle PI's DMSO1 plate before training.

## Layout
`wc/` code (`paths`, `extract`, `validate_cache`, `ceiling`, `compare`), `work/summary/` tables (`*.pkl` case tables
gitignored), `work/quant/` per-well quantiles (gitignored, regenerate with `wc.extract`), `notebooks/`, `STATUS.md`.
