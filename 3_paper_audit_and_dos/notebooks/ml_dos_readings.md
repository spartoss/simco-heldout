## Summary — what we did and what came out (2026-10-07)

**Rule of the comparison (user, 2026-10-07):** only the data CellOT + Stabl used — each visit's measured
**unstimulated** OOL cells. No 15-min stimulations, no clinical variables, no gestational age or visit.

**What we did.**
1. Built one measured feature table from each visit's gated Unstim cells, with no CellOT and no simulation
   (`ml_dos/build_features.py`, 177 × 8288): celltype frequencies, the 10/25/50/75/90th percentiles, mean and SD of
   all 31 markers in 28 celltypes, and the correlations between the 13 functional markers. Same cells as the paper's
   Unstim block, whose 13 functional medians it reproduces (4.7e-7).
2. Ran ~25 learners on it — linear (ridge, lasso, elastic net, PLS, PCR, linear SVR), kernel (RBF SVR, kernel ridge,
   kNN, Gaussian process), bagging (random forest, extra trees, bagged trees, bagged ridge), boosting (XGBoost with the
   paper's `params_43` and tuned, HistGradientBoosting, GradientBoosting, AdaBoost), neural (sklearn MLP, two torch deep
   ensembles, a DeepSets network reading the raw cells), and a stacked super learner.
3. Same 100 splits as the paper (grouped by woman, so no woman is ever in train and test), `36_G1` / `52_G1` removed,
   every hyperparameter tuned inside the training women only.
4. Reran the paper's CellOT + Stabl model (0.415) and CellOT + measured + Stabl (0.448) through the same code, so every
   comparison is paired split by split.
5. Control: the same learners on the CellOT table, to tell "better features" from "better learner".

**`auto`** = in each split, the model with the lowest internal cross-validation error *on the training women only*,
then used once on the test women. It is the honest "best model of the zoo": picking the top row of a table would
choose on the test results.

| | R² | typical error (RMSE) | vs CellOT + Stabl, per split |
|---|---|---|---|
| constant (no information) | 0.00 | 31.3 d | |
| CellOT + Stabl (paper) | 0.415 | 23.8 d | |
| CellOT + measured + Stabl | 0.448 | 23.1 d | |
| **measured Unstim cells only (`full`), `auto`** | **0.545** | **21.0 d** | +0.085, better in 67/100, Holm p 0.006 |
| DeepSets on the raw Unstim cells (no hand-made features) | 0.473 | 22.6 d | +0.08, better in 72/100, Holm p 0.002 |
| gestational age alone (not allowed — a pure clock) | ≈ 0.88 | ≈ 11 d | |

- **Measured Unstim cells, with a dense learner, beat CellOT + Stabl** by ~3 days of error, without simulation.
  18 of the 25 runs on `full` are significantly better per split (Holm). The best families use many weak features at
  once: neural networks, kernel methods, the stacked ensemble (0.51–0.55); random forest 0.39, lasso 0.35.
- **It is the features, not only the learner.** The same dense learners on the paper's CellOT table give 0.28–0.39 —
  *below* CellOT + Stabl — and adding CellOT's simulated stims to the measured table does not help (§8). Stabl's sparse
  selection suits the CellOT table; the measured table needs a dense learner (Stabl + XGBoost on measured quantiles gave
  only 0.37 before).
- **The gain is beyond gestational age**: partial correlation of the prediction with DOS given EGA 0.29–0.33 for the
  measured models vs 0.17 for CellOT + Stabl (§5). Still far below EGA alone.
- Richer summaries help: `full` > `qtl` > `freqmed` for most dense learners (auto 0.545 / 0.486 / 0.474).

**Interpretability.**
- *Readable directly*: linear models (ridge, elastic net, PLS — one weight per feature, or a few components) and tree
  ensembles (feature importances). They agree on where the signal is: **naive CD4 T cells, pSTAT3/pSTAT5** — the
  pSTAT3–pSTAT5 correlation, pSTAT3 median, pSTAT1 upper tail — then naive CD8 T and central-memory CD4 T on the same
  STAT axis (§6). That is the axis that carries the paper's model too, now from measured cells.
- *Not readable directly*: the best models (kernels, Gaussian process, neural networks, stack, `auto`). They work by
  combining hundreds of weakly informative features (the kernel models keep the 200–500 most correlated each split),
  so the signal is diffuse — which is also why Stabl, built to keep ~20 features, does worse here. Explaining them needs
  post-hoc tools (grouped permutation importance, SHAP), which become unstable with thousands of correlated features;
  importance per celltype or per feature block is the realistic level.
- Trade-off in one line: CellOT + Stabl gives ~20 named features and R² 0.415; the dense models give R² 0.55 and an
  explanation at the level of "which celltypes / which markers", not a short list.

**Corrections.** (1) A bug made the first `auto` the median over all models rather than a per-split choice (fixed;
corrected numbers above). (2) An earlier version of this notebook also reported models with the OOL 15-min
stimulations and clinical variables (R² up to 0.61); they are outside the comparison and removed. (3) `params_43` is
no longer described as the best of 97 configs (CLAUDE.md, 2026-10-06).

**Learner-vs-features control complete (2026-10-08).** All 8 learners on the CellOT table give 0.28–0.39, the two
deep networks included (`dnn` 0.393, `dnn_wide` 0.387; vs CellOT + Stabl −0.03 / −0.04 per split, 41 / 37 of 100,
n.s.). On measured + CellOT stims the deep networks give 0.536 / 0.544, the same as on measured alone (0.535 / 0.546).

**Preterm detection, no refit (§9, 2026-10-08).** Rule: predicted pregnancy length L = 7·EGA − predicted DOS;
preterm if L < 37 weeks. 5 preterm women (14 visits), all 34–35 weeks. Over all visits even a no-information model
scores 0.71 (preterm women are sampled earlier), so read the EGA-detrended score or G1:

| | detrended AUROC (p) | G1 AUROC (p) | preterm visits / women caught at 37 wk | term visits correct |
|---|---|---|---|---|
| CellOT + Stabl (paper) | 0.65 (0.10) | 0.74 (0.04) | 57 % / 4 of 5 | 80 % |
| wide deep network | **0.82 (0.002)** | 0.78 (0.02) | 71 % / 4 of 5 | 83 % |
| kernel ridge / SVR | 0.84 / 0.83 | 0.78 / 0.78 | 71 % / 57 % | 78 % / 79 % |
| stacked ensemble / auto | 0.78 / 0.80 | 0.78 / 0.79 | 71 % | 80 % / 81 % |
| constant DOS (no information) | — (no variation left) | 0.65 (0.14) | | |

The measured-feature models separate preterm from term better than CellOT + Stabl once visit timing is removed
(0.78–0.84 vs 0.65), but with **5 preterm women** the CIs are wide and one woman changes the picture; these are
leads, not a validated classifier, and the model (chosen on DOS R²) was not tuned for preterm.

## Possible next steps

1. **Interpretation of the best models**: grouped permutation importance (by celltype, by marker, by block) for `stack`
   and the GP on `full`, and a check that a small, readable model (PLS with few components, or elastic net restricted
   to the top celltypes) keeps most of the R².
2. **External validation.** Fix one model now (e.g. `stack` on `full`) and test it unchanged on an independent cohort
   — PRINCE (`outcome_table_prince*.csv`), if its Unstim cells can be summarised the same way.
3. **Robustness to the split draw**: repeat the pre-specified model with other `random_state`s.
4. **The gestational-age-free target** `DOS_offset`, where Stabl found nothing.
5. **Preterm at G1** with the new predictions (5 women).
6. **Drug arm** with the Unstim-only model, perturbing its measured features through the Unstim→drug maps.
