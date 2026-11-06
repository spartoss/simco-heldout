# OOL longitudinal: one delivery-date prediction per woman from her G1 / G2 / G3 visits

Separate from `simco` (reads its feature tables in `simco_work/` read-only, writes only under `work/`).

**Question.** The OOL women were sampled up to three times (G1 ~28 wk, G2 ~34 wk, G3 ~37 wk). The paper treats every
visit as an independent sample predicting days-to-labour (DOS). Here the target is per woman, her **gestational age at
delivery** (`GA_del` = 7·EGA − DOS, identical across her visits to ~0.2 d), and the visits are used together: as
successive updates (G1, then G1+G2, then all three) or as a trajectory.

**Data** (2026-10-01). 63 women / 175 visits after removing 36_G1 and 52_G1: 51 women with 3 visits, 10 with 2, 2 with 1.
GA at delivery 39.0 ± 1.6 wk (34.0–41.2). **Only 5 preterm women** (34.0–35.1 wk), so preterm is reported descriptively;
the main outcome is the continuous delivery date, whose baseline is the constant 39.0 wk (~11 d error).

**Traps.**
- Visit timing leaks the outcome: a preterm woman's last visit is just before delivery, and she has none after. The
  honest evaluation is **landmarking**: at t weeks use only visits before t, score only women still pregnant at t.
- Everything is leave-one-woman-out: a held-out woman's visits are all in the test set together.

**Methods** (ladder, cheapest first)
- M0 `lgt/m012.py`: constant and clinical baselines.
- M1: per-visit ElasticNet; features as measured + EGA (`raw`) or detrended on gestational age (`dev`, "deviation from the clock").
- M2: per-woman updating, equal mean or Bayes precision weights.
- M3 (next): trajectory features (G1 level + changes) and mixed-effects summaries.
- `lgt/evaluate.py`: per-woman RMSE / MAE / Spearman / preterm AUROC per visit set and per landmark, paired vs constant.

Features: measured Unstim only to start (`freqmed`: log frequencies + 31-marker medians), no CellOT map.

## First results (2026-10-01, measured `freqmed` features, leave-one-woman-out)
- **No signal.** The per-visit ElasticNet keeps 0 features in nearly every fold, so every visit set (G1, G2, G3,
  G1+G2, G1+G2+G3, mean or Bayes) predicts the training mean: RMSE 10.95 d vs 10.95 d for the constant (51 women with
  three visits), 11.39 vs 11.38 (59 women with G1+G2). Leak-free landmarks at 30 / 33 / 35 wk: the same.
  Ridge (no sparsity) is worse than the constant (14.7 d); PCA-10 + ridge 11.3 d.
- **Trajectories (M3) do not help either.** G1 level + G1→G2 change: 11.87 vs 11.40 d. Adding G2→G3: one absurd
  prediction (woman 112, the woman who is atypical on every visit: −1029 d); without her 12.5 vs 11.0 d. The preterm
  women are predicted at term.
- Consistent with simco's night2 `DOS_offset` runs (the same target per visit: R² ≤ 0 for every feature set).

## With the paper's simulated features (2026-10-02, jobs 46219989)
`sim` (the paper's 852 CellOT stim medians) and `freqmed_sim` (+ measured freq / medians): identical to the measured-only
run. The ElasticNet keeps at most 1 feature per fold (0 in most), so every visit set and every landmark predicts the
training mean (RMSE 10.95 vs 10.95 d; 11.39 vs 11.38 d). Trajectories: `sim` 11.45 vs 11.40 (G1+G2), 11.01 vs 10.96
(all three, 0 features kept); `freqmed_sim` 11.83 vs 11.40, and with G2→G3 one absurd prediction again (155.8 d RMSE).
The model rows' Spearman of −1 / AUROC 0 are the same leave-one-out artefact as the constant's: a model that falls back
to the training mean ranks women backwards.
