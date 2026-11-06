<!-- ANSWER -->
## Answer (2026-10-06, updated 2026-10-07 with §7–§8)

**Why the plateau.** At k = 15, the error left by the shift baseline is **95 % real donor-to-donor differences in
the response and only 5 % sampling noise** (§2). So the plateau is not a measurement limit. Each donor responds
differently, and that difference is mostly **not visible in her unstimulated cells**:
- CellOT captures **4.6 %** of it, gauss_ot −4 %, quantile_map −36 %;
- explicit regressions on the donor's Unstim profile (this celltype, or the whole blood) capture 3–8 % (`work/lc/residual_predict`);
- donor contexts (PCs, a fine-tuned encoder, phenotype clusters) add nothing (`work/lc/ctx_score`).

CellOT's donor-specific predictions point the right way (correlation 0.30) but are too weak and too noisy to remove much error.
Going from 3 to 15 training donors moves the specific part only from −3 % to +5 %. This is an **information
limit of the input**, not a model-capacity limit.

**Where CellOT is clearly above the plateau** on the patient-specific part (specific R², positive in most of the 5
repeats):
- **IL-2/4/6 in effector/memory CD8, NKT and CD56hi NK cells:** CD8 Th1 0.51, CD8Tcm CCR2+ 0.47, NKT 0.43,
  CD56hi NK 0.41, CD8Tem CCR2+ 0.39;
- **TNFa in naive/memory T cells:** CD4Tnaive 0.42, CD4Tcm 0.42, CD4Tem 0.37, CD8Tnaive 0.33. These are positive in
  5/5 repeats and tied with gauss_ot;
- **IFNa in CD4Tnaive (0.34) and Granulocytes (0.31); LPS in CD8 Th1 (0.34) and CD4Tnaive (0.31).**

By marker it is the **canonical pathway readout of each stim**:
- pSTAT1 (0.27 overall; IFNa 0.33, LPS 0.31);
- pSTAT6 under GMCSF (0.53) and IFNa (0.34);
- pSTAT3 under TNFa (0.38) and GMCSF (0.30);
- pERK under LPS (0.33).

So where the stim triggers a strong, specific signalling response in a well-defined lymphoid population, the donor's
own response is partly predictable from her Unstim cells.

**Where it is clearly below a plain shift, and why:**
- **PI (−0.09 overall), in monocytes, mDCs, NK cells and CD4 Tregs/Teff.** This is gating, not biology: 23 of the 168
  maps have populations that change size ≥ 1.4× under their stim. Their median specific R² is −0.05, against +0.09
  for stable populations. These include PI cMCs −0.46 (monocytes lost), PI mDCs −0.29 (mDCs 5×) and PI CD4Teff −0.20.
  The "response" there is partly a change of who is in the gate, which the donor's Unstim cells cannot predict.
- **Test donors outside the training range:** where her Unstim value is > 2 training SDs away, CellOT is worse than
  the shift (−0.17 at 2–3 SD, −0.45 beyond 3 SD). It extrapolates with too large an amplitude (SD ratio 1.15). This
  is the out-of-sample failure, and it is what the OOL women face (their inputs sit 2–5 atlas SDs off, see
  `atlas_vs_ool.ipynb`).
- **IL-2/4/6 in CD11c− NK cells, DN T cells, CD56lo NK and B cells (−0.15 to −0.69):** CellOT predicts large donor
  differences (SD ratio 0.86–1.04) with no relation to the truth (correlation 0.04).
- **CD25 in every stim (−0.18; LPS −0.90):** mostly noise (ceiling 0.64), but CellOT predicts a large spread.
- **IL33:** nothing real to predict (ceiling 0.64).
- **Small responses** (mean response < 0.1): negative, while responses > 0.5 reach +0.07. **Pregnant test donors**
  0.03 vs 0.07 for HV. **Few cells**: the bottom cell quartile 0.02 vs 0.09.

**What gets past the plateau now (§5), without a new model:** keep the shift for the average response, add CellOT's
donor-specific prediction only where it held on the other repeats ("gated"), or scaled by a factor learnt on the
other repeats.
- Skill: 0.673 / 0.676, against 0.651 for CellOT and 0.664 for the shift.
- Error: 13 % below CellOT, 7 % below the shift.
- Specific R²: 0.10.
- It beats CellOT for 17 of 20 donor-repeats, but the shift for only 10–12 of 20.

So the gain over a no-network shift is real but small.

**What could move it further.** Each follows from where the signal is and is not:
- **shrink** the donor-specific output, especially for out-of-range donors (amplitude, not direction, is what hurts);
- **don't trust maps on populations that change size** under the stim; handle them at the gate/frequency level;
- **more donors**: slow;
- **an input that carries the donor's responsiveness.** The Unstim cells barely do; a measured response to one stim
  would be the obvious candidate. The cross-celltype correlation notebook found that donor responses are coupled
  across celltypes.

**Two retraining fixes tested (2026-10-07, §7–§8). Neither helps.** Both are paired tests over the 20 test donor-repeats. Skill is on the learning-curve scale, against CellOT 0.596, paired control 0.615 and shift 0.604.
- **Per-donor normalisation (§7).** Each donor's cells are expressed relative to her own Unstim median.
  - It removes the pooled-OT stretch: on stretched features the slope falls from 1.92 to 1.01, against a true 0.69.
  - But the donor-specific signal goes with it. Specific R² falls from +0.045 to −0.14, and the arm is worse than the paired control for 19/20 donor-repeats (skill 0.586; with clipping 0.587, 18/20).
  - Clipping test cells to the training range is harmless. It is better for 16/20 donor-repeats, but by only 0.9 % (skill 0.612). It does not fix out-of-range donors, where the shift stays clearly better.
- **CondOT-style sharing of one map across stims (§8).** The map is conditioned on the perturbation, not the donor.
  - Per celltype, it is worse than per-pair CellOT for 19/20 donor-repeats (skill 0.562).
  - One map for all 168 pairs is worse for 20/20 (skill 0.421, under-fit).
  - The loss is entirely in the average response. The donor-specific part is unchanged.
  - There is no gain for weak or small pairs either (IL33, TNFa, small responses, small celltypes).

So the plateau is not a property of how the per-pair map is set up. The best predictor remains a shift for the average plus a damped CellOT donor-specific part (§5; also the 0.5 CellOT + 0.5 shift ensemble, 0.627, in `mfm_heldout`).

**Caveats.**
- The 14 test donors appear in more than one repeat, so the cross-fitted gating is not fully donor-disjoint.
- Single (stim, celltype, marker) cells rest on 40–100 cases from 14 donors, so the top-20 list is a list of
  candidates.
- The "shared part" of 4 test donors also contains part of their own deviation from the population, so not all of it
  is fixable by a delta.

<!-- READ2 -->
**How to read §2.** First table: the mean squared error per case, split into the shared part (how far the 4 test
donors' average is from the prediction's average) and the patient-specific part. The last row is the sampling noise
in the truth, the part nobody can predict. Second: the specific-part statistics at k = 15. Third: the learning
curve, for overall skill and for each part.

**Reading.**
- **The shift's error is 0.0346; noise is 0.0017 of it (5 %).** The rest is real. The patient-specific part alone is
  0.0233, against 0.0017 noise, so the between-donor differences are 93 % real (ceiling 0.93).
- **CellOT's patient-specific error is 0.0222, hardly below the shift's 0.0233** (specific R² 0.046). Its shared part
  is worse than the shift's (0.0150 vs 0.0113), which is why its total error is higher.
- **The direction is right but the amplitude is not.** Correlation 0.30, SD ratio 0.51: CellOT predicts half the true
  between-donor spread, and only part of what it predicts is right.
- **Learning curve:** from k = 3 to 15 the shared error falls (0.138 → 0.106), which is what more donors mainly buy.
  The specific R² moves −0.03 → +0.05, still rising but slowly. gauss_ot stays negative.

<!-- READ3 -->
**How to read §3.** Specific R² by one factor, then the stim × celltype and stim × marker heatmaps (red = CellOT
captures real between-donor differences, blue = worse than the shift; numbers shown where |value| ≥ 0.25), then the
best and worst maps and the best single features. *repeats > 0* counts how many of the 5 repeats agree in sign.

**Reading.**
- **Stims:** IFNa 0.19, IL246 0.14, LPS 0.14 lead; GMCSF 0.08, TNFa 0.03; IL33 −0.03 (nothing to predict, ceiling
  0.64); PI −0.09.
- **Markers:** pSTAT1 0.27 is the best everywhere, then pCREB, pS6, IkB (0.11–0.13). pSTAT5, pSTAT6 and CD25 are
  negative overall, but pSTAT6 is the best marker *within* GMCSF (0.53) and IFNa (0.34). Its negative total comes
  from PI (−0.50).
- **Celltypes:** T and NKT subsets lead (CD8Tcm CCR2+ 0.28, CD8 Th1 0.25, NKT 0.18, CD4Tnaive 0.14). Myeloid and NK
  subsets trail (cMCs −0.18, DN T −0.19, CD11c− NK −0.14).
- **Quantiles:** q25 / q10 / median 0.08–0.11; q90 −0.01. The upper tail is where CellOT's donor-specific output is
  least reliable.
- **Best maps:** IL246 in CD8 Th1 / CD8Tcm CCR2+ / NKT / CD56hi NK / CD8Tem CCR2+ (0.39–0.51), TNFa in naive and
  memory CD4/CD8 T (0.33–0.42, 5/5 repeats), IFNa CD4Tnaive (0.34, 5/5), LPS CD8 Th1 / CD4Tnaive.
  - gauss_ot is as good on the TNFa T-cell maps (0.36–0.45). There the donor-specific signal is affine: CellOT is not
    needed for it.
  - CellOT is clearly ahead of gauss_ot on IL246 CD56hi NK (0.41 vs 0.17), IL246 CD8Tem CCR2+ (0.39 vs 0.14),
    IFNa CD4Tnaive (0.34 vs 0.12) and LPS CD4Tnaive (0.31 vs 0.14).
- **Best single features**, 40–100 cases each, so candidates:
  - **pSTAT6 under GMCSF in every myeloid population (0.59–0.77);**
  - pSTAT6/pSTAT5 under IL246 in NK/NKT/CD8 (0.60–0.75);
  - IkB under TNFa/PI in NK and T cells (~0.57);
  - two PI features (Granulocytes CD25 0.85, CD4Teff HLADR 0.83).

<!-- READ4 -->
**How to read §4.** Each table splits the k = 15 cases by one candidate explanation. *CellOT closer than shift* =
share of cases where CellOT's donor-specific error is smaller. The population-size table groups the (stim,
celltype) maps by how much the population grows or shrinks under the stim. Figure, left: by how far the test donor
is from the training donors. Right: one dot per map, x = log2 population size change.

**Reading.**
- **Out of range:** up to 2 training SDs CellOT adds a little (0.04–0.06, closer in 52–54 %). Beyond that it is worse
  than the shift: −0.17 at 2–3 SD, −0.45 beyond 3 SD (closer in only 39 %). Its SD ratio there exceeds 1, so it
  over-extrapolates. This is 6 % of cases here, but OOL inputs are 2–5 atlas SDs off on several phospho-markers.
- **Gating:** maps whose population changes size ≥ 1.4× have a median specific R² of −0.05, against +0.09 for stable
  ones. 16 of the 23 are PI. In the figure every good map (> 0.3) sits at a size change near 0.
- **Noise and cells:** the ceiling is high almost everywhere (0.84–0.99) except few-cell wells (0.84) and small
  responses (0.68–0.73). More cells helps a little (0.02 → 0.09).
- **Response size:** specific R² rises with the mean response (−0.05 for < 0.05, +0.07 for > 0.5). Strong
  stimulations carry more predictable donor variation.
- **Cohort:** HV test donors 0.07, PV 0.03; the direction is similar (correlation 0.28 vs 0.30).

<!-- READ5 -->
**How to read §5.** One row per way of predicting:
- *skill*: as in the learning curve;
- *specific R²*;
- *MSE*, and *MSE vs CellOT* (negative = less error than CellOT);
- *donor-repeats better than …*: in how many of the 20 test donor-repeats the mean squared error is smaller.

The gated and scaled versions are cross-fitted: the choice for repeat r uses only the other four repeats.

**Reading.**
- **Replacing CellOT's average with the shift's ("hybrid") is the cheapest fix.** Error falls 10 % below CellOT,
  skill rises 0.651 → 0.671, and it beats CellOT in 17 of 20 donor-repeats. CellOT's weak point is the average
  response, not the donor-specific one.
- **Using CellOT's donor-specific part only where it held elsewhere ("gated"), or scaled ("scaled"), doubles the
  specific R² to 0.10.** Error is 13 % below CellOT and 7 % below the shift; skill 0.673 / 0.676. It is used for 56 %
  of cases.
- **Against the shift the gain is modest:** better in 10–12 of 20 donor-repeats. Per stim it is IFNa 0.19, LPS 0.16,
  IL246 0.13, TNFa 0.09, and about 0 for GMCSF, PI and IL33.
- So with the current inputs and 15 donors, about 10 % of the real donor-specific response is recoverable, and it
  lives in a known set of stims, celltypes and markers.

<!-- READ7 -->
**How to read §7.** First table: one row per arm, all on the same cases.
- *skill*, *MSE*, *specific R²*, *corr* and *SD ratio* are as in §2–§5.
- *shared MSE* is the error on the 4 test donors' average response.
- *slope vs Unstim* measures how strongly the prediction follows the donor's own Unstim value across the test donors. The true slope is in the column name.

Then specific R² by stim and by |z|. §7b gives:
- paired tests over the **20 test donor-repeats**, against the paired control (`paired`: same training, raw cells);
- the stretch feature by feature. A feature is *stretched* when the paired control's slope exceeds the true slope by more than 0.5;
- slope and skill by stim, specific R² by quantile and marker, and MSE and paired tests by |z| bin.

**Scale.** Skill here keeps only groups with at least 3 test donors above the floor. That drops 16 % of cases, mostly small effects, so every level is about 0.06 above the learning curve's. On the learning-curve scale:

| arm | skill |
|---|---|
| shift | 0.604 |
| CellOT | 0.596 |
| paired | 0.615 |
| dnorm | 0.586 |
| dnorm_clip | 0.587 |
| clip | 0.612 |

**Reading (2026-10-07).**
- **Per-donor normalisation does remove the stretch.** In the 15 % of features where the paired control is stretched, its median slope is 1.92 against a true 0.69. With dnorm it is 1.01. The share of features with slope > truth + 0.5 falls from 14.5 % to 6.8 %. By stim, dnorm's slope is 0.99–1.01; the paired control's is 1.00–1.05; the truth is 0.90–0.98.
- **But it removes the donor-specific signal with it, and is worse overall.**
  - Specific R² falls from +0.045 to −0.14 (dnorm_clip −0.11), and corr from 0.29 to −0.03.
  - dnorm is worse than the paired control for **19 of 20** donor-repeats (MSE +12 %, p 4e-6). dnorm_clip is worse for 18/20 (+11 %, p 6e-4).
  - Neither beats the shift on any donor-repeat (0/20).
  - Even inside the stretched features it does not help: specific R² −0.12 vs −0.15, skill 0.744 vs 0.768, better on 6/20.
- **Why.** What CellOT gets right about a donor sits in her baseline level, and that is exactly what normalisation removes. After normalisation the map sees only the shape of her cells around her own median, and that shape does not predict her response.
  - Specific R², paired → dnorm: pSTAT1 0.27 → 0.00, pSTAT3 0.03 → −0.41, pSTAT6 −0.12 → −0.41.
  - Low tails (q10 / q25): 0.09 / 0.10 → −0.43 / −0.45.
  - Worst stim: IL246, −0.48.
  - So the stretch and the useful signal come through the same channel, and removing the channel loses more than it gains.
- **Out-of-range donors.** dnorm is worst beyond 3 SD (specific R² −0.70, MSE 0.097 vs paired 0.071).
- **Clipping alone is harmless and helps slightly.**
  - Better than the paired control for **16/20** donor-repeats, but by only 0.9 % (median MSE ratio 0.991, p 0.006).
  - Skill 0.667 vs 0.666; specific R² 0.051 vs 0.045.
  - The gain is mostly for out-of-range donors: 14/20 at 2–3 SD, 12/18 beyond 3 SD.
  - It does not close the gap with the shift there. Beyond 3 SD the MSE is clip 0.069, shift 0.049. At 2–3 SD it is 0.043 vs 0.039. Clamping single cells to the training range does not move a donor whose whole distribution is shifted.
- **Net.** Neither fix removes the out-of-range failure. The robust way to handle the average response and out-of-range donors is still to fall back on the shift (§5 hybrid/gated; also the 0.5 CellOT + 0.5 shift ensemble in `mfm_heldout`). Clipping can be kept as a free safeguard.

<!-- READ8 -->
**How to read §8.** The first table and the specific-R² tables are laid out as in §7.

`CellOT` is the per-pair unpaired run of the learning curve. It is the like-for-like reference, because CondOT also pools unpaired cells. `paired` is the paired control.

§8b gives:
- paired tests over the 20 test donor-repeats;
- the error split into shared and patient-specific parts;
- skill and paired tests by stim;
- results by response size and celltype size;
- per map, how often CondOT beats CellOT;
- the training-side validation error.

The figure shows one dot per donor-repeat: MSE divided by CellOT's.

On the learning-curve scale the skills are:

| arm | skill |
|---|---|
| CellOT | 0.596 |
| paired | 0.615 |
| shift | 0.604 |
| **condot_ct** | **0.562** |
| **condot_global** | **0.421** |

**Training check.** No run failed or diverged: there was no NaN, and all 140 + 5 tasks completed.
- **condot_ct fits less well than per-pair CellOT even on the training donors.** Its best validation median error is a median 0.042 (range 0.030–0.071; mDCs and B cells are worst). Per-pair CellOT is at 0.021 at k = 15.
  - condot_ct was still improving slowly at 50k iterations: the mean error over successive 10k blocks went 0.065 → 0.059 → 0.056 → 0.055, and the median best step was 40k.
  - Each stim gets about 8k of the 50k iterations, against 50k for each per-pair CellOT map.
- **condot_global is clearly under-fit.** Its best validation error is 0.088–0.096, about 4× CellOT's.
  - It plateaued by about 60k iterations and then oscillated between 0.095 and 0.19. More iterations alone would not fix it.
  - Each of the 168 pairs gets about 900 iterations.
  - There is no identity initialisation: the smoke test started at an error of 143.
  - CondOT's checkpoint is a single one shared by all its conditions, while CellOT picks the best checkpoint for each map.

**Reading (2026-10-07).**
- **Sharing one map across stims does not help.**
  - condot_ct is worse than per-pair CellOT for **19 of 20** donor-repeats (median MSE +10 %, p 6e-6). It is worse than the paired control for 20/20 (+12 %).
  - Skill is 0.621 vs 0.651 (paired 0.666, shift 0.664). Specific R² is 0.037 vs 0.046.
  - condot_global is worse for 20/20 (+77 %), with skill 0.498.
- **The loss is in the average response, not in the donor-specific part.** The patient-specific error is unchanged: 0.0224 for condot_ct and 0.0234 for condot_global, against 0.0222 for CellOT. The shared error is worse: 0.0179 and 0.0418, against 0.0150. So sharing adds no donor-specific information, as expected since the context describes only the stim. It costs accuracy on each stim's mean response.
- **Per stim, condot_ct has lower skill than CellOT in all 7 stims.**
  - GMCSF 0.606 vs 0.632; IFNa 0.686 vs 0.718; IL246 0.601 vs 0.646; LPS 0.604 vs 0.633; PI 0.700 vs 0.715.
  - TNFa is the only stim where its specific R² is higher (0.051 vs 0.025; global 0.072). Its total error there is still worse (better for 5/20).
- **No borrowing of strength for weak or small pairs. They are where sharing hurts most.**
  - IL33: skill 0.038 vs 0.118, better for 2/20 donor-repeats.
  - TNFa: 0.421 vs 0.465, 5/20.
  - Responses under 0.05: 0.117 vs 0.216. 0.05–0.1: 0.442 vs 0.540.
  - The 7 smallest celltypes (pDCs, mDCs, CD56hi NK, CD8Tcm CCR2+, intMCs, ncMCs, CD4Teff): 0.557 vs 0.584, 1/20.
  - Per map, condot_ct beats CellOT on 27 of 168 maps, mostly where CellOT is already worse than the shift (29 % of that third, against 9–11 % elsewhere). There the shift is better still.
- **Answer.** On these data, a CondOT-style map conditioned on the perturbation does not beat per-pair CellOT on held-out donors, overall or on the weak or small pairs. It also cannot reach the donor-specific part, which is the plateau.

**Caveats.**
- The comparison uses CondOT's own budget, not the same number of iterations per condition (about 6× fewer per stim for condot_ct, about 56× fewer per pair for condot_global).
- CondOT uses one checkpoint for all its conditions.
- There are 5 repeats and 14 distinct test donors, so the donor-repeats are not fully independent.

A matched-budget condot_ct (about 300k iterations per celltype) would settle the under-fit question. It would only bring the average response up to CellOT's level, which the shift already provides for free.

<!-- READ6 -->
## 6. Files and commands

- Case table: `lc/where_skill.py` → `work/lc/where_skill/cases.pkl` (one row per repeat × map × k × test donor ×
  quantile × marker, with the truth, the noise, the training means and SDs, and all predictions).
- This notebook: `lc/build_where_skill_notebook.py` + `lc/where_skill_readings.md` →
  `notebooks/where_cellot_skill.ipynb`; figures in `work/lc/where_skill/`.
- Related: `work/lc/residual_predict` (linear predictors of the residual from Unstim), `work/lc/ctx_score`
  (donor contexts), `notebooks/phenotype_shifts.ipynb` (population sizes), `notebooks/atlas_vs_ool.ipynb` (OOL inputs
  off the atlas range).
