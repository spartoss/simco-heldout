<!-- ANSWER -->
## Answer (2026-10-07)

**Short answer: no. Using DMSO1 only (or any single plate) does not reduce the noise, and it makes held-out
prediction worse.** Pooling the three plates, as every pipeline does today, is the better choice. The only place where
one plate "helps" is when the thing to predict is that same plate's well under PI, and there the gain is the plate's
own batch offset, not the donor.

**Which DMSO is used for a stim: all three, pooled, everywhere.** The paper's stim maps, the learning-curve caches
and every held-out score take Unstim *and* stim cells from the DMSO1 + DMSO2 + DMSO3 wells together (about
36 / 33 / 30 % of the cells), with no plate label kept and no plate matching between source and target. Only the drug
arm uses a single control (its plate's `use_DMSO`).

**Consequence.** The three wells of one donor and stim differ by more than sampling:
- **PI**: plate-to-plate SD of the response 0.34 against a between-donor SD of 0.21 (variance ratio 2.6). It is
  plate 1: its PI response is 0.35 larger along the response direction, and it is the odd plate out in 69 % of cases.
  In the pooled truth this plate noise is about as large as the whole between-donor variance (ratio 1.0).
- **Other stims**: plate-only variance is 0.23–0.26 × the between-donor variance (IFNa, IL246, LPS), 0.47–0.53
  (TNFa, GMCSF), 0.76 (IL33). After averaging three plates it is still 8–26 % of the between-donor variance of the
  pooled truth. DMSO1 is only mildly odd outside PI (odd plate out in 35–39 % of cases against 33 % by chance).
- The split-half noise ceiling (0.87) cannot see any of this, because both halves hold the same mix of plates. So
  part of what `where_cellot_skill.ipynb` calls "real donor-to-donor differences" is plate noise. well_ceiling
  estimates a realistic ceiling of ~0.73 on the pooled target.
- Plate matching helps somewhat: Unstim and stim from the same plate share part of the plate offset. The plate-only
  variance of the matched response is 0.60–0.86 × that of the stim level alone (GMCSF, IFNa, IL246, LPS, TNFa),
  but 0.96 for PI, where plate 1 differs in its *response*, not in its baseline.

**Part 2: one plate everywhere (5 repeats × 4 held-out donors, k = 15, CellOT and four closed-form maps).**

| CellOT, held-out donors | pool | poolN1 (pooled, DMSO1 cell count) | DMSO1 | DMSO2 | DMSO3 |
|---|---|---|---|---|---|
| median cells per stim block | 2498 | 948 | 953 | 820 | 712 |
| skill on its own truth | **0.611** | 0.562 | 0.567 | 0.529 | 0.489 |
| skill on the donor's pooled response | **0.612** | 0.600 | 0.442 | 0.553 | 0.489 |
| … PI only | 0.717 | 0.722 | 0.432 | 0.631 | 0.561 |
| … the other six stims | 0.571 | 0.557 | 0.443 | 0.528 | 0.465 |
| better than pool, donor-repeats (pooled response) | — | 0/20 | 0/20 | 0/20 | 0/20 |

1. **Against the donor's real response (the pooled three wells), every single-plate pipeline is worse than the pooled
   one, in 20 of 20 donor-repeats, for every method** (CellOT 0.61 → 0.44 with DMSO1; gauss_ot 0.60 → 0.43;
   quantile_map 0.54 → 0.32). DMSO1 is the worst of the three plates, and not only because of PI: on the other six
   stims it is also last (0.44 vs 0.53 / 0.47).
2. **It is the single plate that costs, not the fewer cells.** Cutting the pooled data to DMSO1's cell count (poolN1,
   2.6× fewer cells) lowers the skill by only 0.01 (0.612 → 0.600; nothing at all for PI). Taking those cells from
   one plate instead lowers it by 0.06–0.17.
3. **On its own one-well truth, DMSO1 reports 0.567, below the pooled pipeline's 0.611 (pool better in 17 of 20)
   and the same as an equally small pooled sample (poolN1 0.562; DMSO1 better in 11 of 20, p 0.62).** One well is
   not a cleaner measurement: its split-half noise is the same as poolN1's (≈ 0.03 arcsinh), larger than the
   pooled block's (≈ 0.02), and two plates of the same donor differ by another 0.045–0.072 on top of that outside PI
   (0.22–0.31 under PI when plate 1 is involved). Averaging three plates removes part of that; one plate keeps all of it.
4. **Plate matching helps only for PI, and only for predicting that plate.** A map trained on plate p predicts the
   donor's plate-p PI well better than the pooled map in 18–20 of 20 donor-repeats for each plate (DMSO1: 0.757 vs
   0.577), because each plate has its own systematic PI offset. For the other stims it is no better (12, 10, 8 of 20;
   p 0.23–0.67). A plain feature-level shift fitted on the same plate does as well as CellOT (0.563 vs 0.567), so what
   is learnt is the plate's batch offset, shared by all donors.
5. **No arm captures more of the real between-donor differences.** CellOT's specific R² is 0.05 (pool), 0.04 (poolN1),
   −0.01 (DMSO1), 0.08 / 0.08 (DMSO2 / DMSO3) on own truth; it varies from −0.16 to +0.19 between repeats, far more
   than between arms. Against the pooled response DMSO1 is clearly negative (−0.12).

**What to do.** Keep pooling the three plates for stim maps and truths. If plate noise matters for a claim, the
replicate reference (§2.3: a donor's other plates, plate-matched) is the honest one-well ceiling (0.55–0.62), and PI
on plate 1 should be flagged as a batch effect (it is the `DMSO1` outlier already seen in the drug arm).

**Caveats.** One k (15). The 20 donor-repeats are not fully independent (14 donors over 5 repeats), so the p-values are
optimistic; the 0/20 and 20/20 results do not depend on that. Skill on own truth is relative to each arm's own effect
size, so arms are not directly comparable there (DMSO1's PI effect is 0.45 vs 0.29 pooled, which inflates its PI
skill); the pooled-response and one-well comparisons are the fair ones. Caches are capped at 10k cells per donor ×
condition; single wells are mostly below the cap, so for large celltypes the pooled arm has ~3× the cells.

<!-- READ1 -->
**Reading 1.** Shown as SDs, the plate effect is the same size as the sampling noise of one well for most stims
(0.07–0.10), and below the between-donor SD (0.13–0.19). PI is the exception: its plate SD (0.34) is larger than
everything else, and it is driven by plate 1. The systematic part (the same offset for every donor) is 0.54 for
DMSO1 under PI, and 0.03–0.07 for every other plate × stim. So plate 1 under PI is a batch shift shared by all
donors. Elsewhere, plate differences are mostly donor-specific well noise. Caveats: features with tiny cell counts
inflate the RMS values, so the medians in the table are smaller. The between-donor SD itself contains some plate and
sampling noise.

<!-- READ21 -->
**Reading 2.1.** *Sanity check:* the `pool` arm, retrained here, gives CellOT 0.611 against 0.615 for the learning
curve's paired CellOT on the same cases, and gauss_ot 0.595, quantile_map 0.538, median_shift 0.512, the learning
curve's k = 15 values. So the arm reproduces the reference.

Each arm judged on its own wells: pool 0.611, poolN1 0.562, DMSO1 0.567, DMSO2 0.529, DMSO3 0.489 (CellOT; the same
order for every method). Paired over the 20 donor-repeats:
- fewer cells costs 0.05: poolN1 is below pool in 20 of 20;
- one plate instead of a mixed sample of the same size changes nothing for DMSO1 (better in 11 of 20, p 0.62; gauss_ot
  12 of 20, p 0.23) and costs 0.03–0.07 for DMSO2 / DMSO3 (worse in 15 and 19 of 20). DMSO2/3 wells are smaller
  (820 / 712 cells) and their ceilings lower (0.78 / 0.76 vs DMSO1 0.81);
- DMSO1 is ahead of DMSO2 / DMSO3 (15 and 19 of 20) only because of PI: DMSO1's PI response is larger (0.45 vs
  0.21–0.26), and skill is measured relative to the effect, so a bigger effect gives a bigger skill. DMSO1 is the only
  arm above pool on any stim, and only on PI (0.757 vs 0.717). Without PI: pool 0.571, poolN1 0.510, DMSO1 0.497,
  DMSO2 0.479, DMSO3 0.442.

So a DMSO1-only pipeline would report a lower skill than today's, and the drop is what one would get from simply
having fewer cells. The split-half ceilings (0.87 pool, 0.76–0.81 for the others) follow the cell counts and cannot
see plate effects.

<!-- READ22 -->
**Reading 2.2.** This is the main test: same target for everyone, the test donor's pooled three-well response.
- **Every single-plate arm is worse than pool in 20 of 20 donor-repeats, for every method** (CellOT 0.612 → DMSO1
  0.442, DMSO2 0.553, DMSO3 0.489). The method differences (CellOT 0.612, feature shift 0.605, gauss_ot 0.597,
  quantile_map 0.538) are much smaller than the arm differences.
- **Fewer cells is nearly free**: poolN1 0.600 vs 0.612 (better than pool in 2 of 20 on the other stims; on PI 0.722 vs
  0.717, 9 of 20, p 0.7).
- **One plate is not**: against poolN1, which has the same number of cells, DMSO1 is worse in 20 of 20 (PI 0.432 vs
  0.722; other stims 0.443 vs 0.557), DMSO2 in 16–18 of 20, DMSO3 in 18–20 of 20.
- **Plate 1 specifically is the worst**, on PI (0.43 vs 0.63 / 0.56) and on the other stims (0.44 vs 0.53 / 0.47;
  worse than DMSO2 in 18 of 20). IL33 is negative in every single-plate arm (−0.23 for DMSO1): a single well's IL33
  response is mostly plate noise.

Plate matching (Unstim and stim from the same plate) does not compensate for losing the average over three plates.

<!-- READ23 -->
**Reading 2.3.** Target = one well. The *replicate* (the donor's own plate-matched response measured on the other two
plates) is the reference no model can be expected to beat for donor-specific reasons: 0.565 for a DMSO1 target,
0.622 for DMSO2, 0.551 for DMSO3. Paired, CellOT, plate-matched arm vs the others:

| target well | stims | matched | pool | poolN1 | replicate | matched better than pool | than replicate |
|---|---|---|---|---|---|---|---|
| DMSO1 | PI | 0.757 | 0.577 | 0.568 | 0.445 | 18/20 | 20/20 |
| DMSO1 | other six | 0.498 | 0.510 | 0.494 | 0.604 | 12/20 (p 0.23) | 0/20 |
| DMSO2 | PI | 0.688 | 0.630 | 0.631 | 0.725 | 18/20 | 9/20 |
| DMSO2 | other six | 0.479 | 0.501 | 0.484 | 0.597 | 10/20 (p 0.67) | 0/20 |
| DMSO3 | PI | 0.644 | 0.368 | 0.373 | 0.281 | 20/20 | 20/20 |
| DMSO3 | other six | 0.442 | 0.505 | 0.490 | 0.597 | 8/20 (p 0.37) | 0/20 |

- **Under PI each plate has its own systematic offset**, so a map trained on that plate predicts that plate's well
  better than the pooled map (18–20 of 20 for every plate) and, for plates 1 and 3, better than the donor's own wells
  on the other plates. Here the plate effect is larger than the donor effect. The feature-level shift fitted on DMSO1
  does as well as CellOT (0.563 vs 0.567): the gain is a batch offset shared by all donors, not donor biology.
- **For the other six stims, matching gives nothing** (8–12 of 20, n.s.), and all models stay far below the replicate
  (0.44–0.51 vs 0.60, 0 of 20): there is real donor-specific signal in a well that none of the maps capture. Against an equally small mixed sample (poolN1) the matched map is slightly better at
  predicting its own plate (13–16 of 20; p 0.02 for DMSO1, 0.08 and 0.13 for DMSO2/3), but its skill is not higher
  (0.50 vs 0.49, 0.48 vs 0.48, 0.44 vs 0.49): a small, inconsistent edge.

<!-- READ25 -->
**Reading 2.4–2.5.** *Noise.* The split-half noise of one well (0.027–0.038) is the same as that of a pooled sample of
the same size (poolN1, 0.027–0.035) and larger than the pooled block's (0.017–0.024). A single plate is therefore not
less noisy; it only has fewer cells. On top of sampling, two plates of the same held-out donor differ by 0.045–0.072
in the plate-matched response outside PI (about 1.5–2× one well's sampling noise, and a third to two thirds of the
pooled effect; more than the effect for IL33), and by 0.22–0.31 under PI when plate 1 is one of them (0.089 for
DMSO2 vs DMSO3). The pooled truth averages this; a one-plate truth carries all of it, and the split-half ceiling does
not see it.

*Specific R²* (share of the real between-donor differences, beyond the average response, that CellOT predicts):
0.05 for pool, 0.04 poolN1, −0.01 DMSO1, 0.08 DMSO2, 0.08 DMSO3 on own truth; against the pooled response 0.05,
0.02, −0.12, 0.01, −0.01. Per repeat it ranges from −0.16 to +0.19 within one arm, so the arm differences are not
meaningful, except that DMSO1 is the lowest in 3 of 5 repeats. No choice of plate makes the donor-specific part
predictable.

<!-- READ3 -->
**How to read Part 3.**
- **First table:** how often the plate is guessed right on donors the classifier never saw (chance 0.33). It does this
  from one cell, from a batch of 100 cells, or from a batch compared with the same donor's other batches. In that last
  case each batch is centred on the donor's mean batch, which removes her own level.
- **Second table:** the recall for each plate in that last setting.
- **3b:** the size of each component, all as standard deviations in arcsinh units:
  - *SD_donor*: real differences between donors;
  - *SD_well*: the well effect shared by every donor in the well, which is learnable;
  - *SD_dxw*: donor × well, including cell sampling, which is not learnable.
  The *level* rows are one well's values; the *response* rows are the stim well minus the Unstim well of the same plate.
  The first magnitude table is superseded by 3b (2026-10-07: it mixed an uncorrected RMS, a corrected SD, a mean of SDs
  and a mean absolute value, so its columns could not be compared).

**Reading.**
- **One cell does not tell its plate** (0.37–0.40), and neither does a lone batch of 100 cells (0.44–0.56).
- **Batches of the same donor do:** 0.75–0.85, Unstim included, best for plate 1 (recall 0.86–0.95). So every well
  carries a consistent signature shared by all its donors.
- **But it is small in the response.**
  - Outside PI the shared well effect is SD 0.035–0.05 on one well. In the response (stim − Unstim on the same plate) it
    is 0.026–0.036, and only 0.011–0.013 for IL33 / TNFa.
  - That is all a plate-matched map can learn, against average responses of 0.38–0.68 (GMCSF, IFNa, IL246, LPS). This
    is why matching gained nothing outside PI. Under PI the shared well effect is 0.28, as large as the real donor
    differences.
- **Donor × well is 2–4× the shared part:** SD 0.054–0.084 in the response. It cannot be learned, and pooling three
  wells divides its variance by 3.
- **How much of a donor's apparent response is real** (response rows):
  - IFNa / IL246 / LPS: about 80 % from one well, about 92 % from the 3-well average (real donor SD 0.11–0.13, against
    donor × well 0.054–0.064).
  - GMCSF / TNFa: about 54 % from one well, about 78 % from three.
  - IL33: 21 % from one well, 44 % from three. Most of an IL33 donor's apparent individuality is well noise.
- **Implication:**
  - On the 3-well averages the targets are mostly real for the strong stims, so the noise does not explain most of the
    plateau there. The unexplained donor-specific response is real but not predictable from Unstim (where_cellot_skill).
  - For TNFa, IL33 and one-well targets, well noise sets a low ceiling.
  - Split-half noise misses all of it.
