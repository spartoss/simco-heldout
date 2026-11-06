<!-- ANSWER -->
## Answer (2026-10-06, held-out donors only)

**Question.** For the drug arm, is a quantile map a better drug map than a trained CellOT drug map, above all on each
patient's own drug effect, and where do they differ?

**Short answer.** On new donors **no drug map predicts how one patient's drug response differs from another's**. So
per-patient accuracy comes down to getting the *average* drug effect right. On the medians, which are the features the
DOS model reads, quantile_map does that better than CellOT for nearly every donor. CellOT keeps a real but
small edge on the **low tails (q10/q25)** and on large effects, but that edge disappears once the drug map
sits inside the full pipeline.

1. **The full pipeline, as an OOL woman goes through it:** CellOT stim map followed by quantile_map on the features
   (the proposal), against the paper's CellOT stim map followed by a CellOT drug map on cells. Out of 20 donor-repeats,
   the proposal has the smaller median error in:
   - **19 of 20** on the DOS features: error 0.031 vs 0.036, closer in 58 % of cases, median skill 0.44 vs 0.34;
   - **20 of 20** on all real cases: 0.050 vs 0.053;
   - **18 of 20** on effects > 0.1 (tiny margin, 0.133 vs 0.135).
   The paper's chain also loses to CellOT stim → *constant shift* (median_shift) in 20 of 20 donors on all real cases.
2. **The drug map alone** (applied to the donor's real control cells):
   - **On medians, quantile_map wins:** 20 of 20 donors (error 0.030 vs 0.035, 14 % smaller); DOS features 18 of 20
     (0.030 vs 0.036, 15 % smaller).
   - **Over all quantiles it is a tie:** 12 of 20 donors, p 0.25.
   - **On effects > 0.1, CellOT is better in all 20 donors**, but only by 2 % of error (0.132 vs 0.135).
   - gauss_ot beats CellOT in 20 of 20 donors on all real cases.
3. **Per patient: shared part vs patient-specific part.** Only donor groups whose differences are real are scored
   (34 % of groups). Across all groups, 84 % of the between-donor spread of the true effect is sampling noise.
   - **Shared part (the average over donors):** quantile_map 0.36 vs CellOT 0.32. On the DOS features 0.64 vs 0.47
     (median_shift 0.61). quantile_map is better on 58 % of drug maps (63 % on the DOS features).
   - **Patient-specific part:** every map captures between −7 % and +6 % of the real donor-to-donor differences.
     CellOT −1 %, quantile_map −7 %, gauss_ot +1 %, the paper's chain −4 %. On the DOS features 0–6 % for every map,
     none distinguishable from giving everyone the same effect.
   - CellOT's donor-to-donor variation points weakly the right way (correlation 0.17 across donors). It is too small
     and too noisy to help.
   - quantile_map's variation adds error, mostly in the low tail.
   - On the DOS features, **46 % of CellOT's per-patient error is getting the average wrong** (quantile_map 32 %).
4. **Where CellOT wins:**
   - **Low tails:** q10 (quantile_map closer in only 44 % of cases) and q25 (48.5 %), in every stim and every drug.
     Strongest on pSTAT5 in lymphocytes, CD25, NK cells, and under Unstim, TNFa, RIF and SA. quantile_map's low-tail
     skill there is about 0.
   - **Large effects:** a small edge, as in point 2.
   - **A few DOS features with few cases:** CD8Tcm CCR2+ pNFkB PI, CD4Tnaive pSTAT3 IFNa, intMCs pSTAT5 GMCSF.
5. **Where quantile_map wins:**
   - **On medians, in almost every celltype × marker:** Bcells, CD4Tcm and pDCs 60–61 %; pp38 and CD25 63 %.
   - **On the Granulocyte pS6 DOS features:** TNFa/GMCSF/PI 64–69 %.
   - **On naive CD4 pSTAT3 under IL246/GMCSF:** 73–77 %.
   - **On small effects at the median:** 62 %.
   The most-selected feature, CD4Tnaive_pSTAT3_TNFa, is a tie (49 %).
6. **For the DOS features, quantile_map and a constant shift are equivalent** (pipeline: 11 of 20 donors, p 0.96).
   quantile_map's patient-dependence adds nothing there; it wins over median_shift only on the other quantiles.
7. **On the OOL women** (no truth): CellOT drug maps spread the women 3× more than quantile_map. The held-out test
   (point 3) gives no support for that extra spread being real. They also make 58 % of treated women implausible for
   XGBoost, against 14 %. THF and SALPZ are robust to the choice of drug map; MAP's effect needs CellOT; RIF, SA and
   PRA flip sign.

**What changed from the in-sample version (§9):** CellOT's "13 % of real patient differences captured" was
memorisation: on new donors it is about 0. The quantile_map advantage holds on new donors, but only on the medians;
over all quantiles the two maps tie, and CellOT wins the low tails.

**Caveats.**
- 14 distinct donors, 4 per repeat, and ≤ 5k cells per well, so the truths are noisy (most between-donor variance is
  noise).
- The held-out CellOT drug maps are retrained (paper architecture, last iterate), not the paper's checkpoints.
- Case-level p-values are far too small. Quote the donor counts (x of 20) and the drug-map p-values.
- LPZ/SA under PI (the DMSO1 plate artefact) are excluded from every number above.

<!-- READ3 -->
**How to read §3.** First table: median skill of each drug map on new donors (columns = subsets of the real cases).
Second: the DMSO1 plate cases alone. Third: paired comparisons with CellOT as A:
- *B better (cases)* = share of cases where B's error is smaller;
- *drug maps: B better* = share of repeat × drug × stim × celltype maps where B's mean error is smaller;
- *donors: B better* = in how many of the 20 donor-repeats B has the smaller median error.
Quote the donor and drug-map columns; case-level p-values are not meaningful.

**Reading.**
- **All real cases: tie.** CellOT and quantile_map have the same median error (0.0519 vs 0.0517), quantile_map is
  closer in 50.0 % of cases and in 12 of 20 donors (p 0.25). gauss_ot is the best drug map overall (20 of 20
  donors, error 0.049).
  On *median skill* (figure in §4) CellOT looks ahead, 0.17 vs 0.14. Skill divides the error by the effect, so it
  weighs the small low-tail effects, where quantile_map is relatively worst. On absolute error, and counted per donor,
  the two maps tie.
- **Medians: quantile_map.** 20 of 20 donors, error 0.030 vs 0.035; DOS features 18 of 20, 0.030 vs 0.036.
  median_shift and gauss_ot also beat CellOT on medians in 16–20 of 20 donors. **The CellOT drug map is weakest on
  the one quantity the DOS model reads.**
- **Effects > 0.1: CellOT, narrowly.** CellOT has the lower median error in all 20 donors, but by 2 % (0.132 vs
  0.135), and it is closer in only 52.5 % of cases. It ties gauss_ot there.
- **DMSO1 plate (LPZ/SA under PI):** much easier for CellOT, quantile_map and gauss_ot (skill 0.55–0.63 on all real cases; median_shift 0.05), because
  the effect is a plate offset shared by all donors. Excluded from the headline numbers.

<!-- READ4 -->
**How to read §4.** Same layout as §3, but each donor starts from her unstimulated cells and goes through a stim map
and then a drug map, as an OOL woman does. The figure shows median skill for §3 (left) and §4 (right).

**Reading.**
- **The proposal (CellOT stim → quantile_map on features) beats the paper's chain everywhere.** It wins in 20 of 20
  donors on all real cases (error 0.050 vs 0.053), 18 of 20 on effects > 0.1, 20 of 20 on medians (0.031 vs 0.036),
  and 19 of 20 on the DOS features (0.031 vs 0.036; median skill 0.435 vs 0.342).
- **CellOT's edge on large effects in §3 does not survive the pipeline.** Inside the chain the CellOT drug map
  receives simulated cells, not real ones. quantile_map works on feature values and is not exposed to that.
- **The paper's chain is beaten even by a constant shift** (CellOT stim → median_shift): 20 of 20 donors on all real
  cases, 16 of 20 on the DOS features. It is also beaten by the all-gauss_ot chain (20 of 20).
- **On the DOS features, quantile_map = median_shift** (11 of 20 donors, p 0.96); on medians generally 12 of 20.
  quantile_map's advantage over a constant shift is on the other quantiles (19 of 20 donors on all real cases).

<!-- READ5 -->
**How to read §5.** Three tables (all groups; groups with real between-donor differences; DOS features), then paired
tests per drug map (*B better* = share of drug maps where B's error on that part is smaller), then the breakdown by
stim and by quantile, then the figure.
- *real differences captured*: 0 % = no better than giving every donor the same effect, negative = worse.
- *tracking r*: does the map's donor-to-donor variation point the right way?
- *SD ratio*: how much the map varies between donors relative to the truth.
- *error from shared part*: the share of the per-patient error that comes from getting the average wrong.

**Reading.**
- **Most of the between-donor spread is noise.** Over all groups the noise share is 84 %, and every map does worse
  than a constant (−8 % to −47 %). Only 34 % of groups have real donor differences (pooled reliability > 0.5); the
  rest of this section is on those.
- **Shared part: quantile_map 0.36 vs CellOT 0.32** (gauss_ot 0.37, median_shift 0.31). quantile_map is better on
  58 % of drug maps.
  - On the DOS features: quantile_map 0.64, gauss_ot 0.62, median_shift 0.61, CellOT 0.47; quantile_map better on
    63 % of drug maps.
  - In the pipelines: CellOT stim → quantile_map 0.36 / 0.65 vs the paper's chain 0.30 / 0.47.
- **Patient-specific part: no map captures it.**
  - Reliable groups: CellOT −1.2 %, quantile_map −7.4 %, gauss_ot +1.2 %, paper chain −3.9 %, CellOT → quantile_map
    −7.0 %.
  - DOS features: 0.8–6.0 % for every map; the CellOT vs quantile_map difference is not significant (p 0.12, and
    p 0.22 for the pipelines).
  - No stim and no quantile exceeds ~6 %.
- **CellOT does beat quantile_map on the specific part** (63 % of drug maps). That is because quantile_map's
  donor-dependence adds error, above all in the low tail (q10: −17 %), not because CellOT captures much. CellOT's
  donor variation correlates 0.17 with the truth, at 0.28 of its size; that is right in direction but too weak to
  reduce error. gauss_ot beats CellOT on the specific part in 78 % of drug maps.
- **So per patient, accuracy is accuracy on the shared part.** On the DOS features that is 46 % of CellOT's
  per-patient error against 32 % for quantile_map, and it is where quantile_map wins.

<!-- READ6 -->
**How to read §6.** Each table splits the real cases of the drug map alone by one factor: *qmap better* > 0.5 means
quantile_map is more often closer. The cross-table gives the same by quantile × effect size. The heatmaps are red
where quantile_map wins and blue where CellOT wins: medians first, then the low tails.

**Reading.**
- **Quantile is the main axis.**

  | quantile | q10 | q25 | q50 | q75 | q90 |
  |---|---|---|---|---|---|
  | quantile_map closer | 44 % | 48.5 % | 58 % | 54 % | 52 % |

  CellOT wins the low tails, quantile_map the median and upper tails. On medians quantile_map wins at every effect size
  (52–62 %, strongest on small effects). In the q10 tail CellOT wins at every effect size (54–57 %).
- **Low tails: CellOT wins in every stim (52–57 %) and every drug (51–58 %).** Most clearly under Unstim, TNFa and
  IL33, and for RIF and SA, where quantile_map's low-tail skill is about 0 (it extrapolates badly where the control
  distribution is bunched near zero). The low-tail heatmap is blue in the **pSTAT5** column for nearly every
  lymphocyte, in **CD25** for T cells, and for **CD56hi NK** cells.
- **Medians: quantile_map wins in nearly every celltype × marker.**
  - No celltype is CellOT's (best is CD56hi NK, a tie, 51 %).
  - Strongest: Bcells, CD4Tcm, pDCs, CD4Tem CCR2+ (60–61 %); by marker pp38 and CD25 (63 %), pCREB (60 %).
  - The stim × drug median heatmap is red almost everywhere; only Unstim, SALPZ under GMCSF and a few cells are
    neutral.
- **Drugs (all quantiles):** CellOT is relatively better for RIF, MF, SA and MAP (46.5–48.6 %), quantile_map for LPZ
  (52.6 %).
- **Cohort:** CellOT is relatively better on pregnant donors (49.5 % vs 51.3 % for HV); this does not change the
  picture.

<!-- READ7 -->
**How to read §7.** One column per test donor-repeat:
- the number of real DOS-feature cases;
- the median error of A (CellOT) and B (quantile_map);
- the share of cases where B is closer.
Then one row per DOS feature, with its selection frequency in the Fig 5b folds. Few cases per feature, so
per-feature p-values are mostly non-significant.

**Reading.**
- **Per donor, on the DOS features, quantile_map wins for almost everyone.**
  - Drug map alone: smaller median error in 18 of 20 donor-repeats (exceptions HV11 in repeat 0 and PV07 in repeat 4),
    closer in more than half the cases for all 20.
  - Pipelines: 19 of 20 and 19 of 20.
  - Pregnant and non-pregnant donors alike.
- **Per feature:** quantile_map is closer more often on 60 of the 88 scored DOS features, CellOT on 24; weighted by
  how often Stabl selects each feature, 56 %.
- **quantile_map's clearest wins:** the Granulocyte pS6 features (TNFa 69 %, GMCSF 68 %, PI 64 %),
  CD4Tnaive_pSTAT3_IL246 (77 %) and _GMCSF (73 %), CD4Tregs_pSTAT5_PI (82 %).
- **CellOT's:** CD8Tcm_CCR2pos_pNFkB_PI (41 %), CD4Tnaive_pSTAT3_IFNa (44 %), intMCs_pSTAT5_GMCSF (39 %), and
  NK_cells_CD11cpos_HLADR_PI (30 %, 10 cases).
- **The most-selected feature, CD4Tnaive_pSTAT3_TNFa, is a tie** (49 % of cases), although quantile_map's median skill
  is higher (0.30 vs 0.18).

<!-- READ8 -->
**How to read §8.** First table: how much each drug map spreads the OOL women. *het* = SD of the per-woman effect
relative to the SD between women; *own R²* = how much of a woman's effect is a function of her own value.
Second table: mean predicted DOS shift in days and the share of fold models agreeing on its sign.
Third: implausible treated women and out-of-range values. Maps are the deployed ones (all 19 atlas donors).

**Reading.**
- **CellOT spreads the women 3× more than quantile_map** (het 0.27 vs 0.09), and half of that spread is not a
  function of the woman's own feature value. §5 found no map that predicts donor-specific drug effects on new donors,
  so there is no held-out evidence that this extra spread is real.
- **XGBoost has to extrapolate much more with CellOT drug maps:** 58 % of treated women are implausible, against 14 %
  (quantile_map) and 7 % (median_shift); 12 % of values are out of range, against 3 %.
- **Drug shifts:**
  - THF (−8.0 / −10.4 / −7.4 days) and SALPZ (−6.3 / −5.2 / −4.9) are robust to the choice of drug map; CHT is +2.2 to
    +3.4 days with all three.
  - MAP's −4.7 days needs CellOT (−1.8 with quantile_map, −0.4 with median_shift).
  - RIF, SA and PRA change sign with the map.

<!-- READ9 -->
## 9. What changed from the in-sample version (archived 2026-10-06)

Removed, because each scored a map on donors it was trained on: the pooled 80/20 cell split, per-patient accuracy
on the 19 drug-screen patients, in-sample patient-specificity, and the in-sample shared/specific split and "where"
tables. The archived copy is `notebooks/archive/drug_map_choice_insample_2026-10-06.ipynb`. Consequences:

1. **"CellOT captures 13 % of the real between-patient differences" was memorisation.** On new donors it is −1 %
   (DOS features 3.5 %, not distinguishable from quantile_map or from a constant).
2. **The earlier held-out figure of "4 %" (§11 of the old version) does not survive a better reliability
   estimate.** It selected groups with a reliability computed from the 4 test donors of one repeat (3 degrees of
   freedom). Pooling all 5 repeats gives −1 %.
3. **"quantile_map is better per patient" holds on new donors for the medians, not in general.** Medians and DOS
   features: 18–20 of 20 donors. Over all quantiles the drug maps alone tie (12 of 20), and CellOT wins the low tails
   in every stim and drug.
4. **CellOT's large-effect edge looks different out of sample.** In-sample it showed up on medians. On new donors it
   is small (2 %) but consistent across donors, comes from the tails, and disappears inside the pipeline.
5. **New:** the paper's pipeline loses to the quantile_map pipeline in 18–20 of 20 donors on every subset, and to a
   constant shift on all real cases. For the DOS features a constant shift is as good as quantile_map.
