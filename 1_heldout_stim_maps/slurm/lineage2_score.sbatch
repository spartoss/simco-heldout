#!/bin/bash
# Score lineage round 2 (slurm/lineage2.sbatch lc / lodo, slurm/lineage2_lgbm.sbatch): prediction dirs (tab_ens, lambda 1),
# case tables per comparison set (tab_compare --dump; PI / IL33 out, q50-q90), summaries (lc/lineage2_summary.py).
#   k = 15 (5 repeats): work/lc/tab/lin2_{ct,cur,big,T}/   k = 18 (19 LODO folds): work/lc/tab/lin2_lodo_{ct,cur,big,T}/
#   sbatch --dependency=afterok:<lc>:<lodo>:<lgbm> slurm/lineage2_score.sbatch
#SBATCH -p normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -J lineage2_score
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/lineage2_score_%j.out
set -euo pipefail
source /scratch/users/bmerwan/atlas_heldout/env.sh
module load viz py-matplotlib/3.8.3_py312 py-scikit-learn/1.5.1_py312 >/dev/null 2>&1
W=/scratch/users/bmerwan/atlas_heldout/work/lc; T=$W/tab; P=tab/pred; R="repeat_{r}"
DZ=$W/design_lodo.json; RL=$(seq -s " " 0 18)
CUR="L1cat L1F L1all_raw L1M L1allM"; BIG="L1cat L1all_raw L1M L1allM"; TT="L1cat L1M L1allM"; X="L1X L1M L1allM"
names() { local o=""; for L in $2; do o="$o $1$L"; done; echo $o; }
# prediction dirs (shift + full residual)
python3 -m lc.tab_ens --lambdas 1 --res-tag _lin2 --names $(names v2_x_ "$X") $(names v2lgb_x_ "$X") \
  $(names v2_gcur_G_ "$CUR") $(names v2_gbig_G_ "$BIG") $(names v2_gT_G_ "$TT")
python3 -m lc.tab_ens --lambdas 1 --res-tag _lin2lodo --design $DZ --repeats $RL --names $(names v2_x_lodo_ "$X") \
  $(names v2lgb_x_lodo_ "$X") $(names v2_gcur_lodo_G_ "$CUR") $(names v2_gbig_lodo_G_ "$BIG") $(names v2_gT_lodo_G_ "$TT")
O="--exclude-stims IL33 PI --qs 50 75 90 --rule-qs 50 75 90 --dump"
for MODE in lc lodo; do
  if [[ $MODE == lc ]]; then
    A="--repeats 0 1 2 3 4 --ks 15"; S=""; SL=""; OUT=lin2
    REF="CellOT=cond_train/$R/none CellOT+shift=../../../mfm_heldout/work/ens/w0.5/$R"
  else
    A="--design $DZ --repeats $RL --ks 18"; S="_lodo"; SL="lodo_"; OUT=lin2_lodo
    REF="CellOT=cond_train_lodo/$R/none"
  fi
  CT="ct_L1cat=$P/v2_${SL}L1cat_l1/$R ct_L1F=$P/v2_${SL}L1F_l1/$R ct_L1all_raw=$P/v2_${SL}L1all_raw_l1/$R"
  for L in $X; do CT="$CT ct_$L=$P/v2_x${S}_${L}_l1/$R"; done
  LG="lgb_L1cat=$P/v2lgb_${SL}L1cat_l1/$R lgb_L1all_raw=$P/v2lgb_${SL}L1all_raw_l1/$R"
  for L in $X; do LG="$LG lgb_$L=$P/v2lgb_x${S}_${L}_l1/$R"; done
  python3 -m lc.tab_compare --methods $REF $CT $LG $A $O --out $T/${OUT}_ct
  G=""; for L in $CUR; do G="$G Gcur_$L=$P/v2_gcur${S}_G_${L}_l1/$R"; done
  OLD=""; [[ $MODE == lc ]] && OLD="old_G_S4_5=$P/v2_G_S4_5_l1/$R old_G_S1_5=$P/v2_G_S1_5_l1/$R"
  python3 -m lc.tab_compare --methods $REF $CT $G $OLD $A $O --out $T/${OUT}_cur
  G=""; for L in $BIG; do G="$G Gbig_$L=$P/v2_gbig${S}_G_${L}_l1/$R"; done
  python3 -m lc.tab_compare --methods $REF $CT $G $A $O --out $T/${OUT}_big
  G=""; for L in $TT; do G="$G GT_$L=$P/v2_gT${S}_G_${L}_l1/$R"; done
  python3 -m lc.tab_compare --methods $REF $CT $G $A $O --out $T/${OUT}_T
  for C in ct cur big T; do python3 -m lc.lineage2_summary --cases $T/${OUT}_$C/cases.pkl --cellot CellOT --ref ct_L1cat; done
done
