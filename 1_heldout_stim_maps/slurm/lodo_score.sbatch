#!/bin/bash
# Score the leave-one-donor-out run: tab_ens (shift + rhat) for TabPFN / LightGBM, lc/tab_compare.py (k = 18, no PI/IL33,
# q50-90, --dump), then lc/lodo_summary.py (mixes, pooled between_R2).   sbatch --dependency=afterok:<cellot>:<tabpfn>
#SBATCH -p normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -J lodo_score
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/lodo_score_%j.out
set -euo pipefail
source /scratch/users/bmerwan/atlas_heldout/env.sh
module load viz py-matplotlib/3.8.3_py312 py-scikit-learn/1.5.1_py312 >/dev/null 2>&1
W=/scratch/users/bmerwan/atlas_heldout/work/lc; DZ=$W/design_lodo.json; RS=$(seq -s " " 0 18)
LAY="L1cat L1F L1all_raw L1all_pc"
N=""; for L in $LAY; do N="$N v2_lodo_$L v2lgb_lodo_$L"; done
python3 -m lc.tab_ens --design $DZ --names $N --repeats $RS --lambdas 1 --res-tag _lodo
P=tab/pred; R="repeat_{r}"
M="CellOT_orig=train_lodo/$R CellOT_paired=cond_train_lodo/$R/none"
MO=""; for L in $LAY; do M="$M pfn_$L=$P/v2_lodo_${L}_l1/$R lgb_$L=$P/v2lgb_lodo_${L}_l1/$R"; MO="$MO pfn_$L lgb_$L"; done
python3 -m lc.tab_compare --design $DZ --methods $M --repeats $RS --ks 18 --exclude-stims IL33 PI --qs 50 75 90 \
  --rule-qs 50 75 90 --dump --out $W/tab/lodo_compare
python3 -m lc.lodo_summary --cases $W/tab/lodo_compare/cases.pkl --cellots CellOT_orig CellOT_paired --models $MO \
  --weights 0.4 --ref CellOT_paired+shift
