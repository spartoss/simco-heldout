#!/bin/bash
# Score the second tab_v2 round (L1cat + marker axes, L1cat + frequency / whole-donor blocks) with lc/tab_compare.py.
#   sbatch --dependency=afterok:<ct job> slurm/tab_v3_score.sbatch   -> work/lc/tab/v3_compare_{k15,curve}
#SBATCH -p normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -J tab_v3_score
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/tab_v3_score_%j.out
set -euo pipefail
source /scratch/users/bmerwan/atlas_heldout/env.sh
module load viz py-matplotlib/3.8.3_py312 py-scikit-learn/1.5.1_py312 >/dev/null 2>&1
NEW="L1cat_mx L1cat_mxc L1F L1fpc L1all_pc L1all_raw L1all_raw_perm"
N=""; for L in $NEW; do N="$N v2_$L"; done
python3 -m lc.tab_ens --names $N --repeats 0 1 2 3 4 --lambdas 1 --cellot --w-cellot 0.4 --res-tag _v3
P=tab/pred; R="repeat_{r}"
REF="CellOT=cond_train/$R/none CellOT+shift=../../../mfm_heldout/work/ens/w0.5/$R"
REF="$REF L1cat=$P/v2_L1cat_l1/$R 0.4CellOT+0.6L1cat=$P/v2_L1cat_l1_c0.40/$R 0.4CellOT+0.6S4_3=$P/v2_S4_3_l1_c0.40/$R"
REF="$REF 0.4CellOT+0.6lgbm_L1=$P/lgbm_L1_l1_c0.40/$R"
A=""; M=""; for L in $NEW; do A="$A $L=$P/v2_${L}_l1/$R"; M="$M 0.4CellOT+0.6$L=$P/v2_${L}_l1_c0.40/$R"; done
O="--repeats 0 1 2 3 4 --exclude-stims IL33 PI --qs 50 75 90 --rule-qs 50 75 90 --dump"
W=/scratch/users/bmerwan/atlas_heldout/work/lc/tab
python3 -m lc.tab_compare --methods $REF $A $M --ks 15 $O --vs CellOT+shift L1cat 0.4CellOT+0.6L1cat --out $W/v3_compare_k15
python3 -m lc.tab_compare --methods $REF $A $M --ks 3 5 8 15 $O --vs CellOT+shift L1cat --out $W/v3_compare_curve
