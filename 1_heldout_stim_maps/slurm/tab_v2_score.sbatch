#!/bin/bash
# Score the tab_v2 runs (k15 and learning curve; q50/q75/q90; PI and IL33 left out) with lc/tab_compare.py.
#   sbatch --dependency=afterok:<ct>:<group> slurm/tab_v2_score.sbatch   -> work/lc/tab/v2_compare_*
#SBATCH -p normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -J tab_v2_score
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/tab_v2_score_%j.out
set -euo pipefail
source /scratch/users/bmerwan/atlas_heldout/env.sh
module load viz py-matplotlib/3.8.3_py312 py-scikit-learn/1.5.1_py312 >/dev/null 2>&1
CT="S0 L1cat S1_5 S1_7 S2 S3 S4_3 S4_5 S4_7 S4perm"; GR="G_S0 G_S1_5 G_S4_5"
N=""; for L in $CT $GR; do N="$N v2_$L"; done
python3 -m lc.tab_ens --names $N --repeats 0 1 2 3 4 --lambdas 1 --cellot --w-cellot 0.4 --res-tag _v2
P=tab/pred; R="repeat_{r}"
REF="CellOT=cond_train/$R/none CellOT+shift=../../../mfm_heldout/work/ens/w0.5/$R 0.4CellOT+0.6lgbm_L1=$P/lgbm_L1_l1_c0.40/$R"
A=""; M=""; for L in $CT; do A="$A $L=$P/v2_${L}_l1/$R"; M="$M 0.4CellOT+0.6$L=$P/v2_${L}_l1_c0.40/$R"; done
O="--repeats 0 1 2 3 4 --exclude-stims IL33 PI --qs 50 75 90 --rule-qs 50 75 90 --vs CellOT+shift"
W=/scratch/users/bmerwan/atlas_heldout/work/lc/tab
python3 -m lc.tab_compare --methods $REF $A $M --ks 15 $O --out $W/v2_compare_k15
python3 -m lc.tab_compare --methods $REF $M --ks 3 5 8 15 $O --out $W/v2_compare_curve
G=""; for L in S0 S1_5 S4_5; do G="$G ct_$L=$P/v2_${L}_l1/$R grp_$L=$P/v2_G_${L}_l1/$R"; done
python3 -m lc.tab_compare --methods $G --ks 3 5 8 15 $O --vs ct_S0 ct_S1_5 ct_S4_5 --out $W/v2_compare_groups
