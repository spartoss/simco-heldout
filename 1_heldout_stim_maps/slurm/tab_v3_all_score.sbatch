#!/bin/bash
# Score the second round for both models together: TabPFN (v2_<layer>) and LightGBM (v2lgb_<layer>), alone and as
# 0.4 CellOT + 0.6 model.   sbatch --dependency=afterok:<tabpfn>:<lgbm> slurm/tab_v3_all_score.sbatch
#   -> work/lc/tab/v3all_compare_{k15,curve}
#SBATCH -p normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -J tab_v3all_score
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/tab_v3all_score_%j.out
set -euo pipefail
source /scratch/users/bmerwan/atlas_heldout/env.sh
module load viz py-matplotlib/3.8.3_py312 py-scikit-learn/1.5.1_py312 >/dev/null 2>&1
NEW="L1cat_mx L1cat_mxc L1F L1fpc L1all_pc L1all_raw L1all_raw_perm"
N=""; for L in L1cat $NEW; do N="$N v2lgb_$L"; done
python3 -m lc.tab_ens --names $N --repeats 0 1 2 3 4 --lambdas 1 --cellot --w-cellot 0.4 --res-tag _v3lgb
P=tab/pred; R="repeat_{r}"
REF="CellOT=cond_train/$R/none CellOT+shift=../../../mfm_heldout/work/ens/w0.5/$R"
A=""; M=""
for L in L1cat $NEW; do
  A="$A pfn_$L=$P/v2_${L}_l1/$R lgb_$L=$P/v2lgb_${L}_l1/$R"
  M="$M 0.4CellOT+0.6pfn_$L=$P/v2_${L}_l1_c0.40/$R 0.4CellOT+0.6lgb_$L=$P/v2lgb_${L}_l1_c0.40/$R"
done
O="--repeats 0 1 2 3 4 --exclude-stims IL33 PI --qs 50 75 90 --rule-qs 50 75 90 --dump"
W=/scratch/users/bmerwan/atlas_heldout/work/lc/tab
python3 -m lc.tab_compare --methods $REF $A $M --ks 15 $O --vs CellOT+shift pfn_L1cat lgb_L1cat --out $W/v3all_compare_k15
python3 -m lc.tab_compare --methods $REF $M --ks 3 5 8 15 $O --vs CellOT+shift --out $W/v3all_compare_curve
