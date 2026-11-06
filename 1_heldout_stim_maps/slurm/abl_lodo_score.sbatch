#!/bin/bash
# Score the L1allM ablations on LODO against L1allM, alone and mixed 0.4 with the subcluster CellOT -> work/lc/tab/abl_lodo
#SBATCH -p normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -J abl_score
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/abl_score_%j.out
set -euo pipefail
source /scratch/users/bmerwan/atlas_heldout/env.sh
module load viz py-matplotlib/3.8.3_py312 py-scikit-learn/1.5.1_py312 >/dev/null 2>&1
W=/scratch/users/bmerwan/atlas_heldout/work/lc; DZ=$W/design_lodo.json; RL=$(seq -s " " 0 18); P=tab/pred; R="repeat_{r}"
AB="L1allM_noOwn L1allM_noFreq L1allM_noBPC Mmin"
N=""; for L in $AB; do N="$N v2_abl_lodo_$L v2lgb_abl_lodo_$L"; done
python3 -m lc.tab_ens --design $DZ --repeats $RL --lambdas 1 --res-tag _abl --names $N
M="CellOT_paired=cond_train_lodo/$R/none CellOT_cluster=cond_train_lodo/$R/cluster_K4_stable"
M="$M pfn_L1allM=$P/v2_x_lodo_L1allM_l1/$R pfn_L1M=$P/v2_x_lodo_L1M_l1/$R pfn_L1cat=$P/v2_lodo_L1cat_l1/$R lgb_L1allM=$P/v2lgb_x_lodo_L1allM_l1/$R"
for L in $AB; do M="$M pfn_$L=$P/v2_abl_lodo_${L}_l1/$R lgb_$L=$P/v2lgb_abl_lodo_${L}_l1/$R"; done
python3 -m lc.tab_compare --methods $M --design $DZ --repeats $RL --ks 18 --exclude-stims IL33 PI --qs 50 75 90 \
  --rule-qs 50 75 90 --dump --out $W/tab/abl_lodo
python3 -m lc.lineage2_summary --cases $W/tab/abl_lodo/cases.pkl --cellot CellOT_cluster --ref pfn_L1allM
