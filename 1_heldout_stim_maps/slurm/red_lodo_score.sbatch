#!/bin/bash
# Score the column-reduction variants on LODO against L1allM without whole-blood PCs (ref), alone and mixed 0.4 with
# the subcluster CellOT -> work/lc/tab/red_lodo (cases.pkl, lin_summary.csv)
#SBATCH -p normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -J red_score
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/red_score_%j.out
set -euo pipefail
source /scratch/users/bmerwan/atlas_heldout/env.sh
module load viz py-matplotlib/3.8.3_py312 py-scikit-learn/1.5.1_py312 >/dev/null 2>&1
W=/scratch/users/bmerwan/atlas_heldout/work/lc; DZ=$W/design_lodo.json; RL=$(seq -s " " 0 18); P=tab/pred; R="repeat_{r}"
N=""; for L in R_ownq3 R_ownq50 R_ownMS R_ownPath R_ownPC5 R_ownPC10 R_ownPC15 R_ownPLS3 R_ownPLS5 R_ownPLS10 L1M_q3; do N="$N v2_redA_lodo_$L"; done; for L in R_frPC3 R_frPC5 R_frPC10 R_frPLS3 R_frPLS5 R_frLin C_q3_lin C_ms_lin C_path_lin C_pls5_pls3; do N="$N v2_redB_lodo_$L"; done
python3 -m lc.tab_ens --design $DZ --repeats $RL --lambdas 1 --res-tag _red --names $N
M="CellOT_paired=cond_train_lodo/$R/none CellOT_cluster=cond_train_lodo/$R/cluster_K4_stable"
M="$M pfn_L1allM_noBPC=$P/v2_abl_lodo_L1allM_noBPC_l1/$R pfn_L1allM=$P/v2_x_lodo_L1allM_l1/$R pfn_L1M=$P/v2_x_lodo_L1M_l1/$R pfn_L1cat=$P/v2_lodo_L1cat_l1/$R"
for L in R_ownq3 R_ownq50 R_ownMS R_ownPath R_ownPC5 R_ownPC10 R_ownPC15 R_ownPLS3 R_ownPLS5 R_ownPLS10 L1M_q3; do M="$M pfn_$L=$P/v2_redA_lodo_${L}_l1/$R"; done; for L in R_frPC3 R_frPC5 R_frPC10 R_frPLS3 R_frPLS5 R_frLin C_q3_lin C_ms_lin C_path_lin C_pls5_pls3; do M="$M pfn_$L=$P/v2_redB_lodo_${L}_l1/$R"; done
python3 -m lc.tab_compare --methods $M --design $DZ --repeats $RL --ks 18 --exclude-stims IL33 PI --qs 50 75 90 \
  --rule-qs 50 75 90 --dump --out $W/tab/red_lodo
python3 -m lc.lineage2_summary --cases $W/tab/red_lodo/cases.pkl --cellot CellOT_cluster --ref pfn_L1allM_noBPC
