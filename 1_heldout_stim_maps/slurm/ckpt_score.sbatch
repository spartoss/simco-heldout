#!/bin/bash
# Checkpoint averages + scoring of the compact set on LODO -> work/lc/tab/ckpt_lodo (lin_summary.csv)
#SBATCH -p normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -J ckpt_score
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/ckpt_score_%j.out
set -euo pipefail
source /scratch/users/bmerwan/atlas_heldout/env.sh
module load viz py-matplotlib/3.8.3_py312 py-scikit-learn/1.5.1_py312 >/dev/null 2>&1
W=/scratch/users/bmerwan/atlas_heldout/work/lc; DZ=$W/design_lodo.json; RL=$(seq -s " " 0 18); P=tab/pred; R="repeat_{r}"
A="v2_ck09_lodo_C_ms_lin v2_ck2no_lodo_C_ms_lin v2_ckwyl_lodo_C_ms_lin"
python3 -m lc.avg_preds --names $A --out v2_ckavg3_lodo_C_ms_lin --repeats $RL
python3 -m lc.avg_preds --names $A v2_redB_lodo_C_ms_lin --out v2_ckavg4_lodo_C_ms_lin --repeats $RL
python3 -m lc.tab_ens --design $DZ --repeats $RL --lambdas 1 --res-tag _ckpt --names $A v2_ckavg3_lodo_C_ms_lin v2_ckavg4_lodo_C_ms_lin
M="CellOT_paired=cond_train_lodo/$R/none CellOT_cluster=cond_train_lodo/$R/cluster_K4_stable"
M="$M pfn_default=$P/v2_redB_lodo_C_ms_lin_l1/$R pfn_ck09=$P/v2_ck09_lodo_C_ms_lin_l1/$R pfn_ck2no=$P/v2_ck2no_lodo_C_ms_lin_l1/$R"
M="$M pfn_ckwyl=$P/v2_ckwyl_lodo_C_ms_lin_l1/$R pfn_avg3=$P/v2_ckavg3_lodo_C_ms_lin_l1/$R pfn_avg4=$P/v2_ckavg4_lodo_C_ms_lin_l1/$R"
M="$M pfn_L1allM_noBPC=$P/v2_abl_lodo_L1allM_noBPC_l1/$R"
python3 -m lc.tab_compare --methods $M --design $DZ --repeats $RL --ks 18 --exclude-stims IL33 PI --qs 50 75 90 \
  --rule-qs 50 75 90 --dump --out $W/tab/ckpt_lodo
python3 -m lc.lineage2_summary --cases $W/tab/ckpt_lodo/cases.pkl --cellot CellOT_cluster --ref pfn_default
