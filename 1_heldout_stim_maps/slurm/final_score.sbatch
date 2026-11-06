#!/bin/bash
# Final comparison on leave-one-donor-out: prediction dirs for the TabPFN steps, case table with every method
# (tab_compare --dump -> work/lc/tab/final/cases.pkl), lc/final_compare.py, notebooks/final_comparison.ipynb.
#   sbatch --dependency=afterok:<lodo_covot>:<lodo_steps> slurm/final_score.sbatch
#SBATCH -p normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -J final_score
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/final_score_%j.out
set -euo pipefail
cd /scratch/users/bmerwan/atlas_heldout
bash -lc '
set -euo pipefail
source env.sh; module load viz py-matplotlib/3.8.3_py312 py-scikit-learn/1.5.1_py312 >/dev/null 2>&1
DZ=work/lc/design_lodo.json; RL=$(seq -s " " 0 18); P=tab/pred; R="repeat_{r}"
python3 -m lc.tab_ens --design $DZ --repeats $RL --lambdas 1 --res-tag _final --names v2_lodo_S0 v2_lodo_S1_5 v2_lodo_S2 v2_lodo_S3 v2_lodo_S4_3
M="CellOT_paper=train_lodo/$R CellOT_paired=cond_train_lodo/$R/none CellOT_pcblood=cond_train_lodo/$R/pcblood_wd1"
M="$M CellOT_film=cond_train_lodo/$R/pcb_film_wd1 CellOT_cluster=cond_train_lodo/$R/cluster_K4_stable"
for L in S0 S1_5 S2 S3 S4_3; do M="$M pfn_$L=$P/v2_lodo_${L}_l1/$R"; done
for L in L1cat L1F L1all_pc L1all_raw; do M="$M pfn_$L=$P/v2_lodo_${L}_l1/$R lgb_$L=$P/v2lgb_lodo_${L}_l1/$R"; done
for L in L1X L1M L1allM; do M="$M pfn_$L=$P/v2_x_lodo_${L}_l1/$R lgb_$L=$P/v2lgb_x_lodo_${L}_l1/$R"; done
python3 -m lc.tab_compare --methods $M --design $DZ --repeats $RL --ks 18 --exclude-stims IL33 PI --qs 50 75 90 \
  --rule-qs 50 75 90 --dump --out work/lc/tab/final
python3 -m lc.final_compare
'
bash -lc 'source env_cpu.sh >/dev/null 2>&1; export PYTHONPATH=/scratch/users/bmerwan/atlas_heldout
python3 -m lc.build_final_notebook && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/final_comparison.ipynb'
