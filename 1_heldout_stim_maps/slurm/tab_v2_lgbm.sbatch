#!/bin/bash
# lc/tab_v2.py --model lgbm (same rows / features as the TabPFN rounds, pooled over celltypes), one repeat per task.
#   sbatch --array=0-4 slurm/tab_v2_lgbm.sbatch --layers L1cat L1F ...   -> work/lc/tab/v2lgb_<layer>/repeat_<r>/
#SBATCH -p normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -J tab_v2lgb
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/tab_v2lgb_%A_%a.out
set -euo pipefail
source /scratch/users/bmerwan/atlas_heldout/env.sh
module load viz py-matplotlib/3.8.3_py312 py-lightgbm/4.5.0_py312 py-scikit-learn/1.5.1_py312 >/dev/null 2>&1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python3 -m lc.tab_v2 --model lgbm --mode ct --repeat $SLURM_ARRAY_TASK_ID "$@"
