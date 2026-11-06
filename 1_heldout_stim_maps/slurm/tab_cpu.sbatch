#!/bin/bash
# lc/tab_predict.py, CPU models (lgbm, ridge), one repeat per array task -> work/lc/tab/<model>_<layer>/repeat_<r>/
#   sbatch --array=0-4 slurm/tab_cpu.sbatch
#SBATCH -p normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -J tab_cpu
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/tab_cpu_%A_%a.out
set -euo pipefail
source /scratch/users/bmerwan/atlas_heldout/env.sh
module load viz py-matplotlib/3.8.3_py312 py-lightgbm/4.5.0_py312 py-scikit-learn/1.5.1_py312 >/dev/null 2>&1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
R=$SLURM_ARRAY_TASK_ID
python3 -m lc.tab_predict --model ridge --repeat $R
python3 -m lc.tab_predict --model lgbm --repeat $R
