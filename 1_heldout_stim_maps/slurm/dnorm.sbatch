#!/bin/bash
# Per-donor normalisation / input clipping experiment (2026-10-06): paired context-free CellOT (lc/cond_train.py
# --ctx none), k = 15 only, one repeat per array task.
#   sbatch --array=0-4 slurm/dnorm.sbatch <name> <cond_train args...>   -> work/lc/cond_train/repeat_<r>/<name>/
# Resumable from state.pt (resubmit the same task).
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:16GB|GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -J dnorm
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/dnorm_%x_%A_%a.out
set -euo pipefail
W=/scratch/users/bmerwan/atlas_heldout/work/lc
source /scratch/users/bmerwan/atlas_heldout/env.sh
module load py-scikit-learn/1.5.1_py312 >/dev/null 2>&1
nvidia-smi --query-gpu=name,memory.total --format=csv
N=$1; shift; R=$SLURM_ARRAY_TASK_ID
python3 -m lc.cond_train --repeat $R --ctx none --ks 15 --n-iters 20000 --out $W/cond_train/repeat_$R/$N "$@"
