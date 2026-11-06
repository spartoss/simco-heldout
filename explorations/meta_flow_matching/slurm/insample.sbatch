#!/bin/bash
# In-sample fit (training donors' val-side cells) of CellOT_paired and fm / mfm / mfm_blood (mfm/insample.py).
# Task t = repeat t + 1 (repeats 1-4), k = 3 and 15 -> work/insample_r<r>/ ; merge with mfm/insample_merge (inline).
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:16GB|GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --array=0-3
#SBATCH -J insample
#SBATCH -o /scratch/users/bmerwan/mfm_heldout/work/logs/%x_%A_%a.out
set -euo pipefail
source /scratch/users/bmerwan/mfm_heldout/env.sh
R=$((SLURM_ARRAY_TASK_ID + 1))
python3 -c "
import sys, mfm.insample as I
I.OUT = I.OUT.parent / 'insample_r$R'
sys.argv = ['insample', '--repeats', '$R', '--ks', '3', '15']
I.main()"
