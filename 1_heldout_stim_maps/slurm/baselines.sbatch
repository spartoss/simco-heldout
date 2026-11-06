#!/bin/bash
# Closed-form maps on the same design, one task per repeat (0-4).
#SBATCH -p normal
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -J lc_base
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/base_%A_%a.out
set -euo pipefail
source /scratch/users/bmerwan/atlas_heldout/env_cpu.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python3 -m lc.baselines --repeat $SLURM_ARRAY_TASK_ID
