#!/bin/bash
# Closed-form maps on the uncapped cache, same two parts as train_full, one task per repeat (0-4).
#SBATCH -p normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -J lcfull_base
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/basefull_%A_%a.out
set -euo pipefail
W=/scratch/users/bmerwan/atlas_heldout/work/lc
source /scratch/users/bmerwan/atlas_heldout/env_cpu.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python3 -m lc.baselines --repeat $SLURM_ARRAY_TASK_ID --cache-dir $W/cache_full --caps 0 --out $W/baselines_full/curve
python3 -m lc.baselines --repeat $SLURM_ARRAY_TASK_ID --cache-dir $W/cache_full --ks 1 2 4 15 --caps 1000 5000 20000 \
  --out $W/baselines_full/cells
