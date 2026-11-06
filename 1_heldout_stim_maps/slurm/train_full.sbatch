#!/bin/bash
# Uncapped learning curve. Task t (0-9): repeat t/2; t%2 = 0 -> donor curve (k = 1..15, all cells),
# t%2 = 1 -> cells axis (k = 1, 2, 4, 15 x 1k / 5k / 20k training cells per donor). 20k iterations: the capped run's
# trajectory shows the same skill at 20k and 50k. Resumable from state.pt.
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:16GB|GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -J lcfull_train
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/trainfull_%A_%a.out
set -euo pipefail
W=/scratch/users/bmerwan/atlas_heldout/work/lc
R=$((SLURM_ARRAY_TASK_ID / 2))
source /scratch/users/bmerwan/atlas_heldout/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
if (( SLURM_ARRAY_TASK_ID % 2 == 0 )); then
  python3 -m lc.train --repeat $R --cache-dir $W/cache_full --caps 0 --n-iters 20000 \
    --out $W/train_full/repeat_$R/curve
else
  python3 -m lc.train --repeat $R --cache-dir $W/cache_full --ks 1 2 4 15 --caps 1000 5000 20000 --n-iters 20000 \
    --out $W/train_full/repeat_$R/cells
fi
