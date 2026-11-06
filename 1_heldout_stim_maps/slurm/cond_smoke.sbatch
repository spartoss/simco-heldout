#!/bin/bash
# Smoke test of lc/cond_train.py: pDCs + CD4Tnaive, all arms, 400 iterations; ms/iter for sizing the real run.
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:16GB|GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -J cond_smoke
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/cond_smoke_%j.out
set -euo pipefail
W=/scratch/users/bmerwan/atlas_heldout/work/lc
source /scratch/users/bmerwan/atlas_heldout/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
rm -rf $W/smoke/cond_train_* $W/smoke/cond_oracle*
for c in none pc; do
  python3 -m lc.cond_train --repeat 0 --ctx $c --celltypes pDCs CD4Tnaive --ks 3 15 --n-iters 400 --eval-every 200 \
    --out $W/smoke/cond_train_$c
done
for c in none onehot; do
  python3 -m lc.cond_train --oracle --ctx $c --celltypes pDCs --n-iters 400 --eval-every 200 --out $W/smoke/cond_oracle_$c
done
