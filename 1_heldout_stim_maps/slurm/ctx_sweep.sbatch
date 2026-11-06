#!/bin/bash
# Context weight-decay sweep for lc/cond_train.py --ctx ft (repeat 0, 4 celltypes, k 5 15): picks --ctx-wd.
# Task t -> lambda WD[t] -> work/lc/ctx_sweep/ft_pretrained_wd<lambda>/
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:16GB|GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-5
#SBATCH -J ctx_sweep
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/ctx_sweep_%A_%a.out
set -euo pipefail
W=/scratch/users/bmerwan/atlas_heldout/work/lc
source /scratch/users/bmerwan/atlas_heldout/env.sh
WD=(0 0.001 0.01 0.1 1 10); L=${WD[$SLURM_ARRAY_TASK_ID]}
python3 -m lc.cond_train --repeat 0 --ctx ft --ft-init pretrained --ctx-wd $L --celltypes pDCs cMCs CD4Tnaive Bcells \
  --ks 5 15 --n-iters 20000 --out $W/ctx_sweep/ft_pretrained_wd$L
