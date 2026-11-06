#!/bin/bash
# Held-out project GPU training. --export=EXP=<drugs|stims_all>:
#   drugs      array 0-9 : repeat i/2, celltypes CELLTYPES[i%2::2]; drug maps (control -> drug under a stim), 128-128-64-32,
#                          trained on the 15 non-test donors; test donors scored on the drug step alone at each eval
#   stims_all  single    : Unstim -> stim for all 196 pairs, all 19 donors (features for the OOL women)
#   score_drugs array 0-4 : held-out drug-effect scores of repeat i (step / chain / feature-level) -> drugs/scores/
#   ool_features single  : apply those maps (and gauss_ot) to the 177 OOL women -> features/ool_*_allq.csv
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:16GB|GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/heldout/logs/%x_%A_%a.out
set -euo pipefail
W=/scratch/users/bmerwan/atlas_heldout/work/heldout; i=${SLURM_ARRAY_TASK_ID:-0}
source /scratch/users/bmerwan/atlas_heldout/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
case "$EXP" in
  drugs)
    R=$((i / 2)); CTS=$(python3 -c "from lc.paths import CELLTYPES; print(' '.join(CELLTYPES[$i % 2::2]))")
    python3 -m lc.train --spec $W/designs/drugs/r$R.json --celltypes $CTS --hidden 128 128 64 32 \
      --cache-dir $W/cache_drug --n-iters 20000 --eval-every 5000 --max-test-cells 5000 --out $W/drugs/maps/r$R/h$((i % 2)) ;;
  score_drugs)
    python3 -m heldout.score_drugs --repeat $i ;;
  ool_features)
    python3 -m heldout.ool_features ;;
  stims_all)
    python3 -m lc.train --spec $W/designs/stims_all.json --cache-dir $W/cache_stim_all --n-iters 20000 --eval-every 20000 \
      --out $W/stims_all ;;
esac
