#!/bin/bash
# CondOT-faithful, one PICNN for all 168 pairs, context = one-hot stim + one-hot celltype (lc/condot_train.py --mode global).
#   sbatch --array=0-4 slurm/condot_global.sbatch
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -J condot_global
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/condot_global_%A_%a.out
set -euo pipefail
source /scratch/users/bmerwan/atlas_heldout/env.sh
R=$SLURM_ARRAY_TASK_ID
python3 -m lc.condot_train --repeat $R --mode global --n-iters 150000 --eval-every 5000 \
    --out /scratch/users/bmerwan/atlas_heldout/work/lc/condot/global/repeat_$R
