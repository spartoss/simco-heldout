#!/bin/bash
# CondOT-faithful, one PICNN per celltype with one-hot stim context (lc/condot_train.py --mode celltype), CPU.
# Array index = repeat * 28 + celltype index.   sbatch --array=0-139 slurm/condot_ct.sbatch
#SBATCH -p normal
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH -J condot_ct
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/condot_ct_%A_%a.out
set -euo pipefail
source /scratch/users/bmerwan/atlas_heldout/env.sh
CTS=($(python3 -c "from lc.paths import PAIRS; print(' '.join(sorted({c for _, c in PAIRS})))"))
R=$((SLURM_ARRAY_TASK_ID / 28)); C=${CTS[$((SLURM_ARRAY_TASK_ID % 28))]}
python3 -m lc.condot_train --repeat $R --mode celltype --celltypes $C --n-iters 50000 --threads 4 \
    --out /scratch/users/bmerwan/atlas_heldout/work/lc/condot/celltype/repeat_$R/$C
