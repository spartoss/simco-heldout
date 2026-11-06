#!/bin/bash
# Leave-one-donor-out CellOT maps (work/lc/design_lodo.json: 19 folds, k = 18), the 112 pairs of the 5 kept stims
# (GMCSF IFNa IL246 LPS TNFa). Tasks 0-18: original CellOT (lc/train.py, 20k it. as the paired one, best-checkpoint selection) for fold t
# -> work/lc/train_lodo/repeat_<t>; tasks 19-37: paired CellOT (lc/cond_train.py --ctx none, 20k it.) for fold t-19
# -> work/lc/cond_train_lodo/repeat_<f>/none. Resumable from state.pt.
#   sbatch --array=0-37 slurm/lodo_cellot.sbatch
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_GEN:VLT|GPU_GEN:TUR|GPU_GEN:AMP|GPU_GEN:LOV|GPU_GEN:HPR
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -J lodo_cellot
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/lodo_cellot_%A_%a.out
set -euo pipefail
W=/scratch/users/bmerwan/atlas_heldout/work/lc
source /scratch/users/bmerwan/atlas_heldout/env.sh
module load py-scikit-learn/1.5.1_py312 >/dev/null 2>&1
nvidia-smi --query-gpu=name,memory.total --format=csv
PAIRS=$(awk -F, '$1 ~ /^(GMCSF|IFNa|IL246|LPS|TNFa)$/ {printf "%s:%s ", $1, $2}' /scratch/users/bmerwan/atlas_heldout/meta/pairs.txt)
t=$SLURM_ARRAY_TASK_ID
if (( t < 19 )); then
  python3 -m lc.train --design $W/design_lodo.json --repeat $t --ks 18 --n-iters 20000 --pairs $PAIRS --out $W/train_lodo/repeat_$t
else
  f=$((t - 19))
  python3 -m lc.cond_train --design $W/design_lodo.json --repeat $f --ctx none --ks 18 --n-iters 20000 --pairs $PAIRS \
    --out $W/cond_train_lodo/repeat_$f/none
fi
