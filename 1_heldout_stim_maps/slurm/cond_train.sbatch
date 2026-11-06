#!/bin/bash
# Patient-covariate experiment, steps 2-3 (lc/cond_train.py). Capped cache, 20k iterations, paper ICNN.
# Tasks 0-9 : repeat t/2, ctx none (t even, the control) or pc (t odd), k = 3 5 8 15  -> cond_train/repeat_<r>/<ctx>/
# Tasks 10-11: donor-ID oracle (all 19 donors, scored on their own val side), ctx none / onehot -> cond_train/oracle/<ctx>/
# Resumable from state.pt.
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:16GB|GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -J cond_train
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/cond_train_%A_%a.out
set -euo pipefail
W=/scratch/users/bmerwan/atlas_heldout/work/lc
source /scratch/users/bmerwan/atlas_heldout/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
t=$SLURM_ARRAY_TASK_ID
if (( t < 10 )); then
  R=$((t / 2)); C=$([ $((t % 2)) -eq 0 ] && echo none || echo pc)
  python3 -m lc.cond_train --repeat $R --ctx $C --ks 3 5 8 15 --n-iters 20000 --out $W/cond_train/repeat_$R/$C
else
  C=$([ $t -eq 10 ] && echo none || echo onehot)
  python3 -m lc.cond_train --oracle --ctx $C --n-iters 20000 --out $W/cond_train/oracle/$C
fi
