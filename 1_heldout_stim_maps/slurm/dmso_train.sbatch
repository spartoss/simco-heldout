#!/bin/bash
# DMSO-plate experiment, step 2a: paired context-free CellOT (lc/cond_train --ctx none), k = 15, 20k iterations,
# one repeat per array task (0-4), all five arms in turn -> work/lc/dmso/train/repeat_<r>/<arm>/.
# Resumable (an arm with pred_q.npy is skipped; an interrupted arm resumes from state.pt).
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:16GB|GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -J dmso_train
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/dmso_train_%A_%a.out
set -euo pipefail
L=/scratch/users/bmerwan/atlas_heldout/work/lc; W=$L/dmso
source /scratch/users/bmerwan/atlas_heldout/env.sh
module load py-scikit-learn/1.5.1_py312 >/dev/null 2>&1
nvidia-smi --query-gpu=name,memory.total --format=csv
R=$SLURM_ARRAY_TASK_ID
for ARM in DMSO1 pool poolN1 DMSO2 DMSO3; do
    O=$W/train/repeat_$R/$ARM
    [ -f $O/pred_q.npy ] && { echo "[SKIP ] $ARM"; continue; }
    C=$W/cache_$ARM; [ $ARM = pool ] && C=$L/cache
    python3 -m lc.cond_train --repeat $R --ctx none --ks 15 --n-iters 20000 --cache-dir $C --out $O
done
