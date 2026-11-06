#!/bin/bash
# Conditioning-architecture / penalty variants of paired CellOT (2026-10-13), k = 15 only, 20k it.
# Runs u = 0..54: variant line u / 5 + 1 of work/lc/cond_var/variants.tsv (name <tab> cond_train args), repeat u % 5.
# One run per task (Sherlock GPUs are exclusive-process: a second process on the same GPU fails at CUDA init, which
# is what killed every second run of the first, packed submission 47708953). Task t runs u = line t + 1 of
# work/lc/cond_var/runs_todo.txt.
#   sbatch --array=0-<n-1> slurm/cond_var.sbatch   -> work/lc/cond_train/repeat_<r>/<name>/   (resumable from state.pt)
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -J cond_var
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/cond_var_%A_%a.out
set -euo pipefail
W=/scratch/users/bmerwan/atlas_heldout/work/lc
source /scratch/users/bmerwan/atlas_heldout/env.sh
module load py-scikit-learn/1.5.1_py312 >/dev/null 2>&1
nvidia-smi --query-gpu=name,memory.total --format=csv

run() {
    local U=$1 R L LINE N ARGS
    R=$((U % 5)); L=$((U / 5 + 1))
    LINE=$(sed -n "${L}p" $W/cond_var/variants.tsv); N=$(echo "$LINE" | cut -f1); ARGS=$(echo "$LINE" | cut -f2)
    echo "run $U: variant $N repeat $R args $ARGS"
    python3 -m lc.cond_train --repeat $R --ks 15 --n-iters 20000 --out $W/cond_train/repeat_$R/$N $ARGS > $W/logs/cond_var_${N}_r${R}.out 2>&1
}
T=$SLURM_ARRAY_TASK_ID
U=$(sed -n "$((T + 1))p" $W/cond_var/runs_todo.txt)
run $U
