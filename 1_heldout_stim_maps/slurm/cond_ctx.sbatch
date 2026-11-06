#!/bin/bash
# lc/cond_train.py on the learning-curve design, one repeat per array task, any context.
#   sbatch --array=0-4 slurm/cond_ctx.sbatch <name> <cond_train args...>   -> work/lc/cond_train/repeat_<r>/<name>/
# e.g. cluster_K4_stable --ctx cluster --K 4 --stable ;  ft_pre_wd0.1 --ctx ft --ft-init pretrained --ctx-wd 0.1
# Resumable from state.pt (resubmit the same task).
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:16GB|GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -J cond_ctx
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/cond_ctx_%x_%A_%a.out
set -euo pipefail
W=/scratch/users/bmerwan/atlas_heldout/work/lc
source /scratch/users/bmerwan/atlas_heldout/env.sh
module load py-scikit-learn/1.5.1_py312 >/dev/null 2>&1
nvidia-smi --query-gpu=name,memory.total --format=csv
N=$1; shift; R=$SLURM_ARRAY_TASK_ID
python3 -m lc.cond_train --repeat $R --ks 3 5 8 15 --n-iters 20000 --out $W/cond_train/repeat_$R/$N "$@"
