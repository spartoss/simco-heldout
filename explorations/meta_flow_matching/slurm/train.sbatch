#!/bin/bash
# Meta Flow Matching on the atlas learning-curve design (mfm/train.py). Array task t over (repeat, k):
# repeat t / 4, k = KS[t % 4]  -> work/runs/<name>/repeat_<r>/k<k>/.
#   sbatch --array=0-19 slurm/train.sbatch <arm> [<name> [args]]      (name defaults to the arm)
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:16GB|GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -J mfm
#SBATCH -o /scratch/users/bmerwan/mfm_heldout/work/logs/%x_%A_%a.out
set -euo pipefail
source /scratch/users/bmerwan/mfm_heldout/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
KS=(3 5 8 15); T=$SLURM_ARRAY_TASK_ID; R=$((T / 4)); K=${KS[$((T % 4))]}; A=$1; N=${2:-$1}; shift; [ $# -gt 0 ] && shift
python3 -m mfm.train --repeat $R --k $K --arm $A --out work/runs/$N/repeat_$R/k$K "$@"
