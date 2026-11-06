#!/bin/bash
# One task per repeat (0-4): 168 pairs x 10 training-set sizes = 1680 CellOT maps on one GPU. Resumes from state.pt.
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:16GB|GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -J lc_train
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/train_%A_%a.out
set -euo pipefail
W=/scratch/users/bmerwan/atlas_heldout/work/lc
if [[ ! -f $W/PARITY_PASS ]]; then echo "[STOP ] no PARITY_PASS (see logs/check_*.out)"; exit 1; fi
source /scratch/users/bmerwan/atlas_heldout/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -m lc.train --repeat $SLURM_ARRAY_TASK_ID --out $W/train/repeat_$SLURM_ARRAY_TASK_ID
