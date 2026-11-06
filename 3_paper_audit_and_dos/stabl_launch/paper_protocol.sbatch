#!/bin/bash
# Array over paper_protocol/arms.txt (line i+1 = task i).  sbatch --array=0-3 stabl_launch/paper_protocol.sbatch
#SBATCH -p normal
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -J paper_proto
#SBATCH -o /scratch/users/bmerwan/simco_work/onset_test/paper_protocol/logs/arm_%A_%a.out
#SBATCH -e /scratch/users/bmerwan/simco_work/onset_test/paper_protocol/logs/arm_%A_%a.err
set -euo pipefail
ARM=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" /scratch/users/bmerwan/simco_work/onset_test/paper_protocol/arms.txt | cut -d'|' -f1)
echo "[INFO ] task ${SLURM_ARRAY_TASK_ID} -> ${ARM} on $(hostname)"
/scratch/users/bmerwan/simco/stabl_launch/paper_protocol_arm.sh "${ARM}"
