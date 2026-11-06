#!/bin/bash
# Array over night2/arms.txt (line i+1 = task i).  sbatch --array=0-10 stabl_launch/night2.sbatch
#SBATCH -p normal
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -J night2
#SBATCH -o /scratch/users/bmerwan/simco_work/onset_test/night2/logs/arm_%A_%a.out
#SBATCH -e /scratch/users/bmerwan/simco_work/onset_test/night2/logs/arm_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
ARM=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" /scratch/users/bmerwan/simco_work/onset_test/night2/arms.txt | cut -d'|' -f1)
echo "[INFO ] task ${SLURM_ARRAY_TASK_ID} -> ${ARM} on $(hostname)"
/scratch/users/bmerwan/simco/stabl_launch/night2_arm.sh "${ARM}"
