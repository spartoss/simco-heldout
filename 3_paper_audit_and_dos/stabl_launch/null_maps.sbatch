#!/bin/bash
# Array over null_maps/arms.txt (line i+1 = task i). Each task: build table -> Stabl x2.
#   sbatch --array=0-8 stabl_launch/null_maps.sbatch     (task 9 = cellot_repro, build-only)
#SBATCH -p normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -J null_maps
#SBATCH -o /scratch/users/bmerwan/simco_work/onset_test/null_maps/logs/arm_%A_%a.out
#SBATCH -e /scratch/users/bmerwan/simco_work/onset_test/null_maps/logs/arm_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
ARM=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" /scratch/users/bmerwan/simco_work/onset_test/null_maps/arms.txt | awk '{print $1}')
echo "[INFO ] task ${SLURM_ARRAY_TASK_ID} -> ${ARM} on $(hostname)"
/scratch/users/bmerwan/simco/stabl_launch/null_maps_arm.sh "${ARM}" all
