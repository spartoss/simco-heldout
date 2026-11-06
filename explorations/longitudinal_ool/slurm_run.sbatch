#!/bin/bash
# M0-M2 + M3 + evaluation for one feature set (array index -> FEATS below)
#SBATCH -p normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -J lgt
#SBATCH -o /scratch/users/bmerwan/ool_longitudinal/work/logs/lgt_%A_%a.out
set -euo pipefail
FEATS=(sim freqmed_sim)
F=${FEATS[$SLURM_ARRAY_TASK_ID]}
cd /scratch/users/bmerwan/ool_longitudinal
module load python/3.9.0; source /scratch/users/bmerwan/venv-stabl/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python -m lgt.m012 --features $F
python -m lgt.evaluate --features $F > work/m012_$F/evaluate.txt
python -m lgt.m3 --features $F > work/m3_$F.txt
