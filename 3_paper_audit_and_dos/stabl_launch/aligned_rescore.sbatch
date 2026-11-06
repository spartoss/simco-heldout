#!/bin/bash
# 97-config XGBoost sweep on stored 100-split selections, rows/groups aligned (arms: paper_protocol/aligned/arms.txt)
#SBATCH -p normal
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -J aligned_rs
#SBATCH -o /scratch/users/bmerwan/simco_work/onset_test/paper_protocol/aligned/logs/arm_%A_%a.out
#SBATCH -e /scratch/users/bmerwan/simco_work/onset_test/paper_protocol/aligned/logs/arm_%A_%a.err
set -euo pipefail
L=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" /scratch/users/bmerwan/simco_work/onset_test/paper_protocol/aligned/arms.txt)
IFS='|' read -r ARM FEATS SELS <<< "$L"
module load python/3.9.0
source /scratch/users/bmerwan/venv-stabl/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python /scratch/users/bmerwan/simco/stabl_launch/aligned_rescore.py --arm "$ARM" --features "$FEATS" --selections "$SELS"
