#!/bin/bash
#SBATCH -J ml_dos_sum
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 1:00:00
#SBATCH -o /scratch/users/bmerwan/simco_work/onset_test/ml_dos/logs/summary_%j.out
set -euo pipefail
OUT=/scratch/users/bmerwan/simco_work/onset_test/ml_dos
module load python/3.9.0
source /scratch/users/bmerwan/venv-stabl/bin/activate
cd "$OUT/code"
python summarize.py
echo "$(date '+%F %T') summary job $SLURM_JOB_ID done -> summary/" >> "$OUT/STATUS.md"
