#!/bin/bash
#SBATCH -J ml_dos_refresh
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 1:00:00
#SBATCH -o /scratch/users/bmerwan/simco_work/onset_test/ml_dos/logs/refresh_%j.out
# Re-summarise every finished run and re-execute notebooks/ml_dos_supervised.ipynb (tables/figures only;
# the written readings in notebooks/ml_dos_readings.md are not changed by this job).
set -euo pipefail
OUT=/scratch/users/bmerwan/simco_work/onset_test/ml_dos
NB=/scratch/users/bmerwan/simco/notebooks
module load python/3.9.0
source /scratch/users/bmerwan/venv-stabl/bin/activate
cd "$OUT/code" && python summarize.py
deactivate
source /scratch/users/bmerwan/venv-cellot/bin/activate
python "$NB/build_ml_dos.py"
jupyter nbconvert --to notebook --execute --inplace --allow-errors --ExecutePreprocessor.kernel_name=cellot \
    "$NB/ml_dos_supervised.ipynb"
echo "$(date '+%F %T') refresh job $SLURM_JOB_ID: summary + notebook rebuilt" >> "$OUT/STATUS.md"
