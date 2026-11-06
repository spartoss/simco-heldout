#!/bin/bash
# One task per drug: drug-treated OOL tables under the CellOT, gauss_ot and measured feature sets.
#   sbatch --array=0-8 stabl_launch/drug_models_build.sbatch
#SBATCH -p normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -J drug_build
#SBATCH -o /scratch/users/bmerwan/simco_work/onset_test/drug_models/logs/build_%A_%a.out
#SBATCH -e /scratch/users/bmerwan/simco_work/onset_test/drug_models/logs/build_%A_%a.err
set -euo pipefail
DRUGS=(SA RIF SALPZ CHT THF LPZ MAP PRA MF)
D=${DRUGS[$SLURM_ARRAY_TASK_ID]}
M=/scratch/users/bmerwan/simco_work/onset_test/drug_models/tables
if [[ -s $M/${D}_cellot.csv && -s $M/${D}_gaussot.csv && -s $M/${D}_distrib.csv ]]; then echo "[SKIP ] $D"; exit 0; fi
module load python/3.9.0
source /scratch/users/bmerwan/venv-cellot/bin/activate
cd /scratch/users/bmerwan/simco
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK HDF5_USE_FILE_LOCKING=FALSE
python -m baselines.drug_features --drug "$D" --out "$M"
