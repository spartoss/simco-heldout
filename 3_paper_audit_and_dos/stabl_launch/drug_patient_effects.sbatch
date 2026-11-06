#!/bin/bash
# Per-patient true vs predicted drug effects on the atlas, one task per celltype used by Fig 5b (21).
#SBATCH -p normal
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -J drug_pat
#SBATCH -o /scratch/users/bmerwan/simco_work/drug_choice/logs/pat_%A_%a.out
set -euo pipefail
O=/scratch/users/bmerwan/simco_work/drug_choice
CT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $O/celltypes_used.txt)
module load python/3.9.0; source /scratch/users/bmerwan/venv-cellot/bin/activate
cd /scratch/users/bmerwan/simco; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK HDF5_USE_FILE_LOCKING=FALSE
python -m baselines.drug_patient_effects --celltype "$CT" --pairs $O/pairs_used.csv --out $O/patient_effects
