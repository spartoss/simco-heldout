#!/bin/bash
# gauss_ot drug maps, one task per celltype (simco_work/baselines/gaussot_celltypes.txt).
#SBATCH -p normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -J gdrug_fit
#SBATCH -o /scratch/users/bmerwan/simco_work/onset_test/drug_models/logs/gfit_%A_%a.out
#SBATCH -e /scratch/users/bmerwan/simco_work/onset_test/drug_models/logs/gfit_%A_%a.err
set -euo pipefail
CT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" /scratch/users/bmerwan/simco_work/baselines/gaussot_celltypes.txt)
module load python/3.9.0; source /scratch/users/bmerwan/venv-cellot/bin/activate
cd /scratch/users/bmerwan/simco; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK HDF5_USE_FILE_LOCKING=FALSE
python -m baselines.gauss_drug_maps --celltype "$CT" --outdir /scratch/users/bmerwan/simco_work/onset_test/drug_models/gauss_drug_maps
