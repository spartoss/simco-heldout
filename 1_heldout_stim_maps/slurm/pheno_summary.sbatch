#!/bin/bash
# Per-well summaries of all 38 markers, one task per celltype (0-27) -> work/heldout/pheno/<ct>.csv.gz
#SBATCH -p normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH -J ho_pheno
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/heldout/logs/pheno_%A_%a.out
set -euo pipefail
cd /scratch/users/bmerwan/atlas_heldout
source /scratch/users/bmerwan/atlas_heldout/env_cpu.sh; export HDF5_USE_FILE_LOCKING=FALSE
CT=$(python3 -c "from lc.paths import CELLTYPES; print(CELLTYPES[$SLURM_ARRAY_TASK_ID])")
python3 -m heldout.pheno_summary --celltype "$CT"
