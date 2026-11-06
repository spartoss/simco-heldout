#!/bin/bash
# Batch-corrected per-donor drug-screen cells, one task per celltype (0-27) -> work/heldout/cache_drug/<ct>.npz
#SBATCH -p normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -J ho_drugcache
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/heldout/logs/drugcache_%A_%a.out
set -euo pipefail
source /scratch/users/bmerwan/atlas_heldout/env_cpu.sh; export HDF5_USE_FILE_LOCKING=FALSE
CT=$(python3 -c "from lc.paths import CELLTYPES; print(CELLTYPES[$SLURM_ARRAY_TASK_ID])")
python3 -m heldout.drug_cache --celltype "$CT"
