#!/bin/bash
# One task per celltype (0-27): per-donor cells -> WORK/cache/<ct>.npz
#SBATCH -p normal
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -J lc_cache
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/cache_%A_%a.out
set -euo pipefail
source /scratch/users/bmerwan/atlas_heldout/env_cpu.sh
export HDF5_USE_FILE_LOCKING=FALSE
CT=$(python3 -c "from lc.paths import CELLTYPES; print(CELLTYPES[$SLURM_ARRAY_TASK_ID])")
python3 -m lc.cache --celltype "$CT"
