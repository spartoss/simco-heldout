#!/bin/bash
# DMSO-plate experiment (notebooks/dmso_plate.ipynb), step 1: one task per celltype (0-27).
# Single-plate caches (Unstim + stims of one plate control, <= 10k per donor x condition) -> work/lc/dmso/cache_DMSO{1,2,3}/
# and the pooled cache cut to the DMSO1 counts -> work/lc/dmso/cache_poolN1/. Reads the atlas h5ad read-only.
#SBATCH -p normal
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -J dmso_cache
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/dmso_cache_%A_%a.out
set -euo pipefail
source /scratch/users/bmerwan/atlas_heldout/env_cpu.sh
export HDF5_USE_FILE_LOCKING=FALSE
W=/scratch/users/bmerwan/atlas_heldout/work/lc/dmso
CT=$(python3 -c "from lc.paths import CELLTYPES; print(CELLTYPES[$SLURM_ARRAY_TASK_ID])")
python3 -m lc.cache --celltype "$CT" --treatments DMSO1 DMSO2 DMSO3 --outdir "$W/cache_{t}"
python3 -m lc.match_cache --celltype "$CT" --like $W/cache_DMSO1 --out $W/cache_poolN1
