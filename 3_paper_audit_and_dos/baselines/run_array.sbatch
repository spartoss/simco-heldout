#!/bin/bash
# Leave-one-patient-out baseline benchmark, one array task per celltype.
#
#   sbatch --array=0-26 baselines/run_array.sbatch celltypes_small.txt
#   sbatch --array=0-0 -c 24 --mem=192G --time=12:00:00 baselines/run_array.sbatch celltypes_granulocytes.txt
#
# Granulocytes_HVPV.h5ad is 30 GB (next largest is 2.4 GB) so it gets its own
# submission with more memory rather than sizing all 28 tasks for the outlier.
#SBATCH -p normal
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -J simco_baseline
#SBATCH -o /scratch/users/bmerwan/simco_work/baselines/logs/base_%A_%a.out
#SBATCH -e /scratch/users/bmerwan/simco_work/baselines/logs/base_%A_%a.err

set -euo pipefail
module load python/3.9.0
source /scratch/users/bmerwan/venv-cellot/bin/activate
cd /scratch/users/bmerwan/simco

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

WORK=/scratch/users/bmerwan/simco_work/baselines
LIST="${WORK}/${1:?usage: sbatch run_array.sbatch <celltype-list-file>}"
CELLTYPE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
echo "[INFO] task ${SLURM_ARRAY_TASK_ID} -> ${CELLTYPE}"

DATA=/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets
python -m baselines.run_baselines \
  --celltype "$CELLTYPE" \
  --data-dir "$DATA/ptb_concatenated_batchcorrected" \
  --features "$DATA/ptb_concatenated_per_condition_celltype/13features.txt" \
  --pairs-file "$WORK/pairs_168.txt" \
  --outdir "$WORK/lopo" \
  --cellot-results /home/groups/gbrice/ptb-drugscreen/ot/cellot/results/cross_validation_original_13_HVPV
