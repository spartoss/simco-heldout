#!/bin/bash
#SBATCH -p normal
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH -J simco_smoke
#SBATCH -o /scratch/users/bmerwan/simco_work/baselines/logs/smoke_%j.out
#SBATCH -e /scratch/users/bmerwan/simco_work/baselines/logs/smoke_%j.err

set -euo pipefail
module load python/3.9.0
source /scratch/users/bmerwan/venv-cellot/bin/activate
cd /scratch/users/bmerwan/simco

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "=== unit tests ==="
python -m baselines.test_baselines

echo
echo "=== smoke run: pDCs (smallest celltype) ==="
DATA=/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets
python -m baselines.run_baselines \
  --celltype pDCs \
  --data-dir "$DATA/ptb_concatenated_batchcorrected" \
  --features "$DATA/ptb_concatenated_per_condition_celltype/13features.txt" \
  --pairs-file /scratch/users/bmerwan/simco_work/baselines/pairs_168.txt \
  --outdir /scratch/users/bmerwan/simco_work/baselines/smoke \
  --cellot-results /home/groups/gbrice/ptb-drugscreen/ot/cellot/results/cross_validation_original_13_HVPV
