#!/bin/bash
#SBATCH -p normal
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH -J simco_inspect
#SBATCH -o /scratch/users/bmerwan/simco_work/baselines/logs/inspect_%j.out
#SBATCH -e /scratch/users/bmerwan/simco_work/baselines/logs/inspect_%j.err

set -euo pipefail
module load python/3.9.0
source /scratch/users/bmerwan/venv-cellot/bin/activate

DATA=/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_batchcorrected
python /scratch/users/bmerwan/simco/baselines/inspect_h5ad.py "$DATA/pDCs_HVPV.h5ad"
