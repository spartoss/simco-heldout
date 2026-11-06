#!/bin/bash
# Build the six feature sets (heldout/quantile_arms.py) from the OOL features.
#SBATCH -p normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -J ho_arms
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/heldout/logs/arms_%j.out
set -euo pipefail
source /scratch/users/bmerwan/atlas_heldout/env_cpu.sh
python3 -m heldout.quantile_arms
