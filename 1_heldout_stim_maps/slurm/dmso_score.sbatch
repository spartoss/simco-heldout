#!/bin/bash
# DMSO-plate experiment, step 3: per-case table (lc/dmso_score.py -> work/lc/dmso/cases.pkl), then rebuild and execute
# notebooks/dmso_plate.ipynb. Scores whatever exists (missing CellOT arms show as pending), so it can run twice:
# after the baselines (interim) and after the GPU training (final).
#SBATCH -p normal
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -J dmso_score
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/dmso_score_%j.out
set -uo pipefail
source /scratch/users/bmerwan/atlas_heldout/env_cpu.sh
python3 -m lc.dmso_score
python3 -m lc.build_dmso_plate_notebook && jupyter nbconvert --to notebook --execute --inplace notebooks/dmso_plate.ipynb
