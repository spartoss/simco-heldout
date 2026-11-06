#!/bin/bash
# Score whatever finished, then build and execute the notebook. Safe to rerun by hand at any time.
#SBATCH -p normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -J lc_summary
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/summary_%j.out
set -uo pipefail
source /scratch/users/bmerwan/atlas_heldout/env_cpu.sh
python3 -m lc.score
python3 -m lc.build_notebook && jupyter nbconvert --to notebook --execute --inplace notebooks/learning_curve.ipynb
