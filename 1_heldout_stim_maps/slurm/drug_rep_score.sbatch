#!/bin/bash
# Score TabPFN (drug_rep / drug_steprep) and the drug maps on the repeated design, same cases:
# ~/simco_interview_figures/drug_maps_mse.py -> data/drug_maps_cases.csv
#   sbatch --dependency=afterok:<drug_rep>:<drug_steprep> slurm/drug_rep_score.sbatch
#SBATCH -p normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH -J drug_rep_score
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/drug_rep_score_%j.out
set -euo pipefail
cd /scratch/users/bmerwan/atlas_heldout
source env_cpu.sh >/dev/null 2>&1
python ~/simco_interview_figures/drug_maps_mse.py
