#!/bin/bash
# Runs after the null_maps array: paired tests vs CellOT / gauss_ot.
#   sbatch --dependency=afterany:<array_jobid> stabl_launch/null_maps_summary.sbatch
#SBATCH -p normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -J null_maps_sum
#SBATCH -o /scratch/users/bmerwan/simco_work/onset_test/null_maps/logs/summary_%j.out
module load python/3.9.0
source /scratch/users/bmerwan/venv-stabl/bin/activate
python /scratch/users/bmerwan/simco/stabl_launch/null_maps_compare.py
