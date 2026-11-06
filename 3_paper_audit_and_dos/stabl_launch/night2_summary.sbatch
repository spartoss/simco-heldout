#!/bin/bash
#SBATCH -p normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -J night2_sum
#SBATCH -o /scratch/users/bmerwan/simco_work/onset_test/night2/logs/summary_%j.out
module load python/3.9.0
source /scratch/users/bmerwan/venv-stabl/bin/activate
python /scratch/users/bmerwan/simco/stabl_launch/night2_compare.py
