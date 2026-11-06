#!/bin/bash
# After simq_stabl_n100: aligned params_43 rescore of both simq arms with 36_G1 / 52_G1 excluded -> paper_protocol/aligned_excl/
#SBATCH -p normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -J simq100_excl
#SBATCH -o /scratch/users/bmerwan/simco_work/onset_test/paper_protocol/logs/simq100_excl_%j.out
set -euo pipefail
O=/scratch/users/bmerwan/simco_work/onset_test; P=$O/paper_protocol
module load python/3.9.0; source /scratch/users/bmerwan/venv-stabl/bin/activate
cd /scratch/users/bmerwan/simco/stabl_launch; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
for ARM in comb_cellot_simq comb_gaussot_simq; do
  python aligned_rescore.py --arm $ARM --features $P/tables/$ARM.csv --selections $P/stabl_n100/$ARM \
    --out $P/aligned_excl --params-glob $O/param_grid_files/params_43.json --exclude 36_G1 52_G1
done
