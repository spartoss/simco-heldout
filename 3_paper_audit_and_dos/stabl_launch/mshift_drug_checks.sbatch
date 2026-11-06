#!/bin/bash
# Drug checks a-d, CellOT stim model + median_shift drug maps, 36_G1/52_G1 excluded -> drug_models/excl/cellot_mshiftdrug
#SBATCH -p normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -J mshift_chk
#SBATCH -o /scratch/users/bmerwan/simco_work/onset_test/drug_models/logs/mshift_chk_%j.out
set -euo pipefail
O=/scratch/users/bmerwan/simco_work/onset_test; M=$O/drug_models
module load python/3.9.0; source /scratch/users/bmerwan/venv-stabl/bin/activate
cd /scratch/users/bmerwan/simco; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python stabl_launch/drug_arm_checks.py --features $O/ina_13OG_final_long_allstims_filtered.csv \
  --selections "$O/foldfeats_GSS/Training CV/Selected Features STABL ALasso.csv" \
  --drug-table "$M/tables_qmap/{drug}_mshift_cellot.csv" --out $M/excl/cellot_mshiftdrug --exclude 36_G1 52_G1 --no-validate
