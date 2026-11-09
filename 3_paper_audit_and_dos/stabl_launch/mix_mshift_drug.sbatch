#!/bin/bash
# Drug arm of our pipeline (mix_both: 0.4 CellOT + 0.6 TabPFN stims) with the AVERAGE drug effect: each feature gets
# the pooled atlas drug effect on the median (baselines.drug_features_qmap --method median_shift), then drug checks a-d
# with 36_G1 / 52_G1 excluded -> drug_models/excl/mix_both_mshiftdrug/.
#SBATCH -p normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -J mix_mshift_drug
#SBATCH -o /scratch/users/bmerwan/simco_work/onset_test/drug_models/logs/mix_mshift_drug_%j.out
set -euo pipefail
O=/scratch/users/bmerwan/simco_work/onset_test; P=$O/paper_protocol; M=$O/drug_models
module load python/3.9.0; source /scratch/users/bmerwan/venv-stabl/bin/activate
cd /scratch/users/bmerwan/simco; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ARM=mix_both
python -m baselines.drug_features_qmap --method median_shift --maps $M/qmap_drug_maps --features $P/tables/$ARM.csv \
  --out $M/tables_qmap --tag mshift_$ARM
python stabl_launch/drug_arm_checks.py --features $P/tables/$ARM.csv \
  --selections "$P/stabl_n100/$ARM/Training CV/Selected Features STABL ALasso.csv" \
  --drug-table "$M/tables_qmap/{drug}_mshift_$ARM.csv" --out $M/excl/${ARM}_mshiftdrug --exclude 36_G1 52_G1 --no-validate
