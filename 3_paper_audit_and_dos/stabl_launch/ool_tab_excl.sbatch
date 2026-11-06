#!/bin/bash
# After the aligned sweep of tab_both / mix_both: params_43 rescore without 36_G1 / 52_G1 -> aligned_excl/, then the summary
#SBATCH -p normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -J ool_tab_excl
#SBATCH -o /scratch/users/bmerwan/simco_work/onset_test/paper_protocol/logs/ool_tab_excl_%j.out
set -euo pipefail
O=/scratch/users/bmerwan/simco_work/onset_test; P=$O/paper_protocol
module load python/3.9.0; source /scratch/users/bmerwan/venv-stabl/bin/activate
cd /scratch/users/bmerwan/simco/stabl_launch; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
for ARM in ${ARMS:-tab_both mix_both}; do
  python aligned_rescore.py --arm $ARM --features $P/tables/$ARM.csv --selections $P/stabl_n100/$ARM \
    --out $P/aligned_excl --params-glob $O/param_grid_files/params_43.json --exclude 36_G1 52_G1
done
python ool_tab_summary.py
# drug checks (quantile-map drug maps on the feature values, as drug_models/excl/*_qmapdrug)
M=$O/drug_models; cd /scratch/users/bmerwan/simco
for ARM in ${DRUG_ARMS:-}; do
  python -m baselines.drug_features_qmap --maps $M/qmap_drug_maps --features $P/tables/$ARM.csv --out $M/tables_qmap --tag qmap_$ARM
  python stabl_launch/drug_arm_checks.py --features $P/tables/$ARM.csv --selections "$P/stabl_n100/$ARM/Training CV/Selected Features STABL ALasso.csv" \
    --drug-table "$M/tables_qmap/{drug}_qmap_$ARM.csv" --out $M/excl/${ARM}_qmapdrug --exclude 36_G1 52_G1 --no-validate
done
