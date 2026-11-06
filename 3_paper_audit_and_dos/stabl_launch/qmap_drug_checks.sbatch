#!/bin/bash
# After qmap_drug_fit: drug checks a-d, 36_G1 / 52_G1 excluded, for 4 (stim model, drug map) combinations:
# 0 CellOT stims + CellOT drug maps (paper), 1 CellOT stims + quantile_map drug, 2 gauss_ot + gauss_ot drug,
# 3 gauss_ot stims + quantile_map drug. Out: drug_models/excl/<name>/.
#SBATCH -p normal
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -J qdrug_chk
#SBATCH -o /scratch/users/bmerwan/simco_work/onset_test/drug_models/logs/qchk_%A_%a.out
#SBATCH -e /scratch/users/bmerwan/simco_work/onset_test/drug_models/logs/qchk_%A_%a.err
set -euo pipefail
O=/scratch/users/bmerwan/simco_work/onset_test; M=$O/drug_models
module load python/3.9.0; source /scratch/users/bmerwan/venv-stabl/bin/activate
cd /scratch/users/bmerwan/simco; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
CELLOT_F=$O/ina_13OG_final_long_allstims_filtered.csv; CELLOT_S="$O/foldfeats_GSS/Training CV/Selected Features STABL ALasso.csv"
GAUSS_F=$O/gaussot_13OG_both.csv; GAUSS_S="$O/paper_protocol/stabl_n100/gaussot_both/Training CV/Selected Features STABL ALasso.csv"
case $SLURM_ARRAY_TASK_ID in
  0) NAME=cellot_cellotdrug; F=$CELLOT_F; S=$CELLOT_S; D="/scratch/users/bmerwan/simco_work/cellot_results/ina_13OG_{drug}_pred_unstim.csv";;
  1) NAME=cellot_qmapdrug;   F=$CELLOT_F; S=$CELLOT_S; D="$M/tables_qmap/{drug}_qmap_cellot.csv"
     python -m baselines.drug_features_qmap --maps $M/qmap_drug_maps --features $F --out $M/tables_qmap --tag qmap_cellot;;
  2) NAME=gauss_gaussdrug;   F=$GAUSS_F; S=$GAUSS_S; D="$M/tables/{drug}_gaussfull.csv";;
  3) NAME=gauss_qmapdrug;    F=$GAUSS_F; S=$GAUSS_S; D="$M/tables_qmap/{drug}_qmap_gauss.csv"
     python -m baselines.drug_features_qmap --maps $M/qmap_drug_maps --features $F --out $M/tables_qmap --tag qmap_gauss;;
esac
python stabl_launch/drug_arm_checks.py --features "$F" --selections "$S" --drug-table "$D" \
  --out $M/excl/$NAME --exclude 36_G1 52_G1 --no-validate
