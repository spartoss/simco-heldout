#!/bin/bash
# After gauss_drug_fit: build the gauss_ot-stim + gauss_ot-drug tables, run checks a-d with the
# gauss_ot DOS model (100-split selections, params_43, aligned), then the cross-model comparison.
#SBATCH -p normal
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -J gdrug_build
#SBATCH -o /scratch/users/bmerwan/simco_work/onset_test/drug_models/logs/gbuild_%j.out
#SBATCH -e /scratch/users/bmerwan/simco_work/onset_test/drug_models/logs/gbuild_%j.err
set -uo pipefail
O=/scratch/users/bmerwan/simco_work/onset_test; M=$O/drug_models
module load python/3.9.0; cd /scratch/users/bmerwan/simco
export HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
if [[ ! -s $M/tables/MF_gaussfull.csv ]]; then
  source /scratch/users/bmerwan/venv-cellot/bin/activate
  python -m baselines.drug_features_gauss --drug-maps $M/gauss_drug_maps --out $M/tables || exit 1
  deactivate
fi
source /scratch/users/bmerwan/venv-stabl/bin/activate
python stabl_launch/drug_arm_checks.py --features $O/gaussot_13OG_both.csv \
  --selections "$O/paper_protocol/stabl_n100/gaussot_both/Training CV/Selected Features STABL ALasso.csv" \
  --drug-table "$M/tables/{drug}_gaussfull.csv" --out $M/checks_gaussot_full --no-validate > $M/logs/checks_gaussot_full.log 2>&1
echo "[CHECKS] rc=$?"
python stabl_launch/drug_models_compare.py
