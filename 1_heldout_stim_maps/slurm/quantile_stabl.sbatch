#!/bin/bash
# Stabl (simco's run_regression_cv.py, unchanged) on one feature set, 100 splits, then the aligned XGBoost rescoring over
# all 97 settings (simco's aligned_rescore.py). Array 0-8 over the sets in ARMS below.
#SBATCH -p normal
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -J ho_qstabl
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/heldout/logs/qstabl_%A_%a.out
set -euo pipefail
ARMS=(ref852 rule_med rule_rawq rule_shape rule_rep rule_corr rule_selq selq_corr selq_corr_eff)
A=${ARMS[$SLURM_ARRAY_TASK_ID]}
Q=/scratch/users/bmerwan/atlas_heldout/work/heldout/quantile_arms; O=/scratch/users/bmerwan/simco_work/onset_test
module load python/3.9.0; source /scratch/users/bmerwan/venv-stabl/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
BLOCKS=$(python -c "import json; print(' '.join(json.load(open('$Q/blocks.json'))['$A']['blocks']))")
cd /scratch/users/bmerwan/simco/stabl_launch
if [[ ! -s "$Q/stabl/$A/Summary/Scores training CV.csv" ]]; then
  python run_regression_cv.py --features_path $Q/tables/$A.csv --results_dir $Q/stabl/$A \
    --artificial_type knockoff --model_chosen xgboost --n_splits 100 --blocks $BLOCKS
fi
python aligned_rescore.py --arm $A --features $Q/tables/$A.csv --selections $Q/stabl/$A --out $Q/aligned \
  --params-glob "$O/param_grid_files/params_*.json"
