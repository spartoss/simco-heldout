#!/bin/bash
# Stabl (unchanged pipeline) on the paper's 100 splits for the combined tables with
# simulated quantiles. Task 0 = comb_cellot_simq, task 1 = comb_gaussot_simq.
#SBATCH -p normal
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -J simq_stabl100
#SBATCH -o /scratch/users/bmerwan/simco_work/onset_test/paper_protocol/logs/simq_stabl100_%A_%a.out
#SBATCH -e /scratch/users/bmerwan/simco_work/onset_test/paper_protocol/logs/simq_stabl100_%A_%a.err
set -euo pipefail
P=/scratch/users/bmerwan/simco_work/onset_test/paper_protocol
ARM=$([[ $SLURM_ARRAY_TASK_ID == 0 ]] && echo comb_cellot_simq || echo comb_gaussot_simq)
module load python/3.9.0; source /scratch/users/bmerwan/venv-stabl/bin/activate
cd /scratch/users/bmerwan/simco/stabl_launch; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
[[ -s "$P/stabl_n100/$ARM/Summary/Scores training CV.csv" ]] && exit 0
python run_regression_cv.py --features_path $P/tables/$ARM.csv --results_dir $P/stabl_n100/$ARM \
  --artificial_type knockoff --model_chosen xgboost --n_splits 100 \
  --blocks TNFa LPS IL246 IFNa GMCSF PI IL33 freq med qtl TNFaq LPSq IL246q IFNaq GMCSFq PIq IL33q
