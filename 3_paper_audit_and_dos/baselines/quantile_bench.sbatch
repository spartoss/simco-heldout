#!/bin/bash
# Atlas benchmark re-run with per-marker quantile errors (metrics.QUANTILES), cell-weighted arm.
#   ARM=stim sbatch --array=0-27 baselines/quantile_bench.sbatch   -> baselines/pooled_q/
#   ARM=drug sbatch --array=0-27 baselines/quantile_bench.sbatch   -> baselines/pooled_drug_q/
#SBATCH -p normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH -J qbench
#SBATCH -o /scratch/users/bmerwan/simco_work/baselines/logs/qbench_%x_%A_%a.out
#SBATCH -e /scratch/users/bmerwan/simco_work/baselines/logs/qbench_%x_%A_%a.err
set -euo pipefail
W=/scratch/users/bmerwan/simco_work/baselines; DATA=/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets
CT=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $W/celltypes.txt)
module load python/3.9.0; source /scratch/users/bmerwan/venv-cellot/bin/activate
cd /scratch/users/bmerwan/simco; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK HDF5_USE_FILE_LOCKING=FALSE
if [[ "${ARM}" == stim ]]; then
  OUT=$W/pooled_q; [[ -s $OUT/$CT.csv ]] && exit 0
  python -m baselines.run_baselines --celltype "$CT" --data-dir $DATA/ptb_concatenated_batchcorrected \
    --features $DATA/ptb_concatenated_per_condition_celltype/13features.txt --pairs-file $W/pairs_168.txt \
    --outdir $OUT --eval-mode pooled --eval-split test --weighting cell \
    --cellot-results $DATA/../results/cross_validation_original_13_HVPV
else
  OUT=$W/pooled_drug_q; [[ -s $OUT/$CT.csv ]] && exit 0
  LM=""; [[ "$CT" == Granulocytes ]] && LM="--low-memory"
  python -m baselines.run_baselines --arm drug --celltype "$CT" --data-dir $DATA/ptb_concatenated_batchcorrected \
    --features $DATA/ptb_concatenated_per_condition_celltype/13features.txt --pairs-file $W/pairs_168.txt \
    --outdir $OUT --eval-mode pooled --eval-split test --weighting cell $LM \
    --cellot-results $DATA/../results_drug/cross_validation_drug_OG13 \
    --batch-correction /scratch/users/bmerwan/simco/cross_validation/batchcorrection_2.csv
fi
