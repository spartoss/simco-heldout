#!/usr/bin/bash
# Matched comparison: Stabl on measured Unstim features vs CellOT-simulated stim features.
# Same outcome (DOS), same 177 rows / 63 patients, same GroupShuffleSplit(random_state=42),
# same artificial_type and downstream regressor -> the only thing that differs is the feature block.
#
# Usage: ./run_unstim_vs_simstim.sh [n_splits] [tag]
# Run directly (interactive allocation), NOT via sbatch.
set -u

NSPLITS="${1:-3}"
TAG="${2:-n${NSPLITS}}"

ONSET=/scratch/users/bmerwan/simco_work/onset_test
OUT="${ONSET}/cmp_unstim_vs_simstim_${TAG}"
LOGS="${OUT}/logs"
mkdir -p "${LOGS}"

module load python/3.9.0
source /scratch/users/bmerwan/venv-stabl/bin/activate
cd /scratch/users/bmerwan/simco/stabl_launch

ARTIFICIAL=knockoff
MODEL=xgboost

ARMS="${ARMS:-unstim_only no_Unstim both}"

for arm in ${ARMS}; do
    case "${arm}" in
        unstim_only) feats="${ONSET}/ina_13OG_final_long_allstims_filtered_unstim_only.csv" ;;
        no_Unstim)   feats="${ONSET}/ina_13OG_final_long_allstims_filtered_no_Unstim.csv" ;;
        # 'both' = Unstim + 7 simulated stim blocks; this is the table behind the paper's Fig 5b
        both)        feats="${ONSET}/ina_13OG_final_long_allstims_filtered.csv" ;;
    esac
    echo "=== ${arm}: ${feats}  (n_splits=${NSPLITS})"
    start=$(date +%s)
    python run_regression_cv.py \
        --features_path "${feats}" \
        --results_dir "${OUT}/${arm}" \
        --artificial_type "${ARTIFICIAL}" \
        --model_chosen "${MODEL}" \
        --n_splits "${NSPLITS}" \
        > "${LOGS}/${arm}.out" 2> "${LOGS}/${arm}.err"
    rc=$?
    echo "=== ${arm} rc=${rc} elapsed=$(( $(date +%s) - start ))s"
done
