#!/usr/bin/bash
# One overnight arm: Stabl (20 splits, knockoff, xgboost -- same settings as every other _n20
# run) on a feature table, for target DOS or DOS_offset (= DOS - 7*(EGA-40), days relative to the
# 40-week due date: the gestational-age clock removed, no fitted parameter). Idempotent.
#   night2_arm.sh <arm>      arms: simco_work/onset_test/night2/arms.txt  (name|target|features|blocks)
set -euo pipefail
ARM="${1:?arm}"
N=/scratch/users/bmerwan/simco_work/onset_test/night2
LINE=$(awk -F'|' -v a="${ARM}" '$1==a' "${N}/arms.txt")
[[ -n "${LINE}" ]] || { echo "unknown arm ${ARM}"; exit 2; }
IFS='|' read -r _ TARGET FEATS BLOCKS <<< "${LINE}"
OUT="${N}/stabl_n20/${ARM}"
if [[ -s "${OUT}/Summary/Scores training CV.csv" ]]; then echo "[SKIP ] ${ARM} done"; exit 0; fi
for i in $(seq 1 60); do [[ -s "${FEATS}" ]] && break; echo "[WAIT ] ${FEATS}"; sleep 60; done
[[ -s "${FEATS}" ]] || { echo "[ERROR] missing ${FEATS}"; exit 3; }
case "${TARGET}" in
    DOS) OUTCOME=/scratch/users/bmerwan/simco_work/onset_test/outcome_table_all_pre.csv ;;
    DOS_offset) OUTCOME="${N}/outcome_offset.csv" ;;
esac
module load python/3.9.0
source /scratch/users/bmerwan/venv-stabl/bin/activate
cd /scratch/users/bmerwan/simco/stabl_launch
echo "[STABL] ${ARM}: target=${TARGET} feats=${FEATS} blocks='${BLOCKS}' start $(date)"
python run_regression_cv.py --features_path "${FEATS}" --results_dir "${OUT}" \
    --artificial_type knockoff --model_chosen xgboost --n_splits 20 \
    --outcome_path "${OUTCOME}" --outcome_col "${TARGET}" \
    ${BLOCKS:+--blocks ${BLOCKS}}
echo "[STABL] ${ARM} done $(date)"
