#!/usr/bin/bash
# One null-map arm end to end: build the 177 x 1216 feature table (if missing), derive the
# 852-col no_Unstim table, run Stabl (20 splits, knockoff, xgboost -- the same settings as
# cmp_unstim_vs_simstim_n20/ and gaussot_stabl_n20/) on both. Idempotent: every step skips
# when its output exists, so a killed task is simply re-run.
#
#   null_maps_arm.sh <arm-name> [build|all]     arms and their flags: null_maps/arms.txt
# `cellot_repro` is build-only: its Stabl runs already exist (cmp_unstim_vs_simstim_n20/,
# cellot_both_n20/); the rebuilt table is the full-size pipeline check.
set -euo pipefail
ARM="${1:?arm}"; MODE="${2:-all}"
ONSET=/scratch/users/bmerwan/simco_work/onset_test
W="${ONSET}/null_maps"
FLAGS=$(awk -v a="${ARM}" '$1==a{$1=""; print}' "${W}/arms.txt")
[[ -n "${FLAGS}" ]] || { echo "unknown arm ${ARM}"; exit 2; }
NSPLITS="${NSPLITS:-20}"
TABLE="${W}/tables/${ARM}_both.csv"
NOUN="${W}/tables/${ARM}_no_Unstim.csv"
export HDF5_USE_FILE_LOCKING=FALSE  # group-home NFS: lock fails intermittently (errno 121)
mkdir -p "${W}/tables" "${W}/logs"
module load python/3.9.0
cd /scratch/users/bmerwan/simco

if [[ ! -s "${TABLE}" ]]; then
    source /scratch/users/bmerwan/venv-cellot/bin/activate
    echo "[BUILD] ${ARM}: ${FLAGS}"
    python -m baselines.null_map_features ${FLAGS} --out "${TABLE}" \
        --like "${ONSET}/ina_13OG_final_long_allstims_filtered.csv"
    deactivate
fi
source /scratch/users/bmerwan/venv-stabl/bin/activate
if [[ ! -s "${NOUN}" ]]; then
    python - "${TABLE}" "${NOUN}" <<'PY'
import sys, pandas as pd
t = pd.read_csv(sys.argv[1], index_col=0)
t[[c for c in t.columns if not c.endswith("_Unstim")]].to_csv(sys.argv[2])
PY
fi
[[ "${MODE}" == build || "${ARM}" == cellot_repro ]] && { echo "[DONE ] ${ARM} built"; exit 0; }

cd stabl_launch
for tab in no_Unstim both; do
    out="${W}/stabl_n${NSPLITS}/${ARM}/${tab}"
    if [[ -s "${out}/Summary/Scores training CV.csv" ]]; then echo "[SKIP ] ${ARM}/${tab}"; continue; fi
    feats="${W}/tables/${ARM}_${tab}.csv"
    echo "[STABL] ${ARM}/${tab} n_splits=${NSPLITS} start $(date)"
    python run_regression_cv.py --features_path "${feats}" --results_dir "${out}" \
        --artificial_type knockoff --model_chosen xgboost --n_splits "${NSPLITS}"
    echo "[STABL] ${ARM}/${tab} done $(date)"
done
