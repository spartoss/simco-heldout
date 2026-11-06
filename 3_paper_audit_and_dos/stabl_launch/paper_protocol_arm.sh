#!/usr/bin/bash
# Put a feature table on the PAPER's Fig 5b protocol, in three idempotent steps:
#   1. Stabl, 100 splits (run_regression_cv.py) -> per-fold selected features
#      (skipped when arms.txt gives an existing 100-split fold dir, e.g. CellOT's foldfeats_GSS)
#   2. rescore with params_43.json -- the paper's XGBoost config, tuned for CellOT (fair version)
#   3. rescore with all 97 param_grid_files/params_*.json, keep min RMSE -- the same argmin the
#      lab ran for CellOT (grid_search_model.py:268-272), so every arm gets the same optimism
#
#   paper_protocol_arm.sh <arm>     arms: simco_work/onset_test/paper_protocol/arms.txt
#                                   (name|features|blocks|existing_fold_dir)
set -euo pipefail
ARM="${1:?arm}"
O=/scratch/users/bmerwan/simco_work/onset_test
P="${O}/paper_protocol"
LINE=$(awk -F'|' -v a="${ARM}" '$1==a' "${P}/arms.txt")
[[ -n "${LINE}" ]] || { echo "unknown arm ${ARM}"; exit 2; }
IFS='|' read -r _ FEATS BLOCKS FOLDS <<< "${LINE}"
module load python/3.9.0
source /scratch/users/bmerwan/venv-stabl/bin/activate
cd /scratch/users/bmerwan/simco/stabl_launch
NCPU="${SLURM_CPUS_PER_TASK:-4}"
BARG=""; [[ -n "${BLOCKS}" ]] && BARG="--blocks ${BLOCKS}"

# 1. Stabl, 100 splits
if [[ -z "${FOLDS}" ]]; then
    FOLDS="${P}/stabl_n100/${ARM}"
    if [[ ! -s "${FOLDS}/Summary/Scores training CV.csv" ]]; then
        echo "[STABL] ${ARM} 100 splits start $(date)"
        OMP_NUM_THREADS="${NCPU}" python run_regression_cv.py --features_path "${FEATS}" \
            --results_dir "${FOLDS}" --artificial_type knockoff --model_chosen xgboost \
            --n_splits 100 ${BARG}
        echo "[STABL] ${ARM} done $(date)"
    fi
fi
export FEATS FOLDS BARG

rescore () {   # $1 params json, $2 out dir
    [[ -s "$2/Summary/Scores training CV.csv" ]] && return 0
    OMP_NUM_THREADS=2 python get_foldinfo_on_existing_run.py --features_path "${FEATS}" ${BARG} \
        --stabl_models "STABL ALasso" --fold_feats_path "${FOLDS}" --xgb_config_path "$1" \
        --model_chosen xgboost --results_dir "$2" --artificial_type knockoff > "$2.log" 2>&1 \
        || echo "[FAIL ] rescore $1 -> $2 (see $2.log)"
}
export -f rescore

# 2. params_43 (fair version)
R="${P}/rescore/${ARM}"
mkdir -p "${R}"
rescore "${O}/param_grid_files/params_43.json" "${R}/params_43"
echo "[P43  ] ${ARM}: $(sed -n 2p "${R}/params_43/Summary/Scores training CV.csv" | cut -c1-80)"

# 3. all 97 configs in parallel (2 threads each), then the argmin
ls "${O}"/param_grid_files/params_*.json | \
    xargs -P $(( NCPU / 2 > 0 ? NCPU / 2 : 1 )) -I{} bash -c 'rescore "$1" "$2/$(basename "$1" .json)"' _ {} "${R}"
python - "${R}" <<'PY'
import sys, glob, os, pandas as pd
rows = []
for f in glob.glob(os.path.join(sys.argv[1], "params_*/Summary/Scores training CV.csv")):
    r = pd.read_csv(f, index_col=0).loc["STABL ALasso"]
    rows.append(dict(params=f.split("/")[-3], R2=float(r.R2.split()[0]),
                     RMSE=float(r.RMSE.split()[0]), R2_full=r.R2))
t = pd.DataFrame(rows).sort_values("RMSE")
t.to_csv(os.path.join(sys.argv[1], "sweep.csv"), index=False)
print(f"[SWEEP] {len(t)} configs; best by RMSE: {t.iloc[0].params} "
      f"R2 {t.iloc[0].R2_full} RMSE {t.iloc[0].RMSE}")
PY
