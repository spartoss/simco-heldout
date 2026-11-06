#!/usr/bin/bash
# Matched comparison at the PAPER's configuration.
#
# The paper's Fig 5b number is a *rescore*: Stabl's per-fold feature selection is reused
# unchanged and only the downstream XGBoost regressor is refit, with the grid-searched
# params_43.json. Verified: this reproduces GS_cellot_longallstims_foldinfo exactly
# (R2 0.415, RMSE 24.120, MAE 18.947, N feats 21 [18, 30.25]) in ~51 s.
#
# So the measured-vs-simulated comparison can be put on the paper's footing without
# refitting Stabl at all -- all three arms already have stored 100-split runs whose
# per-fold feature lists are staged under $O/foldfeats_*.
#
# Run directly (interactive allocation), NOT via sbatch.
set -u

O=/scratch/users/bmerwan/simco_work/onset_test
OUT="${OUT:-$O/rescore_params43}"
PARAMS="${PARAMS:-$O/param_grid_files/params_43.json}"
MODELS="${MODELS:-STABL ALasso}"

module load python/3.9.0
source /scratch/users/bmerwan/venv-stabl/bin/activate
cd /scratch/users/bmerwan/simco/stabl_launch
export OMP_NUM_THREADS=1

mkdir -p "$OUT/logs"

# arm : feature table : staged fold-feature dir
run_one () {
    local arm="$1" feats="$2" folds="$3"
    [ -s "$OUT/$arm/Summary/Scores training CV.csv" ] && { echo "[SKIP ] $arm"; return 0; }
    echo "[START] $arm $(date +%T)"
    local t0=$(date +%s)
    python get_foldinfo_on_existing_run.py \
        --features_path "$feats" \
        --stabl_models "$MODELS" \
        --fold_feats_path "$folds" \
        --xgb_config_path "$PARAMS" \
        --model_chosen xgboost \
        --results_dir "$OUT/$arm" \
        --artificial_type knockoff \
        > "$OUT/logs/$arm.log" 2>&1
    echo "[DONE ] $arm rc=$? elapsed=$(( $(date +%s) - t0 ))s"
}

run_one unstim_only "$O/ina_13OG_final_long_allstims_filtered_unstim_only.csv" "$O/foldfeats_unstim_only"
run_one no_Unstim   "$O/ina_13OG_final_long_allstims_filtered_no_Unstim.csv"   "$O/foldfeats_no_Unstim"
run_one both        "$O/ina_13OG_final_long_allstims_filtered.csv"             "$O/foldfeats_GSS"

echo
for arm in unstim_only no_Unstim both; do
    echo "=== $arm"; cat "$OUT/$arm/Summary/Scores training CV.csv" 2>/dev/null
done
