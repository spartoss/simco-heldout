#!/bin/bash
# Second pass of the LOPO benchmark with CellOT's cell-level leak removed:
# baselines fit on the train side of the reproduced 80/20 split, every method scored
# on the held-out patient's test cells only. See run_baselines.py --eval-split.
#
#   bash baselines/run_local_testsplit.sh
set -uo pipefail

WORK=/scratch/users/bmerwan/simco_work/baselines
DATA=/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets
REPO=/scratch/users/bmerwan/simco
NPROC="${NPROC:-4}"

cd "$REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$WORK/lopo_test" "$WORK/logs"

run_one() {
  local ct="$1"
  [ -f "$WORK/lopo_test/${ct}.csv" ] && { echo "[SKIP ] $ct already done"; return 0; }
  echo "[START] $ct $(date +%H:%M:%S)"
  python -m baselines.run_baselines \
    --celltype "$ct" \
    --data-dir "$DATA/ptb_concatenated_batchcorrected" \
    --features "$DATA/ptb_concatenated_per_condition_celltype/13features.txt" \
    --pairs-file "$WORK/pairs_168.txt" \
    --outdir "$WORK/lopo_test" \
    --eval-split test \
    --cellot-results /home/groups/gbrice/ptb-drugscreen/ot/cellot/results/cross_validation_original_13_HVPV \
    > "$WORK/logs/test_${ct}.out" 2> "$WORK/logs/test_${ct}.err"
  echo "[DONE ] $ct rc=$? $(date +%H:%M:%S)"
}
export -f run_one
export WORK DATA REPO

while read -r ct; do
  f="$DATA/ptb_concatenated_batchcorrected/${ct}_HVPV.h5ad"
  echo "$(stat -c %s "$f") $ct"
done < "$WORK/celltypes_small.txt" | sort -rn | awk '{print $2}' \
  | xargs -I{} -P "$NPROC" bash -c 'run_one "$@"' _ {}

echo "[ALL DONE] $(ls "$WORK/lopo_test" | wc -l) celltype CSVs"
