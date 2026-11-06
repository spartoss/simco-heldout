#!/bin/bash
# model.pt vs last.pt on the same test cells -- how much of CellOT's score is the
# checkpoint argmin over ~200 test evaluations rather than the trained map.
# Runs inside the existing allocation -- no sbatch.
#
#   ARM=stim NPROC=4 bash baselines/run_selection_leak.sh              # all 28
#   ARM=drug NPROC=4 bash baselines/run_selection_leak.sh              # all 28 x 9 drugs
#   ARM=drug bash baselines/run_selection_leak.sh Granulocytes         # the 30 GB outlier
#
# Resumable: a celltype whose CSV exists is skipped. Granulocytes holds ~7.6 GB in the
# drug arm (one pass over 196M rows, 13 markers, float32) so it wants the node to
# itself; the other 27 are under 1 GB each and run four-up.
set -uo pipefail

WORK=/scratch/users/bmerwan/simco_work/baselines
DATA=/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets
REPO=/scratch/users/bmerwan/simco
ARM="${ARM:-stim}"
OUT="${OUT:-$WORK/selection_leak_$ARM}"
NPROC="${NPROC:-4}"
LIST="${LIST:-$WORK/celltypes.txt}"

# Bash arrays do not survive `export -f`, so the drug arm's extra flag is carried as a
# plain string and left unquoted at the call site -- it holds no whitespace.
if [ "$ARM" = "drug" ]; then
  RESULTS="$DATA/../results_drug/cross_validation_drug_OG13"
  EXTRA="--batch-correction $REPO/cross_validation/batchcorrection_2.csv"
else
  RESULTS="$DATA/../results/cross_validation_original_13_HVPV"
  EXTRA=""
fi
# LOWMEM=1 reads one stim at a time. Granulocytes needs it: the single-pass form peaks
# over 23 GB on the drug arm and is OOM-killed (rc=137) in this 32 GB allocation, which
# is why its drug CSV was missing rather than merely unfinished.
[ -n "${LOWMEM:-}" ] && EXTRA="$EXTRA --low-memory"

cd "$REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT" "$WORK/logs"

run_one() {
  local ct="$1"
  [ -s "$OUT/${ct}.csv" ] && { echo "[SKIP ] $ct already done"; return 0; }
  echo "[START] $ct $(date +%T)"
  python -m baselines.selection_leak \
    --arm "$ARM" \
    --celltype "$ct" \
    --data-dir "$DATA/ptb_concatenated_batchcorrected" \
    --features "$DATA/ptb_concatenated_per_condition_celltype/13features.txt" \
    --pairs-file "$WORK/pairs_168.txt" \
    --results "$RESULTS" \
    --outdir "$OUT" \
    $EXTRA \
    > "$WORK/logs/selleak_${ARM}_${ct}.out" 2> "$WORK/logs/selleak_${ARM}_${ct}.err"
  echo "[DONE ] $ct rc=$? $(date +%T)"
}
export -f run_one
export WORK DATA REPO OUT ARM RESULTS EXTRA

if [ $# -gt 0 ]; then
  for ct in "$@"; do run_one "$ct"; done
  exit 0
fi

while read -r ct; do
  f="$DATA/ptb_concatenated_batchcorrected/${ct}_HVPV.h5ad"
  echo "$(stat -c %s "$f") $ct"
done < "$LIST" | sort -n | awk '{print $2}' \
  | xargs -I{} -P "$NPROC" bash -c 'run_one "$@"' _ {}

echo "[ALL DONE] $(ls "$OUT" | wc -l) celltype CSVs in $OUT"
