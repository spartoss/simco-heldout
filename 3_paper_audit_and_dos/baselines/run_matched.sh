#!/bin/bash
# Phase 2: the full 2x2 arm cross (fit-patients x weighting) on the stim arm, all 28
# celltypes including Granulocytes. Runs inside the existing allocation -- no sbatch.
#
#   NPROC=4 bash baselines/run_matched.sh              # all 28, smallest first
#   bash baselines/run_matched.sh Granulocytes         # the 30 GB outlier, alone
#
# Resumable: a celltype whose CSV already exists is skipped, so this can be stopped by
# a walltime and picked up in the next allocation.
#
# OUT is lopo_test_v3 because gauss_ot and linear_median* were added after the v2 run.
# The 26 CSVs in lopo_test_v2 carry the old three-method schema, and the resume check
# below is "does the CSV exist", not "does it have the right columns" -- pointing this
# at v2 would skip them and leave a directory summarize.py silently averages over two
# different method sets.
set -uo pipefail

WORK=/scratch/users/bmerwan/simco_work/baselines
DATA=/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets
REPO=/scratch/users/bmerwan/simco
OUT="${OUT:-$WORK/lopo_test_v3}"
NPROC="${NPROC:-4}"

cd "$REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT" "$WORK/logs"

run_one() {
  local ct="$1"
  [ -s "$OUT/${ct}.csv" ] && { echo "[SKIP ] $ct already done"; return 0; }
  echo "[START] $ct $(date +%T)"
  python -m baselines.run_baselines \
    --celltype "$ct" \
    --data-dir "$DATA/ptb_concatenated_batchcorrected" \
    --features "$DATA/ptb_concatenated_per_condition_celltype/13features.txt" \
    --pairs-file "$WORK/pairs_168.txt" \
    --outdir "$OUT" \
    --eval-split test \
    --fit-patients both \
    --weighting both \
    --cellot-results "$DATA/../results/cross_validation_original_13_HVPV" \
    > "$WORK/logs/v2_${ct}.out" 2> "$WORK/logs/v2_${ct}.err"
  echo "[DONE ] $ct rc=$? $(date +%T)"
}
export -f run_one
export WORK DATA REPO OUT

if [ $# -gt 0 ]; then
  for ct in "$@"; do run_one "$ct"; done
  exit 0
fi

# Smallest first so a short allocation still banks as many celltypes as possible.
while read -r ct; do
  f="$DATA/ptb_concatenated_batchcorrected/${ct}_HVPV.h5ad"
  echo "$(stat -c %s "$f") $ct"
done < "$WORK/celltypes.txt" | sort -n | awk '{print $2}' \
  | xargs -I{} -P "$NPROC" bash -c 'run_one "$@"' _ {}

echo "[ALL DONE] $(ls "$OUT" | wc -l) celltype CSVs"
