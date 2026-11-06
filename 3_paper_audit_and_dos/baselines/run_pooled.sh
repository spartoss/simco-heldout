#!/bin/bash
# CellOT's own protocol, applied to the baselines: fit once on the 80% train cells
# pooled over all 19 patients, score once on the 20% test cells. No held-out patient.
# One row per (stim, celltype, method). Runs inside the existing allocation -- no sbatch.
#
#   NPROC=4 bash baselines/run_pooled.sh               # all 28, smallest first
#   bash baselines/run_pooled.sh Granulocytes          # the 30 GB outlier, alone
#
# Much cheaper than run_matched.sh: one fit per (stim, celltype) instead of 19, so the
# per-celltype cost is dominated by reading the .h5ad rather than by refitting.
#
# Resumable: a celltype whose CSV already exists is skipped, so this can be stopped by
# a walltime and picked up in the next allocation. Its output dir is its own -- the
# pooled rows carry patient="pooled" and are not comparable row-for-row with the lopo
# dirs, so never merge the two.
set -uo pipefail

WORK=/scratch/users/bmerwan/simco_work/baselines
DATA=/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets
REPO=/scratch/users/bmerwan/simco
OUT="${OUT:-$WORK/pooled}"
NPROC="${NPROC:-4}"
# celltypes_small.txt is the 27 that fit comfortably alongside each other; Granulocytes
# is 30 GB on disk and wants the node to itself.
LIST="${LIST:-$WORK/celltypes.txt}"

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
    --eval-mode pooled \
    --eval-split test \
    --weighting both \
    --cellot-results "$DATA/../results/cross_validation_original_13_HVPV" \
    > "$WORK/logs/pooled_${ct}.out" 2> "$WORK/logs/pooled_${ct}.err"
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
done < "$LIST" | sort -n | awk '{print $2}' \
  | xargs -I{} -P "$NPROC" bash -c 'run_one "$@"' _ {}

echo "[ALL DONE] $(ls "$OUT" | wc -l) celltype CSVs"
