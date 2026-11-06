#!/bin/bash
# The noise floor: DMSO_i vs DMSO_j within a stim, scored with the identity map.
# Answers "how far apart are two samples that should be identical?", which is the
# scale every distance in the benchmark has to be read against. Runs inside the
# existing allocation -- no sbatch.
#
#   NPROC=3 bash baselines/run_noise_floor.sh          # all 28, smallest first
#   bash baselines/run_noise_floor.sh Granulocytes     # the 30 GB outlier, alone
#
# Cheaper than run_pooled.sh: nothing is fit, so the cost is the .h5ad read plus a
# few hundred MMDs on 2000-cell subsamples. ~1 min for a small celltype.
#
# Resumable: a celltype whose CSV already exists is skipped. Its own output dir --
# these rows have `scope`/`pair_type`/`source`/`target` columns that no benchmark dir
# has, so never point summarize.py at it.
set -uo pipefail

WORK=/scratch/users/bmerwan/simco_work/baselines
DATA=/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets
REPO=/scratch/users/bmerwan/simco
OUT="${OUT:-$WORK/noise_floor}"
NPROC="${NPROC:-3}"
LIST="${LIST:-$WORK/celltypes.txt}"

cd "$REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT" "$WORK/logs"

run_one() {
  local ct="$1"
  [ -s "$OUT/${ct}.csv" ] && { echo "[SKIP ] $ct already done"; return 0; }
  echo "[START] $ct $(date +%T)"
  python -m baselines.noise_floor \
    --celltype "$ct" \
    --data-dir "$DATA/ptb_concatenated_batchcorrected" \
    --features "$DATA/ptb_concatenated_per_condition_celltype/13features.txt" \
    --outdir "$OUT" \
    > "$WORK/logs/noise_${ct}.out" 2> "$WORK/logs/noise_${ct}.err"
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
