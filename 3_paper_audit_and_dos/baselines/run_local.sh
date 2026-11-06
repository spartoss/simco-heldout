#!/bin/bash
# Run the LOPO baseline benchmark inside an existing allocation (no sbatch).
#
#   bash baselines/run_local.sh            # 27 celltypes, 4 at a time
#   bash baselines/run_local.sh Granulocytes   # the 30 GB outlier, alone
#
# Each worker is pinned to a single BLAS thread: the numpy work here is small
# (quantiles over <=13 markers, MMD on a 2000-cell subsample), so process-level
# parallelism across celltypes beats thread-level parallelism within one.
set -uo pipefail

WORK=/scratch/users/bmerwan/simco_work/baselines
DATA=/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets
REPO=/scratch/users/bmerwan/simco
NPROC="${NPROC:-4}"

cd "$REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$WORK/lopo" "$WORK/logs"

run_one() {
  local ct="$1"
  echo "[START] $ct $(date +%H:%M:%S)"
  python -m baselines.run_baselines \
    --celltype "$ct" \
    --data-dir "$DATA/ptb_concatenated_batchcorrected" \
    --features "$DATA/ptb_concatenated_per_condition_celltype/13features.txt" \
    --pairs-file "$WORK/pairs_168.txt" \
    --outdir "$WORK/lopo" \
    --cellot-results /home/groups/gbrice/ptb-drugscreen/ot/cellot/results/cross_validation_original_13_HVPV \
    > "$WORK/logs/local_${ct}.out" 2> "$WORK/logs/local_${ct}.err"
  echo "[DONE ] $ct rc=$? $(date +%H:%M:%S)"
}
export -f run_one
export WORK DATA REPO

if [ $# -gt 0 ]; then
  for ct in "$@"; do run_one "$ct"; done
  exit 0
fi

# Largest files first so the long poles start early and the tail is short.
while read -r ct; do
  f="$DATA/ptb_concatenated_batchcorrected/${ct}_HVPV.h5ad"
  echo "$(stat -c %s "$f") $ct"
done < "$WORK/celltypes_small.txt" | sort -rn | awk '{print $2}' \
  | xargs -I{} -P "$NPROC" bash -c 'run_one "$@"' _ {}

echo "[ALL DONE] $(ls "$WORK/lopo" | wc -l) celltype CSVs"
