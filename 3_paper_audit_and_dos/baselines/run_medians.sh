#!/bin/bash
# Per-(stim, patient, marker) medians for all 28 celltypes, inside the existing
# allocation -- no sbatch, no srun. See CLAUDE.md "Do not submit jobs".
#
#   bash baselines/run_medians.sh                # all 28, 4 at a time
#   bash baselines/run_medians.sh Granulocytes   # the 30 GB outlier, alone
#
# Sized for ~32 GB / ~4 cores: each worker holds at most cap x 19 x 8 x 13 float32
# (~0.8 GB) plus one 2M-row read buffer, so four in parallel stay well inside the node.
set -uo pipefail

WORK=/scratch/users/bmerwan/simco_work/baselines
DATA=/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets
REPO=/scratch/users/bmerwan/simco
NPROC="${NPROC:-4}"

cd "$REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$WORK/medians" "$WORK/logs"

run_one() {
  local ct="$1"
  [ -s "$WORK/medians/${ct}.csv" ] && { echo "[SKIP ] $ct already done"; return 0; }
  echo "[START] $ct $(date +%T)"
  python -m baselines.stim_effect_medians \
    --celltype "$ct" \
    --data-dir "$DATA/ptb_concatenated_batchcorrected" \
    --features "$DATA/ptb_concatenated_per_condition_celltype/13features.txt" \
    --outdir   "$WORK/medians" \
    > "$WORK/logs/med_${ct}.out" 2> "$WORK/logs/med_${ct}.err"
  echo "[DONE ] $ct rc=$? $(date +%T)"
}
export -f run_one
export WORK DATA REPO

if [ $# -gt 0 ]; then
  for ct in "$@"; do run_one "$ct"; done
  exit 0
fi

# Smallest first: the quick wins land while Granulocytes is still reading.
while read -r ct; do
  f="$DATA/ptb_concatenated_batchcorrected/${ct}_HVPV.h5ad"
  echo "$(stat -c %s "$f") $ct"
done < "$WORK/celltypes.txt" | sort -n | awk '{print $2}' \
  | xargs -I{} -P "$NPROC" bash -c 'run_one "$@"' _ {}

echo "[ALL DONE] $(ls "$WORK/medians" | wc -l) celltype CSVs"
