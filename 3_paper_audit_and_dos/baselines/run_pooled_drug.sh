#!/bin/bash
# The drug arm of the pooled benchmark: DMSO -> drug within each stim, 9 drugs x 168
# pairs, against the paper's existing checkpoints in results_drug/cross_validation_drug_OG13.
# Nothing is trained here -- those 1512 maps already exist; only the baselines are fit.
# Runs inside the existing allocation -- no sbatch.
#
#   NPROC=3 bash baselines/run_pooled_drug.sh          # all 28, smallest first
#   bash baselines/run_pooled_drug.sh Granulocytes     # the 30 GB outlier, alone
#   DRUGS=CHT,MF bash baselines/run_pooled_drug.sh pDCs
#
# ~9x the work of run_pooled.sh per celltype: the .h5ad is read once but every one of the
# 9 drugs gets its own batch-corrected copy, its own split and its own set of fits. NPROC
# is lower than the stim arm's 4 for the same reason -- each worker holds the whole
# celltype as float32 plus a per-pair float64 copy.
#
# Resumable: a celltype whose CSV exists is skipped. The output dir is its own and must
# stay that way: drug rows carry a `drug` column that stim rows do not, so the two are
# not schema-compatible and summarize.py must never be pointed at both.
set -uo pipefail

WORK=/scratch/users/bmerwan/simco_work/baselines
DATA=/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets
REPO=/scratch/users/bmerwan/simco
OUT="${OUT:-$WORK/pooled_drug}"
NPROC="${NPROC:-3}"
LIST="${LIST:-$WORK/celltypes.txt}"
DRUGS="${DRUGS:-}"

cd "$REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT" "$WORK/logs"

# Bash arrays do not survive `export -f`, so the optional subset is carried as a plain
# string and left unquoted at the call site -- it holds no whitespace.
EXTRA=""
[ -n "$DRUGS" ] && EXTRA="--drugs $DRUGS"

run_one() {
  local ct="$1"
  [ -s "$OUT/${ct}.csv" ] && { echo "[SKIP ] $ct already done"; return 0; }
  echo "[START] $ct $(date +%T)"
  python -m baselines.run_baselines \
    --arm drug \
    --celltype "$ct" \
    --data-dir "$DATA/ptb_concatenated_batchcorrected" \
    --features "$DATA/ptb_concatenated_per_condition_celltype/13features.txt" \
    --pairs-file "$WORK/pairs_168.txt" \
    --outdir "$OUT" \
    --eval-mode pooled \
    --eval-split test \
    --weighting both \
    --cellot-results "$DATA/../results_drug/cross_validation_drug_OG13" \
    --batch-correction "$REPO/cross_validation/batchcorrection_2.csv" \
    $EXTRA \
    > "$WORK/logs/pooled_drug_${ct}.out" 2> "$WORK/logs/pooled_drug_${ct}.err"
  echo "[DONE ] $ct rc=$? $(date +%T)"
}
export -f run_one
export WORK DATA REPO OUT EXTRA

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
