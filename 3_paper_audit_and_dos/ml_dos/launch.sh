#!/bin/bash
# Submit the supervised-DOS benchmark (measured Unstim features only).
#   bash ml_dos/launch.sh            # snapshot code, write tasks.txt, submit array + summary
# Code is copied to $OUT/code so the queued jobs do not depend on this checkout; every task writes one
# file per split and skips existing ones, so resubmitting the same array index resumes it.
set -euo pipefail
OUT=/scratch/users/bmerwan/simco_work/onset_test/ml_dos
SRC=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/code" "$OUT/logs"
cp "$SRC"/*.py "$SRC"/*.sh "$SRC"/*.sbatch "$OUT/code/"

ARMS="freqmed qtl full full_plus"
SK="mean ridge lasso enet pls pcr svr_lin svr_rbf krr_rbf knn gp rf et bag_tree bag_ridge xgb_p43 xgb hgb gbr ada mlp stack"
: > "$OUT/tasks.txt"
for arm in $ARMS; do
  for m in $SK; do echo "run_cv $arm $m" >> "$OUT/tasks.txt"; done
  for m in dnn dnn_wide; do echo "run_dnn $arm $m" >> "$OUT/tasks.txt"; done
done
echo "run_deepsets cells deepsets" >> "$OUT/tasks.txt"
N=$(($(wc -l < "$OUT/tasks.txt") - 1))
jid=$(sbatch --parsable --array=0-$N%24 "$OUT/code/task.sbatch")
sid=$(sbatch --parsable --dependency=afterany:$jid "$OUT/code/summary.sbatch")
echo "array $jid (0-$N), summary $sid"
echo "$(date '+%F %T') submitted array $jid (0-$N), summary $sid" >> "$OUT/STATUS.md"
