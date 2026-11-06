#!/bin/bash
# Score the uncapped run, both parts: summary_full/{curve,cells}/ (CSVs + learning_curve.png). Safe to rerun by hand.
#SBATCH -p normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -J lcfull_sum
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/summaryfull_%j.out
set -uo pipefail
W=/scratch/users/bmerwan/atlas_heldout/work/lc
source /scratch/users/bmerwan/atlas_heldout/env_cpu.sh
for P in curve cells; do
  python3 -m lc.score --train-dir $W/train_full --part $P --baseline-dir $W/baselines_full/$P \
    --cache-dir $W/cache_full --out $W/summary_full/$P
done
