#!/bin/bash
# Lineage round 2, LightGBM (pooled over celltypes) with the new per-celltype layers L1X L1M L1allM:
# the 5 learning-curve repeats at k = 15 and the 19 leave-one-donor-out folds at k = 18, in one CPU job (~30-60 min).
#   -> work/lc/tab/v2lgb_x_<L>, v2lgb_x_lodo_<L>
#SBATCH -p normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -J lineage2_lgbm
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/lineage2_lgbm_%j.out
set -euo pipefail
W=/scratch/users/bmerwan/atlas_heldout/work/lc
source /scratch/users/bmerwan/atlas_heldout/env.sh
module load viz py-matplotlib/3.8.3_py312 py-lightgbm/4.5.0_py312 py-scikit-learn/1.5.1_py312 >/dev/null 2>&1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
for r in 0 1 2 3 4; do
  python3 -m lc.tab_v2 --model lgbm --mode ct --repeat $r --ks 15 --tag _x --layers L1X L1M L1allM
done
for f in $(seq 0 18); do
  python3 -m lc.tab_v2 --model lgbm --mode ct --design $W/design_lodo.json --repeat $f --ks 18 --tag _x_lodo --layers L1X L1M L1allM
done
