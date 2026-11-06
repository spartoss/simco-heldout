#!/bin/bash
# L1allM ablations with LightGBM on the 19 LODO folds (k = 18), one CPU job -> work/lc/tab/v2lgb_abl_lodo_<L>
#SBATCH -p normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -J abl_lgbm
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/abl_lgbm_%j.out
set -euo pipefail
W=/scratch/users/bmerwan/atlas_heldout/work/lc
source /scratch/users/bmerwan/atlas_heldout/env.sh
module load viz py-matplotlib/3.8.3_py312 py-lightgbm/4.5.0_py312 py-scikit-learn/1.5.1_py312 >/dev/null 2>&1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
for f in $(seq 0 18); do
  python3 -m lc.tab_v2 --model lgbm --mode ct --design $W/design_lodo.json --repeat $f --ks 18 --tag _abl_lodo \
    --layers L1allM_noOwn L1allM_noFreq L1allM_noBPC Mmin
done
