#!/bin/bash
# Score the quantile tests (slurm/tab_v2_qtest.sbatch), k = 15, 5 repeats, no PI/IL33:
#   q3 = scored on q50-q90 (5-quantile L1cat vs 3-quantile L1cat vs L1cat_q3f), q5 = scored on all five quantiles
#   (5-quantile L1cat vs CellOT / shift). Alone and 0.4 CellOT + 0.6 model.   -> work/lc/tab/qtest_{q3,q5}
#   sbatch --dependency=afterok:<qtest> slurm/tab_qtest_score.sbatch
#SBATCH -p normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -J tab_qscore
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/tab_qscore_%j.out
set -euo pipefail
source /scratch/users/bmerwan/atlas_heldout/env.sh
module load viz py-matplotlib/3.8.3_py312 py-scikit-learn/1.5.1_py312 >/dev/null 2>&1
python3 -m lc.tab_ens --names v2_q5_L1cat v2_q3f_L1cat_q3f --repeats 0 1 2 3 4 --lambdas 1 --cellot --w-cellot 0.4 --res-tag _qtest
P=tab/pred; R="repeat_{r}"; W=/scratch/users/bmerwan/atlas_heldout/work/lc/tab
REF="CellOT=cond_train/$R/none CellOT+shift=../../../mfm_heldout/work/ens/w0.5/$R"
python3 -m lc.tab_compare --methods $REF L1cat_q5rows=$P/v2_q5_L1cat_l1/$R L1cat=$P/v2_L1cat_l1/$R L1cat_q3f=$P/v2_q3f_L1cat_q3f_l1/$R \
  mix_L1cat_q5rows=$P/v2_q5_L1cat_l1_c0.40/$R mix_L1cat=$P/v2_L1cat_l1_c0.40/$R mix_L1cat_q3f=$P/v2_q3f_L1cat_q3f_l1_c0.40/$R \
  --repeats 0 1 2 3 4 --ks 15 --exclude-stims IL33 PI --qs 50 75 90 --rule-qs 50 75 90 --vs L1cat mix_L1cat --dump --out $W/qtest_q3
python3 -m lc.tab_compare --methods $REF L1cat_q5rows=$P/v2_q5_L1cat_l1/$R mix_L1cat_q5rows=$P/v2_q5_L1cat_l1_c0.40/$R \
  --repeats 0 1 2 3 4 --ks 15 --exclude-stims IL33 PI --qs 10 25 50 75 90 --rule-qs 10 25 50 75 90 --vs CellOT+shift --dump --out $W/qtest_q5
