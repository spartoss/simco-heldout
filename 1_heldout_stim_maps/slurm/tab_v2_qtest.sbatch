#!/bin/bash
# Quantile tests for TabPFN L1cat at k = 15, one repeat per task (both runs in one task, ~20 min):
#   (1) trained and predicted on all five quantile rows  -> work/lc/tab/v2_q5_L1cat/repeat_<r>/
#   (2) own-profile features restricted to q50-q90 (39 columns), q50-q90 rows -> work/lc/tab/v2_q3f_L1cat_q3f/repeat_<r>/
#   sbatch --array=0-4 slurm/tab_v2_qtest.sbatch
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_GEN:VLT|GPU_GEN:TUR|GPU_GEN:AMP|GPU_GEN:LOV|GPU_GEN:HPR
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -J tab_qtest
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/tab_qtest_%A_%a.out
set -euo pipefail
source /scratch/users/bmerwan/atlas_heldout/env.sh
module load viz py-matplotlib/3.8.3_py312 py-scikit-learn/1.5.1_py312 >/dev/null 2>&1
V=$GROUP_HOME/bmerwan/venv-tabpfn
source $V/bin/activate
export PYTHONPATH=$V/lib/python3.12/site-packages:${PYTHONPATH:-}
export TABPFN_MODEL_CACHE_DIR=$GROUP_HOME/bmerwan/tabpfn_ckpt
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv
R=$SLURM_ARRAY_TASK_ID
python3 -m lc.tab_v2 --mode ct --repeat $R --ks 15 --qkeep 10 25 50 75 90 --tag _q5 --layers L1cat
python3 -m lc.tab_v2 --mode ct --repeat $R --ks 15 --tag _q3f --layers L1cat_q3f
