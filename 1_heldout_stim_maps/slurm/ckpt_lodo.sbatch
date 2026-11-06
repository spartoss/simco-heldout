#!/bin/bash
# TabPFN on the compact feature set C_ms_lin with the 3 alternative v2 regressor checkpoints, one LODO fold per task
# -> work/lc/tab/v2_ck{09,2no,wyl}_lodo_C_ms_lin/repeat_<fold>/      sbatch --array=0-18 slurm/ckpt_lodo.sbatch
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_GEN:VLT|GPU_GEN:TUR|GPU_GEN:AMP|GPU_GEN:LOV|GPU_GEN:HPR
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -J ckpt
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/ckpt_%A_%a.out
set -euo pipefail
W=/scratch/users/bmerwan/atlas_heldout/work/lc
source /scratch/users/bmerwan/atlas_heldout/env.sh
module load viz py-matplotlib/3.8.3_py312 py-scikit-learn/1.5.1_py312 >/dev/null 2>&1
V=$GROUP_HOME/bmerwan/venv-tabpfn
source $V/bin/activate
export PYTHONPATH=$V/lib/python3.12/site-packages:${PYTHONPATH:-}
export TABPFN_MODEL_CACHE_DIR=$GROUP_HOME/bmerwan/tabpfn_ckpt
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv
F=$SLURM_ARRAY_TASK_ID
for pair in 09:09gpqh39 2no:2noar4o2 wyl:wyl4o83o; do
  t=${pair%%:*}; c=${pair#*:}
  python3 -m lc.tab_v2 --mode ct --design $W/design_lodo.json --repeat $F --ks 18 --tag _ck${t}_lodo --layers C_ms_lin \
    --ckpt tabpfn-v2-regressor-$c.ckpt
done
