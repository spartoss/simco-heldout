#!/bin/bash
# lc/tab_predict.py --model tabpfn (TabPFN v2 regressor, $GROUP_HOME/bmerwan/venv-tabpfn), one repeat per array task.
#   sbatch --array=0-4 slurm/tab_pfn.sbatch
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:16GB|GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -J tab_pfn
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/tab_pfn_%A_%a.out
set -euo pipefail
source /scratch/users/bmerwan/atlas_heldout/env.sh
module load viz py-matplotlib/3.8.3_py312 py-scikit-learn/1.5.1_py312 >/dev/null 2>&1
V=$GROUP_HOME/bmerwan/venv-tabpfn
source $V/bin/activate
export PYTHONPATH=$V/lib/python3.12/site-packages:${PYTHONPATH:-}     # venv typing_extensions before the module's
export TABPFN_MODEL_CACHE_DIR=$GROUP_HOME/bmerwan/tabpfn_ckpt         # tabpfn-v2-regressor.ckpt, fetched 2026-10-08
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -m lc.tab_predict --model tabpfn --repeat $SLURM_ARRAY_TASK_ID
