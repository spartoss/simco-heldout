#!/bin/bash
# lc/tab_v2.py (TabPFN v2, step-by-step feature sets), one repeat per array task; resumable (rerun the same task).
#   sbatch --array=0-4 slurm/tab_v2.sbatch ct      -> work/lc/tab/v2_<layer>/repeat_<r>/   (10 layers, ~4.5k fits)
#   sbatch --array=0-4 --time=03:00:00 slurm/tab_v2.sbatch group   (lineage-group tables, 3 layers)
#   extra arguments are passed on, e.g. ct --layers L1cat_mx L1F
# Pascal GPUs (P100) excluded: about 2x slower than V100 for TabPFN (tab_pfn r3/r4 timed out on them).
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_GEN:VLT|GPU_GEN:TUR|GPU_GEN:AMP|GPU_GEN:LOV|GPU_GEN:HPR
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -J tab_v2
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/tab_v2_%x_%A_%a.out
set -euo pipefail
source /scratch/users/bmerwan/atlas_heldout/env.sh
module load viz py-matplotlib/3.8.3_py312 py-scikit-learn/1.5.1_py312 >/dev/null 2>&1
V=$GROUP_HOME/bmerwan/venv-tabpfn
source $V/bin/activate
export PYTHONPATH=$V/lib/python3.12/site-packages:${PYTHONPATH:-}     # venv typing_extensions before the module's
export TABPFN_MODEL_CACHE_DIR=$GROUP_HOME/bmerwan/tabpfn_ckpt
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -m lc.tab_v2 --mode $1 --repeat $SLURM_ARRAY_TASK_ID "${@:2}"
