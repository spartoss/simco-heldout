#!/bin/bash
# Best held-out model (shift + TabPFN C_ms_lin, trained on all 19 atlas donors) applied to the 177 OOL samples, one stim
# per task (lc/ool_tab.py) -> work/lc/ool/pred_q_<stim>.pkl.   sbatch --array=0-6 slurm/ool_tab.sbatch  (resumable)
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_GEN:VLT|GPU_GEN:TUR|GPU_GEN:AMP|GPU_GEN:LOV|GPU_GEN:HPR
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -J ool_tab
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/ool_tab_%A_%a.out
set -euo pipefail
cd /scratch/users/bmerwan/atlas_heldout
source env.sh
module load viz py-matplotlib/3.8.3_py312 py-scikit-learn/1.5.1_py312 >/dev/null 2>&1
V=$GROUP_HOME/bmerwan/venv-tabpfn
source $V/bin/activate
export PYTHONPATH=$V/lib/python3.12/site-packages:${PYTHONPATH:-}
export TABPFN_MODEL_CACHE_DIR=$GROUP_HOME/bmerwan/tabpfn_ckpt
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv
STIMS=(GMCSF IFNa IL246 IL33 LPS PI TNFa)
S=${STIMS[$SLURM_ARRAY_TASK_ID]}
python3 -m lc.ool_tab --stims $S --tag _$S "$@"
