#!/bin/bash
# Drug arm, step setting (lc/drug_step.py: measured control), LODO: task t = fold t % 19, stim STIMS[t / 19]
#   -> work/lc/drug_tab/step_<stim>/      sbatch --array=0-94 slurm/drug_step.sbatch   (resumable)
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_GEN:VLT|GPU_GEN:TUR|GPU_GEN:AMP|GPU_GEN:LOV|GPU_GEN:HPR
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -J drug_step
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/drug_step_%A_%a.out
set -euo pipefail
source /scratch/users/bmerwan/atlas_heldout/env.sh
module load viz py-matplotlib/3.8.3_py312 py-scikit-learn/1.5.1_py312 >/dev/null 2>&1
V=$GROUP_HOME/bmerwan/venv-tabpfn
source $V/bin/activate
export PYTHONPATH=$V/lib/python3.12/site-packages:${PYTHONPATH:-}
export TABPFN_MODEL_CACHE_DIR=$GROUP_HOME/bmerwan/tabpfn_ckpt
export HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv
STIMS=(GMCSF IFNa IL246 LPS TNFa)
t=$SLURM_ARRAY_TASK_ID; F=$((t % 19)); S=${STIMS[$((t / 19))]}
python3 -m lc.drug_step --fold $F --stims $S --tag _$S
