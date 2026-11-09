#!/bin/bash
# Drug arm on the REPEATED held-out design (design.json: 5 repeats x 4 test donors, 15 training donors), lc.drug_step --design repeat
# task t = repeat t % 5, stim STIMS[t / 5] -> work/lc/drug_tab/steprep_<stim>/fold_<repeat>.pkl   (resumable)
#   sbatch --array=0-24 slurm/drug_steprep.sbatch
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_GEN:VLT|GPU_GEN:TUR|GPU_GEN:AMP|GPU_GEN:LOV|GPU_GEN:HPR
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -J drug_steprep
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/drug_steprep_%A_%a.out
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
t=$SLURM_ARRAY_TASK_ID; R=$((t % 5)); S=${STIMS[$((t / 5))]}
python3 -m lc.drug_step --design repeat --fold $R --stims $S --tag _$S
