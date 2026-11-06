#!/bin/bash
# Lineage round 2 (lc/tab_v2.py), TabPFN, one repeat (lc) or fold (lodo) per task, all runs resumable:
#   ct  : per-celltype tables with row-aligned other-celltype features   L1X L1M L1allM         -> v2_x<sfx>_<L>
#   cur : curated stim-specific groups   G_L1cat G_L1F G_L1all_raw G_L1M G_L1allM              -> v2_gcur<sfx>_<L>
#   big : whole lineages (more rows)     G_L1cat G_L1all_raw G_L1M G_L1allM                    -> v2_gbig<sfx>_<L>
#   T   : all 16 T celltypes (most rows) G_L1cat G_L1M G_L1allM                                -> v2_gT<sfx>_<L>
#   sbatch --array=0-4  slurm/lineage2.sbatch lc     (k = 15, learning-curve repeats; sfx = "")
#   sbatch --array=0-18 slurm/lineage2.sbatch lodo   (k = 18, leave-one-donor-out folds; sfx = "_lodo")
# >= 24 GB GPUs: the all-T tables have ~9-11k training rows.
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -J lineage2
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/lineage2_%x_%A_%a.out
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
R=$SLURM_ARRAY_TASK_ID
if [[ $1 == lc ]]; then D="--ks 15"; S=""; else D="--design $W/design_lodo.json --ks 18"; S="_lodo"; fi
python3 -m lc.tab_v2 --mode ct --repeat $R $D --tag _x$S --layers L1X L1M L1allM
python3 -m lc.tab_v2 --mode group --groupset cur --repeat $R $D --tag _gcur$S --layers G_L1cat G_L1F G_L1all_raw G_L1M G_L1allM
python3 -m lc.tab_v2 --mode group --groupset big --repeat $R $D --tag _gbig$S --layers G_L1cat G_L1all_raw G_L1M G_L1allM
python3 -m lc.tab_v2 --mode group --groupset T --repeat $R $D --tag _gT$S --layers G_L1cat G_L1M G_L1allM
