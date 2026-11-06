#!/bin/bash
# ImmuneFM phase 4 (fm/finetune.py): continued masked pre-training on the atlas, quantile tokens.
# Task t: repeat t/2, init pretrained (t even) or random (t odd, the ablation) -> work/lc/fm/finetune/repeat_<r>/<init>/
# Then the donor covariates of that fine-tune (fm/embed_donors.py --ckpt) -> .../donor_emb.npz
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-9
#SBATCH -J fm_ft
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/fm_ft_%A_%a.out
set -euo pipefail
source /scratch/users/bmerwan/atlas_heldout/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
T=$SLURM_ARRAY_TASK_ID; R=$((T / 2)); I=$([ $((T % 2)) -eq 0 ] && echo pretrained || echo random)
O=/scratch/users/bmerwan/atlas_heldout/work/lc/fm/finetune/repeat_$R/$I
python3 -m fm.finetune --repeat $R --init $I --out $O "$@"
python3 -m fm.embed_donors --ckpt $O/ckpt.pt --out $O/donor_emb.npz
