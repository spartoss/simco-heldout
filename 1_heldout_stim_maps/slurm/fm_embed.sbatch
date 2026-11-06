#!/bin/bash
# ImmuneFM phase 1: donor-level mean Unstim embeddings (fm/embed_donors.py) -> work/lc/fm/donor_emb.npz.
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -J fm_embed
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/fm_embed_%j.out
set -euo pipefail
source /scratch/users/bmerwan/atlas_heldout/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -m fm.embed_donors "$@"
