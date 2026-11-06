#!/bin/bash
# ImmuneFM phase 0 go/no-go (fm/sanity.py): masked reconstruction, linear probes, token resolution on the atlas.
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -J fm_sanity
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/fm_sanity_%j.out
set -euo pipefail
source /scratch/users/bmerwan/atlas_heldout/env.sh
module load py-scikit-learn/1.5.1_py312 >/dev/null 2>&1
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -m fm.sanity "$@"
