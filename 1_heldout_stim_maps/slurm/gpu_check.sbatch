#!/bin/bash
# Throughput (M = 168 / 840 / 1680 maps) then parity (4 pairs, all 19 donors, 50k iterations, vs the paper's model.pt).
# Writes WORK/PARITY_PASS only if our maps are in line with the paper's; the training array checks for it.
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:16GB|GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -J lc_check
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/check_%j.out
set -euo pipefail
source /scratch/users/bmerwan/atlas_heldout/env.sh
W=/scratch/users/bmerwan/atlas_heldout/work/lc
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -m lc.test_equivalence
python3 -m lc.train --bench --out $W/check/bench
python3 -m lc.train --parity --out $W/check/parity
