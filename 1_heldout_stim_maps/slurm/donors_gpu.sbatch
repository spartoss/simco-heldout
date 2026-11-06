#!/bin/bash
# Donor experiments (lc/design_donors.py), CellOT on GPU, 20k iterations. Submit with --export=EXP=<hvpv|small|pairwise>:
#   hvpv     array 0-5  : one repeat each (1008 maps)
#   small    array 0-4  : one repeat each (51 small-celltype pairs x 6 k x 6 caps = 1836 maps)
#   pairwise array 0-3  : 19 single-donor training sets x all 19 test donors, celltypes split in 4 groups (~800 maps);
#                         one evaluation at the end, test donors at <= 5k cells
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:16GB|GPU_MEM:24GB|GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -J lc_donors
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/donors_%x_%A_%a.out
set -euo pipefail
W=/scratch/users/bmerwan/atlas_heldout/work/lc; D=$W/designs; i=$SLURM_ARRAY_TASK_ID
source /scratch/users/bmerwan/atlas_heldout/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
case "$EXP" in
  hvpv)  python3 -m lc.train --spec $D/hvpv/r$i.json --n-iters 20000 --eval-every 5000 --out $W/donors/hvpv/r$i ;;
  small) python3 -m lc.train --spec $D/small/r$i.json --n-iters 20000 --eval-every 5000 --out $W/donors/small/r$i ;;
  pairwise)
    CTS=$(python3 -c "from lc.paths import CELLTYPES; print(' '.join(CELLTYPES[$i::4]))")
    python3 -m lc.train --spec $D/pairwise.json --celltypes $CTS --n-iters 20000 --eval-every 20000 \
      --max-test-cells 5000 --out $W/donors/pairwise/g$i ;;
  *) echo "unknown EXP=$EXP"; exit 2 ;;
esac
