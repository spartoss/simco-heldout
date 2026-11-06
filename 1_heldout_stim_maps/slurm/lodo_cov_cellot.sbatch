#!/bin/bash
# Leave-one-donor-out covariate CellOTs (work/lc/design_lodo.json: 19 folds, k = 18), the 112 pairs of the 5 kept stims,
# same settings as the learning-curve runs (cond_train/repeat_r/<name>/meta.json):
#   tasks  0-18: pcblood_wd1        (--ctx pcblood --n-pc 3 --ctx-wd 1)
#   tasks 19-37: pcb_film_wd1       (--ctx pcblood --n-pc 3 --ctx-wd 1 --arch film)
#   tasks 38-56: cluster_K4_stable  (--ctx cluster --K 4 --stable)
#   -> work/lc/cond_train_lodo/repeat_<fold>/<name>. Resumable from state.pt.   sbatch --array=0-56 slurm/lodo_cov_cellot.sbatch
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_GEN:VLT|GPU_GEN:TUR|GPU_GEN:AMP|GPU_GEN:LOV|GPU_GEN:HPR
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -J lodo_covot
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/lodo_covot_%A_%a.out
set -euo pipefail
W=/scratch/users/bmerwan/atlas_heldout/work/lc
source /scratch/users/bmerwan/atlas_heldout/env.sh
module load py-scikit-learn/1.5.1_py312 >/dev/null 2>&1
nvidia-smi --query-gpu=name,memory.total --format=csv
PAIRS=$(awk -F, '$1 ~ /^(GMCSF|IFNa|IL246|LPS|TNFa)$/ {printf "%s:%s ", $1, $2}' /scratch/users/bmerwan/atlas_heldout/meta/pairs.txt)
t=$SLURM_ARRAY_TASK_ID; f=$((t % 19)); v=$((t / 19))
case $v in
  0) N=pcblood_wd1;       ARGS="--ctx pcblood --n-pc 3 --ctx-wd 1" ;;
  1) N=pcb_film_wd1;      ARGS="--ctx pcblood --n-pc 3 --ctx-wd 1 --arch film" ;;
  2) N=cluster_K4_stable; ARGS="--ctx cluster --K 4 --stable" ;;
esac
python3 -m lc.cond_train --design $W/design_lodo.json --repeat $f --ks 18 --n-iters 20000 --pairs $PAIRS $ARGS \
  --out $W/cond_train_lodo/repeat_$f/$N
