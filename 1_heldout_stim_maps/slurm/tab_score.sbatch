#!/bin/bash
# Score the tab_predict runs: residual readouts + pred dirs (lc/tab_ens.py), then lc.ctx_score on common cases.
#   sbatch --dependency=afterok:<cpu>:<pfn> slurm/tab_score.sbatch
#SBATCH -p normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -J tab_score
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/tab_score_%j.out
set -euo pipefail
source /scratch/users/bmerwan/atlas_heldout/env.sh
module load viz py-matplotlib/3.8.3_py312 py-scikit-learn/1.5.1_py312 >/dev/null 2>&1
N=""; for m in ridge lgbm tabpfn; do for L in L0 L1 L2 L2perm Lperm; do N="$N ${m}_$L"; done; done
python3 -m lc.tab_ens --names $N --lambdas 0.5 1 --cellot
ENS=../../../mfm_heldout/work/ens/w0.5/repeat_{r}
M="none=cond_train/repeat_{r}/none ens=$ENS"
for m in ridge lgbm tabpfn; do for L in L0 L1 L2 Lperm; do
  M="$M ${m}_$L=tab/pred/${m}_${L}_l1/repeat_{r} ${m}_${L}_h=tab/pred/${m}_${L}_l0.5/repeat_{r}"; done
  M="$M ${m}_L1_ens=tab/pred/${m}_L1_l1_ens/repeat_{r} ${m}_L2_ens=tab/pred/${m}_L2_l1_ens/repeat_{r}"
  M="$M ${m}_L1_hens=tab/pred/${m}_L1_l0.5_ens/repeat_{r} ${m}_L2_hens=tab/pred/${m}_L2_l0.5_ens/repeat_{r}"; done
W=/scratch/users/bmerwan/atlas_heldout/work/lc
python3 -m lc.ctx_score --methods $M --repeats 0 1 2 3 4 --vs ens shift --out $W/tab/score_all
python3 -m lc.ctx_score --methods $M --repeats 1 2 3 4 --vs ens shift --out $W/tab/score_r1-4
