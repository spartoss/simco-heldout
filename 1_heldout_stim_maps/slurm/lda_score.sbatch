#!/bin/bash
# Score the donor-signature (LDA) context arms against the context-free paired CellOT, pcblood, pc_wd1 and the shift.
#   sbatch --dependency=afterany:<lda jobs> slurm/lda_score.sbatch
#SBATCH -p normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -J lda_score
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/lda_score_%j.out
set -uo pipefail
source /scratch/users/bmerwan/atlas_heldout/env_cpu.sh
M="none=cond_train/repeat_{r}/none pc_wd1=cond_train/repeat_{r}/pc_wd1 pcblood=cond_train/repeat_{r}/pcblood_wd1 \
   lda2=cond_train/repeat_{r}/lda2_wd1 lda3=cond_train/repeat_{r}/lda3_wd1"
python3 -m lc.ctx_score --methods $M --repeats 0 1 2 3 4 --vs none shift pcblood --out work/lc/ctx_score/lda_r01234
python3 -m lc.ctx_score --methods $M --repeats 1 2 3 4 --vs none shift pcblood --out work/lc/ctx_score/lda_r1234
