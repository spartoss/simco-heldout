#!/bin/bash
# Score the per-patient checkpoint arms (2026-10-12): paired CellOT at k = 15, centred (dnorm) or not, checkpoint by the
# pooled or the per-donor median error, against the existing paired control (none) and the old dnorm run.
#   sbatch --dependency=afterany:<train jobs> slurm/pdonor_score.sbatch
#SBATCH -p normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -J pdonor_score
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/pdonor_score_%j.out
set -uo pipefail
cd /scratch/users/bmerwan/atlas_heldout
source env_cpu.sh
M="none=cond_train/repeat_{r}/none dnorm=cond_train/repeat_{r}/dnorm:dnorm dnorm_pool=cond_train/repeat_{r}/dnorm_pool:dnorm \
   donor50=cond_train/repeat_{r}/donor50 dnorm_donor50=cond_train/repeat_{r}/dnorm_donor50:dnorm"
python3 -m lc.ctx_score --methods $M --ks 15 --qs 50 75 90 --vs none dnorm_pool shift --out work/lc/ctx_score/pdonor_q50_75_90
python3 -m lc.ctx_score --methods $M --ks 15 --vs none dnorm_pool shift --out work/lc/ctx_score/pdonor_allq
