#!/bin/bash
# Score the cond_var variants (lc/var_score.py) on the feature set, k = 15, repeats 0-4 (afterany the array).
#SBATCH -p normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -J cond_var_score
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/cond_var_score_%j.out
set -euo pipefail
cd /scratch/users/bmerwan/atlas_heldout
python -m lc.var_score --methods none=cond_train/repeat_{r}/none pcb_gate_wd1=cond_train/repeat_{r}/pcblood_wd1 pcb_bias_wd1=cond_train/repeat_{r}/pcb_bias_wd1 pcb_film_wd1=cond_train/repeat_{r}/pcb_film_wd1 pcb_trans_wd1=cond_train/repeat_{r}/pcb_trans_wd1 pcb_trans_wd0.1=cond_train/repeat_{r}/pcb_trans_wd0.1 pcb_affine_wd1=cond_train/repeat_{r}/pcb_affine_wd1 pcb_gate_wd10=cond_train/repeat_{r}/pcb_gate_wd10 pcb_gate_drop0.5=cond_train/repeat_{r}/pcb_gate_drop0.5 pcb_gate_sr0.3=cond_train/repeat_{r}/pcb_gate_sr0.3 pcb_trans_sr0.3=cond_train/repeat_{r}/pcb_trans_sr0.3 none_sr0.3=cond_train/repeat_{r}/none_sr0.3 none_sr1=cond_train/repeat_{r}/none_sr1 --ref none --out work/lc/cond_var/score_k15_r01234
