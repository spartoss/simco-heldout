#!/bin/bash
# Score the dnorm / dnorm_clip / clip arms with the paired control and the baselines, then rebuild the notebook.
#SBATCH -p normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -J dnorm_score
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/dnorm_score_%j.out
set -uo pipefail
source /scratch/users/bmerwan/atlas_heldout/env_cpu.sh
python3 -m lc.where_skill --ks 15 --out cases_dnorm.pkl --extra paired=cond_train/repeat_{r}/none \
    dnorm=cond_train/repeat_{r}/dnorm:dnorm dnorm_clip=cond_train/repeat_{r}/dnorm_clip:dnorm clip=cond_train/repeat_{r}/clip
python3 -m lc.build_where_skill_notebook && jupyter nbconvert --to notebook --execute --inplace notebooks/where_cellot_skill.ipynb
