#!/bin/bash
# Merge the per-celltype CondOT outputs, score both CondOT arms on the held-out design, rebuild the notebook.
#SBATCH -p normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -J condot_score
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/condot_score_%j.out
set -uo pipefail
source /scratch/users/bmerwan/atlas_heldout/env_cpu.sh
W=/scratch/users/bmerwan/atlas_heldout/work/lc
for R in 0 1 2 3 4; do python3 -m lc.condot_merge --indir $W/condot/celltype/repeat_$R --out $W/condot/celltype_merged/repeat_$R; done
python3 -m lc.where_skill --ks 15 --out cases_condot.pkl --extra paired=cond_train/repeat_{r}/none \
    condot_ct=condot/celltype_merged/repeat_{r} condot_global=condot/global/repeat_{r}
python3 -m lc.build_where_skill_notebook && jupyter nbconvert --to notebook --execute --inplace notebooks/where_cellot_skill.ipynb
