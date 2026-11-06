#!/bin/bash
# Closed-form maps on the donor designs (CPU). Tasks 0-5 hvpv repeats, 6-10 small repeats, 11 pairwise.
#SBATCH -p normal
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -J lc_donors_base
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/donorsbase_%A_%a.out
set -euo pipefail
W=/scratch/users/bmerwan/atlas_heldout/work/lc; D=$W/designs; i=$SLURM_ARRAY_TASK_ID
source /scratch/users/bmerwan/atlas_heldout/env_cpu.sh; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
if (( i < 6 )); then python3 -m lc.baselines --repeat $i --spec $D/hvpv/r$i.json --out $W/donors_base/hvpv
elif (( i < 11 )); then r=$((i-6)); python3 -m lc.baselines --repeat $r --spec $D/small/r$r.json --out $W/donors_base/small
else python3 -m lc.baselines --repeat 0 --spec $D/pairwise.json --out $W/donors_base/pairwise; fi
