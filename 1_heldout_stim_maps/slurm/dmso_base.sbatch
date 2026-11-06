#!/bin/bash
# DMSO-plate experiment, step 2b: closed-form maps (identity, median_shift, quantile_map, gauss_ot), k = 15,
# one repeat per array task (0-4), all five arms -> work/lc/dmso/baselines/<arm>/repeat_<r>.npz.
#SBATCH -p normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -J dmso_base
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/dmso_base_%A_%a.out
set -euo pipefail
L=/scratch/users/bmerwan/atlas_heldout/work/lc; W=$L/dmso
source /scratch/users/bmerwan/atlas_heldout/env_cpu.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
R=$SLURM_ARRAY_TASK_ID
for ARM in pool poolN1 DMSO1 DMSO2 DMSO3; do
    [ -f $W/baselines/$ARM/repeat_$R.npz ] && { echo "[SKIP ] $ARM"; continue; }
    C=$W/cache_$ARM; [ $ARM = pool ] && C=$L/cache
    python3 -m lc.baselines --repeat $R --ks 15 --cache-dir $C --out $W/baselines/$ARM
done
