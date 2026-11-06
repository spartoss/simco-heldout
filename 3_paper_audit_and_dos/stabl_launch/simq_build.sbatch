#!/bin/bash
# CellOT-simulated quantile features for the OOL women, then merge with the combined CellOT table.
#SBATCH -p normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -J simq_build
#SBATCH -o /scratch/users/bmerwan/simco_work/onset_test/paper_protocol/logs/simq_build_%j.out
#SBATCH -e /scratch/users/bmerwan/simco_work/onset_test/paper_protocol/logs/simq_build_%j.err
set -euo pipefail
P=/scratch/users/bmerwan/simco_work/onset_test/paper_protocol
module load python/3.9.0; source /scratch/users/bmerwan/venv-cellot/bin/activate
cd /scratch/users/bmerwan/simco; export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK HDF5_USE_FILE_LOCKING=FALSE
[[ -s $P/tables/cellot_simq.csv ]] || python -m baselines.sim_quantile_features --out $P/tables/cellot_simq.csv
python - <<PY
import pandas as pd
P="$P"
a=pd.read_csv(f"{P}/tables/comb_cellot.csv",index_col=0); q=pd.read_csv(f"{P}/tables/cellot_simq.csv",index_col=0)
t=a.join(q.reindex(a.index)); assert list(t.index)==list(a.index)
blocks=["TNFa","LPS","IL246","IFNa","GMCSF","PI","IL33","freq","med","qtl"]+[s+"q" for s in ["TNFa","LPS","IL246","IFNa","GMCSF","PI","IL33"]]
n={b:sum(c.endswith(b) for c in t.columns) for b in blocks}; assert sum(n.values())==t.shape[1], n
t.to_csv(f"{P}/tables/comb_cellot_simq.csv"); print("[DONE ] comb_cellot_simq", t.shape, n)
PY
