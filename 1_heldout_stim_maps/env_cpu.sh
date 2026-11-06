# Scoring, plots and the notebook. Reuses simco venv-cellot python read-only (pandas, scipy, matplotlib, jupyter);
# nothing is installed into it. Training uses env.sh (cluster PyTorch 2.4 module) instead.
deactivate 2>/dev/null || true
unset PYTHONPATH
module purge >/dev/null 2>&1
module load python/3.9.0 >/dev/null 2>&1
source /scratch/users/bmerwan/venv-cellot/bin/activate
cd /scratch/users/bmerwan/atlas_heldout
