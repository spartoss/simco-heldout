# Scoring (lc.ctx_score needs scipy / matplotlib): simco venv-cellot read-only, as atlas_heldout/env_cpu.sh.
deactivate 2>/dev/null || true
unset PYTHONPATH
module purge >/dev/null 2>&1
module load python/3.9.0 >/dev/null 2>&1
source /scratch/users/bmerwan/venv-cellot/bin/activate
export PYTHONPATH=/scratch/users/bmerwan/mfm_heldout:/scratch/users/bmerwan/atlas_heldout${PYTHONPATH:+:$PYTHONPATH}
cd /scratch/users/bmerwan/mfm_heldout
