# Source before running anything here. Cluster PyTorch module (as atlas_heldout/env.sh); atlas_heldout is on the
# PYTHONPATH for READ-ONLY imports (lc.paths, lc.score, lc.ctx_score, design, caches). Nothing is written there.
deactivate 2>/dev/null || true
unset PYTHONPATH
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/scratch/users/bmerwan/venv-' | paste -sd:)
module purge >/dev/null 2>&1
module load math py-pytorch/2.4.1_py312 py-pandas/2.2.1_py312 >/dev/null 2>&1
export PYTHONPATH=/scratch/users/bmerwan/mfm_heldout:/scratch/users/bmerwan/atlas_heldout${PYTHONPATH:+:$PYTHONPATH}
cd /scratch/users/bmerwan/mfm_heldout
