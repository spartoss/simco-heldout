# Source before running anything in this project. Uses cluster modules only, never the simco venvs.
deactivate 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '/scratch/users/bmerwan/venv-' | paste -sd:)
unset VIRTUAL_ENV PYTHONPATH
module purge >/dev/null 2>&1
module load math py-pytorch/2.4.1_py312 py-pandas/2.2.1_py312 >/dev/null 2>&1
cd /scratch/users/bmerwan/atlas_heldout
