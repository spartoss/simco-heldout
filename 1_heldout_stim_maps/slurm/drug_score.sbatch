#!/bin/bash
# Score the drug arm (lc/drug_score.py) -> work/lc/drug_tab/score/; then the drug notebook if its builder exists.
#SBATCH -p normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -J drug_score
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/drug_score_%j.out
set -euo pipefail
cd /scratch/users/bmerwan/atlas_heldout
bash -lc 'set -euo pipefail; source env.sh; module load viz py-matplotlib/3.8.3_py312 py-scikit-learn/1.5.1_py312 >/dev/null 2>&1; python3 -m lc.drug_score'
if [[ -f lc/build_drug_notebook.py ]]; then
  bash -lc 'source env_cpu.sh >/dev/null 2>&1; export PYTHONPATH=/scratch/users/bmerwan/atlas_heldout
  python3 -m lc.build_drug_notebook && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/drug_heldout.ipynb'
fi
