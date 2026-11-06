#!/bin/bash
# OOL predictions -> Fig 5b tables tab_both / mix_both (lc/ool_stabl_tables.py) in simco_work/onset_test/paper_protocol/tables
#SBATCH -p normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -J ool_tables
#SBATCH -o /scratch/users/bmerwan/atlas_heldout/work/lc/logs/ool_tables_%j.out
set -euo pipefail
cd /scratch/users/bmerwan/atlas_heldout
source env.sh
module load viz py-matplotlib/3.8.3_py312 py-scikit-learn/1.5.1_py312 >/dev/null 2>&1
python3 -m lc.ool_stabl_tables "$@"
