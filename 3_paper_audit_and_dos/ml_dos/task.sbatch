#!/bin/bash
#SBATCH -J ml_dos
#SBATCH -p normal
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 16:00:00
#SBATCH -o /scratch/users/bmerwan/simco_work/onset_test/ml_dos/logs/task_%A_%a.out
# One line of tasks.txt: "<runner> <arm> <model>". Resumable (one file per split).
set -euo pipefail
OUT=/scratch/users/bmerwan/simco_work/onset_test/ml_dos
read -r runner arm model < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$OUT/tasks.txt")
module load python/3.9.0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$OUT/code"
case $runner in
  run_cv)       source /scratch/users/bmerwan/venv-stabl/bin/activate
                python run_cv.py --arm "$arm" --model "$model" --n-jobs "$SLURM_CPUS_PER_TASK" ;;
  run_dnn)      source /scratch/users/bmerwan/venv-cellot/bin/activate
                python run_dnn.py --arm "$arm" --model "$model" --n-jobs "$SLURM_CPUS_PER_TASK" ;;
  run_deepsets) source /scratch/users/bmerwan/venv-cellot/bin/activate
                [ -f "$OUT/cells.npz" ] || python run_deepsets.py prepare --n-jobs 8
                python run_deepsets.py fit --n-jobs "$SLURM_CPUS_PER_TASK" ;;
esac
