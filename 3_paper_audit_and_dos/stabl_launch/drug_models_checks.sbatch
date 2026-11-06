#!/bin/bash
# After the build: checks a-d for each DOS model (same CellOT drug maps, different features).
#SBATCH -p normal
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -J drug_checks
#SBATCH -o /scratch/users/bmerwan/simco_work/onset_test/drug_models/logs/checks_%j.out
#SBATCH -e /scratch/users/bmerwan/simco_work/onset_test/drug_models/logs/checks_%j.err
set -uo pipefail
module load python/3.9.0
source /scratch/users/bmerwan/venv-stabl/bin/activate
cd /scratch/users/bmerwan/simco
O=/scratch/users/bmerwan/simco_work/onset_test; M=$O/drug_models; S=$O/paper_protocol/stabl_n100
export OMP_NUM_THREADS=2
run () {  # name features selections drug-table
    [[ -s $M/checks_$1/summary_by_drug.csv ]] && { echo "[SKIP ] $1"; return; }
    python stabl_launch/drug_arm_checks.py --features "$2" --selections "$3" --drug-table "$4" \
        --out $M/checks_$1 --no-validate > $M/logs/checks_$1.log 2>&1; echo "[DONE ] $1 rc=$?"
}
run cellot_rebuilt $O/ina_13OG_final_long_allstims_filtered.csv "$O/foldfeats_GSS/Training CV/Selected Features STABL ALasso.csv" "$M/tables/{drug}_cellot.csv" &
run gaussot        $O/gaussot_13OG_both.csv        "$S/gaussot_both/Training CV/Selected Features STABL ALasso.csv" "$M/tables/{drug}_gaussot.csv" &
run meas_qtl       $O/night2/distrib_all_v2.csv    "$S/d_qtl/Training CV/Selected Features STABL ALasso.csv"        "$M/tables/{drug}_distrib.csv" &
run meas_freqmed   $O/night2/distrib_all_v2.csv    "$S/d_freqmed/Training CV/Selected Features STABL ALasso.csv"    "$M/tables/{drug}_distrib.csv" &
wait
python stabl_launch/drug_models_compare.py
