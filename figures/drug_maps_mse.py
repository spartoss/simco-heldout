"""Drug-effect predictions of the drug MAPS on held-out donors, in MSE and between-donor R2, on the selected features.

Design = atlas_heldout learning-curve repeats (5 repeats x 4 test donors, maps fitted on the 15 others;
heldout/score_drugs.py -> work/heldout/drugs/scores/r<r>.csv.gz). Effect E = Q(drug + stim) - Q(the plate's DMSO + stim).
Cases = the selected features of lc/drug_score.py (q50-q90, 5 stims, 9 drugs), taken from work/lc/drug_tab/score/cases.pkl,
which also holds every donor's measured E; the reference "mean effect" E0 of a repeat = mean E of its 15 training donors.
Methods: step:<X> = drug map X applied to the donor's MEASURED stimulated control cells (stim -> drug + stim);
chain:CellOT>CellOT = resting cells -> CellOT stim map -> CellOT drug map (the paper's chain);
feature:CellOT>quantile_map = per-marker quantile drug map applied to the simulated stim features.
TabPFN on the same repeats (lc/drug_tab.py / lc/drug_step.py --design repeat -> work/lc/drug_tab/rep_<stim>/,
steprep_<stim>/): E_A = direct (resting -> drug + stim, minus the predicted control), E_B = two-step (resting -> control
-> effect), E_S / E_S65 = step (the donor's MEASURED stimulated control -> drug + stim); added when present, and every
method is then scored on the cases common to all.
Between-donor R2: each case centred across the repeat's 4 test donors. Writes data/drug_maps_cases.csv.

    cd /scratch/users/bmerwan/atlas_heldout; source env_cpu.sh; python ~/simco_interview_figures/drug_maps_mse.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/scratch/users/bmerwan/atlas_heldout")
from lc.paths import WORK  # noqa: E402

KEY = ["drug", "stim", "celltype", "q", "marker"]
METH = ["step:quantile_map", "step:gauss_ot", "step:CellOT", "chain:CellOT>CellOT", "feature:CellOT>quantile_map"]
OUT = Path(__file__).parent / "data" / "drug_maps_cases.csv"

C = pd.read_pickle(WORK / "drug_tab/score/cases.pkl")[["donor"] + KEY + ["E"]]
EW = C.pivot_table(index=KEY, columns="donor", values="E")               # measured effect, every donor
rows = []
for dz in json.load(open(WORK / "design.json")):
    r, tests, train = dz["repeat"], dz["test"], dz["order"][:15]
    e0 = EW[[d for d in train if d in EW]].mean(1).rename("E0").reset_index()
    S = pd.read_csv(WORK.parent / "heldout/drugs/scores" / f"r{r}.csv.gz")
    S = S[S.donor.isin(tests)].merge(C, on=["donor"] + KEY).merge(e0, on=KEY)
    rows.append(S.assign(repeat=r))
D = pd.concat(rows)
print("cases", len(D), "| corr(score-file truth, LODO-table E) = %.4f" % np.corrcoef(D.true, D.E)[0, 1])
TABP = {"rep": ["E_A", "E_B"], "steprep": ["E_S", "E_S65"]}
for kind, cols in TABP.items():
    fs = [f for st in ["GMCSF", "IFNa", "IL246", "LPS", "TNFa"]
          for f in sorted((WORK / "drug_tab" / f"{kind}_{st}").glob("fold_*.pkl"))]      # not the _smoke runs
    if not fs:
        print(f"[SKIP ] no {kind} outputs"); continue
    R = pd.concat([pd.read_pickle(f) for f in fs]).rename(columns={"fold": "repeat"})
    D = D.merge(R[["repeat", "donor"] + KEY + cols], on=["repeat", "donor"] + KEY, how="inner")
    METH = METH + cols
    print(f"[TABPF] {kind}: {len(fs)} files, {len(D)} common cases")
res = []
for m in ["E0"] + METH:
    per = D.assign(se=(D[m] - D.E) ** 2).groupby(["repeat", "donor"]).se.mean()
    g = ["repeat"] + KEY
    pc = D[m] - D.groupby(g)[m].transform("mean"); tc = D.E - D.groupby(g).E.transform("mean")
    res.append({"method": m, "MSE": per.mean(), "vs_E0_%": np.nan, "between_R2": 1 - ((pc - tc) ** 2).sum() / (tc ** 2).sum(),
                "better_than_E0": np.nan, "_per": per})
ref = res[0]["_per"]
for x in res:
    x["vs_E0_%"] = 100 * (x["MSE"] / res[0]["MSE"] - 1); x["better_than_E0"] = int((x["_per"] < ref).sum())
T = pd.DataFrame([{k: v for k, v in x.items() if k != "_per"} for x in res])
T["n_units"] = len(ref)
T.to_csv(OUT, index=False)
print(T.round(4).to_string(index=False)); print("[DONE ]", OUT)
