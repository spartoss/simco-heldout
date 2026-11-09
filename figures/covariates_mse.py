"""Figures 15-16: conditional CellOT with donor covariates, in MSE, on common cases.

Same case construction as k15_methods_mse.py (atlas_heldout design.json repeats 1-4, 4 held-out donors each, k = 15,
5 stims without PI / IL33, q50-q90, effect > FLOOR); repeat 0 was used to set the covariate penalty and is left out.
Runs: atlas_heldout lc/cond_train.py (--ctx none / pc / pcblood / ft / lda / cluster, --ctx-wd, --arch, --ctx-drop).
Writes data/covariates_cases.csv (per repeat x donor x method MSE) and prints MSE, % vs paired CellOT, donors better,
and between-donor R2 (each case centred across the repeat's 4 test donors).

    cd /scratch/users/bmerwan/atlas_heldout; source env_cpu.sh; python ~/simco_interview_figures/covariates_mse.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/scratch/users/bmerwan/atlas_heldout")
from lc.cond_train_score import load_train  # noqa: E402
from lc.paths import DONORS, PAIRS, QS, WORK  # noqa: E402
from lc.score import FLOOR, truth_tables  # noqa: E402

REPEATS, K, QKEEP, EXCL = [1, 2, 3, 4], 15, [50, 75, 90], {"PI", "IL33"}
CT = "cond_train/repeat_{r}/"
METH = {"CellOT_paper": "train/repeat_{r}", "paired": CT + "none",
        "pc_noreg": CT + "pc", "pc": CT + "pc_wd1", "pcblood": CT + "pcblood_wd1", "ft_immunefm": CT + "ft_pre_wd1",
        "ft_scratch": CT + "ft_rand_wd1", "lda": CT + "lda3_wd1", "cluster": CT + "cluster_K4_stable",
        "arch_bias": CT + "pcb_bias_wd1", "arch_film": CT + "pcb_film_wd1", "arch_trans": CT + "pcb_trans_wd1",
        "arch_affine": CT + "pcb_affine_wd1", "gate_wd10": CT + "pcb_gate_wd10", "gate_drop0.5": CT + "pcb_gate_drop0.5"}
OUT = Path(__file__).parent / "data" / "covariates_cases.csv"

T = truth_tables(sorted({c for _, c in PAIRS}), set(DONORS))
qmask = np.isin(np.repeat(QS, 13), QKEEP)
rows, cases = [], []
for dz in json.load(open(WORK / "design.json")):
    r, tests = dz["repeat"], dz["test"]
    if r not in REPEATS:
        continue
    maps = [(s, c, K) for s, c in PAIRS if s not in EXCL]
    P = {n: load_train(WORK / d.format(r=r), maps) for n, d in METH.items()}
    for m in maps:
        if any(m not in D for D in P.values()):
            continue
        s, c, _ = m
        for j, d in enumerate(tests):
            if (c, s, d) not in T or (c, "Unstim", d) not in T:
                continue
            u, t = T[(c, "Unstim", d)][0], T[(c, s, d)][0]
            preds = {n: D[m][j] for n, D in P.items()}
            if any(np.isnan(p).all() for p in preds.values()):
                continue
            keep = (np.abs(t - u).ravel() > FLOOR) & qmask
            idx = np.flatnonzero(keep)
            for n, p in preds.items():
                e = (p - t).ravel()[keep]
                rows.append((n, r, d, (e ** 2).sum(), keep.sum()))
                cases.append(pd.DataFrame({"method": n, "repeat": r, "donor": d, "stim": s, "celltype": c, "cell": idx,
                                           "pred": p.ravel()[keep], "true": t.ravel()[keep]}))
D = pd.DataFrame(rows, columns=["method", "repeat", "donor", "sse", "n"])
D = D.groupby(["method", "repeat", "donor"])[["sse", "n"]].sum().reset_index(); D["mse"] = D.sse / D.n
C = pd.concat(cases); g = ["method", "repeat", "stim", "celltype", "cell"]
C["pc"] = C.pred - C.groupby(g).pred.transform("mean"); C["tc"] = C.true - C.groupby(g).true.transform("mean")
B = {m: 1 - ((x.pc - x.tc) ** 2).sum() / (x.tc ** 2).sum() for m, x in C.groupby("method")}
D["between_R2"] = D.method.map(B)
D.to_csv(OUT, index=False)
W = D.pivot_table(index=["repeat", "donor"], columns="method", values="mse")
S = pd.DataFrame({"MSE": W.mean(), "vs_paired_%": 100 * (W.mean() / W.mean()["paired"] - 1),
                  "better_than_paired": [(W[m] < W.paired).sum() for m in W.columns], "between_R2": pd.Series(B)})
print(S.round(4).sort_values("MSE").to_string()); print("units", len(W))
