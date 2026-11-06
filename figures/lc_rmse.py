"""Learning curve in RMSE for figure 04: paper CellOT vs covariate CellOTs, on common cases.

Same case construction as atlas_heldout lc/ctx_score.py (design.json repeats, 4 held-out donors each, effect > FLOOR),
restricted to repeats 1-4 (repeat 0 was used to set the covariate regularisation), k = 3 5 8 15 for all methods
(paper CellOT and the shift also at k = 1 2 4 6 10 12), the 5 stims of the
other figures (PI, IL33 out) and q50-q90. RMSE = sqrt(mean over (repeat, test donor) of the MSE over its cases).
Writes data/lc_rmse_cases.csv (per repeat x donor x k x method MSE), read by make_figures.py.

    cd /scratch/users/bmerwan/atlas_heldout; source env_cpu.sh; python ~/simco_interview_figures/lc_rmse.py
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

REPEATS, KS, QKEEP, EXCL = [1, 2, 3, 4], [3, 5, 8, 15], [50, 75, 90], {"PI", "IL33"}
ALLK = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]   # paper CellOT is trained at every k; the covariate runs only at KS
METH = {"CellOT_paper": "train/repeat_{r}", "CellOT_pcblood": "cond_train/repeat_{r}/pcblood_wd1",
        "CellOT_cluster": "cond_train/repeat_{r}/cluster_K4_stable"}
OUT = Path(__file__).parent / "data" / "lc_rmse_cases.csv"

T = truth_tables(sorted({c for _, c in PAIRS}), set(DONORS))
qmask = np.isin(np.repeat(QS, 13), QKEEP)
rows = []
for dz in json.load(open(WORK / "design.json")):
    r, tests = dz["repeat"], dz["test"]
    if r not in REPEATS:
        continue
    maps = [(s, c, k) for s, c in PAIRS for k in ALLK if s not in EXCL]
    P = {n: load_train(WORK / d.format(r=r), maps) for n, d in METH.items()}
    for (s, c, k) in maps:
        need = METH if k in KS else ["CellOT_paper"]
        if any((s, c, k) not in P[n] for n in need):
            continue
        trd = [d for d in dz["order"][:k] if (c, s, d) in T and (c, "Unstim", d) in T]
        if len(trd) < 2:
            continue
        mu_u = np.mean([T[(c, "Unstim", d)][0] for d in trd], 0); mu_s = np.mean([T[(c, s, d)][0] for d in trd], 0)
        for j, d in enumerate(tests):
            if (c, s, d) not in T or (c, "Unstim", d) not in T:
                continue
            u, t = T[(c, "Unstim", d)][0], T[(c, s, d)][0]
            preds = {"shift": mu_s + (u - mu_u)}
            preds.update({n: P[n][(s, c, k)][j] for n in need})
            if any(np.isnan(p).all() for p in preds.values()):
                continue
            keep = (np.abs(t - u).ravel() > FLOOR) & qmask
            for n, p in preds.items():
                se = ((p - t).ravel() ** 2)[keep]
                rows.append((n, r, d, k, se.sum(), keep.sum()))
D = pd.DataFrame(rows, columns=["method", "repeat", "donor", "k", "sse", "n"])
D = D.groupby(["method", "repeat", "donor", "k"])[["sse", "n"]].sum().reset_index()
D["mse"] = D.sse / D.n
D.to_csv(OUT, index=False)
C = D.groupby(["method", "k"]).mse.mean().pipe(np.sqrt).unstack("k")
print(C.round(4).to_string()); print("[DONE ]", OUT)
