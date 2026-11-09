"""Figure 03: paper CellOT vs closed-form transport maps on new donors, in MSE, on common cases.

Same case construction as lc_rmse.py (atlas_heldout design.json, 4 held-out donors per repeat, effect > FLOOR, 5 stims
without PI / IL33, q50-q90), all 5 repeats, k = 15 training donors (and every k for the curve). Closed-form maps fitted on
the same training donors' cells (atlas_heldout lc/baselines.py -> work/lc/baselines/repeat_<r>.npz): median shift,
per-marker quantile map, Gaussian OT (affine). Between-donor R2: each case centred across the repeat's 4 test donors
(how well the differences between new donors are predicted). Writes data/baselines_cases.csv (per repeat x donor x k
x method MSE) and data/baselines_between.csv.

    cd /scratch/users/bmerwan/atlas_heldout; source env_cpu.sh; python ~/simco_interview_figures/baselines_mse.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/scratch/users/bmerwan/atlas_heldout")
from lc.cond_train_score import load_npz, load_train  # noqa: E402
from lc.paths import DONORS, PAIRS, QS, WORK  # noqa: E402
from lc.score import FLOOR, truth_tables  # noqa: E402

KS, QKEEP, EXCL = [2, 3, 4, 5, 6, 8, 10, 12, 15], [50, 75, 90], {"PI", "IL33"}
CF = ["median_shift", "quantile_map", "gauss_ot"]
OUT = Path(__file__).parent / "data"

T = truth_tables(sorted({c for _, c in PAIRS}), set(DONORS))
qmask = np.isin(np.repeat(QS, 13), QKEEP)
rows, cases = [], []
for dz in json.load(open(WORK / "design.json")):
    r, tests = dz["repeat"], dz["test"]
    maps = [(s, c, k) for s, c in PAIRS for k in KS if s not in EXCL]
    P = {"CellOT_paper": load_train(WORK / f"train/repeat_{r}", maps)}
    P.update(load_npz(WORK / f"baselines/repeat_{r}.npz", CF, maps))
    for (s, c, k) in maps:
        if any((s, c, k) not in D for D in P.values()):
            continue
        for j, d in enumerate(tests):
            if (c, s, d) not in T or (c, "Unstim", d) not in T:
                continue
            u, t = T[(c, "Unstim", d)][0], T[(c, s, d)][0]
            preds = {n: D[(s, c, k)][j] for n, D in P.items()}
            if any(np.isnan(p).all() for p in preds.values()):
                continue
            keep = (np.abs(t - u).ravel() > FLOOR) & qmask
            for n, p in preds.items():
                e = (p - t).ravel()[keep]
                rows.append((n, r, d, k, (e ** 2).sum(), keep.sum()))
                if k == 15:
                    idx = np.flatnonzero(keep)
                    cases.append(pd.DataFrame({"method": n, "repeat": r, "donor": d, "stim": s, "celltype": c, "cell": idx,
                                               "pred": p.ravel()[keep], "true": t.ravel()[keep]}))
D = pd.DataFrame(rows, columns=["method", "repeat", "donor", "k", "sse", "n"])
D = D.groupby(["method", "repeat", "donor", "k"])[["sse", "n"]].sum().reset_index(); D["mse"] = D.sse / D.n
D.to_csv(OUT / "baselines_cases.csv", index=False)
C = pd.concat(cases); g = ["method", "repeat", "stim", "celltype", "cell"]
C["pc"] = C.pred - C.groupby(g).pred.transform("mean"); C["tc"] = C.true - C.groupby(g).true.transform("mean")
B = C.groupby("method").apply(lambda x: 1 - ((x.pc - x.tc) ** 2).sum() / (x.tc ** 2).sum()).rename("between_R2").reset_index()
B.to_csv(OUT / "baselines_between.csv", index=False)
W = D[D.k == 15].pivot_table(index=["repeat", "donor"], columns="method", values="mse")
S = pd.DataFrame({"MSE": W.mean(), "CellOT_better_units": [(W.CellOT_paper < W[m]).sum() for m in W.columns]}).join(B.set_index("method"))
print(S.round(4).to_string()); print("units", len(W))
print(np.sqrt(D.groupby(["method", "k"]).mse.mean()).unstack("k").round(4).to_string())
