"""MSE of the methods in figure 11 ("what did not work"), on common cases, k = 15 training donors.

Same case construction as lc_rmse.py / atlas_heldout lc/ctx_score.py: repeats 1-4 of design.json (4 held-out donors each),
5 stims (PI, IL33 out), q50-q90, effect > FLOOR; only cases predicted by every method. Locations as in lc/model_zoo.py
(dnorm = trained on donor-normalised cells: the test donor's resting-median offset is added back).
Writes data/k15_methods_cases.csv (per repeat x donor x method MSE), read by make_figures.py.

    cd /scratch/users/bmerwan/atlas_heldout; source env_cpu.sh; python ~/simco_interview_figures/k15_methods_mse.py
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
MFM = "/scratch/users/bmerwan/mfm_heldout/work/runs"
METH = {"CellOT_paper": "train/repeat_{r}", "CellOT_paired": "cond_train/repeat_{r}/none",
        "ctx_pc_noreg": "cond_train/repeat_{r}/pc", "ctx_ft_immunefm": "cond_train/repeat_{r}/ft_pre_wd1",
        "ctx_ft_scratch": "cond_train/repeat_{r}/ft_rand_wd1", "fm": MFM + "/fm_sel/repeat_{r}",
        "mfm": MFM + "/mfm_sel/repeat_{r}", "condot_global": "condot/global/repeat_{r}",
        "dnorm": "cond_train/repeat_{r}/dnorm"}
DNORM = {"dnorm"}
OUT = Path(__file__).parent / "data" / "k15_methods_cases.csv"

T = truth_tables(sorted({c for _, c in PAIRS}), set(DONORS))
off = np.load(WORK / "cache_dnorm/offsets.npz")
qmask = np.isin(np.repeat(QS, 13), QKEEP)
rows = []
for dz in json.load(open(WORK / "design.json")):
    r, tests = dz["repeat"], dz["test"]
    if r not in REPEATS:
        continue
    maps = [(s, c, K) for s, c in PAIRS if s not in EXCL]
    P = {n: load_train(WORK / d.format(r=r), maps) for n, d in METH.items()}
    for (s, c, k) in maps:
        if any((s, c, k) not in D for D in P.values()):
            continue
        trd = [d for d in dz["order"][:k] if (c, s, d) in T and (c, "Unstim", d) in T]
        mu_u = np.mean([T[(c, "Unstim", d)][0] for d in trd], 0); mu_s = np.mean([T[(c, s, d)][0] for d in trd], 0)
        for j, d in enumerate(tests):
            if (c, s, d) not in T or (c, "Unstim", d) not in T:
                continue
            u, t = T[(c, "Unstim", d)][0], T[(c, s, d)][0]
            preds = {"shift": mu_s + (u - mu_u)}
            preds.update({n: D[(s, c, k)][j] + (off[f"{c}|{d}"][None, :] if n in DNORM else 0) for n, D in P.items()})
            if any(np.isnan(p).all() for p in preds.values()):
                continue
            keep = (np.abs(t - u).ravel() > FLOOR) & qmask
            for n, p in preds.items():
                rows.append((n, r, d, ((p - t).ravel() ** 2)[keep].sum(), keep.sum()))
D = pd.DataFrame(rows, columns=["method", "repeat", "donor", "sse", "n"])
D = D.groupby(["method", "repeat", "donor"])[["sse", "n"]].sum().reset_index()
D["mse"] = D.sse / D.n
D.to_csv(OUT, index=False)
W = D.pivot_table(index=["repeat", "donor"], columns="method", values="mse")
S = pd.DataFrame({"MSE": W.mean(), "vs_paired_%": 100 * (W.mean() / W.mean()["CellOT_paired"] - 1),
                  "better_than_paired": (W.lt(W["CellOT_paired"], axis=0)).sum(), "n_units": len(W)})
print(S.round(4).sort_values("MSE").to_string()); print("[DONE ]", OUT)
