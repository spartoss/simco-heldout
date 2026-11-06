"""How correlated are the errors of CellOT and the (corrected) shift? Why averaging them helps.

Cases as lc.ctx_score: (repeat, stim, celltype, k, test donor, q, marker) with effect |S - U| > FLOOR, present for every
method. Error e = pred - true. Per pair (a, b): Pearson corr(e_a, e_b), RMSE of each, RMSE of the 0.5/0.5 average, and
w* = the in-sample best weight on a in w * a + (1 - w) * b (descriptive only: fitted on the cases it is reported on).
Breakdowns: by k, and at k = 15 by stim, by q, by marker.
    python -m lc.err_corr --repeats 0 1 2 --out WORK/tab/err_corr
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from lc.cond_train_score import load_train
from lc.paths import DONORS, MARKERS, PAIRS, QS, WORK
from lc.score import FLOOR, truth_tables

METHODS = {"CellOT": "cond_train/repeat_{r}/none", "tabpfn_L2": "tab/pred/tabpfn_L2_l1/repeat_{r}",
           "lgbm_L2": "tab/pred/lgbm_L2_l1/repeat_{r}"}
PAIRS_CMP = [("CellOT", "shift"), ("CellOT", "tabpfn_L2"), ("CellOT", "lgbm_L2"), ("tabpfn_L2", "lgbm_L2"),
             ("shift", "tabpfn_L2")]


def stats(g, a, b):
    ea, eb = g[a].to_numpy(), g[b].to_numpy(); d = ea - eb
    w = float(np.clip(-(eb * d).sum() / (d ** 2).sum(), 0, 1)) if (d ** 2).sum() > 0 else np.nan
    rm = lambda e: np.sqrt((e ** 2).mean())
    return pd.Series({"corr": np.corrcoef(ea, eb)[0, 1], f"rmse_a": rm(ea), f"rmse_b": rm(eb),
                      "rmse_avg": rm(0.5 * (ea + eb)), "w_a_best": w, "n": len(g)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--ks", nargs="*", type=int, default=[3, 5, 8, 15])
    ap.add_argument("--out", default=str(WORK / "tab" / "err_corr"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    T = truth_tables(sorted({c for _, c in PAIRS}), set(DONORS))
    rows = []
    for dz in json.load(open(WORK / "design.json")):
        r, tests = dz["repeat"], dz["test"]
        if r not in a.repeats: continue
        maps = [(s, c, k) for s, c in PAIRS for k in a.ks]
        P = {n: load_train(WORK / d.format(r=r), maps) for n, d in METHODS.items()}
        for (s, c, k) in maps:
            if any((s, c, k) not in D for D in P.values()): continue
            trd = [d for d in dz["order"][:k] if (c, s, d) in T and (c, "Unstim", d) in T]
            if len(trd) < 2: continue
            mu_u = np.mean([T[(c, "Unstim", d)][0] for d in trd], 0); mu_s = np.mean([T[(c, s, d)][0] for d in trd], 0)
            for j, d in enumerate(tests):
                if (c, s, d) not in T or (c, "Unstim", d) not in T: continue
                u, t = T[(c, "Unstim", d)][0], T[(c, s, d)][0]
                pr = {"shift": mu_s + u - mu_u}; pr.update({n: D[(s, c, k)][j] for n, D in P.items()})
                if any(np.isnan(p).all() for p in pr.values()): continue
                f = {"repeat": r, "stim": s, "celltype": c, "k": k, "donor": d, "q": np.repeat(QS, 13),
                     "marker": np.tile(MARKERS, len(QS)), "eff": np.abs(t - u).ravel()}
                f.update({n: (p - t).ravel() for n, p in pr.items()})
                rows.append(pd.DataFrame(f))
    D = pd.concat(rows); D = D[D.eff > FLOOR].dropna()
    print(f"[CASES] {len(D)}", flush=True)
    res = {}
    for by, sub in [("k", D), ("stim", D[D.k == 15]), ("q", D[D.k == 15]), ("marker", D[D.k == 15])]:
        tabs = []
        for x, y in PAIRS_CMP:
            S = sub.groupby(by).apply(lambda g: stats(g, x, y), include_groups=False).reset_index()
            S.insert(0, "pair", f"{x} vs {y}"); tabs.append(S)
        res[by] = pd.concat(tabs); res[by].round(4).to_csv(out / f"by_{by}.csv", index=False)
    pd.set_option("display.width", 220)
    for by, S in res.items():
        print(f"--- corr of errors by {by}" + ("" if by == "k" else " (k = 15)"))
        print(S.pivot(index="pair", columns=by, values="corr").round(3).to_string())
    print("--- k = 15 detail"); print(res["k"][res["k"].k == 15].round(4).to_string(index=False))
    print("--- best weight on the first method, by stim (k = 15, in-sample)")
    print(res["stim"].pivot(index="pair", columns="stim", values="w_a_best").round(2).to_string())
    print(f"[DONE ] -> {out}")


if __name__ == "__main__":
    main()
