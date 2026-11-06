"""Score conditional CellOT (lc/cond_train.py) against its control and the baselines; and the donor-ID oracle.

Learning curve (5 repeats, k = 3 5 8 15, held-out donors), per (stim, celltype, k, test donor, quantile, marker):
methods CellOT_pc (donor context), CellOT_none (same code, no context), CellOT_lc (original learning-curve run),
gauss_ot_pool / gauss_ot_cond (cond_baselines/med_z), shift (feature level: train stim mean + test Unstim deviation
from the train Unstim mean; no cells, no test values), identity. Three readouts, effects > FLOOR:
- skill       median of 1 - err / effect (per repeat, mean over repeats), as lc/score.py;
- dev_R2      1 - sum (pred - true)^2 / sum (true - train stim mean)^2: the test donor's deviation from the TRAINING
              donors' mean (only training statistics used);
- between_R2  predictions and truth each centred across the repeat's test donors (offset-free, as lc/cond_between.py).
Paired CellOT_pc vs CellOT_none: % of cases closer, Wilcoxon over (repeat, map, donor) median errors.
Oracle (cond_train/oracle/{none,onehot}): all 19 donors trained, scored on each donor's own val side (last 20 % of its
cached cells, Unstim and stim); dev_R2 / between_R2 across the 19 donors; shift there = mean over donors + deviation.
Outputs WORK/cond_train_summary/: curve.csv, paired.csv, oracle.csv.
    python -m lc.cond_train_score
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from lc.paths import CACHE, DONORS, PAIRS, QS, VAL_FRAC, WORK
from lc.score import FLOOR, MIN_CELLS, truth_tables

KS = [3, 5, 8, 15]
OUT = WORK / "cond_train_summary"


def load_train(d, maps):
    cm = pd.read_csv(d / "maps.csv"); P = np.load(d / "pred_q.npy", mmap_mode="r")
    pos = {(s, c, k): i for i, (s, c, k) in enumerate(zip(cm.stim, cm.celltype, cm.k))}
    last = P[P.shape[0] - 1]
    return {m: np.asarray(last[pos[m]]) for m in maps if m in pos}


def load_npz(f, names, maps):
    z = np.load(f, allow_pickle=True)
    pos = {tuple(m)[:3]: i for i, m in enumerate(z["maps"])}
    return {n: {m: z[n][pos[m]] for m in maps if m in pos} for n in names}


def curve():
    rows = []
    for dz in json.load(open(WORK / "design.json")):
        r, tests = dz["repeat"], dz["test"]
        maps = [(s, c, k) for s, c in PAIRS for k in KS]
        P = {"CellOT_pc": load_train(WORK / f"cond_train/repeat_{r}/pc", maps),
             "CellOT_none": load_train(WORK / f"cond_train/repeat_{r}/none", maps),
             "CellOT_lc": load_train(WORK / f"train/repeat_{r}", maps)}
        P.update(load_npz(WORK / f"cond_baselines/med_z/repeat_{r}.npz", ["gauss_ot_pool", "gauss_ot_cond"], maps))
        T = truth_tables(sorted({c for _, c in PAIRS}), set(DONORS))
        for (s, c, k) in maps:
            trd = [d for d in dz["order"][:k] if (c, s, d) in T and (c, "Unstim", d) in T]
            if len(trd) < 2: continue
            mu_u = np.mean([T[(c, "Unstim", d)][0] for d in trd], 0); mu_s = np.mean([T[(c, s, d)][0] for d in trd], 0)
            for j, d in enumerate(tests):
                if (c, s, d) not in T or (c, "Unstim", d) not in T: continue
                u, t = T[(c, "Unstim", d)][0], T[(c, s, d)][0]
                preds = {"identity": u, "shift": mu_s + (u - mu_u)}
                for name, D in P.items():
                    if (s, c, k) in D and not np.isnan(D[(s, c, k)][j]).all(): preds[name] = D[(s, c, k)][j]
                for name, p in preds.items():
                    rows.append(pd.DataFrame({"method": name, "repeat": r, "stim": s, "celltype": c, "k": k, "donor": d,
                                              "q": np.repeat(QS, 13), "marker": np.tile(np.arange(13), len(QS)),
                                              "pred": p.ravel(), "true": t.ravel(), "mu_s": mu_s.ravel(),
                                              "eff": np.abs(t - u).ravel()}))
        print(f"[LOAD ] repeat {r}", flush=True)
    D = pd.concat(rows); D = D[D.eff > FLOOR]
    keep = D.groupby(["repeat", "stim", "celltype", "k", "donor", "q", "marker"]).method.transform("nunique")
    D = D[keep == keep.max()]                       # identical cases for every method
    D["err"] = (D.pred - D.true).abs(); D["skill"] = 1 - D.err / D.eff
    grp = ["repeat", "stim", "celltype", "k", "q", "marker", "method"]
    D["pc"] = D.pred - D.groupby(grp).pred.transform("mean"); D["tc"] = D.true - D.groupby(grp).true.transform("mean")
    D["ntest"] = D.groupby(grp).pred.transform("size")
    res = []
    for (m, k), g in D.groupby(["method", "k"]):
        gb = g[g.ntest >= 3]
        res.append({"method": m, "k": k,
                    "skill": g.groupby("repeat").skill.median().mean(),
                    "dev_R2": 1 - ((g.pred - g.true) ** 2).sum() / ((g.true - g.mu_s) ** 2).sum(),
                    "between_R2": 1 - ((gb.pc - gb.tc) ** 2).sum() / (gb.tc ** 2).sum(),
                    "n_cases": len(g)})
    C = pd.DataFrame(res); C.round(4).to_csv(OUT / "curve.csv", index=False)
    W = D.pivot_table(index=["repeat", "stim", "celltype", "k", "donor", "q", "marker"], columns="method", values="err")
    pr = []
    for a, b in [("CellOT_pc", "CellOT_none"), ("CellOT_pc", "shift"), ("CellOT_none", "shift"), ("CellOT_none", "CellOT_lc")]:
        for k, g in W.groupby(level="k"):
            gm = g.groupby(level=["repeat", "stim", "celltype", "donor"]).median()
            pr.append({"a": a, "b": b, "k": k, "pct_a_closer": 100 * (g[a] < g[b]).mean(),
                       "median_err_a": g[a].median(), "median_err_b": g[b].median(),
                       "p_wilcoxon": wilcoxon(gm[a], gm[b]).pvalue})
    Pd = pd.DataFrame(pr); Pd.round(4).to_csv(OUT / "paired.csv", index=False)
    by = D[D.k == 15].groupby(["method", "stim"]).apply(
        lambda g: 1 - ((g.pc - g.tc) ** 2).sum() / (g.tc ** 2).sum(), include_groups=False).unstack("stim")
    by.round(3).to_csv(OUT / "between_R2_by_stim_k15.csv")
    return C, Pd, by


def oracle():
    """Truth = val side (last 20 %) of each donor's cached cells, as the oracle's test cells."""
    rows = []
    cm = pd.read_csv(WORK / "cond_train/oracle/none/maps.csv")
    P = {c: np.load(WORK / f"cond_train/oracle/{c}/pred_q.npy", mmap_mode="r") for c in ("none", "onehot")}
    for i, (s, ct) in enumerate(zip(cm.stim, cm.celltype)):
        z = np.load(CACHE / f"{ct}.npz")
        val = lambda key: (lambda x: x[max(1, int(round(len(x) * (1 - VAL_FRAC)))):])(z[key]) if key in z.files else None
        U, S, ok = [], [], []
        for j, d in enumerate(DONORS):
            u, t = val(f"Unstim|{d}"), val(f"{s}|{d}")
            if u is None or t is None or len(u) < MIN_CELLS or len(t) < MIN_CELLS: continue
            U.append(np.percentile(u, QS, 0)); S.append(np.percentile(t, QS, 0)); ok.append(j)
        if len(ok) < 5: continue
        U, S = np.stack(U), np.stack(S)
        preds = {"identity": U, "shift": S.mean(0) + (U - U.mean(0)),
                 "CellOT_none": np.asarray(P["none"][-1, i, ok]), "CellOT_onehot": np.asarray(P["onehot"][-1, i, ok])}
        eff = np.abs(S - U)
        for m, p in preds.items():
            rows.append(pd.DataFrame({"method": m, "stim": s, "celltype": ct, "pred": p.ravel(), "true": S.ravel(),
                                      "eff": eff.ravel(), "pc": (p - p.mean(0)).ravel(), "tc": (S - S.mean(0)).ravel(),
                                      "mu_s": np.broadcast_to(S.mean(0), S.shape).ravel()}))
    D = pd.concat(rows); D = D[D.eff > FLOOR]
    D["skill"] = 1 - (D.pred - D.true).abs() / D.eff
    O = D.groupby("method").apply(lambda g: pd.Series({
        "skill": g.skill.median(), "between_R2": 1 - ((g.pc - g.tc) ** 2).sum() / (g.tc ** 2).sum(),
        "dev_R2": 1 - ((g.pred - g.true) ** 2).sum() / ((g.true - g.mu_s) ** 2).sum(), "n": len(g)}), include_groups=False)
    O.round(4).to_csv(OUT / "oracle.csv")
    return O


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    C, Pd, by = curve()
    for col in ("skill", "dev_R2", "between_R2"):
        print(f"--- {col}"); print(C.pivot(index="method", columns="k", values=col).round(3).to_string())
    print("--- paired"); print(Pd.round(4).to_string(index=False))
    print("--- between_R2 by stim, k = 15"); print(by.round(3).to_string())
    print("--- oracle"); print(oracle().round(3).to_string())
    print(f"[DONE ] -> {OUT}")


if __name__ == "__main__":
    main()
