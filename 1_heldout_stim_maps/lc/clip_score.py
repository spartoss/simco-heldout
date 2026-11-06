"""Test-input clipping at several levels (2026-10-13): paired CellOT k15 maps of cond_train/repeat_<r>/clip, test cells
clamped per marker to the map's training-Unstim [P, 100-P] percentiles (cond_train --clip-test P, resumed at the last
step, i.e. the same trained maps, predictions only). Readouts as lc.dnorm_slope_score, plus the between-donor slope by
within-donor widening bin (gam from lc.slope_calib.gammas on all donors).
    python -m lc.clip_score
"""
import json

import numpy as np
import pandas as pd

import lc.dnorm_slope_score as dss
from heldout.common import WORK as HW
from lc.cond_train_score import load_train
from lc.dnorm_slope import OUT as DSO
from lc.paths import MARKERS, PAIRS, QS, WORK
from lc.score import FLOOR

LEVELS = ["clip", "clip2.5", "clip5", "clip10", "clip25"]
OUT = WORK / "clip_levels"
F = ["stim", "celltype", "q", "marker"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D = pd.read_pickle(DSO / "cases.pkl")[["repeat", "stim", "celltype", "donor", "q", "marker", "u", "t", "mu_s", "shift", "paired", "ens"]]
    parts = []
    for dz in json.load(open(WORK / "design.json")):
        r = dz["repeat"]; maps = [(s, c, 15) for s, c in PAIRS]
        P = {n: load_train(WORK / f"cond_train/repeat_{r}/{n}", maps) for n in LEVELS}
        rows = []
        for (s, c, k) in maps:
            for j, d in enumerate(dz["test"]):
                row = {"repeat": r, "stim": s, "celltype": c, "donor": d, "q": np.repeat(QS, len(MARKERS)), "marker": np.tile(MARKERS, len(QS))}
                ok = True
                for n, X in P.items():
                    if (s, c, k) not in X: ok = False; break
                    row[n] = np.asarray(X[(s, c, k)][j], float).ravel()
                if ok: rows.append(pd.DataFrame(row))
        parts.append(pd.concat(rows))
    D = D.merge(pd.concat(parts), on=["repeat", "stim", "celltype", "donor", "q", "marker"], how="inner")
    D = D.dropna(subset=LEVELS)
    D["eff"] = (D.t - D.u).abs(); D = D[D.eff > FLOOR].copy()
    inc = pd.read_csv(HW / "features/inclusion.csv")
    L = inc.melt(id_vars=["stim", "celltype", "marker"], value_vars=[f"shift_q{q}" for q in (10, 25, 50, 75, 90)],
                 var_name="q", value_name="rule_eff"); L["q"] = L.q.str[7:].astype(int)
    D = D.merge(L, on=["stim", "celltype", "marker", "q"], how="left")
    rule = D.q.isin([50, 75, 90]) & (D.rule_eff > 0.1)
    dss.ARMS = ["shift", "paired", "ens"] + LEVELS
    pd.set_option("display.width", 250)
    for name, X in [("all", D), ("rule", D[rule])]:
        R = dss.readout(X, refs=("paired", "ens")); R.round(4).to_csv(OUT / f"score_{name}.csv")
        print(f"\n=== {name}: {len(X)} cases"); print(R.round(4).to_string())
    # slopes by gamma bin (all features)
    X = D.copy(); G = ["repeat"] + F; X["n"] = X.groupby(G).t.transform("size"); X = X[X.n >= 3]; g = [X[c] for c in G]
    X["uc"] = X.u - X.u.groupby(g).transform("mean")
    A = ["t", "paired"] + LEVELS
    for c in A: X["c_" + c] = X[c] - X[c].groupby(g).transform("mean")
    S = X.groupby(F).apply(lambda d: pd.Series({c: (d["c_" + c] * d.uc).sum() / (d.uc ** 2).sum() for c in A})).dropna()
    gm = pd.read_pickle(OUT / "gam.pkl") if (OUT / "gam.pkl").exists() else None
    if gm is None:
        from lc.slope_calib import gammas
        gm = gammas(sorted({d for dz in json.load(open(WORK / "design.json")) for d in dz["order"] + dz["test"]}), k=10 ** 6).set_index(F).gam
        gm.to_pickle(OUT / "gam.pkl")
    S = S.join(gm, how="inner")
    S["bin"] = pd.cut(S.gam, [0, 0.8, 1.2, 1.5, 2, 3, np.inf])
    print("\n=== median between-donor slope by within-donor widening (gam) bin")
    print(S.groupby("bin", observed=True).agg(n=("t", "size"), **{c: (c, "median") for c in A}).round(2).to_string())
    print("share stretched (slope > true + 0.5):", {c: round(((S[c] - S.t) > 0.5).mean(), 3) for c in A[1:]})
    S.to_pickle(OUT / "slopes.pkl")


if __name__ == "__main__":
    main()
