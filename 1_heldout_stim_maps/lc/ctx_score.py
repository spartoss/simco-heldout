"""Score any set of lc/cond_train.py runs on the learning-curve design, on their COMMON cases (as lc/cond_train_score.py).

Methods are name=dir pairs, dir relative to WORK with {r} for the repeat; identity and shift (feature level) are added.
name=dir:dnorm = trained on the donor-normalised cache (lc/dnorm_cache.py): the test donor's Unstim median offset is
added back to the predicted quantiles. --qs restricts every readout to these quantiles.
Only (repeat, stim, celltype, k, donor, q, marker) cases present for every method and with effect > FLOOR are kept.
Readouts per (method, k): skill (median per repeat, mean over repeats), dev_R2, between_R2 (centred across the
repeat's test donors); paired vs each --vs method: % of cases closer, Wilcoxon over (repeat, map, donor) median errors.
    python -m lc.ctx_score --methods none=cond_train/repeat_{r}/none ft=ctx_sweep/... --repeats 0 --vs none shift \
        --out WORK/ctx_score/<tag>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from lc.cond_train_score import load_train
from lc.paths import DONORS, PAIRS, QS, WORK
from lc.score import FLOOR, truth_tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--repeats", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--ks", nargs="*", type=int, default=[3, 5, 8, 15])
    ap.add_argument("--vs", nargs="*", default=["shift"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--qs", nargs="*", type=int, default=list(QS))
    ap.add_argument("--exclude-stims", nargs="*", default=[], help="leave these stims out of every readout")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meth = dict(m.split("=", 1) for m in a.methods)
    dn = {n for n, d in meth.items() if d.endswith(":dnorm")}
    meth = {n: d.removesuffix(":dnorm") for n, d in meth.items()}
    off = np.load(WORK / "cache_dnorm/offsets.npz") if dn else None
    T = truth_tables(sorted({c for _, c in PAIRS}), set(DONORS))
    rows = []
    for dz in json.load(open(WORK / "design.json")):
        r, tests = dz["repeat"], dz["test"]
        if r not in a.repeats: continue
        maps = [(s, c, k) for s, c in PAIRS for k in a.ks if s not in a.exclude_stims]
        P = {n: load_train(WORK / d.format(r=r), maps) for n, d in meth.items()}
        for (s, c, k) in maps:
            if any((s, c, k) not in D for D in P.values()): continue
            trd = [d for d in dz["order"][:k] if (c, s, d) in T and (c, "Unstim", d) in T]
            if len(trd) < 2: continue
            mu_u = np.mean([T[(c, "Unstim", d)][0] for d in trd], 0); mu_s = np.mean([T[(c, s, d)][0] for d in trd], 0)
            for j, d in enumerate(tests):
                if (c, s, d) not in T or (c, "Unstim", d) not in T: continue
                u, t = T[(c, "Unstim", d)][0], T[(c, s, d)][0]
                preds = {"identity": u, "shift": mu_s + (u - mu_u)}
                preds.update({n: D[(s, c, k)][j] + (off[f"{c}|{d}"][None, :] if n in dn else 0)
                              for n, D in P.items() if not np.isnan(D[(s, c, k)][j]).all()})
                for n, p in preds.items():
                    rows.append(pd.DataFrame({"method": n, "repeat": r, "stim": s, "celltype": c, "k": k, "donor": d,
                                              "q": np.repeat(QS, 13), "marker": np.tile(np.arange(13), len(QS)),
                                              "pred": p.ravel(), "true": t.ravel(), "mu_s": mu_s.ravel(),
                                              "eff": np.abs(t - u).ravel()}))
    D = pd.concat(rows); D = D[(D.eff > FLOOR) & D.q.isin(a.qs)]
    keep = D.groupby(["repeat", "stim", "celltype", "k", "donor", "q", "marker"]).method.transform("nunique")
    D = D[keep == len(meth) + 2]
    D["err"] = (D.pred - D.true).abs(); D["skill"] = 1 - D.err / D.eff
    grp = ["repeat", "stim", "celltype", "k", "q", "marker", "method"]
    D["pc"] = D.pred - D.groupby(grp).pred.transform("mean"); D["tc"] = D.true - D.groupby(grp).true.transform("mean")
    D["ntest"] = D.groupby(grp).pred.transform("size")
    res = []
    for (m, k), g in D.groupby(["method", "k"]):
        gb = g[g.ntest >= 3]
        res.append({"method": m, "k": k, "skill": g.groupby("repeat").skill.median().mean(),
                    "dev_R2": 1 - ((g.pred - g.true) ** 2).sum() / ((g.true - g.mu_s) ** 2).sum(),
                    "between_R2": 1 - ((gb.pc - gb.tc) ** 2).sum() / (gb.tc ** 2).sum() if len(gb) else np.nan,
                    "n_cases": len(g)})
    C = pd.DataFrame(res); C.round(4).to_csv(out / "curve.csv", index=False)
    W = D.pivot_table(index=["repeat", "stim", "celltype", "k", "donor", "q", "marker"], columns="method", values="err")
    pr = []
    for m in meth:
        for b in a.vs:
            if b == m: continue
            for k, g in W.groupby(level="k"):
                gm = g.groupby(level=["repeat", "stim", "celltype", "donor"]).median()
                pr.append({"a": m, "b": b, "k": k, "pct_a_closer": 100 * (g[m] < g[b]).mean(),
                           "p_wilcoxon": wilcoxon(gm[m], gm[b]).pvalue if len(gm) > 5 else np.nan})
    Pd = pd.DataFrame(pr); Pd.round(4).to_csv(out / "paired.csv", index=False)
    pd.set_option("display.width", 200)
    for col in ("skill", "dev_R2", "between_R2"):
        print(f"--- {col}"); print(C.pivot(index="method", columns="k", values=col).round(4).to_string())
    print("--- paired"); print(Pd.round(4).to_string(index=False))
    print(f"[DONE ] {len(D) // (len(meth) + 2)} cases -> {out}")


if __name__ == "__main__":
    main()
