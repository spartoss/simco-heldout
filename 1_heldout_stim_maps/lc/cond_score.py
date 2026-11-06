"""Score the donor-conditioned maps (lc/cond_baselines.py) against the pooled maps on the learning-curve test donors.

Same cases as lc/score.py (truth = test donor's stimulated quantiles; effect > FLOOR only). Adds, from the existing
runs, median_shift / quantile_map / gauss_ot (WORK/baselines) and CellOT paper-style best (WORK/train).
Outputs WORK/cond_summary/<tag>/:
- curve.csv           median skill per method and k (mean +- sd over the 5 repeats), as lc/score.py
- vs_pool.csv         paired vs gauss_ot_pool per method and k: median per-case gain 1 - err/err_pool, % of cases
                      where the method is closer, Wilcoxon p over (map, donor) median errors
- vs_pool_by_stim.csv the same at k = 15 per stim
- chosen.csv          inner-LODO choices of h (knn) and alpha (cond) per k
    python -m lc.cond_score --tag med
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from lc.paths import WORK
from lc.score import FLOOR, cases, truth_tables

REF = "gauss_ot_pool"   # --ref overrides


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="med")
    ap.add_argument("--indir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref", default="gauss_ot_pool", help="reference method of the paired tables")
    a = ap.parse_args()
    REF = a.ref
    indir = Path(a.indir or WORK / "cond_baselines" / a.tag)
    out = Path(a.out or WORK / "cond_summary" / (a.tag + ("" if a.ref == "gauss_ot_pool" else f"_vs_{a.ref}"))); out.mkdir(parents=True, exist_ok=True)
    allc, ch = [], []
    for dz in json.load(open(WORK / "design.json")):
        r, tests = dz["repeat"], dz["test"]
        f = indir / f"repeat_{r}.npz"
        if not f.exists(): continue
        z = np.load(f, allow_pickle=True)
        maps = [tuple(m) + (0,) for m in z["maps"]]
        preds = {m: z[m] for m in z.files if m not in ("maps", "chosen")}
        b = np.load(WORK / "baselines" / f"repeat_{r}.npz", allow_pickle=True)
        bpos = {tuple(m)[:3]: i for i, m in enumerate(b["maps"])}
        bi = [bpos.get(m[:3], -1) for m in maps]
        pick = lambda P, idx: np.stack([P[i] if i >= 0 else np.full(P.shape[1:], np.nan) for i in idx])
        for m in ("identity", "median_shift", "quantile_map"):
            preds[m] = pick(b[m], bi)
        cdir = WORK / "train" / f"repeat_{r}"
        if (cdir / "pred_q.npy").exists():
            cm = pd.read_csv(cdir / "maps.csv"); P = np.load(cdir / "pred_q.npy", mmap_mode="r")
            cpos = {(s, c, k): i for i, (s, c, k) in enumerate(zip(cm.stim, cm.celltype, cm.k))}
            preds["CellOT"] = pick(P[P.shape[0] - 1], [cpos.get(m[:3], -1) for m in maps])
        T = truth_tables(sorted({m[1] for m in maps}), set(tests))
        df = cases(maps, preds, tests, T); df["repeat"] = r
        allc.append(df[df.effect > FLOOR])
        ch.append(pd.DataFrame([(r, *m[:3], *c) for m, c in zip(maps, z["chosen"])],
                               columns=["repeat", "stim", "celltype", "k", "h", "alpha"]))
        print(f"[SCORE] repeat {r}: {len(df)} rows", flush=True)
    D = pd.concat(allc); D["skill"] = 1 - D.err / D.effect
    cur = D.groupby(["method", "repeat", "k"]).skill.median().reset_index()
    cur.groupby(["method", "k"]).skill.agg(["mean", "std"]).round(4).reset_index().to_csv(out / "curve.csv", index=False)
    key = ["repeat", "stim", "celltype", "k", "donor", "q", "marker"]
    W = D.pivot_table(index=key, columns="method", values="err").dropna(subset=[REF])
    rows, rows_s = [], []
    for m in [c for c in W.columns if c != REF]:
        for k, g in W[[m, REF]].dropna().groupby(level="k"):
            gm = g.groupby(level=["repeat", "stim", "celltype", "donor"]).median()
            rows.append({"method": m, "k": k, "gain_median": np.median(1 - g[m] / g[REF]),
                         "pct_closer": 100 * (g[m] < g[REF]).mean(), "pct_farther": 100 * (g[m] > g[REF]).mean(),
                         "p_wilcoxon_maps": wilcoxon(gm[m], gm[REF]).pvalue if len(gm) > 10 and (gm[m] != gm[REF]).any() else np.nan,
                         "n_cases": len(g)})
            if k == W.index.get_level_values("k").max():
                for s, gs in g.groupby(level="stim"):
                    rows_s.append({"method": m, "stim": s, "gain_median": np.median(1 - gs[m] / gs[REF]),
                                   "pct_closer": 100 * (gs[m] < gs[REF]).mean(), "n_cases": len(gs)})
    V = pd.DataFrame(rows); V.round(4).to_csv(out / "vs_pool.csv", index=False)
    pd.DataFrame(rows_s).round(4).to_csv(out / "vs_pool_by_stim.csv", index=False)
    C = pd.concat(ch); C.to_csv(out / "chosen.csv", index=False)
    print(cur.groupby(["method", "k"]).skill.mean().unstack("k").round(3).to_string())
    print(V.pivot(index="method", columns="k", values="pct_closer").round(1).to_string())
    print(C.groupby("k")[["h", "alpha"]].agg(lambda x: x.replace(np.inf, -1).value_counts().to_dict()).to_string())
    print(f"[DONE ] -> {out}")


if __name__ == "__main__":
    main()
