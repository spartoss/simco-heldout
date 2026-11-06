"""Score held-out-donor predictions on the user's feature set with aggregate metrics (2026-10-12).

Cases = (repeat, test donor, stim, celltype, marker, q) for every feature of meta/feature_set.csv (or, --features all,
every marker of every map pair at --qs) (heldout/feature_set.py:
q50/75/90, mean over donors |effect| > 0.1, no PI/IL33) whose pair has a map, at one k; only cases every method
predicts. Metrics over ALL those cases (no median, no effect floor): MAE, RMSE, effect-weighted skill
1 - sum|err| / sum|effect| (effect = |truth - Unstim| of the test donor), MAE per repeat, and per (repeat, donor) MSE
compared with --ref (how many of the donor-repeats are better).
Methods:
    identity, shift          built in (shift = training-donor mean stim + (Unstim - training mean Unstim), per feature)
    name=dir[:vb|:avg][:dnorm]  dir relative to WORK with {r}: lc.train / lc.cond_train layout (pred_q.npy). vb = the
                             run's last slice (val-best checkpoint, default); avg = mean of the checkpoints <= --avg-max
                             iterations; dnorm = add the test donor's Unstim median offset back (cache_dnorm runs).
                             tab/pred/<x> dirs (1-slice pred_q) work the same way (vb).
    --mix name=a+b:w         w * a + (1 - w) * b, a and b are method names defined before.
    python -m lc.feature_score --methods none=cond_train/repeat_{r}/none:avg ... --mix ens=none+shift:0.5 \
        --repeats 0 1 2 3 4 --ref none --out WORK/feature_score/<tag>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lc.paths import DONORS, MARKERS, PAIRS, QS, REPO, WORK
from lc.score import truth_tables


def load(d, k, avg, avg_max):
    cm = pd.read_csv(d / "maps.csv"); P = np.load(d / "pred_q.npy", mmap_mode="r")
    if "cap" in cm: cm = cm[cm.cap == 0]
    n = P.shape[0] - 1
    if avg and n > 0:
        meta = json.load(open(d / "meta.json")); st = meta.get("eval_every", 2500) * np.arange(1, n + 1)
        idx = list(np.flatnonzero(st <= avg_max))
    else:
        idx = [n]
    return {(s, c): np.asarray(P[idx, i]).mean(0) for i, s, c, kk in zip(cm.index, cm.stim, cm.celltype, cm.k) if kk == k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=[])
    ap.add_argument("--mix", nargs="*", default=[])
    ap.add_argument("--repeats", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--avg-max", type=int, default=12500)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--features", choices=["set", "all"], default="set",
                    help="set = meta/feature_set.csv; all = every marker of every map pair (stims of the set) at --qs")
    ap.add_argument("--qs", nargs="*", type=int, default=[50, 75, 90])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    FS = pd.read_csv(REPO / "meta" / "feature_set.csv"); FS = FS[FS.in_map_grid]
    if a.features == "all":
        st = set(FS.stim)
        FS = pd.DataFrame([{"stim": s, "celltype": c, "marker": m, "q": q} for s, c in PAIRS if s in st
                           for m in MARKERS for q in a.qs])
    FS = FS[FS.q.isin(a.qs)]
    qi = {q: QS.index(q) for q in (50, 75, 90)}; mi = {m: i for i, m in enumerate(MARKERS)}
    feats = {}
    for (s, c), g in FS.groupby(["stim", "celltype"]):
        feats[(s, c)] = (np.array([qi[q] for q in g.q]), np.array([mi[m] for m in g.marker]), list(zip(g.marker, g.q)))
    T = truth_tables(sorted({c for _, c in PAIRS}), set(DONORS))
    off = np.load(WORK / "cache_dnorm/offsets.npz")
    specs = {}
    for m in a.methods:
        n, rest = m.split("=", 1); parts = rest.split(":")
        specs[n] = (parts[0], "avg" in parts[1:], "dnorm" in parts[1:])
    mixes = []
    for m in a.mix:
        n, rest = m.split("=", 1); ab, w = rest.rsplit(":", 1); x, y = ab.split("+"); mixes.append((n, x, y, float(w)))
    rows = []
    for dz in json.load(open(WORK / "design.json")):
        r, tests = dz["repeat"], dz["test"]
        if r not in a.repeats: continue
        L = {n: load(WORK / d.format(r=r), a.k, avg, a.avg_max) for n, (d, avg, dn) in specs.items()}
        for (s, c), (fq, fm, lab) in feats.items():
            trd = [d for d in dz["order"][:a.k] if (c, s, d) in T and (c, "Unstim", d) in T]
            if len(trd) < 2: continue
            mu_u = np.mean([T[(c, "Unstim", d)][0] for d in trd], 0); mu_s = np.mean([T[(c, s, d)][0] for d in trd], 0)
            for j, d in enumerate(tests):
                if (c, s, d) not in T or (c, "Unstim", d) not in T: continue
                t, u = T[(c, s, d)][0], T[(c, "Unstim", d)][0]
                P = {"identity": u, "shift": mu_s + u - mu_u}
                for n, (dd, avg, dn) in specs.items():
                    if (s, c) not in L[n] or np.isnan(L[n][(s, c)][j]).all(): continue
                    P[n] = L[n][(s, c)][j] + (off[f"{c}|{d}"][None, :] if dn else 0)
                for n, x, y, w in mixes:
                    if x in P and y in P: P[n] = w * P[x] + (1 - w) * P[y]
                if len(P) < 2 + len(specs) + len(mixes): continue
                tt, uu = t[fq, fm], u[fq, fm]
                for n, p in P.items():
                    rows.append(pd.DataFrame({"method": n, "repeat": r, "donor": d, "stim": s, "celltype": c,
                                              "marker": [x for x, _ in lab], "q": [y for _, y in lab],
                                              "err": p[fq, fm] - tt, "eff": np.abs(tt - uu)}))
        print(f"repeat {r} done", flush=True)
    D = pd.concat(rows, ignore_index=True); D["ae"] = D.err.abs(); D["se"] = D.err ** 2
    D.to_pickle(out / "cases.pkl")
    order = list(dict.fromkeys(["identity", "shift"] + list(specs) + [m[0] for m in mixes]))
    agg = lambda g: pd.Series({"MAE": g.ae.mean(), "RMSE": np.sqrt(g.se.mean()), "eff_skill": 1 - g.ae.sum() / g.eff.sum()})
    S = D.groupby("method").apply(agg)
    S = S.join(D.groupby(["method", "repeat"]).ae.mean().unstack().add_prefix("MAE_r"))
    if a.ref:
        M = D.groupby(["method", "repeat", "donor"]).se.mean().unstack("method")
        S["better_than_ref"] = [f"{(M[m] < M[a.ref]).sum()}/{len(M)}" if m != a.ref else "-" for m in S.index]
    S = S.loc[order]; S.round(4).to_csv(out / "summary.csv")
    Q = D.groupby(["method", "q"]).ae.mean().unstack().loc[order]; Q.round(4).to_csv(out / "mae_by_q.csv")
    B = D.groupby(["method", "stim"]).ae.mean().unstack().loc[order]; B.round(4).to_csv(out / "mae_by_stim.csv")
    D["bin"] = pd.cut(D.eff, [-1, 0.1, 0.3, 1, 99], labels=["<0.1", "0.1-0.3", "0.3-1", ">1"])
    E = D.groupby(["method", "bin"], observed=True).ae.mean().unstack().loc[order]; E.round(4).to_csv(out / "mae_by_effect.csv")
    pd.set_option("display.width", 250)
    print(f"cases per method: {len(D) // len(order)} (repeats {a.repeats}, k {a.k})")
    print(S.round(4).to_string()); print("--- MAE by q"); print(Q.round(4).to_string())
    print("--- MAE by stim"); print(B.round(4).to_string()); print("--- MAE by test-donor effect"); print(E.round(4).to_string())
    print("share of cases by effect:", D[D.method == "identity"].bin.value_counts(normalize=True).sort_index().round(3).to_dict())


if __name__ == "__main__":
    main()
