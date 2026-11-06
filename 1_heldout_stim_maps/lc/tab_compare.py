"""Compare methods on held-out donors with MAE / MSE (and between-donor R2), optionally only on rule-selected features.

Methods are name=dir pairs as in lc.ctx_score (dir relative to WORK, {r} = repeat); shift is added. Cases =
(repeat, stim, celltype, k, test donor, q, marker), kept only if every method has them.
Two case sets:
- "floor": as lc.ctx_score, the test donor's |S - U| > FLOOR (0.0273);
- "rule":  the any-quantile rule of heldout/inclusion.py at the listed quantiles: keep (stim, celltype, marker, q) if
  the mean over the 19 donors of |q(stim) - q(Unstim)| > 0.1 at that quantile (work/heldout/features/inclusion.csv);
  every test donor of a kept feature is scored (no per-case floor).
Readouts per (case set, k, method): MAE, MSE, between_R2 (centred across the repeat's test donors, groups >= 3 donors),
n; paired vs --vs: % of cases with smaller |error| and Wilcoxon over (repeat, stim, celltype, donor) mean |error|.
    python -m lc.tab_compare --methods a=dir ... --repeats 0 1 2 --ks 15 --exclude-stims IL33 PI --rule-qs 50 75 90 \
        --vs CellOT+shift --out WORK/tab/compare_<tag>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from lc.cond_train_score import load_train
from lc.paths import DONORS, MARKERS, PAIRS, QS, WORK
from lc.score import FLOOR, truth_tables

RULE = WORK.parent / "heldout" / "features" / "inclusion.csv"


def readout(D, names, vs):
    res, pr = [], []
    for (cs, k), g in D.groupby(["set", "k"]):
        for m in names:
            e = g[m] - g.true
            gb = g[g.ntest >= 3]
            pc = gb[m] - gb.groupby(["repeat", "stim", "celltype", "q", "marker"])[m].transform("mean")
            res.append({"set": cs, "k": k, "method": m, "MAE": e.abs().mean(), "MSE": (e ** 2).mean(),
                        "between_R2": 1 - ((pc - gb.tc) ** 2).sum() / (gb.tc ** 2).sum(), "n": len(g)})
        for m in names:
            for b in vs:
                if b == m: continue
                ea, eb = (g[m] - g.true).abs(), (g[b] - g.true).abs()
                gm = pd.DataFrame({"a": ea, "b": eb, **{c: g[c] for c in ["repeat", "stim", "celltype", "donor"]}}) \
                    .groupby(["repeat", "stim", "celltype", "donor"])[["a", "b"]].mean()
                pr.append({"set": cs, "k": k, "a": m, "b": b, "pct_a_closer": 100 * (ea < eb).mean(),
                           "p_wilcoxon": wilcoxon(gm.a, gm.b).pvalue})
    return pd.DataFrame(res), pd.DataFrame(pr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--repeats", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--ks", nargs="*", type=int, default=[15])
    ap.add_argument("--exclude-stims", nargs="*", default=[])
    ap.add_argument("--rule-qs", nargs="*", type=int, default=[50, 75, 90])
    ap.add_argument("--qs", nargs="*", type=int, default=list(QS), help="quantiles scored in the floor set")
    ap.add_argument("--rule-thr", type=float, default=0.1)
    ap.add_argument("--vs", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--design", default=str(WORK / "design.json"))
    ap.add_argument("--dump", action="store_true", help="also save the case table (cases.pkl) for breakdowns")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meth = dict(m.split("=", 1) for m in a.methods)
    R = pd.read_csv(RULE)
    keep = {(r.stim, r.celltype, r.marker, q) for r in R.itertuples() for q in a.rule_qs
            if getattr(r, f"shift_q{q}") > a.rule_thr}
    T = truth_tables(sorted({c for _, c in PAIRS}), set(DONORS))
    rows = []
    for dz in json.load(open(a.design)):
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
                pr = {"shift": mu_s + u - mu_u}; pr.update({n: Dm[(s, c, k)][j] for n, Dm in P.items()})
                if any(np.isnan(p).all() for p in pr.values()): continue
                f = {"repeat": r, "stim": s, "celltype": c, "k": k, "donor": d, "q": np.repeat(QS, 13),
                     "marker": np.tile(MARKERS, len(QS)), "true": t.ravel(), "eff": np.abs(t - u).ravel()}
                f.update({n: p.ravel() for n, p in pr.items()})
                rows.append(pd.DataFrame(f))
    D = pd.concat(rows, ignore_index=True)
    names = ["shift"] + list(meth)
    D = D.dropna(subset=names)
    sel = np.array([(s, c, m, q) in keep for s, c, m, q in zip(D.stim, D.celltype, D.marker, D.q)])
    parts = [D[(D.eff > FLOOR) & D.q.isin(a.qs)].assign(set="floor"), D[sel].assign(set=f"rule q{'/'.join(map(str, a.rule_qs))}")]
    D = pd.concat(parts, ignore_index=True)
    grp = ["set", "repeat", "stim", "celltype", "k", "q", "marker"]
    D["tc"] = D.true - D.groupby(grp).true.transform("mean"); D["ntest"] = D.groupby(grp).true.transform("size")
    if a.dump: D.to_pickle(out / "cases.pkl")
    C, Pd = readout(D, names, a.vs)
    C.round(5).to_csv(out / "metrics.csv", index=False); Pd.round(5).to_csv(out / "paired.csv", index=False)
    pd.set_option("display.width", 220)
    for (cs, k), g in C.groupby(["set", "k"]):
        x = g.set_index("method").drop(columns=["set", "k"]).sort_values("MSE", ascending=False)
        for b in a.vs:
            pp = Pd[(Pd.set == cs) & (Pd.k == k) & (Pd.b == b)].set_index("a")
            x[f"%closer_vs_{b}"] = pp.pct_a_closer; x[f"p_vs_{b}"] = pp.p_wilcoxon
        print(f"=== {cs}, k = {k}, n = {int(g.n.iloc[0])} cases"); print(x.drop(columns="n").round(4).to_string())
    print(f"[DONE ] -> {out}")


if __name__ == "__main__":
    main()
