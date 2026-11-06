"""Score conditioning variants (lc/cond_train --arch / --shift-reg / --ctx-drop, 2026-10-13) at one k on the user's
feature set (meta/feature_set.csv, in_map_grid: q50/75/90, mean effect > 0.1, no PI / IL33), common cases only.

Per method m and its 0.5 / 0.5 mix with the shift (m+shift):
  MAE, MSE, skill = 1 - sum|err| / sum|truth - Unstim|
  between_R2  = 1 - sum(pc - tc)^2 / sum tc^2, prediction and truth centred across the repeat's test donors per feature
  specific_R2 = the same on the departure from the shift (pred - shift vs truth - shift), centred: the share of the
                between-donor variation the shift misses that m explains (0 = shift, < 0 = adds noise)
  better/ref  = donor-repeats (repeat, test donor) whose MSE beats --ref's; p = Wilcoxon over those MSEs
    python -m lc.var_score --methods none=cond_train/repeat_{r}/none pcb=cond_train/repeat_{r}/pcblood_wd1 ...
        --ref none --out WORK/cond_var/score_<tag> [--repeats 0 1 2 3 4] [--pairs IFNa:cMCs ...]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from lc.paths import DONORS, MARKERS, QS, REPO, WORK
from lc.score import truth_tables

KEY = ["repeat", "stim", "celltype", "marker", "q"]


def load(d, k):
    cm = pd.read_csv(d / "maps.csv"); P = np.load(d / "pred_q.npy", mmap_mode="r")
    return {(s, c): np.asarray(P[-1, i]) for i, s, c, kk in zip(cm.index, cm.stim, cm.celltype, cm.k) if kk == k}


def metrics(D, m, ref):
    e = D[m] - D.t; out = {"MAE": e.abs().mean(), "MSE": (e ** 2).mean(), "skill": 1 - e.abs().sum() / (D.t - D.u).abs().sum()}
    g = [D[c] for c in KEY]
    cen = lambda v: v - v.groupby(g).transform("mean")
    tc = cen(D.t); out["between_R2"] = 1 - ((cen(D[m]) - tc) ** 2).sum() / (tc ** 2).sum()
    rc = cen(D.t - D["shift"]); out["specific_R2"] = 1 - ((cen(D[m] - D["shift"]) - rc) ** 2).sum() / (rc ** 2).sum()
    if ref and m != ref:
        a = (e ** 2).groupby([D.repeat, D.donor]).mean(); b = ((D[ref] - D.t) ** 2).groupby([D.repeat, D.donor]).mean()
        out[f"better than {ref}"] = f"{(a < b).sum()}/{len(a)}"; out["p"] = wilcoxon(a, b).pvalue if len(a) > 5 else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--repeats", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--pairs", nargs="*", default=None, help="stim:celltype subset (screens)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meth = dict(m.split("=", 1) for m in a.methods)
    FS = pd.read_csv(REPO / "meta" / "feature_set.csv"); FS = FS[FS.in_map_grid]
    if a.pairs: FS = FS[(FS.stim + ":" + FS.celltype).isin(a.pairs)]
    T = truth_tables(sorted(FS.celltype.unique()), set(DONORS))
    qi = {q: i for i, q in enumerate(QS)}; mi = {m: i for i, m in enumerate(MARKERS)}
    rows = []
    for dz in json.load(open(WORK / "design.json")):
        r = dz["repeat"]
        if r not in a.repeats: continue
        P = {}
        for n, d in meth.items():
            dd = WORK / d.format(r=r)
            if not (dd / "pred_q.npy").exists(): print(f"[MISS ] {n} repeat {r}: {dd}", flush=True); continue
            P[n] = load(dd, a.k)
        if len(P) < len(meth): continue
        for (s, c), F in FS.groupby(["stim", "celltype"]):
            if any((s, c) not in X for X in P.values()): continue
            trd = [d for d in dz["order"][:a.k] if (c, s, d) in T and (c, "Unstim", d) in T]
            mu_u = np.mean([T[(c, "Unstim", d)][0] for d in trd], 0); mu_s = np.mean([T[(c, s, d)][0] for d in trd], 0)
            ii, jj = [qi[q] for q in F.q], [mi[m] for m in F.marker]
            for j, d in enumerate(dz["test"]):
                if (c, s, d) not in T or (c, "Unstim", d) not in T: continue
                u, t = T[(c, "Unstim", d)][0], T[(c, s, d)][0]
                row = {"repeat": r, "donor": d, "stim": s, "celltype": c, "marker": F.marker.values, "q": F.q.values,
                       "u": u[ii, jj], "t": t[ii, jj], "shift": (mu_s + u - mu_u)[ii, jj]}
                row.update({n: X[(s, c)][j][ii, jj] for n, X in P.items()})
                rows.append(pd.DataFrame(row))
    D = pd.concat(rows, ignore_index=True).dropna()
    for n in meth: D[f"{n}+shift"] = 0.5 * D[n] + 0.5 * D["shift"]
    D.to_pickle(out / "cases.pkl")
    res = {m: metrics(D, m, a.ref) for m in ["shift"] + list(meth) + [f"{n}+shift" for n in meth]}
    R = pd.DataFrame(res).T; R.to_csv(out / "metrics.csv")
    pd.set_option("display.width", 250)
    print(f"{len(D)} cases, repeats {sorted(D.repeat.unique())}, {D.groupby(['stim', 'celltype']).ngroups} pairs")
    print(R.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
