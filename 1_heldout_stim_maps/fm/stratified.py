"""Phase 2, closed-form first: does conditioning a map on the cell's PHENOTYPE help a new donor?

Per map (stim, celltype, k) of the lc design: k-means (K clusters) on the z-scored phenotypic markers of the training
donors' train-side Unstim cells; source (Unstim) and target (stim) training cells and the test donor's Unstim cells
are assigned to the nearest centroid; one closed-form map per cluster (lc/baselines fit/apply: gauss_ot, median_shift),
fitted on that cluster's source vs target cells (pooled map if either side has < MIN_CL cells); each test cell is
mapped by its cluster's map, and the predicted quantiles are those of the mapped mixture. The cluster mixture of the
test donor is its own (phenotype composition differs between donors: that is the point).
Phenotype = the 25 non-functional panel markers (fm cache), minus, with --stable, those whose training-donor median
moves by > STABLE_THR under that stim (a target cell must land in the cluster its source would have).
--feat fm uses instead the frozen ImmuneFM embedding with the functional markers masked (needs a GPU; not done here).
Output WORK/fm/stratified/<tag>/repeat_<r>.npz: maps (stim, celltype, k), one (M, n_test, n_q, 13) array per method,
the pooled maps included (same cells, so paired). Score: python -m fm.stratified --score <tag>.
    python -m fm.stratified --repeat 0 --K 4 [--stable] --tag raw_K4      (env_cpu.sh)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from lc.baselines import apply, fit
from lc.paths import MARKERS, PAIRS, QS, VAL_FRAC, WORK
from lc.score import FLOOR, cases, truth_tables

CACHE = WORK / "fm" / "cache_full"
OUT = WORK / "fm" / "stratified"
MIN_CL = 50
STABLE_THR = 0.2
MAX_TEST = 50_000
BASE = ["gauss_ot", "median_shift"]


def run(a):
    dz = json.load(open(WORK / "design.json"))[a.repeat]; tests = dz["test"]
    maps, preds = [], {f"{m}{s}": [] for m in BASE for s in ("", "_strat")}
    cur = None
    for stim, ct in PAIRS:
        if a.celltypes and ct not in a.celltypes: continue
        if cur != ct:
            z = dict(np.load(CACHE / f"{ct}.npz")); cur = ct
            panel = list(z["markers"]); fi = [panel.index(m) for m in MARKERS]
            pi_all = [i for i, m in enumerate(panel) if m not in MARKERS]
        tr = lambda cond, d: (lambda x: x[:max(1, int(round(len(x) * (1 - VAL_FRAC))))])(z[f"{cond}|{d}"])
        test_x = {d: z[f"Unstim|{d}"][:MAX_TEST] for d in tests if f"Unstim|{d}" in z}
        for k in a.ks:
            ds = [d for d in dz["order"][:k] if f"Unstim|{d}" in z and f"{stim}|{d}" in z]
            if not ds: continue
            S = np.concatenate([tr("Unstim", d) for d in ds]); T = np.concatenate([tr(stim, d) for d in ds])
            pi = pi_all
            if a.stable:
                mv = np.abs(np.median(T[:, pi_all], 0) - np.median(S[:, pi_all], 0))
                pi = [i for i, v in zip(pi_all, mv) if v <= STABLE_THR]
            mu, sd = S[:, pi].mean(0), S[:, pi].std(0) + 1e-6
            rng = np.random.default_rng(a.repeat * 1000 + k)
            sub = S[rng.permutation(len(S))[:20_000]]
            km = KMeans(a.K, n_init=4, random_state=0).fit((sub[:, pi] - mu) / sd)
            lab = lambda X: km.predict((X[:, pi] - mu) / sd)
            ls, lt = lab(S), lab(T)
            pooled = fit(S[:, fi], T[:, fi])
            per = {c: fit(S[ls == c][:, fi], T[lt == c][:, fi]) if min((ls == c).sum(), (lt == c).sum()) >= MIN_CL else pooled
                   for c in range(a.K)}
            maps.append((stim, ct, k))
            row = {m: np.full((len(tests), len(QS), len(fi)), np.nan, np.float32) for m in preds}
            for j, d in enumerate(tests):
                if d not in test_x or len(test_x[d]) < 20: continue
                x = test_x[d]; lx = lab(x); xf = x[:, fi]
                for m in BASE:
                    row[m][j] = np.percentile(apply(m, xf, pooled), QS, axis=0)
                    y = np.empty_like(xf, dtype=np.float64)
                    for c in range(a.K):
                        if (lx == c).any(): y[lx == c] = apply(m, xf[lx == c], per[c])
                    row[f"{m}_strat"][j] = np.percentile(y, QS, axis=0)
            for m in preds: preds[m].append(row[m])
        print(f"[DONE ] {stim} {ct}", flush=True)
    out = OUT / a.tag; out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"repeat_{a.repeat}.npz", maps=np.array(maps, dtype=object), **{m: np.stack(v) for m, v in preds.items()})


def score(tag):
    out = OUT / tag; rows = []
    design = json.load(open(WORK / "design.json"))
    for dz in design:
        f = out / f"repeat_{dz['repeat']}.npz"
        if not f.exists(): continue
        z = np.load(f, allow_pickle=True); maps = [tuple(m) + (0,) for m in z["maps"]]
        T = truth_tables(sorted({m[1] for m in maps}), set(dz["test"]))
        C = cases(maps, {m: z[m] for m in z.files if m != "maps"}, dz["test"], T); C["repeat"] = dz["repeat"]
        rows.append(C)
    C = pd.concat(rows); C = C[C.effect > FLOOR]; C["skill"] = 1 - C.err / C.effect
    S = C.groupby(["method", "k"]).skill.median().unstack(0)
    W = C.pivot_table(index=["repeat", "stim", "celltype", "k", "donor", "q", "marker"], columns="method", values="err")
    lines = [f"tag {tag}: median skill (cases with effect > FLOOR)", S.round(4).to_string(), ""]
    for m in BASE:
        b = (W[f"{m}_strat"] < W[m]).groupby("k").mean()
        lines.append(f"{m}_strat closer than {m}: " + " ".join(f"k{k} {v:.3f}" for k, v in b.items()))
    by = C[C.k == C.k.max()].groupby(["stim", "method"]).skill.median().unstack(1)
    lines += ["", f"by stim, k = {C.k.max()}", by.round(3).to_string()]
    (out / "summary.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--ks", nargs="*", type=int, default=[3, 5, 8, 15])
    ap.add_argument("--stable", action="store_true")
    ap.add_argument("--celltypes", nargs="*", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--score", default=None, help="score this tag (all repeats present) and exit")
    a = ap.parse_args()
    if a.score: return score(a.score)
    a.tag = a.tag or f"raw_K{a.K}" + ("_stable" if a.stable else "")
    run(a)


if __name__ == "__main__":
    main()
