"""Closed-form maps on exactly the learning-curve design: identity, median_shift, quantile_map, gauss_ot.

Same maps, same training cells (train side of the first k donors, pooled = "cell" weighting as CellOT's sampling),
same test donors as lc/train.py. Formulas vendored from simco/baselines/baselines.py (copied, not imported).
Writes WORK/baselines/repeat_<r>.npz: "maps" (stim, celltype, k) rows and one (M, n_test, n_q, n_markers) array
of predicted test-donor quantiles per method.
    python -m lc.baselines --repeat 0
"""
import argparse
import json
from pathlib import Path

import numpy as np

from lc.paths import CACHE, PAIRS, QS, VAL_FRAC, WORK

EIG_REL_FLOOR = 1e-8
GRID = np.linspace(0, 1, 1001)
METHODS = ["identity", "median_shift", "quantile_map", "gauss_ot"]
MIN_TEST_CELLS = 20


def sym_pow(S, p):
    w, V = np.linalg.eigh(0.5 * (S + S.T))
    fl = max(w.max(), 0.0) * EIG_REL_FLOOR
    w = np.clip(w, fl if fl > 0 else EIG_REL_FLOOR, None)
    return (V * w ** p) @ V.T


def fit(src, tgt):
    s, t = src.astype(np.float64), tgt.astype(np.float64)
    ms, mt, Ss, St = s.mean(0), t.mean(0), np.cov(s, rowvar=False), np.cov(t, rowvar=False)
    h, ih = sym_pow(Ss, 0.5), sym_pow(Ss, -0.5)
    A = ih @ sym_pow(h @ St @ h, 0.5) @ ih
    qs = np.quantile(s, GRID, axis=0)
    qs = qs + np.arange(len(GRID))[:, None] * 1e-9          # strictly increasing, as simco's _strictly_increasing
    return {"shift": np.median(t, 0) - np.median(s, 0), "mu_s": ms, "mu_t": mt, "A": 0.5 * (A + A.T),
            "q_src": qs, "q_tgt": np.quantile(t, GRID, axis=0)}


def apply(method, x, p):
    x = x.astype(np.float64)
    if method == "identity": return x
    if method == "median_shift": return x + p["shift"]
    if method == "gauss_ot": return p["mu_t"] + (x - p["mu_s"]) @ p["A"].T
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        out[:, j] = np.interp(np.interp(x[:, j], p["q_src"][:, j], GRID), GRID, p["q_tgt"][:, j])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, required=True)
    ap.add_argument("--celltypes", nargs="*", default=None, help="smoke tests")
    ap.add_argument("--out", default=str(WORK / "baselines"))
    ap.add_argument("--cache-dir", default=str(CACHE))
    ap.add_argument("--caps", nargs="*", type=int, default=[0], help="training cells per donor, 0 = all cached")
    ap.add_argument("--ks", nargs="*", type=int, default=None)
    ap.add_argument("--spec", default=None, help="custom design JSON (lc/design_donors.py); --repeat then only names the output")
    a = ap.parse_args()
    if a.spec:
        sp = json.load(open(a.spec)); tests = sp["tests"]
        sets = [(ts["name"], ts["donors"]) for ts in sp["train_sets"]]; caps = sp.get("caps", [0])
        only = sp.get("celltypes")
    else:
        dz = json.load(open(WORK / "design.json"))[a.repeat]
        tests = dz["test"]; sets = [(f"k{k}", dz["order"][:k]) for k in (a.ks or dz["ks"])]; caps = a.caps; only = None
    maps, preds = [], {m: [] for m in METHODS}
    cache = {}
    for stim, ct in PAIRS:
        if a.celltypes and ct not in a.celltypes: continue
        if only and ct not in only: continue
        if ct not in cache: cache = {ct: dict(np.load(Path(a.cache_dir) / f"{ct}.npz"))}   # one celltype, loaded once
        z = cache[ct]
        tr = lambda cond, d, cap: (lambda x: x[:max(1, int(round(len(x) * (1 - VAL_FRAC))))][:cap or None])(z[f"{cond}|{d}"])
        test_x = {d: z[f"Unstim|{d}"][:50_000] for d in tests if f"Unstim|{d}" in z}   # as MAX_TEST_CELLS
        for (name, donors), cap in [(st, cap) for st in sets for cap in caps]:
            ds = [d for d in donors if f"Unstim|{d}" in z and f"{stim}|{d}" in z]
            if not ds: continue
            src = np.concatenate([tr("Unstim", d, cap) for d in ds]); tgt = np.concatenate([tr(stim, d, cap) for d in ds])
            if len(src) < 5 or len(tgt) < 5: continue
            p = fit(src, tgt)
            maps.append((stim, ct, len(donors), cap, name, len(src)))
            for m in METHODS:
                arr = np.full((len(tests), len(QS), 13), np.nan, np.float32)
                for j, d in enumerate(tests):
                    if d in test_x and len(test_x[d]) >= MIN_TEST_CELLS:
                        arr[j] = np.percentile(apply(m, test_x[d], p), QS, axis=0)
                preds[m].append(arr)
        print(f"[FIT  ] {stim}:{ct}", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.savez(Path(a.out) / f"repeat_{a.repeat}.npz", maps=np.array(maps, dtype=object),
             **{m: np.stack(v) for m, v in preds.items()})
    print(f"[DONE ] repeat {a.repeat}: {len(maps)} maps")


if __name__ == "__main__":
    main()
