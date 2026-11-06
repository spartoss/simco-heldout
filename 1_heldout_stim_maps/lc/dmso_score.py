"""Per-case table for the DMSO-plate experiment (notebooks/dmso_plate.ipynb, 2026-10-06). Held-out donors only.

Arms = which plate-control wells feed training AND the test donor's input / truth (one cache each):
  pool     WORK/cache                  DMSO1/2/3 pooled (the learning curve's cache)
  poolN1   WORK/dmso/cache_poolN1      pooled, cut to the DMSO1 cell count of every block (lc/match_cache.py)
  DMSO1    WORK/dmso/cache_DMSO1       plate 1 only, Unstim and stim (plate-matched); likewise DMSO2, DMSO3
For each arm, repeat (work/lc/design.json), map (stim, celltype, k = 15), test donor, quantile, marker, the same columns
as lc/where_skill.py (u, t, noise, noise_u, mu_u, mu_s, sd_u, sd_r, n_u, n_s, shift) computed from THAT arm's cache, plus
predictions: CellOT (lc/cond_train --ctx none, WORK/dmso/train/repeat_<r>/<arm>), gauss_ot, quantile_map, median_shift,
identity (lc/baselines, WORK/dmso/baselines/<arm>/repeat_<r>.npz). For arm pool also CellOT_lcnone = the existing
paired run cond_train/repeat_<r>/none (k 3 5 8 15). Missing runs -> NaN (the notebook drops them).
Output WORK/dmso/cases.pkl (no effect filter).
    python -m lc.dmso_score [--arms pool DMSO1 ...] [--celltypes ...] [--out cases.pkl]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lc.cond_train_score import load_npz, load_train
from lc.paths import CACHE, DONORS, MARKERS, PAIRS, QS, WORK
from lc.score import truth_tables

D_ = WORK / "dmso"
ARMS = {"pool": CACHE, "poolN1": D_ / "cache_poolN1", "DMSO1": D_ / "cache_DMSO1", "DMSO2": D_ / "cache_DMSO2",
        "DMSO3": D_ / "cache_DMSO3"}
BASE = ["identity", "median_shift", "quantile_map", "gauss_ot"]
K = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--celltypes", nargs="*", default=None)
    ap.add_argument("--cache-root", default=str(D_), help="single-plate / poolN1 caches: <root>/cache_<arm> (smoke tests)")
    ap.add_argument("--train-root", default=str(D_ / "train"))
    ap.add_argument("--base-root", default=str(D_ / "baselines"))
    ap.add_argument("--out", default=str(D_ / "cases.pkl"))
    a = ap.parse_args()
    cts = sorted({c for _, c in PAIRS if not a.celltypes or c in a.celltypes})
    pairs = [(s, c) for s, c in PAIRS if c in cts]
    design = json.load(open(WORK / "design.json"))
    rows = []
    for arm in a.arms:
        cdir = CACHE if arm == "pool" else Path(a.cache_root) / f"cache_{arm}"
        if not all((cdir / f"{c}.npz").exists() for c in cts):
            print(f"[WARN ] {arm}: cache incomplete in {cdir}, skipped", flush=True); continue
        T = truth_tables(cts, set(DONORS), cdir)
        N = {}
        for c in cts:
            z = np.load(cdir / f"{c}.npz")
            for key in z.files:
                cond, d = key.split("|"); N[(c, cond, d)] = len(z[key])
        for dz in design:
            r, tests = dz["repeat"], dz["test"]
            maps = [(s, c, K) for s, c in pairs]
            P = {}
            td = Path(a.train_root) / f"repeat_{r}" / arm
            if (td / "pred_q.npy").exists(): P["CellOT"] = load_train(td, maps)
            else: print(f"[WARN ] {arm} r{r}: no CellOT in {td}", flush=True)
            bf = Path(a.base_root) / arm / f"repeat_{r}.npz"
            if bf.exists(): P.update(load_npz(bf, BASE, maps))
            else: print(f"[WARN ] {arm} r{r}: no baselines {bf}", flush=True)
            if arm == "pool" and (WORK / f"cond_train/repeat_{r}/none/pred_q.npy").exists():
                P["CellOT_lcnone"] = load_train(WORK / f"cond_train/repeat_{r}/none", maps)
            for (s, c, k) in maps:
                trd = [d for d in dz["order"][:k] if (c, s, d) in T and (c, "Unstim", d) in T]
                if len(trd) < 2: continue
                U = np.stack([T[(c, "Unstim", d)][0] for d in trd]); S = np.stack([T[(c, s, d)][0] for d in trd])
                mu_u, mu_s = U.mean(0), S.mean(0); sd_u = U.std(0, ddof=1); sd_r = (S - U).std(0, ddof=1)
                for j, d in enumerate(tests):
                    if (c, s, d) not in T or (c, "Unstim", d) not in T: continue
                    (u, nzu), (t, nz) = T[(c, "Unstim", d)], T[(c, s, d)]
                    row = {"arm": arm, "repeat": r, "stim": s, "celltype": c, "k": k, "donor": d,
                           "q": np.repeat(QS, len(MARKERS)), "marker": np.tile(MARKERS, len(QS)),
                           "u": u.ravel(), "t": t.ravel(), "noise": nz.ravel(), "noise_u": nzu.ravel(),
                           "mu_u": mu_u.ravel(), "mu_s": mu_s.ravel(), "sd_u": sd_u.ravel(), "sd_r": sd_r.ravel(),
                           "n_u": N[(c, "Unstim", d)], "n_s": N[(c, s, d)], "n_train": len(trd),
                           "shift": (mu_s + u - mu_u).ravel()}
                    for n, Dm in P.items():
                        p = Dm.get((s, c, k))
                        row[n] = np.full(u.size, np.nan) if p is None else np.asarray(p[j], float).ravel()
                    rows.append(pd.DataFrame(row))
            print(f"{arm} repeat {r} done", flush=True)
    D = pd.concat(rows, ignore_index=True)
    for col in ["CellOT", "CellOT_lcnone"] + BASE:
        if col not in D: D[col] = np.nan
    for col in D.select_dtypes("float64"): D[col] = D[col].astype("float32")
    for col in ["arm", "stim", "celltype", "donor", "marker"]: D[col] = D[col].astype("category")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    D.to_pickle(a.out)
    print(D.shape, "->", a.out)


if __name__ == "__main__":
    main()
