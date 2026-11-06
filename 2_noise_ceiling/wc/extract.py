"""Per-well quantile tables for one celltype (the only step that reads the atlas).

Cells: the DMSO plate-control rows of <celltype>_HVPV.h5ad (the paper's stim-arm data), 13 functional markers,
Unstim + the stims of the 168-pair grid for this celltype. Read with h5py (obs category codes, X in row chunks), so
Granulocytes (30 GB) needs a few GB.

Two cell sets per (condition, donor):
- "lc" set: exactly the cells of atlas_heldout's learning-curve cache (lc/cache.py: rng = default_rng(0), one
  permutation per (stim, patient) key in np.unique order, first 10,000), in the same order, now with the well label.
  From it: pooled quantiles (the truth/identity of lc/score.py), the two halves x[:n//2] / x[n//2:] (its split-half
  ceiling), per-well quantiles, and "pseudo-wells" (the pooled cells randomly re-partitioned into groups of the
  per-well sizes: same n as the wells, no well effect).
- "big" set: up to 30,000 cells per well (rng default_rng(1)), per-well and pseudo-well quantiles, to see the well
  effect with less sampling noise.
Writes WORK/quant/<celltype>.npz: arrays "<cond>|<donor>|<kind>" of shape (len(QS), 13) and "...|n" counts.
    python -m wc.extract --celltype pDCs
"""
import argparse
import os
import time

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py  # noqa: E402
import numpy as np  # noqa: E402

from wc.paths import ATLAS, CAP_LC, CAP_WELL, MARKERS, PAIRS, QS, QUANT, WELLS  # noqa: E402

ROW_CHUNK = 2_000_000


def categorical(obs, col):
    g = obs[col]
    cats = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in g["categories"][:]])
    return g["codes"][:], cats


def q(x):
    return np.percentile(x, QS, axis=0).astype(np.float32)


def pseudo(x, sizes, rng):
    """Random partition of x into groups of the given sizes (sampling only, no well effect)."""
    p = rng.permutation(len(x)); out, s = [], 0
    for n in sizes:
        out.append(x[p[s:s + n]]); s += n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--celltype", required=True)
    ap.add_argument("--outdir", default=str(QUANT))
    a = ap.parse_args()
    out = os.path.join(a.outdir, f"{a.celltype}.npz")
    if os.path.exists(out):
        print(f"[SKIP ] {out}"); return
    t0 = time.time()
    conds = ["Unstim"] + sorted({s for s, c in PAIRS if c == a.celltype})
    f = h5py.File(ATLAS / f"{a.celltype}_HVPV.h5ad", "r")
    dc, dcat = categorical(f["obs"], "drug"); sc, scat = categorical(f["obs"], "stim"); pc, pcat = categorical(f["obs"], "patient")
    var = [v.decode() if isinstance(v, bytes) else str(v) for v in f["var"]["_index"][:]]
    vidx = np.array([var.index(m) for m in MARKERS])
    # same row filter and key as atlas_heldout lc/cache.py
    ok_d = np.array([c.startswith("DMSO") for c in dcat]); ok_s = np.isin(scat, conds)
    keep = ok_d[dc] & ok_s[sc] & (dc >= 0) & (sc >= 0) & (pc >= 0)
    rows = np.nonzero(keep)[0]
    key = sc[rows].astype(np.int64) * 10_000 + pc[rows]
    rng0 = np.random.default_rng(0)
    lc_sel = {}
    for k in np.unique(key):
        idx = rows[key == k]
        lc_sel[(scat[k // 10_000], pcat[k % 10_000])] = rng0.permutation(idx)[:CAP_LC]
    # big per-well set
    rng1 = np.random.default_rng(1)
    wkey = key * 100 + dc[rows]
    big_sel = {}
    for k in np.unique(wkey):
        idx = rows[wkey == k]; kk = k // 100
        big_sel[(scat[kk // 10_000], pcat[kk % 10_000], dcat[k % 100])] = rng1.permutation(idx)[:CAP_WELL]
    del key, wkey
    allidx = np.unique(np.concatenate(list(lc_sel.values()) + list(big_sel.values())))
    got = np.empty((len(allidx), len(MARKERS)), np.float32)
    X = f["X"]; N = X.shape[0]
    for s in range(0, N, ROW_CHUNK):
        e = min(s + ROW_CHUNK, N)
        lo, hi = np.searchsorted(allidx, [s, e])
        if hi > lo:
            got[lo:hi] = X[s:e][allidx[lo:hi] - s][:, vidx]
    f.close()
    t_read = time.time() - t0
    well_of = np.full(N, -1, np.int8)
    well_of[rows] = dc[rows]
    rng2 = np.random.default_rng(2)
    arrs = {}
    for (st, pt), idx in lc_sel.items():
        x = got[np.searchsorted(allidx, idx)]; w = dcat[well_of[idx]]
        b = f"{st}|{pt}"; h = len(x) // 2
        arrs[f"{b}|pool"] = q(x); arrs[f"{b}|h1"] = q(x[:h]); arrs[f"{b}|h2"] = q(x[h:])
        arrs[f"{b}|pool|n"] = np.array(len(x))
        present = [wl for wl in WELLS if (w == wl).sum() > 0]
        sizes = [(w == wl).sum() for wl in present]
        for wl, xp in zip(present, pseudo(x, sizes, rng2)):
            arrs[f"{b}|w:{wl}"] = q(x[w == wl]); arrs[f"{b}|w:{wl}|n"] = np.array((w == wl).sum())
            arrs[f"{b}|pw:{wl}"] = q(xp)
        big = {wl: got[np.searchsorted(allidx, big_sel[(st, pt, wl)])] for wl in WELLS if (st, pt, wl) in big_sel}
        if big:
            for wl, xp in zip(big, pseudo(np.concatenate(list(big.values())), [len(v) for v in big.values()], rng2)):
                arrs[f"{b}|W:{wl}"] = q(big[wl]); arrs[f"{b}|W:{wl}|n"] = np.array(len(big[wl]))
                arrs[f"{b}|PW:{wl}"] = q(xp)
    os.makedirs(a.outdir, exist_ok=True)
    tmp = out.replace(".npz", f".{os.getpid()}.tmp.npz")
    np.savez(tmp, **arrs); os.replace(tmp, out)
    print(f"[DONE ] {a.celltype}: {len(lc_sel)} (cond, donor) blocks, {len(allidx)} cells read, read {t_read:.0f}s, "
          f"total {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
