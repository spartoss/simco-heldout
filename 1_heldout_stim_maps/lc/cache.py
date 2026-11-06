"""Extract each donor's cells once, so training never touches the h5ad files again.

For one celltype: DMSO-control rows of `<celltype>_HVPV.h5ad` (the paper's stim-arm data), 13 functional markers
in meta/markers.txt order (--all-markers: the whole panel, var order), split by (condition, donor), shuffled with a fixed seed and capped at CAP_PER_DONOR.
Writes WORK/cache/<celltype>.npz with arrays "<condition>|<donor>" (float32, shuffled; the first 80 % is the
train side and the last 20 % the validation side whenever that donor is used for training).

--treatments DMSO1 [DMSO2 DMSO3] (2026-10-06, notebooks/dmso_plate.ipynb): instead of pooling the three plate
controls, keep only the wells of one plate, Unstim and stims alike (so source and target are plate-matched); one output
per treatment, --outdir then a template with {t} (e.g. work/lc/dmso/cache_{t}). One h5ad pass for all treatments;
each treatment draws with its own default_rng(0), so the result equals a run with that treatment alone.

Reads the h5ad with h5py directly (obs category codes only, X in contiguous row chunks), so even Granulocytes
(149M rows, 23 GB) needs ~2 GB of memory; anndata would load a 149M-string obs index.
    python -m lc.cache --celltype pDCs
"""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py  # noqa: E402
import numpy as np  # noqa: E402

from lc.paths import ATLAS, CACHE, CAP_PER_DONOR, MARKERS, PAIRS  # noqa: E402

ROW_CHUNK = 2_000_000


def categorical(obs, col):
    g = obs[col]
    cats = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in g["categories"][:]])
    return g["codes"][:], cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--celltype", required=True)
    ap.add_argument("--cap", type=int, default=CAP_PER_DONOR, help="cells kept per (condition, donor); 0 = all")
    ap.add_argument("--outdir", default=str(CACHE))
    ap.add_argument("--all-markers", action="store_true",
                    help="keep all panel markers (h5ad var order, saved as 'markers'), e.g. for fm/; same cells as without")
    ap.add_argument("--all-stims", action="store_true",
                    help="all 7 stims for every celltype, including the 28 (stim, celltype) pairs the paper dropped")
    ap.add_argument("--treatments", nargs="*", default=None,
                    help="one plate control per output (e.g. DMSO1 DMSO2 DMSO3) instead of all DMSO wells pooled; "
                         "--outdir must contain {t}")
    a = ap.parse_args()
    cap = a.cap if a.cap > 0 else None
    groups = a.treatments or [None]                      # None = all DMSO wells pooled (the default cache)
    assert a.treatments is None or "{t}" in a.outdir, "--treatments needs an --outdir template with {t}"
    outs = {t: Path(a.outdir.format(t=t)) / f"{a.celltype}.npz" for t in groups}
    groups = [t for t in groups if not outs[t].exists()]
    for t in outs:
        if t not in groups: print(f"[SKIP ] {outs[t]}")
    if not groups: return
    conds = ["Unstim"] + (sorted({s for s, _ in PAIRS}) if a.all_stims else sorted({s for s, c in PAIRS if c == a.celltype}))
    t0 = time.time()
    f = h5py.File(ATLAS / f"{a.celltype}_HVPV.h5ad", "r")
    dc, dcat = categorical(f["obs"], "drug"); sc, scat = categorical(f["obs"], "stim"); pc, pcat = categorical(f["obs"], "patient")
    var = [v.decode() if isinstance(v, bytes) else str(v) for v in f["var"]["_index"][:]]
    markers = var if a.all_markers else MARKERS
    var_idx = np.array([var.index(m) for m in markers])
    ok_s = np.isin(scat, conds)
    chosen_by = {}
    for t in groups:
        ok_d = np.array([c.startswith("DMSO") if t is None else c == t for c in dcat])
        keep = ok_d[dc] & ok_s[sc] & (dc >= 0) & (sc >= 0) & (pc >= 0)
        rows = np.nonzero(keep)[0]
        key = sc[rows].astype(np.int64) * 10_000 + pc[rows]
        rng = np.random.default_rng(0)
        chosen = {}
        for k in np.unique(key):
            idx = rows[key == k]
            chosen[(scat[k // 10_000], pcat[k % 10_000])] = rng.permutation(idx)[:cap]
        chosen_by[t] = chosen
        del keep, rows, key
    allidx = np.unique(np.concatenate([v for ch in chosen_by.values() for v in ch.values()]))
    got = np.empty((len(allidx), len(markers)), np.float32)
    X = f["X"]; N = X.shape[0]; p = 0
    for s in range(0, N, ROW_CHUNK):
        e = min(s + ROW_CHUNK, N)
        lo, hi = np.searchsorted(allidx, [s, e])
        if hi > lo:
            block = X[s:e]
            got[lo:hi] = block[allidx[lo:hi] - s][:, var_idx]
    f.close()
    for t, chosen in chosen_by.items():
        out = outs[t]; outdir = out.parent
        arrs = {f"{st}|{pt}": got[np.searchsorted(allidx, idx)] for (st, pt), idx in chosen.items()}
        outdir.mkdir(parents=True, exist_ok=True)
        tmp = outdir / f"{a.celltype}.{os.getpid()}.tmp.npz"
        if a.all_markers: arrs["markers"] = np.array(markers)
        np.savez(tmp, **arrs); os.replace(tmp, out)
        n = [len(v) for k, v in arrs.items() if k != "markers"]
        print(f"[DONE ] {a.celltype} {t or 'DMSO pooled'}: {len(arrs)} blocks, min {min(n)} / median {int(np.median(n))} "
              f"cells per block, {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
