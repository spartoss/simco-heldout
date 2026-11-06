"""Per-well summary of ALL markers for one celltype: the input of notebooks/phenotype_shifts.ipynb.

A "well" = (stim, treatment, donor), treatment in DMSO1/2/3 (plate controls) and the 9 drugs. For each well: the exact
number of cells of this celltype (from obs), and for every marker in the file (38: 13 functional + phenotypic / other)
the mean, q10, q50, q90 over at most CAP randomly chosen cells. Raw arcsinh values, no batch correction (the notebook
discusses the correction separately). Reads the h5ad with h5py (see lc/cache.py), so Granulocytes fits in memory.
    python -m heldout.pheno_summary --celltype pDCs -> WORK/pheno/<celltype>.csv.gz
"""
import argparse
import os
import time

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from heldout.common import WORK  # noqa: E402
from lc.cache import ROW_CHUNK, categorical  # noqa: E402
from lc.paths import ATLAS  # noqa: E402

CAP = 20_000
OUT = WORK / "pheno"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--celltype", required=True); a = ap.parse_args()
    out = OUT / f"{a.celltype}.csv.gz"
    if out.exists(): print(f"[SKIP ] {out}"); return
    t0 = time.time()
    f = h5py.File(ATLAS / f"{a.celltype}_HVPV.h5ad", "r")
    dc, dcat = categorical(f["obs"], "drug"); sc, scat = categorical(f["obs"], "stim"); pc, pcat = categorical(f["obs"], "patient")
    var = [v.decode() if isinstance(v, bytes) else str(v) for v in f["var"]["_index"][:]]
    ok = (dc >= 0) & (sc >= 0) & (pc >= 0)
    rows = np.nonzero(ok)[0]
    key = (sc[rows].astype(np.int64) * 1000 + dc[rows]) * 1000 + pc[rows]
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    rng = np.random.default_rng(0); chosen = []
    order = np.argsort(inv, kind="stable"); bounds = np.r_[0, np.cumsum(cnt)]
    for g in range(len(uk)):
        idx = rows[order[bounds[g]:bounds[g + 1]]]
        chosen.append(np.sort(rng.permutation(idx)[:CAP]))
    allidx = np.sort(np.concatenate(chosen))
    X = np.empty((len(allidx), len(var)), np.float32); N = f["X"].shape[0]
    for s in range(0, N, ROW_CHUNK):
        e = min(s + ROW_CHUNK, N); lo, hi = np.searchsorted(allidx, [s, e])
        if hi > lo: X[lo:hi] = f["X"][s:e][allidx[lo:hi] - s]
    f.close()
    print(f"[READ ] {a.celltype}: {len(rows)} cells, {len(uk)} wells, {len(allidx)} sampled, {time.time() - t0:.0f}s", flush=True)
    recs = []
    for g, k in enumerate(uk):
        x = X[np.searchsorted(allidx, chosen[g])]
        st, dr, pa = scat[k // 1_000_000], dcat[(k // 1000) % 1000], pcat[k % 1000]
        q = np.percentile(x, [10, 50, 90], axis=0); mu = x.mean(0)
        for j, m in enumerate(var):
            recs.append((a.celltype, st, dr, pa, int(cnt[g]), len(x), m, mu[j], q[0, j], q[1, j], q[2, j]))
    T = pd.DataFrame(recs, columns=["celltype", "stim", "treatment", "donor", "n_cells", "n_sampled", "marker",
                                    "mean", "q10", "q50", "q90"])
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"{a.celltype}.{os.getpid()}.tmp.csv.gz"; T.to_csv(tmp, index=False); os.replace(tmp, out)
    print(f"[DONE ] {a.celltype}: {len(uk)} wells x {len(var)} markers, {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
