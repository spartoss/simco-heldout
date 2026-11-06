"""Per-donor drug-screen cells, batch-corrected as the paper's drug arm, for one celltype.

For every (stim or Unstim, treatment in the 9 drugs + DMSO1/2/3, donor): the 13 functional markers. Each
(patient, population, treatment, stim, marker) group is shifted so its median equals `correctedMedian` of
batchcorrection_2.csv, which is what cellot/data/cell.py:147-194 does at load time for the drug runs (logic copied
from simco/baselines/drug_data.py:batch_correct; the correction of a control plate does not depend on the drug, so
each treatment is corrected once). Medians are taken on all cells, then each block is capped at CAP cells (random).
Stims: Unstim plus the stims of the 168-pair grid for this celltype (the ones with drug maps).
Also writes the plate control of every drug (use_DMSO). Reads the h5ad with h5py (see lc/cache.py).
    python -m heldout.drug_cache --celltype pDCs
"""
import argparse
import json
import os
import time

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from heldout.common import BC, CACHE_DRUG, CONTROLS, DRUGS  # noqa: E402
from lc.cache import categorical  # noqa: E402
from lc.paths import ATLAS, MARKERS, PAIRS  # noqa: E402

CAP = 5_000
ROW_CHUNK = 2_000_000


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--celltype", required=True); a = ap.parse_args()
    out = CACHE_DRUG / f"{a.celltype}.npz"
    if out.exists(): print(f"[SKIP ] {out}"); return
    t0 = time.time()
    conds = ["Unstim"] + sorted({s for s, c in PAIRS if c == a.celltype})
    treats = DRUGS + CONTROLS
    f = h5py.File(ATLAS / f"{a.celltype}_HVPV.h5ad", "r")
    dc, dcat = categorical(f["obs"], "drug"); sc, scat = categorical(f["obs"], "stim")
    pc, pcat = categorical(f["obs"], "patient"); cc, ccat = categorical(f["obs"], "cell_type")
    var = [v.decode() if isinstance(v, bytes) else str(v) for v in f["var"]["_index"][:]]
    vidx = np.array([var.index(m) for m in MARKERS])
    keep = np.isin(dcat, treats)[dc] & np.isin(scat, conds)[sc] & (pc >= 0) & (cc >= 0)
    rows = np.nonzero(keep)[0]
    X = np.empty((len(rows), len(MARKERS)), np.float32); N = f["X"].shape[0]
    for s in range(0, N, ROW_CHUNK):
        e = min(s + ROW_CHUNK, N); lo, hi = np.searchsorted(rows, [s, e])
        if hi > lo: X[lo:hi] = f["X"][s:e][rows[lo:hi] - s][:, vidx]
    f.close()
    st, tr, pa, po = sc[rows], dc[rows], pc[rows], cc[rows]
    print(f"[READ ] {a.celltype}: {len(rows)} cells, {time.time() - t0:.0f}s", flush=True)

    bc = pd.read_csv(BC)
    pops = set(ccat)
    bc = bc[bc.Population.isin(pops) & bc.Stim.isin(conds) & bc.Treatment.isin(treats)]
    table = {(r.Patient, r.Treatment, r.Stim, r.Marker, r.Population): r.correctedMedian for r in bc.itertuples(index=False)}
    plate = bc.groupby("Treatment").use_DMSO.agg(lambda v: v.unique()[0] if v.nunique() == 1 else None).to_dict()

    key = ((st.astype(np.int64) * 1000 + tr) * 1000 + pa) * 1000 + po
    order = np.argsort(key, kind="stable"); ks = key[order]
    bounds = np.flatnonzero(np.diff(ks)) + 1
    n_shift = 0
    for grp in np.split(order, bounds):
        k0 = key[grp[0]]
        s_, t_, p_, o_ = scat[st[grp[0]]], dcat[tr[grp[0]]], pcat[pa[grp[0]]], ccat[po[grp[0]]]
        med = np.median(X[grp], axis=0)
        for j, m in enumerate(MARKERS):
            v = table.get((p_, t_, s_, m, o_))
            if v is not None:
                X[grp, j] += np.float32(v - med[j]); n_shift += 1
    rng = np.random.default_rng(0)
    arrs = {}
    key2 = (st.astype(np.int64) * 1000 + tr) * 1000 + pa
    for k2 in np.unique(key2):
        idx = np.flatnonzero(key2 == k2)
        s_, t_, p_ = scat[k2 // 1_000_000], dcat[(k2 // 1000) % 1000], pcat[k2 % 1000]
        arrs[f"{s_}|{t_}|{p_}"] = X[rng.permutation(idx)[:CAP]]
    CACHE_DRUG.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_DRUG / f"{a.celltype}.{os.getpid()}.tmp.npz"; np.savez(tmp, **arrs); os.replace(tmp, out)
    json.dump({k: v for k, v in plate.items() if k in DRUGS}, open(CACHE_DRUG / "plate_controls.json", "w"), indent=1)
    n = [len(v) for v in arrs.values()]
    print(f"[DONE ] {a.celltype}: {len(arrs)} blocks (median {int(np.median(n))} cells), {n_shift} group-marker shifts, "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
