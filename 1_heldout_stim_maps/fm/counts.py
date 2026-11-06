"""Unstim DMSO cell counts per (celltype, donor) from the atlas h5ad obs (the caches are capped, so they cannot give
frequencies). h5py on obs category codes only, as lc/cache.py. Output WORK/fm/counts.csv (celltype, donor, n, freq);
freq = share of the donor's Unstim DMSO cells over the 28 celltypes.
    python -m fm.counts            (env_cpu.sh)
"""
import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lc.cache import categorical  # noqa: E402
from lc.paths import ATLAS, CELLTYPES, DONORS, WORK  # noqa: E402


def main():
    rows = []
    for c in CELLTYPES:
        with h5py.File(ATLAS / f"{c}_HVPV.h5ad", "r") as f:
            dc, dcat = categorical(f["obs"], "drug"); sc, scat = categorical(f["obs"], "stim")
            pc, pcat = categorical(f["obs"], "patient")
        ok = np.array([x.startswith("DMSO") for x in dcat])[dc] & (scat[sc] == "Unstim") & (dc >= 0) & (sc >= 0) & (pc >= 0)
        n = np.bincount(pc[ok], minlength=len(pcat))
        rows += [(c, d, int(n[list(pcat).index(d)]) if d in pcat else 0) for d in DONORS]
        print(f"[DONE ] {c}", flush=True)
    D = pd.DataFrame(rows, columns=["celltype", "donor", "n"])
    D["freq"] = D.n / D.groupby("donor").n.transform("sum")
    out = WORK / "fm"; out.mkdir(parents=True, exist_ok=True)
    D.to_csv(out / "counts.csv", index=False)
    print(D.pivot(index="celltype", columns="donor", values="freq").round(3).to_string())


if __name__ == "__main__":
    main()
