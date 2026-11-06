"""One-time cache of the 177 OOL women's unstimulated cells (13 functional markers), one .npz per woman, keyed by the
sanitised celltype name. Lets the GPU environment (no anndata / h5py) build features. Run with env_cpu.sh.
    python -m heldout.ool_cache  -> WORK/cache_ool/<woman>.npz
"""
import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import numpy as np  # noqa: E402

from heldout.common import WORK  # noqa: E402
from heldout.ool_features import OOL_LABEL, PATIENTS, patient_cells  # noqa: E402

if __name__ == "__main__":
    out = WORK / "cache_ool"; out.mkdir(parents=True, exist_ok=True)
    inv = {v: k for k, v in OOL_LABEL.items()}
    for n, p in enumerate(PATIENTS, 1):
        f = out / f"{p}.npz"
        if f.exists(): continue
        X, lab = patient_cells(p)
        np.savez(out / f"{p}.tmp.npz", **{inv.get(c, c): X[lab == c] for c in np.unique(lab)})
        os.replace(out / f"{p}.tmp.npz", f)
        if n % 20 == 0: print(f"[INFO ] {n}/{len(PATIENTS)}", flush=True)
    print("[DONE ]", len(list(out.glob("*.npz"))), "women")
