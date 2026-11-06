"""Pooled-plate cache subsampled to the cell counts of a single-plate cache (notebooks/dmso_plate.ipynb, 2026-10-06).

Each block "<condition>|<donor>" of --src (default WORK/cache: DMSO1/2/3 pooled, shuffled) is cut to its first n rows,
n = the size of the same block in --like (e.g. work/lc/dmso/cache_DMSO1). The source cache is shuffled, so the prefix
is a random subsample: same plates mixed as the pooled cache, same number of cells as the single-plate cache. This
separates "fewer cells" (pooled -> pooled_n) from "one plate" (pooled_n -> DMSO1). Blocks absent from --like are dropped.
    python -m lc.match_cache --celltype pDCs --like work/lc/dmso/cache_DMSO1 --out work/lc/dmso/cache_poolN1
"""
import argparse
import os
from pathlib import Path

import numpy as np

from lc.paths import CACHE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--celltype", required=True)
    ap.add_argument("--like", required=True)
    ap.add_argument("--src", default=str(CACHE))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out) / f"{a.celltype}.npz"
    if out.exists(): print(f"[SKIP ] {out}"); return
    src, like = np.load(Path(a.src) / f"{a.celltype}.npz"), np.load(Path(a.like) / f"{a.celltype}.npz")
    arrs = {k: src[k][:len(like[k])] for k in like.files if k in src.files}
    short = [k for k in arrs if len(arrs[k]) < len(like[k])]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / f"{a.celltype}.{os.getpid()}.tmp.npz"
    np.savez(tmp, **arrs); os.replace(tmp, out)
    print(f"[DONE ] {a.celltype}: {len(arrs)} blocks, {len(short)} shorter than --like, "
          f"{sum(len(v) for v in arrs.values())} cells -> {out}", flush=True)


if __name__ == "__main__":
    main()
