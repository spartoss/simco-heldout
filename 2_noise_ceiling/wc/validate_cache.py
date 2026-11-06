"""Check that wc.extract's "lc" set is exactly atlas_heldout's learning-curve cache (read-only):
pooled and half quantiles recomputed from work/lc/cache/<celltype>.npz must equal ours.
    python -m wc.validate_cache pDCs CD4Teff
"""
import sys

import numpy as np

from wc.paths import QS, QUANT

LC_CACHE = "/scratch/users/bmerwan/atlas_heldout/work/lc/cache"

for ct in sys.argv[1:]:
    ours = np.load(QUANT / f"{ct}.npz"); theirs = np.load(f"{LC_CACHE}/{ct}.npz")
    worst, n = 0.0, 0
    for k in theirs.files:
        x = theirs[k]; h = len(x) // 2
        for kind, ref in (("pool", np.percentile(x, QS, 0)), ("h1", np.percentile(x[:h], QS, 0)),
                          ("h2", np.percentile(x[h:], QS, 0))):
            worst = max(worst, float(np.abs(ours[f"{k}|{kind}"] - ref).max())); n += 1
        assert int(ours[f"{k}|pool|n"]) == len(x)
    print(f"{ct}: {len(theirs.files)} blocks, {n} quantile tables, max |diff| = {worst:.2e}")
