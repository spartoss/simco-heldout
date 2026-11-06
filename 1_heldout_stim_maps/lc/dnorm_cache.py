"""Donor-normalised copy of the learning-curve cell cache (2026-10-06, the per-donor normalisation experiment).

Every cell of donor d in celltype c (Unstim and every stim) is shifted by minus m(c, d) = the median of d's cached
Unstim cells in c, per marker. A map trained on these cells learns the response *around each donor's own baseline*,
so it cannot turn a donor's baseline offset into a response (the pooled-OT stretch found in
notebooks/where_cellot_skill.ipynb). Prediction for a test donor = Q(T(her Unstim cells - m)) + m, which uses only
her own Unstim cells (an input). Row order is unchanged, so the train / val split of lc.train.Cells is the same.
Writes WORK/cache_dnorm/<celltype>.npz and WORK/cache_dnorm/offsets.npz (keys "<celltype>|<donor>" -> (13,)).
    python -m lc.dnorm_cache
"""
import numpy as np

from lc.paths import CACHE, PAIRS, WORK

OUT = WORK / "cache_dnorm"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    offsets = {}
    for c in sorted({c for _, c in PAIRS}):
        z = np.load(CACHE / f"{c}.npz")
        med = {k.split("|")[1]: np.median(z[k], 0) for k in z.files if k.startswith("Unstim|")}
        out = {}
        for k in z.files:
            d = k.split("|")[1]
            if d not in med: continue                      # no Unstim for this donor: cannot normalise, drop
            out[k] = (z[k] - med[d]).astype(np.float32)
        np.savez(OUT / f"{c}.npz", **out)
        offsets.update({f"{c}|{d}": m.astype(np.float32) for d, m in med.items()})
        print(c, len(out), "blocks", flush=True)
    np.savez(OUT / "offsets.npz", **offsets)


if __name__ == "__main__":
    main()
