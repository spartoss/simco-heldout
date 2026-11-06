"""Quantile tables of the drug arm (batch-corrected work/heldout/cache_drug, as the paper's drug arm), for lc/drug_tab.py.

For every block "<stim>|<treatment>|<donor>" with >= 20 cells: the q10..q90 quantiles of the 13 functional markers and
the split-half difference |Q(first half) - Q(second half)|. Treatments: the 9 drugs and DMSO1/2/3; each drug's control
is its plate DMSO (plate_controls.json). Writes WORK/drug_tab/tables.pkl: {(celltype, stim, treatment, donor): (Q, half)}
and plate_controls.
    python -m lc.drug_tables
"""
import json
import pickle
from pathlib import Path

import numpy as np

from lc.paths import QS, WORK

CACHE_DRUG = WORK.parent / "heldout" / "cache_drug"
OUT = WORK / "drug_tab"
MIN_CELLS = 20


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    T = {}
    for f in sorted(CACHE_DRUG.glob("*.npz")):
        ct = f.stem; z = np.load(f)
        for k in z.files:
            s, tr, d = k.split("|"); x = z[k]
            if len(x) < MIN_CELLS: continue
            h = len(x) // 2
            T[(ct, s, tr, d)] = (np.percentile(x, QS, 0).astype(np.float32),
                                 np.abs(np.percentile(x[:h], QS, 0) - np.percentile(x[h:], QS, 0)).astype(np.float32))
        print(f"[TAB  ] {ct}", flush=True)
    plates = json.load(open(CACHE_DRUG / "plate_controls.json"))
    pickle.dump({"T": T, "plates": plates}, open(OUT / "tables.pkl", "wb"))
    print(f"[DONE ] {len(T)} blocks -> {OUT / 'tables.pkl'}")


if __name__ == "__main__":
    main()
