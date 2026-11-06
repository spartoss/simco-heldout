"""Learning-curve design: per repeat, 4 random test donors and a random order of the other 15; train on the first k.
    python -m lc.design  -> WORK/design.json
"""
import json

import numpy as np

from lc.paths import DONORS, WORK

R = 5
N_TEST = 4
KS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15]


def main():
    out = []
    for r in range(R):
        rng = np.random.default_rng(1000 + r)
        perm = [DONORS[i] for i in rng.permutation(len(DONORS))]
        out.append({"repeat": r, "test": sorted(perm[:N_TEST]), "order": perm[N_TEST:], "ks": KS})
    WORK.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(WORK / "design.json", "w"), indent=1)
    for d in out:
        print(d["repeat"], "test", d["test"], "order", " ".join(d["order"]))


if __name__ == "__main__":
    main()
