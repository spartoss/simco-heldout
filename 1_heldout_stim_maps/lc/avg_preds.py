"""Average the predicted residuals (rhat.npy) of several lc/tab_v2.py runs with the same maps, e.g. TabPFN checkpoints.
    python -m lc.avg_preds --names v2_a v2_b v2_c --out v2_avg --repeats 0 1 ...   -> WORK/tab/<out>/repeat_<r>/
"""
import argparse

import numpy as np
import pandas as pd

from lc.paths import WORK

TAB = WORK / "tab"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", nargs="+", type=int, required=True)
    a = ap.parse_args()
    for r in a.repeats:
        ms = [pd.read_csv(TAB / n / f"repeat_{r}" / "maps.csv") for n in a.names]
        assert all(m.equals(ms[0]) for m in ms), f"maps differ in repeat {r}"
        R = np.mean([np.load(TAB / n / f"repeat_{r}" / "rhat.npy") for n in a.names], 0)
        o = TAB / a.out / f"repeat_{r}"; o.mkdir(parents=True, exist_ok=True)
        ms[0].to_csv(o / "maps.csv", index=False); np.save(o / "rhat.npy", R)
    print(f"[DONE ] mean of {a.names} -> {TAB / a.out}")


if __name__ == "__main__":
    main()
