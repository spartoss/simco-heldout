"""Merge the per-k runs of one arm and repeat (work/runs/<arm>/repeat_<r>/k<k>/) into work/runs/<arm>/repeat_<r>/
maps.csv + pred_q.npy, the layout lc.ctx_score (atlas_heldout, read-only) scores. Missing ks are skipped.
    python3 -m mfm.merge --arms fm mfm mfm_blood
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path(__file__).resolve().parents[1] / "work" / "runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--repeats", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--ks", nargs="*", type=int, default=[3, 5, 8, 15])
    a = ap.parse_args()
    for arm in a.arms:
        for r in a.repeats:
            d = RUNS / arm / f"repeat_{r}"
            parts = [(pd.read_csv(d / f"k{k}" / "maps.csv"), np.load(d / f"k{k}" / "pred_q.npy"))
                     for k in a.ks if (d / f"k{k}" / "pred_q.npy").exists()]
            if not parts: continue
            M = pd.concat([m for m, _ in parts], ignore_index=True).drop(columns="map")
            M.to_csv(d / "maps.csv", index_label="map")
            np.save(d / "pred_q.npy", np.concatenate([p for _, p in parts], axis=1))
            print(f"{arm} repeat {r}: {len(parts)} ks, {len(M)} maps")


if __name__ == "__main__":
    main()
