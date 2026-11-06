"""Merge per-celltype CondOT outputs (lc/condot_train.py --mode celltype) into one lc.train-format dir per repeat.
    python -m lc.condot_merge --indir WORK/condot/celltype/repeat_0 --out WORK/condot/celltype_merged/repeat_0
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--indir", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dirs = sorted(d for d in Path(a.indir).iterdir() if (d / "pred_q.npy").exists())
    M = [pd.read_csv(d / "maps.csv") for d in dirs]; P = [np.load(d / "pred_q.npy") for d in dirs]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pd.concat(M, ignore_index=True).drop(columns="map").to_csv(out / "maps.csv", index_label="map")
    np.save(out / "pred_q.npy", np.concatenate(P, axis=1))
    print(f"{len(dirs)} celltypes, {sum(len(m) for m in M)} maps -> {out}")


if __name__ == "__main__":
    main()
