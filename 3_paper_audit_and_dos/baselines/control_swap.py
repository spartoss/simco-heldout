"""Rescore the DMSO1 drugs against the other plate controls.

LPZ and SA are referenced to DMSO1, which under PI sits apart from DMSO2/DMSO3 and does
not co-move with its own drugs (slope 0.92/0.96 on its own control vs ~1.01 on a foreign
one). If their measured effect is the drug, it should be about as large against any
control. If it is the DMSO1 discrepancy, it collapses when scored against DMSO2/DMSO3.

Scored both raw and batch-corrected, since the drug arm scores corrected cells. The
correction is cell.py:147-194's: each (patient, treatment) group's per-marker median is
moved onto its own correctedMedian.
"""
import sys
import numpy as np
import pandas as pd

from baselines.noise_floor import load_dmso  # loads DMSO* cells; extended below
from baselines.metrics import score

DATA = "/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets"
BC = "/scratch/users/bmerwan/simco/cross_validation/batchcorrection_2.csv"
CONTROLS = ["DMSO1", "DMSO2", "DMSO3"]


def load_any(data_dir, celltype, features, stims, treatments):
    """Same chunked read as noise_floor.load_dmso, but for an arbitrary treatment set."""
    import anndata as ad
    from scipy import sparse
    import gc
    from pathlib import Path
    a = ad.read_h5ad(Path(data_dir) / f"{celltype}_HVPV.h5ad", backed="r")
    mask = (a.obs["drug"].astype(str).isin(treatments)
            & a.obs["stim"].astype(str).isin(stims)).to_numpy()
    vi = np.array([a.var_names.get_loc(f) for f in features])
    X = np.empty((int(mask.sum()), len(features)), dtype=np.float32)
    at = 0
    for s in range(0, a.shape[0], 2_000_000):
        e = min(s + 2_000_000, a.shape[0])
        bm = mask[s:e]
        n = int(bm.sum())
        if not n:
            continue
        blk = a.X[s:e]
        blk = blk.toarray() if sparse.issparse(blk) else np.asarray(blk)
        X[at:at + n] = blk[bm][:, vi]
        at += n
        del blk
    a.file.close()
    obs = a.obs.loc[mask, ["patient", "stim", "drug"]].reset_index(drop=True).astype(str)
    del a
    gc.collect()
    return X, obs


def correct(X, obs, lut, pop, stim, features):
    Y = X.copy()
    for (pat, treat), idx in obs.groupby(["patient", "drug"], observed=True).indices.items():
        for j, mk in enumerate(features):
            tgt = lut.get((pat, treat, stim, mk, pop))
            if tgt is None:
                continue
            col = Y[idx, j]
            Y[idx, j] = col + np.float32(tgt - np.median(col))
    return Y


def main():
    celltypes = sys.argv[1:]
    feats = open(f"{DATA}/ptb_concatenated_per_condition_celltype/13features.txt").read().split()
    bc = pd.read_csv(BC)
    lut = {(r.Patient, r.Treatment, r.Stim, r.Marker, r.Population): r.correctedMedian
           for r in bc.itertuples(index=False)}
    drugs = ["LPZ", "SA", "CHT"]          # CHT (a DMSO2 drug) as the behaved comparison
    rows = []
    for ct in celltypes:
        X, obs = load_any(f"{DATA}/ptb_concatenated_batchcorrected", ct, feats,
                          ["PI"], drugs + CONTROLS)
        Xc = correct(X, obs, lut, ct, "PI", feats)
        for label, A in [("raw", X), ("corrected", Xc)]:
            for drug in drugs:
                d = A[(obs.drug == drug).to_numpy()].astype(np.float64)
                for ctrl in CONTROLS:
                    c = A[(obs.drug == ctrl).to_numpy()].astype(np.float64)
                    if len(d) < 50 or len(c) < 50:
                        continue
                    s = score(d, c, feats)
                    rows.append({"celltype": ct, "cells": label, "drug": drug, "control": ctrl,
                                 "own": ctrl == {"LPZ": "DMSO1", "SA": "DMSO1", "CHT": "DMSO2"}[drug],
                                 "mmd": s["mmd"], "median_error": s["median_error"]})
        print(f"[DONE] {ct}", flush=True)
        del X, Xc, obs
    out = pd.DataFrame(rows)
    out.to_csv("/scratch/users/bmerwan/simco_work/baselines/control_swap_PI.csv", index=False)
    print(out.pivot_table(index=["cells", "drug", "celltype"], columns="control",
                          values="mmd").to_string(float_format=lambda v: f"{v:8.4f}"))


if __name__ == "__main__":
    main()
