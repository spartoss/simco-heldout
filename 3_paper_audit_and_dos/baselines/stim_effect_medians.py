"""Per-(stim, patient, marker) medians, for testing the paper's pair-selection rule.

The paper states the 168 (stim, celltype) pairs were kept by thresholding a marker
response: a pair survives if at least one of the 13 markers has an arcsinh median
difference between stim and Unstim above 0.1 (0.01 for IL33). This script produces the
raw ingredient -- medians -- so the rule can be replayed against the actual 168/28 split.

Medians are computed over DMSO-prefixed cells only, matching
cellot/data/cell.py:read_single_anndata under ``config.data.drug_used: DMSO``, and over
the 13 functional markers in ``13features.txt`` -- the panel the rule is stated over, and
the one the paper's run actually trains on. The other 25 columns of the h5ad are
phenotypic gating markers plus a few functional ones outside the 13 (GLUT1, pPLCg,
pSTAT4); they are not read.

Cells per (stim, patient) are capped at --cap with a fixed seed. A median over 100k
draws is accurate to far better than the 0.01 threshold being tested; the cap exists so
Granulocytes_HVPV.h5ad (30 GB) fits in memory.
"""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

ROW_CHUNK = 2_000_000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--celltype", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--features", required=True, help="13features.txt")
    p.add_argument("--outdir", required=True)
    p.add_argument("--cap", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    path = Path(args.data_dir) / f"{args.celltype}_HVPV.h5ad"

    markers = Path(args.features).read_text().split()
    adata = ad.read_h5ad(path, backed="r")
    # Same column order training used via data[:, features] in cellot/data/cell.py:63.
    var_idx = np.array([adata.var_names.get_loc(m) for m in markers])
    obs = adata.obs
    keep = obs["drug"].astype(str).str.startswith("DMSO").to_numpy()
    print(f"[INFO] {args.celltype}: {adata.shape[0]:,} rows, {keep.sum():,} DMSO", flush=True)

    # Choose the rows to read up front, so the big file is traversed exactly once.
    rng = np.random.default_rng(args.seed)
    groups, wanted = {}, []
    sub = obs.loc[keep]
    for (stim, patient), idx in sub.groupby(["stim", "patient"], observed=True).indices.items():
        pos = np.flatnonzero(keep)[idx]
        n_total = len(pos)
        if n_total > args.cap:
            pos = rng.choice(pos, size=args.cap, replace=False)
        groups[(str(stim), str(patient))] = (np.sort(pos), n_total)
        wanted.append(pos)

    wanted = np.unique(np.concatenate(wanted))
    print(f"[INFO] {args.celltype}: reading {len(wanted):,} rows in {len(groups)} groups",
          flush=True)

    # Map absolute row position -> position within the materialised matrix.
    where = np.full(adata.shape[0], -1, dtype=np.int64)
    where[wanted] = np.arange(len(wanted))

    blocks = []
    for start in range(0, adata.shape[0], ROW_CHUNK):
        stop = min(start + ROW_CHUNK, adata.shape[0])
        sel = wanted[(wanted >= start) & (wanted < stop)]
        if len(sel) == 0:
            continue
        block = adata.X[start:stop]
        if sparse.issparse(block):
            block = block.toarray()
        blocks.append(np.asarray(block, dtype=np.float32)[sel - start][:, var_idx])
    adata.file.close()

    X = np.concatenate(blocks)
    del blocks
    assert len(X) == len(wanted), (len(X), len(wanted))

    rows = []
    for (stim, patient), (pos, n_total) in groups.items():
        med = np.median(X[where[pos]], axis=0)
        for k, marker in enumerate(markers):
            rows.append({
                "celltype": args.celltype,
                "stim": stim,
                "patient": patient,
                "marker": marker,
                "median": float(med[k]),
                "n_cells": int(n_total),
                "n_used": int(len(pos)),
            })

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{args.celltype}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[WRITE] {out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
