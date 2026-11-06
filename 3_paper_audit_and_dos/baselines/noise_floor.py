"""The noise floor: how far apart are two control samples that should be identical?

Every number in the benchmark is a distance between a predicted and a measured cell
population, but none of them says how large a distance is *unavoidable*. This does.
The data carries three plate controls per (patient, stim) -- ``DMSO1``, ``DMSO2``,
``DMSO3`` -- which are the same cells under the same condition, differing only by
handling. Scoring one against another with the identity map is exactly the benchmark's
``identity`` row with the biology removed, so its value is pure noise:

  ``across_controls``  DMSO_i vs DMSO_j, same patient(s), same stim. Technical
                       replicate variation -- plate/handling effects *plus* finite-cell
                       sampling.
  ``split_half``       one control split in two at random, scored against itself.
                       Finite-cell sampling alone, at the same cell counts.

The gap between the two is what the plate contributes. If they coincide, there is no
plate effect beyond the sampling noise already present.

Two scopes, because the benchmark reports a pooled number and the question was asked
about a single donor:

  ``pooled``       controls pooled over all 19 patients, which is what ``--eval-mode
                   pooled`` scores, so this is the directly comparable null for the
                   0.005 live/null threshold.
  ``per_patient``  one donor at a time. Far fewer cells (often 40-250), so the floor
                   it reports is much higher -- see the bias note below.

All five metrics are computed by ``metrics.score``, the same function the benchmark
uses, so ``median_error`` here is a noise floor for ``median_error`` there.

**MMD is biased upward at small n.** ``compute_scalar_mmd`` is the biased V-statistic,
so E[MMD] > 0 even when both samples come from one distribution, and the bias grows as
n falls. Cell counts are therefore written into every row: compare a floor only against
benchmark numbers computed at similar n. Where a control has fewer than the
``MMD_SUBSAMPLE`` of 2000 cells that the benchmark subsamples to, the floor is
*overestimated* relative to the benchmark -- i.e. conservative, and the direction is
the safe one for a "this pair is indistinguishable from noise" claim.

Cells are read raw. No batch correction is applied: the drug arm's runtime correction
(cellot/data/cell.py:147-194) shifts each patient's per-marker medians onto a shared
value and so would remove part of the very variation being measured. The files under
``ptb_concatenated_batchcorrected/`` are the same ones the benchmark reads.

Usage (one celltype per process, mirroring the other launchers):

    python -m baselines.noise_floor --celltype pDCs \
        --data-dir  $DATA/ptb_concatenated_batchcorrected \
        --features  $DATA/ptb_concatenated_per_condition_celltype/13features.txt \
        --outdir    $WORK/noise_floor
"""

import argparse
import gc
from itertools import combinations
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from baselines.metrics import score

ROW_CHUNK = 2_000_000
# A floor computed on a handful of cells is dominated by MMD's small-n bias and says
# nothing. 30 is the benchmark's own --min-eval-cells floor of 20 with a margin.
MIN_CELLS = 30
OBS_COLS = ["patient", "stim", "drug"]


def load_dmso(data_dir, celltype, features, stims=None):
    """Every DMSO-prefixed cell of ``celltype``, keeping which control it came from.

    ``run_baselines.load_celltype`` merges DMSO1/2/3 into one source population -- that
    is correct there and useless here, so this keeps the label. Chunked out of the
    backed file into a preallocated float32 array, and the obs frame is cut to three
    columns with its index dropped: Granulocytes_HVPV.h5ad has ~146M rows and a
    non-unique string index, which on its own can exhaust a 32 GB allocation.
    """
    path = Path(data_dir) / f"{celltype}_HVPV.h5ad"
    adata = ad.read_h5ad(path, backed="r")

    mask = adata.obs["drug"].astype(str).str.startswith("DMSO").to_numpy()
    if stims is not None:
        mask &= adata.obs["stim"].astype(str).isin(stims).to_numpy()
    var_idx = np.array([adata.var_names.get_loc(f) for f in features])

    X = np.empty((int(mask.sum()), len(features)), dtype=np.float32)
    at = 0
    for start in range(0, adata.shape[0], ROW_CHUNK):
        stop = min(start + ROW_CHUNK, adata.shape[0])
        block_mask = mask[start:stop]
        n = int(block_mask.sum())
        if not n:
            continue
        block = adata.X[start:stop]
        if sparse.issparse(block):
            block = block.toarray()
        X[at:at + n] = np.asarray(block)[block_mask][:, var_idx]
        at += n
        del block
    adata.file.close()
    assert at == len(X), (at, len(X))

    obs = adata.obs.loc[mask, OBS_COLS].reset_index(drop=True)
    for c in OBS_COLS:
        obs[c] = obs[c].astype(str)
    del adata
    gc.collect()
    return X, obs


def halves(n, rng):
    """A random split of ``n`` rows into two disjoint halves."""
    perm = rng.permutation(n)
    cut = n // 2
    return perm[:cut], perm[cut:]


def rows_for(X, obs, features, celltype, stim, scope, patient, groups, seed):
    """Every comparison for one (stim, scope, patient) cell group.

    ``groups`` maps control label -> row indices into ``X``. Emits one row per ordered
    pair of distinct controls (``across_controls``) and one per control split in two
    (``split_half``); ``score`` is called with the same argument order the benchmark
    uses, ``score(pred, true, markers)``.
    """
    out = []
    usable = {k: v for k, v in groups.items() if len(v) >= MIN_CELLS}

    def emit(pair_type, a_name, b_name, a_idx, b_idx):
        row = {
            "celltype": celltype, "stim": stim, "scope": scope, "patient": patient,
            "pair_type": pair_type, "source": a_name, "target": b_name,
            "n_src_cells": len(a_idx), "n_tgt_cells": len(b_idx),
        }
        row.update(score(X[a_idx].astype(np.float64),
                         X[b_idx].astype(np.float64), features))
        out.append(row)

    for a, b in combinations(sorted(usable), 2):
        emit("across_controls", a, b, usable[a], usable[b])

    # Same seed per (celltype, stim, scope, patient, control): the split is arbitrary but
    # reproducible, and no two groups share a draw.
    for name, idx in usable.items():
        if len(idx) < 2 * MIN_CELLS:
            continue
        rng = np.random.default_rng(abs(hash((celltype, stim, scope, patient, name, seed))) % 2**32)
        lo, hi = halves(len(idx), rng)
        emit("split_half", f"{name}_a", f"{name}_b", idx[lo], idx[hi])

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--celltype", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--stims", default=None,
                   help="comma-separated subset; default is every stim in the file")
    p.add_argument("--scopes", default="pooled,per_patient",
                   help="'pooled' matches --eval-mode pooled; 'per_patient' is one donor")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    features = Path(args.features).read_text().split()
    stims = args.stims.split(",") if args.stims else None
    scopes = args.scopes.split(",")

    X, obs = load_dmso(args.data_dir, args.celltype, features, stims)
    print(f"[LOAD ] {args.celltype}: {len(X):,} DMSO cells, "
          f"{obs.drug.nunique()} controls, {obs.patient.nunique()} patients", flush=True)

    rows = []
    for stim, sidx in obs.groupby("stim", observed=True).indices.items():
        sub = obs.iloc[sidx]
        if "pooled" in scopes:
            groups = {d: sidx[i] for d, i in sub.groupby("drug", observed=True).indices.items()}
            rows += rows_for(X, obs, features, args.celltype, stim, "pooled",
                             "pooled", groups, args.seed)
        if "per_patient" in scopes:
            for patient, pidx in sub.groupby("patient", observed=True).indices.items():
                psub = sub.iloc[pidx]
                groups = {d: sidx[pidx[i]]
                          for d, i in psub.groupby("drug", observed=True).indices.items()}
                rows += rows_for(X, obs, features, args.celltype, stim, "per_patient",
                                 patient, groups, args.seed)
        print(f"[STIM ] {stim}: {len(rows)} rows so far", flush=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{args.celltype}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[WRITE] {out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
