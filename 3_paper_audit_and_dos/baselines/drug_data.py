"""Loading and batch-correcting the drug arm, shared by the benchmark and the leak probe.

The drug run differs from the stim run in four ways that all live here, and every one of
them has to be reproduced before a cell is comparable to what CellOT saw:

  * **Source is the drug's own DMSO plate control**, not ``Unstim`` -- DMSO1/2/3 per
    ``batchcorrection_2.csv``'s ``use_DMSO`` column, which matches every dumped config.
    The map is DMSO -> drug *within* one stim, so the frame is restricted to that stim
    first (``condition: drug``, ``target: <drug>``, ``stim: <stim>``).
  * **Cells are batch-corrected at load time** (``data.batch_correction``,
    cellot/data/cell.py:147-194), a per-(patient, population, marker) median shift. The
    stim arm's paper run has no such step. Baselines must be fit on the corrected values
    or they are not seeing CellOT's training data.
  * **The 80/20 split is grouped by ``drug``**, not by ``stim``.
  * ``data.metric`` is absent from all 1512 configs, so ``metric_type`` falls back to mmd
    (train.py:190-193): the drug arm selects checkpoints on MMD, not on median_error.
    Nothing here depends on that, but it is why a drug-arm table reads differently from
    the stim one -- CellOT is selected on a metric it actually wins.

``iter_drug_units`` is the entry point: it yields one unit per (stim, drug) keyed the
same way the stim arm keys its cells, ``(patient, condition)``, so the scoring body in
run_baselines is shared between the arms rather than forked.

One difference from selection_leak.py, which reaches the same cells by a shorter route
and is a useful cross-check (verified: identical to machine precision on 13 of 14 pDCs
pairs). run_baselines pools only patients holding test cells on *both* sides, because a
patient present on one side only would tilt the pooled comparison; selection_leak takes
every test row of the pair. The two agree except where some patient has test cells for
the control but none for the drug -- pDCs/LPS/MF, 359 source cells here against 363
there. Keeping the filter is what makes a drug table comparable to the published stim
one, which was produced by this same code path.
"""

import gc
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split

ROW_CHUNK = 2_000_000


def split_positions(n, test_size, seed):
    """Positional test mask for one group, reproducing split_cell_data_train_test.

    sklearn shuffles positions with ``check_random_state(seed).permutation(n)``, so
    splitting an integer range reproduces exactly what splitting the obs_names Index
    did -- the labels never enter the draw.
    """
    if n < 2:
        return np.zeros(n, dtype=bool)
    is_test = np.zeros(n, dtype=bool)
    _, test_idx = train_test_split(np.arange(n), random_state=seed, test_size=test_size)
    is_test[test_idx] = True
    return is_test


# The only obs columns anything downstream reads. Granulocytes_HVPV.h5ad has ~146M rows
# and a non-unique *string* index, so a bare ``adata.obs.copy()`` materialises tens of
# millions of Python strings plus every unused column -- on its own enough to lose a
# 32 GB allocation before a single cell is read.
OBS_COLS = ["stim", "drug", "patient", "cell_type"]


def load_drug_stims(data_dir, celltype, features, stims):
    """Every cell of ``celltype`` in any of ``stims``, all drugs, 13 markers.

    One chunked pass, not one per stim: the read is a full scan whatever the filter, so
    per-stim passes re-scan the file. Held as float32 and preallocated from an obs-only
    first pass -- ``np.concatenate`` over a block list would transiently hold the result
    twice, which on Granulocytes is the difference between fitting in 32 GB and not.

    Pass one stim at a time (see ``iter_drug_units(per_stim=True)``) when even the
    single-pass form does not fit: the row count drops by the number of stims, at the
    cost of re-scanning.
    """
    path = Path(data_dir) / f"{celltype}_HVPV.h5ad"
    adata = ad.read_h5ad(path, backed="r")
    # Not adata.obs.copy(): take the mask off the live frame, then keep only the needed
    # columns of the needed rows, and drop the string index that is never read.
    mask = adata.obs["stim"].astype(str).isin(stims).to_numpy()
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
    # Positional, so the dropped index cannot change any grouping: every consumer works
    # through ``groupby(...).indices`` (integer positions) or a boolean mask.
    obs = adata.obs.loc[mask, OBS_COLS].reset_index(drop=True)
    del adata
    gc.collect()
    return X, obs


def batch_correct(X, obs, bc, drug, dmso, stim, features):
    """Apply cellot/data/cell.py:147-194's per-patient median shift, in place.

    For each (patient, population, marker) the drug rows and the DMSO rows are each
    shifted so their median for that marker equals ``correctedMedian``. Only the
    ``drug_used != 'DMSO'`` branch is reproduced, which is the one the drug run takes
    -- and the one that actually writes back. The ``drug_used == 'DMSO'`` branch at
    :138-139 assigns through ``data[mask, marker].X += shift``, a view, so it silently
    does nothing; it is unreachable here and is not reimplemented.

    Rows are already restricted to ``stim``, so the original's mask_stim is implied.
    Group row indices are taken once rather than rebuilding a boolean mask per marker:
    at 4M rows for a Granulocytes drug pair the mask form is the whole runtime.
    """
    groups = obs.groupby(
        [obs["patient"].astype(str), obs["cell_type"].astype(str),
         obs["drug"].astype(str)], observed=True, sort=False
    ).indices
    n_shift = 0
    for (patient, population, treat), idx in groups.items():
        for j, marker in enumerate(features):
            # The original reads the drug's own row first and, if its use_DMSO
            # disagrees with the drug's plate control, skips the marker entirely --
            # both the drug shift and the DMSO one. It never fires on
            # batchcorrection_2.csv, where use_DMSO is unique per treatment, but the
            # branch is kept so the reproduction is faithful.
            drug_hit = bc.get((patient, drug, stim, marker, population))
            if drug_hit is not None and drug_hit[1] != dmso:
                continue
            hit = drug_hit if treat == drug else bc.get(
                (patient, dmso, stim, marker, population))
            if hit is None:
                continue
            col = X[idx, j]
            X[idx, j] = col + np.float32(hit[0] - np.median(col))
            n_shift += 1
    return n_shift


def load_bc(path, celltype_pops, stims):
    """(Patient, Treatment, Stim, Marker, Population) -> (correctedMedian, use_DMSO).

    Pre-filtered to the populations and stims this worker touches: the full table is
    1.94M rows and a per-marker pandas mask over it would dominate the runtime.
    """
    df = pd.read_csv(path)
    df = df[df["Population"].isin(celltype_pops) & df["Stim"].isin(stims)]
    return {
        (r.Patient, r.Treatment, r.Stim, r.Marker, r.Population):
            (r.correctedMedian, r.use_DMSO)
        for r in df.itertuples(index=False)
    }


def dmso_for(bc_path, drug, _cache={}):
    """The drug's plate control, read from the table exactly as cell.py:71-76 does."""
    if not _cache:
        df = pd.read_csv(bc_path, usecols=["Treatment", "use_DMSO"]).drop_duplicates()
        for t, grp in df.groupby("Treatment"):
            vals = grp["use_DMSO"].unique()
            _cache[t] = vals[0] if len(vals) == 1 else None
    return _cache.get(drug)


def list_drugs(results, drugs=None):
    """Drug subdirs of the results root, or the requested subset, in a fixed order."""
    if drugs:
        return drugs.split(",")
    return sorted(d for d in os.listdir(results)
                  if os.path.isdir(os.path.join(results, d)))


def model_dir_for(results, drug, stim, celltype):
    return os.path.join(results, drug, stim, celltype, f"model-{stim}_{celltype}")


def pair_frame(X, obs, bc, drug, dmso, stim, features, test_size, split_seed):
    """Batch-corrected cells of one (stim, drug) pair, plus the training-time test mask.

    Returns ``(Xd, obsd, is_test)``. The frame is the drug's rows plus its DMSO control's,
    in file order -- the same rows, in the same order, that read_single_anndata handed to
    split_cell_data for this pair.
    """
    drug_col = obs["drug"].astype(str).to_numpy()
    sel = np.isin(drug_col, [drug, dmso])
    if not sel.any():
        return None, None, None
    # Copy: batch_correct writes in place and each drug re-corrects its own DMSO rows
    # from the raw values.
    Xd, obsd = X[sel].copy(), obs.loc[sel]
    batch_correct(Xd, obsd, bc, drug, dmso, stim, features)

    # split_cell_data_train_test(groupby='drug', random_state=0, test_size=0.2) over the
    # frame in file order, after the stim and transport filters.
    d_col = obsd["drug"].astype(str).to_numpy()
    is_test = np.zeros(len(Xd), dtype=bool)
    for value in (drug, dmso):
        pos = np.flatnonzero(d_col == value)
        is_test[pos[split_positions(len(pos), test_size, split_seed)]] = True
    return Xd, obsd, is_test


def iter_drug_units(data_dir, celltype, features, stims, results, bc_path,
                    drugs=None, test_size=0.2, split_seed=0, per_stim=False):
    """Yield one ready-to-score unit per (stim, drug) that has a checkpoint.

    Each unit carries per-(patient, treatment) cell matrices and their test masks, keyed
    exactly as the stim arm keys its own -- ``(patient, condition)``, with condition being
    the DMSO control or the drug -- so the caller's scoring loop does not branch on arm.

    By default the whole celltype is read once, up front; within it the per-stim slice is
    a view and only the per-pair frame is copied, because batch correction has to write.

    ``per_stim`` re-reads the file once per stim instead, holding only one stim's cells at
    a time. That is ~7x less resident memory for ~7x the I/O, and it is what Granulocytes
    needs: at ~146M rows the single-pass form peaks over 23 GB and is OOM-killed in a
    32 GB allocation. Results are identical either way -- the split, the batch correction
    and the grouping are all computed within a (stim, drug) pair, which neither form
    splits across reads.
    """
    drugs = list_drugs(results, drugs)
    if per_stim:
        for stim in stims:
            yield from iter_drug_units(data_dir, celltype, features, [stim], results,
                                       bc_path, drugs=",".join(drugs),
                                       test_size=test_size, split_seed=split_seed)
        return

    Xall, obsall = load_drug_stims(data_dir, celltype, features, stims)
    bc = load_bc(bc_path, pd.unique(obsall["cell_type"].astype(str)), stims)
    stim_col = obsall["stim"].astype(str).to_numpy()
    print(f"[INFO] {celltype}: {len(Xall)} cells over stims {stims}, drugs "
          f"{sorted(pd.unique(obsall['drug'].astype(str)))}", flush=True)

    try:
        for stim in stims:
            in_stim = stim_col == stim
            X, obs = Xall[in_stim], obsall.loc[in_stim]
            for drug in drugs:
                model_dir = model_dir_for(results, drug, stim, celltype)
                if not os.path.exists(os.path.join(model_dir, "cache", "model.pt")):
                    print(f"[WARN] no checkpoint for {drug}/{stim}/{celltype}", flush=True)
                    continue
                dmso = dmso_for(bc_path, drug)
                if dmso is None:
                    print(f"[WARN] no unique use_DMSO for {drug}", flush=True)
                    continue

                Xd, obsd, is_test = pair_frame(X, obs, bc, drug, dmso, stim, features,
                                               test_size, split_seed)
                if Xd is None:
                    continue

                keys = pd.DataFrame({
                    "patient": obsd["patient"].astype(str).to_numpy(),
                    "treat": obsd["drug"].astype(str).to_numpy(),
                })
                cells, test_mask = {}, {}
                for (patient, treat), idx in keys.groupby(
                        ["patient", "treat"], observed=True, sort=False).indices.items():
                    cells[(patient, treat)] = Xd[idx].astype(np.float64)
                    test_mask[(patient, treat)] = is_test[idx]
                del Xd, obsd, is_test
                gc.collect()

                yield {
                    "stim": stim, "drug": drug, "source": dmso, "target": drug,
                    "cells": cells, "test_mask": test_mask,
                    "patients": sorted({p for (p, _) in cells}),
                    "model_dir": model_dir,
                }
                del cells, test_mask
                gc.collect()
            del X, obs
            gc.collect()
    finally:
        del Xall, obsall
        gc.collect()
