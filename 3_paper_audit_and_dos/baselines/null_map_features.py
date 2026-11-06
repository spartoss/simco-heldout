"""Stabl feature tables from *null* maps -- is Fig 5b about the perturbation, or about nonlinearity?

The Fig 5b features ``<celltype>_<marker>_<stim>`` are per-marker medians of each OOL
patient's Unstim cells pushed through the trained CellOT map for (stim, celltype). Measured
Unstim alone gives R2 ~ 0; the simulated block gives ~0.40. Two explanations:

  (a) the map models the stim response, and that response carries the signal;
  (b) any nonlinear map of the patient's joint cell distribution gives Stabl new, more
      separable views of the same Unstim data -- the perturbation itself is incidental.

Each ``--map-kind`` removes one ingredient of the trained map, keeping the rest:

  cellot            the paper's maps (``model.pt``). Must reproduce the paper's table --
                    run this first, it is the pipeline check.
  cellot_permuted   trained maps, but each stim column is filled by *another* stim's map of
                    the same celltype (a seeded derangement). Real training, real
                    architecture, real magnitude -- wrong perturbation.
  cellot_ctswap     trained map of the *same stim* but a *different celltype* (seeded
                    derangement over the celltypes that have that stim): right stimulation,
                    learned on the wrong cells. Values are new -- not a relabelling.
  cellot_crossswap  trained map of a *different stim AND a different celltype*: a real,
                    trained, correctly-sized map with the wrong perturbation.
                    (``cellot_permuted`` swaps stims within a celltype, so most of its columns
                    are copies of real columns -- it is NOT a valid wrong-perturbation control.)
  random_icnn       same ICNN architecture and init (``load_networks(config)``, no
                    ``load_state_dict``), seeded per pair. Random nonlinearity.
                    With ``--match-scale``: T(x) = x + s * (grad g(x) - x), s per pair set so
                    the mean displacement ||T(x) - x|| on a reference set of OOL Unstim cells
                    equals the trained map's -- random nonlinearity "as large as" CellOT's.

Stabl standardises every column, so a constant per-(pair, marker) shift or rescale of a
feature is invisible downstream; only the nonlinear part applied *before* the median matters.
That is why scale is matched on displacement in cell space, not on feature moments.

Usage (small enough for an interactive smoke test with --patients-limit):
    python -m baselines.null_map_features --map-kind cellot --out <csv> --like <ref.csv>
"""
from __future__ import annotations

import argparse
import gc
import os
import time

# Inputs are read-only on the group-home NFS share, where HDF5's file lock intermittently
# fails with "Remote I/O error" (errno 121); locking buys nothing for reads.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from baselines.simco_features import (CELLTYPE_NAME_MAP, OOL, PERCOND, read_list,
                                      valid_pairs)

RESULTS = Path("/home/groups/gbrice/ptb-drugscreen/ot/cellot/results/"
               "cross_validation_original_13_HVPV")
CHUNK = 100_000


def load_g(stim, ct, trained, seed=None):
    from cellot.models.cellot import load_networks
    from cellot.utils.helpers import load_config

    d = RESULTS / stim / ct / f"model-{stim}_{ct}"
    config = load_config(d / "config.yaml")
    if seed is not None:
        torch.manual_seed(seed)
    _, g = load_networks(config, input_dim=13)
    if trained:
        ckpt = torch.load(d / "cache/model.pt", map_location="cpu")
        g.load_state_dict(ckpt["g_state"])
    g.eval()
    for p in g.parameters():
        p.requires_grad_(False)
    return g


def transport(g, x):
    out = []
    for i in range(0, len(x), CHUNK):
        xb = torch.from_numpy(x[i:i + CHUNK]).float().requires_grad_(True)
        out.append(g.transport(xb).detach().numpy().astype(np.float64))
    return np.concatenate(out) if out else np.empty((0, x.shape[1]))


def derangement(items, rng):
    items = list(items)
    while True:
        perm = list(rng.permutation(items))
        if all(a != b for a, b in zip(items, perm)):
            return dict(zip(items, perm))


def build_maps(args, pairs):
    """{(stim, ct): (g, scale)} -- the network that fills the column, and its scale (None = raw)."""
    by_ct = {}
    for stim, ct in pairs:
        by_ct.setdefault(ct, []).append(stim)
    maps = {}
    if args.map_kind == "cellot":
        for stim, ct in pairs:
            maps[(stim, ct)] = (load_g(stim, ct, trained=True), None)
    elif args.map_kind == "cellot_permuted":
        rng = np.random.default_rng(args.seed)
        for ct, stims in sorted(by_ct.items()):
            perm = derangement(sorted(stims), rng)
            for stim in stims:
                maps[(stim, ct)] = (load_g(perm[stim], ct, trained=True), None)
                print(f"[PERM ] {ct}: {stim} column <- {perm[stim]} map")
    elif args.map_kind == "cellot_ctswap":
        rng = np.random.default_rng(args.seed)
        by_stim = {}
        for stim, ct in pairs:
            by_stim.setdefault(stim, []).append(ct)
        for stim, cts in sorted(by_stim.items()):
            if len(cts) < 2:
                raise ValueError(f"{stim}: one celltype, cannot swap")
            perm = derangement(sorted(cts), rng)
            for ct in cts:
                maps[(stim, ct)] = (load_g(stim, perm[ct], trained=True), None)
                print(f"[SWAP ] {stim}/{ct} column <- {stim}/{perm[ct]} map")
    elif args.map_kind == "cellot_crossswap":
        rng = np.random.default_rng(args.seed)
        allp = sorted(valid_pairs())
        for stim, ct in sorted(pairs):
            cand = [(s2, c2) for s2, c2 in allp if s2 != stim and c2 != ct]
            s2, c2 = cand[rng.integers(len(cand))]
            maps[(stim, ct)] = (load_g(s2, c2, trained=True), None)
            print(f"[SWAP ] {stim}/{ct} column <- {s2}/{c2} map")
    elif args.map_kind == "random_icnn":
        for k, (stim, ct) in enumerate(sorted(pairs)):
            maps[(stim, ct)] = (load_g(stim, ct, trained=False,
                                       seed=args.seed * 100_003 + k), None)
    return maps


def patient_cells(patient, features, tries=5):
    for k in range(tries):
        try:
            a = ad.read_h5ad(OOL / f"{patient}_Unstim.h5ad")
            break
        except OSError as e:
            if k == tries - 1:
                raise
            print(f"[RETRY] {patient}: {e}", flush=True)
            time.sleep(10 * (k + 1))
    var_idx = np.array([a.var_names.get_loc(m) for m in features])
    ct_obs = a.obs["cell_type"].astype(str).to_numpy()
    X = np.asarray(a.X)[:, var_idx].astype(np.float32)
    del a
    return X, ct_obs


def fit_scales(args, maps, pairs, patients, features):
    """s per pair = mean ||T_trained(x) - x|| / mean ||g_rand(x) - x|| on reference cells."""
    rng = np.random.default_rng(12345)
    ref = {}
    for patient in patients[:args.ref_patients]:
        X, ct_obs = patient_cells(patient, features)
        for ct in {c for _, c in pairs}:
            x = X[ct_obs == CELLTYPE_NAME_MAP.get(ct, ct)]
            if len(x) > args.ref_cells:
                x = x[rng.choice(len(x), args.ref_cells, replace=False)]
            ref.setdefault(ct, []).append(x)
        del X
        gc.collect()
    ref = {ct: np.concatenate(v) for ct, v in ref.items()}
    rows = []
    for stim, ct in sorted(pairs):
        x = ref[ct]
        g_rand, _ = maps[(stim, ct)]
        d_train = np.linalg.norm(transport(load_g(stim, ct, trained=True), x) - x, axis=1).mean()
        d_rand = np.linalg.norm(transport(g_rand, x) - x, axis=1).mean()
        s = d_train / d_rand
        maps[(stim, ct)] = (g_rand, s)
        rows.append(dict(stim=stim, celltype=ct, n_ref=len(x), disp_trained=d_train,
                         disp_random=d_rand, scale=s))
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map-kind", required=True,
                   choices=["cellot", "cellot_permuted", "cellot_ctswap", "cellot_crossswap",
                            "random_icnn"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--match-scale", action="store_true",
                   help="random_icnn only: match the trained map's mean displacement")
    p.add_argument("--ref-patients", type=int, default=8)
    p.add_argument("--ref-cells", type=int, default=2000,
                   help="per celltype per reference patient")
    p.add_argument("--out", required=True)
    p.add_argument("--like", default=None, help="reindex columns onto this table")
    p.add_argument("--patients-limit", type=int, default=None, help="smoke tests")
    p.add_argument("--celltypes", nargs="*", default=None, help="smoke tests")
    args = p.parse_args(argv)
    if args.match_scale and args.map_kind != "random_icnn":
        p.error("--match-scale only applies to random_icnn")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))

    features = read_list(PERCOND / "13features.txt")
    pairs = [(s, c) for s, c in valid_pairs()
             if args.celltypes is None or c in args.celltypes]
    patients = read_list(PERCOND / "patients_ina.txt")
    t0 = time.time()
    maps = build_maps(args, pairs)
    print(f"[INFO ] {args.map_kind} seed={args.seed}: {len(maps)} maps "
          f"({time.time() - t0:.0f}s)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.match_scale:
        scales = fit_scales(args, maps, pairs, patients, features)
        scales.to_csv(out.with_suffix(".scales.csv"), index=False)
        print(f"[SCALE] median s={scales.scale.median():.3g} "
              f"[{scales.scale.min():.3g}, {scales.scale.max():.3g}]", flush=True)

    if args.patients_limit:
        patients = patients[:args.patients_limit]
    rows = {}
    for i, patient in enumerate(patients, 1):
        if not (OOL / f"{patient}_Unstim.h5ad").exists():
            print(f"[SKIP ] {patient}: no file")
            continue
        X, ct_obs = patient_cells(patient, features)
        row = {}
        for ct in sorted({c for _, c in pairs}):
            x = X[ct_obs == CELLTYPE_NAME_MAP.get(ct, ct)]
            if not len(x):
                continue
            for marker, val in zip(features, np.median(x.astype(np.float64), axis=0)):
                row[f"{ct}_{marker}_Unstim"] = val
            for stim in [s for s, c in pairs if c == ct]:
                g, s = maps[(stim, ct)]
                y = transport(g, x)
                if s is not None:
                    y = x + s * (y - x)
                for marker, val in zip(features, np.median(y, axis=0)):
                    row[f"{ct}_{marker}_{stim}"] = val
        rows[patient] = row
        del X
        gc.collect()
        print(f"[INFO ] {i}/{len(patients)} {patient} ({time.time() - t0:.0f}s)", flush=True)

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Individual"
    if args.like:
        ref = pd.read_csv(args.like, index_col=0, nrows=1)
        missing = [c for c in ref.columns if c not in table.columns]
        if missing and not (args.patients_limit or args.celltypes):
            print(f"[WARN ] {len(missing)} reference columns not produced, e.g. {missing[:5]}")
        if not args.celltypes:
            table = table.reindex(columns=ref.columns)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    table.to_csv(tmp)
    tmp.rename(out)  # atomic: a concurrent job never reads a half-written table
    print(f"[DONE ] {table.shape} -> {out}")


if __name__ == "__main__":
    main()
