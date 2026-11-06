"""Quantiles of the SIMULATED stimulated cells (CellOT or gauss_ot maps) for the 177 OOL women (Stabl features).

The paper's simulated features are per-marker medians only. This adds the 10th/25th/75th/90th
percentiles of the same transported cells, restricted to the 852 (stim, celltype, marker) columns the
paper keeps (ina_13OG_final_long_allstims_filtered_no_Unstim.csv) -> 852 x 4 = 3408 columns named
``<celltype>_<marker>_q<k>_<stim>q`` so that each stim's quantiles form their own Stabl block
("<stim>q", e.g. "PIq"), next to the stim's median block. The median of the same cells is recomputed
and checked against the paper table (pipeline check).

    python -m baselines.sim_quantile_features --map-kind {cellot,gauss_ot} --out <csv> [--patients-limit 3 --celltypes pDCs]
With --map-kind gauss_ot the stim maps are baselines/gaussot_maps (as in gaussot_13OG_*.csv) and the
median check runs against gaussot_13OG_no_Unstim.csv.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import pandas as pd
import torch

from baselines.baselines import gauss_ot_apply
from baselines.null_map_features import load_g, patient_cells, transport
from baselines.simco_features import CELLTYPE_NAME_MAP, PERCOND, read_list, valid_pairs

ONSET = Path("/scratch/users/bmerwan/simco_work/onset_test")
QS = (10, 25, 75, 90)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--map-kind", choices=["cellot", "gauss_ot"], default="cellot")
    ap.add_argument("--patients-limit", type=int, default=None)
    ap.add_argument("--celltypes", nargs="*", default=None)
    a = ap.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    func = read_list(PERCOND / "13features.txt")
    ref = pd.read_csv(ONSET / ("ina_13OG_final_long_allstims_filtered_no_Unstim.csv" if a.map_kind == "cellot"
                               else "gaussot_13OG_no_Unstim.csv"), index_col=0)
    kept = set(ref.columns)                                   # the paper's 852 simulated columns
    pairs = [(s, c) for s, c in valid_pairs() if a.celltypes is None or c in a.celltypes]
    pairs = [(s, c) for s, c in pairs if any(f"{c}_{m}_{s}" in kept for m in func)]
    cts = sorted({c for _, c in pairs})
    if a.map_kind == "cellot":
        nets = {(s, c): load_g(s, c, trained=True) for s, c in pairs}
        apply = lambda s, c, x: transport(nets[(s, c)], x)
    else:
        G = Path("/scratch/users/bmerwan/simco_work/baselines/gaussot_maps")
        gm = {}
        for c in cts:
            z = np.load(G / f"{c}.npz", allow_pickle=True)
            gm[c] = {st: {k: z[f"{st}|{k}"] for k in ("mu_s", "mu_t", "A")} for st in map(str, z["stims"])}
        apply = lambda s, c, x: gauss_ot_apply(x.astype(np.float64), gm[c][s])
    patients = read_list(PERCOND / "patients_ina.txt")[:a.patients_limit]
    rows, med_check, t0 = {}, [], time.time()
    for n, pt in enumerate(patients, 1):
        X, ct_obs = patient_cells(pt, func)
        r = {}
        for c in cts:
            x = X[ct_obs == CELLTYPE_NAME_MAP.get(c, c)]
            if not len(x): continue
            for s in [s for s, cc in pairs if cc == c]:
                y = apply(s, c, x)
                q = np.percentile(y, QS, axis=0); med = np.median(y, axis=0)
                for k, m in enumerate(func):
                    col = f"{c}_{m}_{s}"
                    if col not in kept: continue
                    med_check.append(abs(med[k] - ref.at[pt, col]) if pt in ref.index else np.nan)
                    if len(x) < 20: continue          # same min-cells rule as distrib_features
                    for j, qk in enumerate(QS):
                        r[f"{c}_{m}_q{qk}_{s}q"] = q[j, k]
        rows[pt] = r
        print(f"[INFO ] {n}/{len(patients)} {pt} ({time.time() - t0:.0f}s)", flush=True)
    t = pd.DataFrame.from_dict(rows, orient="index"); t.index.name = "Individual"
    print(f"[CHECK] simulated median vs paper table: max |diff| {np.nanmax(med_check):.2e} over {len(med_check)} values")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp"); t.to_csv(tmp); tmp.rename(out)
    print(f"[DONE ] {t.shape}, NaN {int(t.isna().sum().sum())} -> {out}")


if __name__ == "__main__":
    main()
