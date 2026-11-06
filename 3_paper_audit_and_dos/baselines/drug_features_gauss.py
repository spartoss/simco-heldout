"""Drug-treated OOL feature tables with NO neural network anywhere: gauss_ot stim maps
(baselines/gaussot_maps, as in gaussot_13OG_both.csv) followed by gauss_ot drug maps
(baselines/gauss_drug_maps.py), and the Unstim block through gauss_ot Unstim x drug maps.
The gauss_ot counterpart of drug_features.py's CellOT chain; one pass over the 177 women for all
9 drugs. Writes <out>/<drug>_gaussfull.csv on the paper table's 1216-column schema.
    python -m baselines.drug_features_gauss --drug-maps <dir> --out <dir> [--patients-limit 3]
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import pandas as pd

from baselines.baselines import gauss_ot_apply
from baselines.null_map_features import patient_cells
from baselines.simco_features import CELLTYPE_NAME_MAP, PERCOND, read_list, valid_pairs

GMAPS = Path("/scratch/users/bmerwan/simco_work/baselines/gaussot_maps")
ONSET = Path("/scratch/users/bmerwan/simco_work/onset_test")
DRUGS = ["SA", "RIF", "SALPZ", "CHT", "THF", "LPZ", "MAP", "PRA", "MF"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--drug-maps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--patients-limit", type=int, default=None)
    ap.add_argument("--celltypes", nargs="*", default=None, help="smoke tests")
    a = ap.parse_args(argv)
    func = read_list(PERCOND / "13features.txt")
    pairs = [(s, c) for s, c in valid_pairs() if a.celltypes is None or c in a.celltypes]
    cts = sorted({c for _, c in pairs})
    K = ("mu_s", "mu_t", "A")
    stim_map, drug_map = {}, {}
    for c in cts:
        z = np.load(GMAPS / f"{c}.npz", allow_pickle=True)
        stim_map[c] = {s: {k: z[f"{s}|{k}"] for k in K} for s in map(str, z["stims"])}
        zd = np.load(Path(a.drug_maps) / f"{c}.npz", allow_pickle=True)
        assert list(zd["features"]) == func, c
        drug_map[c] = {(d, s): {k: zd[f"{d}|{s}|{k}"] for k in K}
                       for d in DRUGS for s in map(str, zd["stims"]) if f"{d}|{s}|A" in zd}
    patients = read_list(PERCOND / "patients_ina.txt")[:a.patients_limit]
    rows = {d: {} for d in DRUGS}; t0 = time.time()
    for n, pt in enumerate(patients, 1):
        X, ct_obs = patient_cells(pt, func)
        r = {d: {} for d in DRUGS}
        for c in cts:
            x = X[ct_obs == CELLTYPE_NAME_MAP.get(c, c)].astype(np.float64)
            if not len(x): continue
            ys = {s: gauss_ot_apply(x, stim_map[c][s]) for s, cc in pairs if cc == c}
            for d in DRUGS:
                if (d, "Unstim") in drug_map[c]:
                    for m, v in zip(func, np.median(gauss_ot_apply(x, drug_map[c][(d, "Unstim")]), 0)):
                        r[d][f"{c}_{m}_Unstim"] = v
                for s, y in ys.items():
                    if (d, s) not in drug_map[c]: continue
                    for m, v in zip(func, np.median(gauss_ot_apply(y, drug_map[c][(d, s)]), 0)):
                        r[d][f"{c}_{m}_{s}"] = v
        for d in DRUGS: rows[d][pt] = r[d]
        print(f"[INFO ] {n}/{len(patients)} {pt} ({time.time() - t0:.0f}s)", flush=True)
    ref = pd.read_csv(ONSET / "ina_13OG_final_long_allstims_filtered.csv", index_col=0, nrows=1).columns
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for d in DRUGS:
        t = pd.DataFrame.from_dict(rows[d], orient="index")
        if a.celltypes is None: t = t.reindex(columns=ref)
        t.index.name = "Individual"
        tmp = Path(a.out) / f"{d}_gaussfull.{os.getpid()}.tmp"; t.to_csv(tmp); tmp.rename(Path(a.out) / f"{d}_gaussfull.csv")
        print(f"[DONE ] {d}: {t.shape}, NaN {int(t.isna().sum().sum())}")


if __name__ == "__main__":
    main()
