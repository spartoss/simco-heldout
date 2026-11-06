"""Measured distributional Unstim features for the OOL cohort -- no map at all.

The null-map controls (CLAUDE.md, "Null-map controls") showed that an untrained ICNN recovers
~80 % of CellOT's Fig 5b gain: the signal is in each patient's Unstim *joint distribution*, and
per-marker medians throw it away. This builds that distribution's summaries directly, so Stabl
can use them without any simulated stimulation.

Blocks (the column suffix is the Stabl block name, see ``run_regression_cv.py --blocks``):

  freq   <ct>_freq                 celltype fraction of all gated cells, as z-scored
                                   log10(fraction + 1e-4)                                 (28)
  med    <ct>_<marker>_med         median, all 31 OOL markers (13 functional + phenotypic) (868)
  qtl    <ct>_<marker>_q<k>_qtl    q10/q25/q75/q90 of the 13 functional markers           (1456)
  corr   <ct>_<m1>__<m2>_corr      Pearson correlation between functional markers          (2184)

Raw fractions have variance < 0.01, so Stabl's VarianceThreshold(0.01) rejected the whole block
and crashed the run (night2, 2026-09-22). log10 handles the skew; the z-score (over all samples,
no outcome involved) is affine and undone by Stabl's own StandardScaler -- it only clears the
variance filter.

Celltypes with fewer than ``--min-cells`` cells get NaN for qtl/corr (Stabl imputes).
The 13 functional ``_med`` columns equal the paper's ``_Unstim`` block (checked on write).

    python -m baselines.distrib_features --out <csv> [--patients-limit 3]
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")  # group-home NFS, errno 121

import numpy as np
import pandas as pd

from baselines.null_map_features import patient_cells
from baselines.simco_features import CELLTYPE_NAME_MAP, PERCOND, read_list

ALL_MARKERS = Path("/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/"
                   "ool_by_patient/features_final.txt")
QS = (10, 25, 75, 90)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--min-cells", type=int, default=20)
    p.add_argument("--patients-limit", type=int, default=None)
    p.add_argument("--check", default="/scratch/users/bmerwan/simco_work/onset_test/"
                   "ina_13OG_final_long_allstims_filtered.csv")
    args = p.parse_args(argv)

    func = read_list(PERCOND / "13features.txt")
    markers = read_list(ALL_MARKERS)
    fidx = [markers.index(m) for m in func]
    pairs = [(i, j) for i in range(len(func)) for j in range(i + 1, len(func))]
    patients = read_list(PERCOND / "patients_ina.txt")[:args.patients_limit]
    t0 = time.time()
    rows = {}
    for n, patient in enumerate(patients, 1):
        X, ct_obs = patient_cells(patient, markers)
        row = {}
        for ct, label in CELLTYPE_NAME_MAP.items():
            sel = ct_obs == label
            row[f"{ct}_freq"] = sel.mean()
            x = X[sel].astype(np.float64)
            if not len(x):
                continue
            for m, v in zip(markers, np.median(x, axis=0)):
                row[f"{ct}_{m}_med"] = v
            if len(x) < args.min_cells:
                continue
            xf = x[:, fidx]
            q = np.percentile(xf, QS, axis=0)
            for k, qk in enumerate(QS):
                for m, v in zip(func, q[k]):
                    row[f"{ct}_{m}_q{qk}_qtl"] = v
            with np.errstate(invalid="ignore", divide="ignore"):
                c = np.corrcoef(xf, rowvar=False)
            for i, j in pairs:
                row[f"{ct}_{func[i]}__{func[j]}_corr"] = c[i, j]
        rows[patient] = row
        print(f"[INFO ] {n}/{len(patients)} {patient} ({time.time() - t0:.0f}s)", flush=True)

    t = pd.DataFrame.from_dict(rows, orient="index")
    t.index.name = "Individual"
    t = t.replace([np.inf, -np.inf], np.nan)
    fq = [c for c in t.columns if c.endswith("_freq")]
    lf = np.log10(t[fq] + 1e-4)
    t[fq] = (lf - lf.mean()) / lf.std(ddof=0)
    ref = pd.read_csv(args.check, index_col=0)
    cols = [c for c in ref.columns if c.endswith("_Unstim")]
    mine = t.reindex(columns=[c.replace("_Unstim", "_med") for c in cols])
    diff = np.nanmax(np.abs(mine.values - ref.loc[t.index, cols].values))
    print(f"[CHECK] functional _med vs paper _Unstim: max |diff| {diff:.2e}")
    for b in ("freq", "med", "qtl", "corr"):
        blk = [c for c in t.columns if c.endswith(f"_{b}")]
        print(f"[INFO ] block {b}: {len(blk)} cols, {int(t[blk].isna().sum().sum())} NaN")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    t.to_csv(tmp)
    tmp.rename(out)
    print(f"[DONE ] {t.shape} -> {out}")


if __name__ == "__main__":
    main()
