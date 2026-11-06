"""Drug-treated feature tables through quantile_map DRUG maps applied directly to the feature values.

A quantile map is monotone marker by marker, so the median (or any quantile) of the transported cells is
the map applied to the median (quantile) of the cells: median(T(x)) = T(median(x)). The treated feature is
therefore T_{drug, stim, celltype, marker}(untreated feature) -- no cells, no second transport. Works on any
untreated table with `<celltype>_<marker>_<stim>` columns (CellOT or gauss_ot simulated stims; `_Unstim`
columns go through the Unstim x drug map) and on simulated-quantile columns `<ct>_<m>_q<k>_<stim>q`.
Values outside the control's training support clamp to the drug's extreme values (quantile_map_apply).
    python -m baselines.drug_features_qmap --maps <dir> --features <csv> --out <dir> --tag <name>
    -> <out>/<drug>_<tag>.csv, same rows and columns as --features
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from baselines.baselines import quantile_map_apply
from baselines.simco_features import PERCOND, read_list

DRUGS = ["SA", "RIF", "SALPZ", "CHT", "THF", "LPZ", "MAP", "PRA", "MF"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--method", choices=["quantile_map", "median_shift"], default="quantile_map",
                    help="median_shift: add the pooled median shift q_tgt(0.5) - q_src(0.5) of the same fitted grids "
                         "to every woman (a constant per feature)")
    a = ap.parse_args(argv)
    func = read_list(PERCOND / "13features.txt")
    X = pd.read_csv(a.features, index_col=0)
    maps = {}
    for f in sorted(Path(a.maps).glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        assert list(z["features"]) == func, f
        maps[f.stem] = z
    # column -> (celltype, marker index, stim)
    spec = {}
    for c, z in maps.items():
        for s in map(str, z["stims"]):
            for j, m in enumerate(func):
                spec[f"{c}_{m}_{s}"] = (c, j, s)
                for k in (10, 25, 75, 90):
                    spec[f"{c}_{m}_q{k}_{s}q"] = (c, j, s)
    cols = [col for col in X.columns if col in spec]
    print(f"[INFO ] {len(cols)}/{X.shape[1]} columns have a drug map; the others are copied unchanged")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for d in DRUGS:
        T = X.copy(); n_map = 0
        for col in cols:
            c, j, s = spec[col]; z = maps[c]
            if f"{d}|{s}|q_src" not in z: continue
            prm = {"u": z[f"{d}|{s}|u"], "q_src": z[f"{d}|{s}|q_src"][:, [j]], "q_tgt": z[f"{d}|{s}|q_tgt"][:, [j]]}
            v = X[col].to_numpy(float); ok = ~np.isnan(v)
            if a.method == "median_shift":
                mid = int(np.argmin(np.abs(prm["u"] - 0.5)))
                T.loc[ok, col] = v[ok] + (prm["q_tgt"][mid, 0] - prm["q_src"][mid, 0])
            else:
                T.loc[ok, col] = quantile_map_apply(v[ok, None], prm)[:, 0]
            n_map += 1
        out = Path(a.out) / f"{d}_{a.tag}.csv"; T.to_csv(out)
        print(f"[DONE ] {d}: {n_map} columns mapped, median |change| {np.nanmedian(np.abs(T[cols] - X[cols]).to_numpy()):.4f} -> {out}")


if __name__ == "__main__":
    main()
