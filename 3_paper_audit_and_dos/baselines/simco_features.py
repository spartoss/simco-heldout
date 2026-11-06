"""Build Stabl feature tables from a *baseline* map instead of CellOT.

Why only ``gauss_ot``
---------------------
The Stabl feature ``<celltype>_<marker>_<stim>`` is the per-marker median of the cells
transported from that patient's Unstim. For a **diagonal** baseline that median carries
no information the Unstim block does not already have:

  * ``median_shift``  T(x) = x + d      -> median(T(x)) = median(x) + d
  * ``quantile_map``  per-marker 1-D OT -> median(T(x)) = F_t^-1(F_s(median(x)))
  * ``linear_median`` T(x) = x + (a + Bu - u), u = median(x) -> median(T(x)) = a + B u

i.e. a constant offset, a monotone function of one Unstim feature, and an affine
function of that celltype's 13 Unstim features respectively. After standardisation the
first is literally the same column; none of the three can beat the ``unstim_only`` arm.

``gauss_ot`` is the cheapest baseline that escapes this. T(x) = mu_t + A (x - mu_s) with
A a full 13x13 matrix, so median_j(T(x)) mixes markers and is a *nonlinear functional of
this patient's joint cell distribution* -- the same property that lets CellOT's features
carry more than the Unstim medians. So the gap between gauss_ot's downstream R2 and
CellOT's is the part of Fig 5b that needs more than a shared affine map to explain.

Two phases, because the training data is per-celltype and the OOL data is per-patient:

  fit    one pass over the 28 batchcorrected .h5ad; fit the 168 maps on the *train*
         side of CellOT's own cell split; write one .npz per celltype.
  apply  one pass over the 177 OOL per-patient Unstim .h5ad; transport each patient's
         cells through every map and take medians; write the wide table.

Usage (inside the existing allocation -- no sbatch):
    python -m baselines.simco_features fit   --celltype pDCs   --outdir <maps>
    python -m baselines.simco_features apply --maps <maps> --out <table.csv>
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from baselines.baselines import gauss_ot_fit, gauss_ot_apply
from baselines.run_baselines import load_celltype

DATA = Path("/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets")
BATCHCORR = DATA / "ptb_concatenated_batchcorrected"
PERCOND = DATA / "ptb_concatenated_per_condition_celltype"
OOL = DATA / "ool_by_patient/unstim_final"

# sanitized celltype (file/pair naming) -> the label carried in the OOL .h5ad obs.
# Copied verbatim from make_prediction/predict_ina_og_h5.py:20-48.
CELLTYPE_NAME_MAP = {
    "Bcells": "Bcells",
    "CD4Tcm": "CD4Tcm",
    "CD4Tcm_CCR2pos": "CD4Tcm CCR2+",
    "CD4Teff": "CD4Teff",
    "CD4Tem": "CD4Tem",
    "CD4Tem_CCR2pos": "CD4Tem CCR2+",
    "CD4Tnaive": "CD4Tnaive",
    "CD4Tregs": "CD4Tregs",
    "CD4negCD8negTcells": "CD4negCD8negTcells",
    "CD56hiCD16negNK": "CD56hiCD16negNK",
    "CD56loCD16posNK": "CD56loCD16posNK",
    "CD8Tcells_Th1": "CD8Tcells Th1",
    "CD8Tcm": "CD8Tcm",
    "CD8Tcm_CCR2pos": "CD8Tcm CCR2+",
    "CD8Teff": "CD8Teff",
    "CD8Tem": "CD8Tem",
    "CD8Tem_CCR2pos": "CD8Tem CCR2+",
    "CD8Tnaive": "CD8Tnaive",
    "Granulocytes": "Granulocytes",
    "MDSCs": "MDSCs",
    "NK_cells_CD11cpos": "NK cells CD11c+",
    "NK_cells_CD11cneg": "NK cells CD11c-",
    "NKT": "NKT",
    "cMCs": "cMCs",
    "intMCs": "intMCs",
    "mDCs": "mDCs",
    "ncMCs": "ncMCs",
    "pDCs": "pDCs",
}


def read_list(path):
    return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]


def valid_pairs():
    """(stim, sanitized_celltype) for the 168 pairs the paper trained."""
    out = []
    for line in read_list(PERCOND / "valid_jobs.txt"):
        stim, ct = line.split(",")[:2]
        out.append((stim, ct))
    return out


# --------------------------------------------------------------------------- #
# phase 1 -- fit
# --------------------------------------------------------------------------- #

def cmd_fit(args):
    features = read_list(PERCOND / "13features.txt")
    pairs = [(s, c) for s, c in valid_pairs() if c == args.celltype]
    if not pairs:
        print(f"[SKIP ] {args.celltype}: no valid pair")
        return
    out = Path(args.outdir) / f"{args.celltype}.npz"
    if out.exists() and not args.force:
        print(f"[SKIP ] {args.celltype}: {out.name} exists")
        return

    cells, test_mask = load_celltype(BATCHCORR, args.celltype, features,
                                     args.test_size, args.split_seed)
    # CellOT trained on the 80% side; fit the baseline on the same cells.
    train = {k: v[~test_mask[k]] for k, v in cells.items()}
    del cells
    gc.collect()

    patients = sorted({p for p, s in train if s == "Unstim"})
    src = [train[(p, "Unstim")] for p in patients if (p, "Unstim") in train]

    blob = {}
    for stim, _ in pairs:
        tgt_pat = [p for p in patients if (p, stim) in train]
        if not tgt_pat:
            print(f"[WARN ] {args.celltype}/{stim}: no target cells")
            continue
        s = [train[(p, "Unstim")] for p in tgt_pat]
        t = [train[(p, stim)] for p in tgt_pat]
        # ``cell`` weighting = the Gaussian of the pooled cells, matching CellOT's
        # cell-level sampling (the arm the benchmark quotes).
        params = gauss_ot_fit(s, t, weighting="cell")
        for k, v in params.items():
            blob[f"{stim}|{k}"] = np.asarray(v)
        print(f"[FIT  ] {args.celltype}/{stim}: {sum(len(x) for x in s)} src, "
              f"{sum(len(x) for x in t)} tgt cells, {len(tgt_pat)} patients")

    blob["features"] = np.array(features)
    blob["stims"] = np.array(sorted({s for s, _ in pairs}))
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    np.savez(out, **blob)
    print(f"[DONE ] {args.celltype} -> {out}")


# --------------------------------------------------------------------------- #
# phase 2 -- apply
# --------------------------------------------------------------------------- #

def cmd_apply(args):
    features = read_list(PERCOND / "13features.txt")
    maps = {}
    for f in sorted(Path(args.maps).glob("*.npz")):
        ct = f.stem
        z = np.load(f, allow_pickle=True)
        assert list(z["features"]) == features, f"marker order differs in {f}"
        maps[ct] = {
            stim: {k: z[f"{stim}|{k}"] for k in ("mu_s", "mu_t", "A")}
            for stim in [str(s) for s in z["stims"]]
            if f"{stim}|A" in z
        }
    print(f"[INFO ] {len(maps)} celltypes, "
          f"{sum(len(v) for v in maps.values())} maps loaded")

    patients = read_list(PERCOND / "patients_ina.txt")
    rows = {}
    for i, patient in enumerate(patients, 1):
        path = OOL / f"{patient}_Unstim.h5ad"
        if not path.exists():
            print(f"[SKIP ] {patient}: no file")
            continue
        a = ad.read_h5ad(path)
        var_idx = np.array([a.var_names.get_loc(m) for m in features])
        ct_obs = a.obs["cell_type"].astype(str).to_numpy()
        X = np.asarray(a.X)[:, var_idx].astype(np.float64)
        del a

        row = {}
        for ct, stim_maps in maps.items():
            label = CELLTYPE_NAME_MAP.get(ct, ct)
            sel = ct_obs == label
            if not sel.any():
                continue
            x = X[sel]
            for marker, val in zip(features, np.median(x, axis=0)):
                row[f"{ct}_{marker}_Unstim"] = val
            for stim, p in stim_maps.items():
                med = np.median(gauss_ot_apply(x, p), axis=0)
                for marker, val in zip(features, med):
                    row[f"{ct}_{marker}_{stim}"] = val
        rows[patient] = row
        del X
        if i % 25 == 0:
            print(f"[INFO ] {i}/{len(patients)} patients", flush=True)
        gc.collect()

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Individual"
    print(f"[INFO ] raw table {table.shape}")

    if args.like:
        ref = pd.read_csv(args.like, index_col=0, nrows=1)
        missing = [c for c in ref.columns if c not in table.columns]
        if missing:
            print(f"[WARN ] {len(missing)} reference columns not produced, e.g. "
                  f"{missing[:5]}")
        table = table.reindex(columns=ref.columns)
        n_nan = int(table.isna().sum().sum())
        print(f"[INFO ] reindexed to {args.like} -> {table.shape}, {n_nan} NaN")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out)
    print(f"[DONE ] -> {args.out}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit")
    f.add_argument("--celltype", required=True)
    f.add_argument("--outdir", required=True)
    f.add_argument("--test-size", type=float, default=0.2)
    f.add_argument("--split-seed", type=int, default=0)
    f.add_argument("--force", action="store_true")
    f.set_defaults(func=cmd_fit)

    a = sub.add_parser("apply")
    a.add_argument("--maps", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--like", default=None,
                   help="reindex the columns onto this CellOT table, so the two feed "
                        "Stabl with an identical schema")
    a.set_defaults(func=cmd_apply)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
