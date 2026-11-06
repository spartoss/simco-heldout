"""Measured OOL Unstim features for supervised DOS prediction -- no map, no timing.

One row per OOL sample (177), built from the gated Unstim cells only. Column names are
``<celltype>__<marker>__<stat>`` (or ``<celltype>__freq``), so a block is a column suffix:

  freq    log10(celltype fraction of all gated cells + 1e-4)                         28
  q10 q25 q50 q75 q90   quantiles of all 31 markers (13 functional + 18 phenotypic)  28*31*5
  mean sd               moments of all 31 markers                                    28*31*2
  corr    Pearson correlation between the 13 functional markers                       28*78

Quantiles beyond the median, moments and correlations need ``--min-cells`` cells (default 20,
same as baselines/distrib_features.py); below that they are NaN and the model pipeline imputes.
No column depends on another sample or on the outcome, so the table can be built once, outside
the cross-validation. ``q50`` of the 13 functional markers equals the paper's ``_Unstim`` block.

    python ml_dos/build_features.py one <patient> --out-dir <dir>     # one parquet-free .csv row
    python ml_dos/build_features.py assemble --out-dir <dir> --out <csv>
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import pandas as pd

DATA = Path("/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets")
OOL = DATA / "ool_by_patient/unstim_final"
PERCOND = DATA / "ptb_concatenated_per_condition_celltype"
MARKERS = DATA / "ool_by_patient/features_final.txt"
QS = (10, 25, 50, 75, 90)
# sanitized celltype -> label in the OOL .h5ad obs (baselines/simco_features.py, from
# make_prediction/predict_ina_og_h5.py:20-48)
CT = {
    "Bcells": "Bcells", "CD4Tcm": "CD4Tcm", "CD4Tcm_CCR2pos": "CD4Tcm CCR2+", "CD4Teff": "CD4Teff",
    "CD4Tem": "CD4Tem", "CD4Tem_CCR2pos": "CD4Tem CCR2+", "CD4Tnaive": "CD4Tnaive",
    "CD4Tregs": "CD4Tregs", "CD4negCD8negTcells": "CD4negCD8negTcells",
    "CD56hiCD16negNK": "CD56hiCD16negNK", "CD56loCD16posNK": "CD56loCD16posNK",
    "CD8Tcells_Th1": "CD8Tcells Th1", "CD8Tcm": "CD8Tcm", "CD8Tcm_CCR2pos": "CD8Tcm CCR2+",
    "CD8Teff": "CD8Teff", "CD8Tem": "CD8Tem", "CD8Tem_CCR2pos": "CD8Tem CCR2+",
    "CD8Tnaive": "CD8Tnaive", "Granulocytes": "Granulocytes", "MDSCs": "MDSCs",
    "NK_cells_CD11cpos": "NK cells CD11c+", "NK_cells_CD11cneg": "NK cells CD11c-", "NKT": "NKT",
    "cMCs": "cMCs", "intMCs": "intMCs", "mDCs": "mDCs", "ncMCs": "ncMCs", "pDCs": "pDCs",
}


def read_list(p):
    return [ln.strip() for ln in Path(p).read_text().splitlines() if ln.strip()]


def load(patient, markers, tries=5):
    import anndata as ad
    for k in range(tries):
        try:
            a = ad.read_h5ad(OOL / f"{patient}_Unstim.h5ad")
            break
        except OSError as e:
            if k == tries - 1:
                raise
            print(f"[RETRY] {patient}: {e}", flush=True)
            time.sleep(10 * (k + 1))
    idx = np.array([a.var_names.get_loc(m) for m in markers])
    ct = a.obs["cell_type"].astype(str).to_numpy()
    X = np.asarray(a.X)[:, idx].astype(np.float64)
    return X, ct


def one(patient, out_dir, min_cells):
    markers = read_list(MARKERS)
    func = read_list(PERCOND / "13features.txt")
    fidx = [markers.index(m) for m in func]
    pairs = [(i, j) for i in range(len(func)) for j in range(i + 1, len(func))]
    X, ct_obs = load(patient, markers)
    row = {"n_cells_total": len(X)}
    for ct, label in CT.items():
        sel = ct_obs == label
        row[f"{ct}__freq"] = np.log10(sel.mean() + 1e-4)
        x = X[sel]
        if not len(x):
            continue
        q = np.percentile(x, QS, axis=0)
        for m, v in zip(markers, q[QS.index(50)]):
            row[f"{ct}__{m}__q50"] = v
        if len(x) < min_cells:
            continue
        for k, qk in enumerate(QS):
            if qk == 50:
                continue
            for m, v in zip(markers, q[k]):
                row[f"{ct}__{m}__q{qk}"] = v
        for m, mu, sd in zip(markers, x.mean(0), x.std(0)):
            row[f"{ct}__{m}__mean"] = mu
            row[f"{ct}__{m}__sd"] = sd
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.corrcoef(x[:, fidx], rowvar=False)
        for i, j in pairs:
            row[f"{ct}__{func[i]}.{func[j]}__corr"] = c[i, j]
    s = pd.Series(row, name=patient)
    out = Path(out_dir) / f"{patient}.csv"
    s.to_frame().T.to_csv(out.with_suffix(".tmp"))
    out.with_suffix(".tmp").rename(out)
    print(f"[DONE ] {patient}: {len(X)} cells, {len(s)} cols", flush=True)


def assemble(out_dir, out):
    patients = read_list(PERCOND / "patients_ina.txt")
    miss = [p for p in patients if not (Path(out_dir) / f"{p}.csv").exists()]
    assert not miss, f"missing {len(miss)}: {miss[:5]}"
    t = pd.concat([pd.read_csv(Path(out_dir) / f"{p}.csv", index_col=0) for p in patients])
    t.index.name = "Individual"
    t = t.replace([np.inf, -np.inf], np.nan)
    # functional q50 must equal the paper's _Unstim block (same cells, same median)
    ref = pd.read_csv("/scratch/users/bmerwan/simco_work/onset_test/"
                      "ina_13OG_final_long_allstims_filtered.csv", index_col=0)
    uc = [c for c in ref.columns if c.endswith("_Unstim")]
    mine = []
    for c in uc:
        ct, m = c[:-len("_Unstim")].rsplit("_", 1)
        mine.append(f"{ct}__{m}__q50")
    d = np.nanmax(np.abs(t.reindex(columns=mine).loc[ref.index].values - ref[uc].values))
    print(f"[CHECK] functional q50 vs paper _Unstim: max |diff| {d:.2e}")
    t = t.loc[ref.index]  # the feature-table row order fixes the GroupShuffleSplit splits
    t.to_csv(out)
    blocks = pd.Series([c.rsplit("__", 1)[-1] for c in t.columns]).value_counts()
    print(blocks.to_string())
    print(f"[DONE ] {t.shape} -> {out}; NaN fraction {t.isna().mean().mean():.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["one", "assemble"])
    ap.add_argument("patient", nargs="?")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out")
    ap.add_argument("--min-cells", type=int, default=20)
    a = ap.parse_args()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    if a.cmd == "one":
        one(a.patient, a.out_dir, a.min_cells)
    else:
        assemble(a.out_dir, a.out)
    sys.exit(0)
