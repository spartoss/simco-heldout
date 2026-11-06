"""Per-patient drug effects on the atlas: true vs predicted by CellOT / quantile_map / median_shift / gauss_ot.

Why: a drug map is applied to each OOL woman separately, so the question behind "is the map patient-dependent?"
is whether the map predicts how a *given* patient's cells respond, not just the average response. On the drug
screen (19 patients) this is measurable. For each (stim or Unstim, celltype, drug) used by the Fig 5b model and each
patient p, using ALL of p's cells (train and test side of CellOT's split, the same cells for every method; test-only
leaves ~20 cells per patient in small celltypes, too few for per-patient quantiles). This is in-sample for every
map, and favours the most flexible one (CellOT), i.e. it is conservative for the claim that CellOT is no more
patient-specific than the simple maps:
    true_p = q(drug cells of p) - q(DMSO cells of p)        per marker, q in 10/25/50/75/90
    pred_p = q(T(DMSO cells of p)) - q(DMSO cells of p)     for each map T
    noise_p = |true on one random half - true on the other half|   (sampling noise of true_p)
CellOT = the paper's checkpoints (results_drug, results_unstim). The other maps are fitted on the train-side cells
pooled over patients ("cell" weighting), with the same control, batch correction and split as CellOT (drug_data).
median_shift moves every cell by the same pooled median shift, so its pred_p is identical for every patient.
    python -m baselines.drug_patient_effects --celltype pDCs --pairs <csv of stim,celltype> --out <dir>
"""
from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import pandas as pd

from baselines.baselines import gauss_ot_apply, gauss_ot_fit, quantile_map_apply, quantile_map_fit
from baselines.drug_data import dmso_for, load_bc, load_drug_stims, pair_frame
from baselines.run_baselines import cellot_transport, load_cellot_g
from baselines.simco_features import PERCOND, read_list

CR = Path("/home/groups/gbrice/ptb-drugscreen/ot/cellot")
DATA = CR / "datasets/ptb_concatenated_batchcorrected"
BC = CR / "cross_validation/batchcorrection_2.csv"
DRUG = CR / "results_drug/cross_validation_drug_OG13"
UNST = CR / "results_unstim/cross_validation_drug_OG13"
DRUGS = ["SA", "RIF", "SALPZ", "CHT", "THF", "LPZ", "MAP", "PRA", "MF"]
QS = (10, 25, 50, 75, 90)
MIN_CELLS = 20


def model_dir(drug, stim, ct):
    return (UNST / drug / ct / f"model-Unstim_{ct}") if stim == "Unstim" else (DRUG / drug / stim / ct / f"model-{stim}_{ct}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--celltype", required=True)
    ap.add_argument("--pairs", required=True, help="csv with columns stim,celltype")
    ap.add_argument("--out", required=True)
    ap.add_argument("--drugs", nargs="*", default=DRUGS)
    ap.add_argument("--stims", nargs="*", default=None, help="subset of this celltype's stims (parallel runs)")
    a = ap.parse_args(argv)
    out = Path(a.out) / (f"{a.celltype}.csv" if not a.stims else f"{a.celltype}__{'_'.join(a.stims)}.csv")
    if out.exists():
        print(f"[SKIP ] {out}"); return
    feats = read_list(PERCOND / "13features.txt")
    pairs = pd.read_csv(a.pairs)
    stims = sorted(pairs.loc[pairs.celltype == a.celltype, "stim"].unique())
    if a.stims: stims = [x for x in stims if x in a.stims]
    rng = np.random.default_rng(0)
    rows, t0 = [], time.time()
    for stim in stims:                                   # one stim at a time: Granulocytes is 146M rows
        X, obs = load_drug_stims(DATA, a.celltype, feats, [stim])
        bc = load_bc(BC, pd.unique(obs["cell_type"].astype(str)), [stim])
        for drug in a.drugs:
            md = model_dir(drug, stim, a.celltype)
            if not (md / "cache/model.pt").exists():
                print(f"[WARN ] no checkpoint {md}", flush=True); continue
            dmso = dmso_for(BC, drug)
            Xd, obsd, is_test = pair_frame(X, obs, bc, drug, dmso, stim, feats, 0.2, 0)
            if Xd is None: continue
            pat = obsd["patient"].astype(str).to_numpy(); trt = obsd["drug"].astype(str).to_numpy()
            pts = [p for p in pd.unique(pat) if ((pat == p) & (trt == dmso)).any() and ((pat == p) & (trt == drug)).any()]
            src_tr = [Xd[(pat == p) & (trt == dmso) & ~is_test].astype(np.float64) for p in pts]
            tgt_tr = [Xd[(pat == p) & (trt == drug) & ~is_test].astype(np.float64) for p in pts]
            try:
                gp = gauss_ot_fit(src_tr, tgt_tr, weighting="cell")
            except np.linalg.LinAlgError:              # degenerate covariance: no gauss_ot prediction for this map
                gp = None
            qp = quantile_map_fit(src_tr, tgt_tr, weighting="cell")
            shift = np.median(np.concatenate(tgt_tr), 0) - np.median(np.concatenate(src_tr), 0)
            g = load_cellot_g(str(md), len(feats))
            for p in pts:
                s = Xd[(pat == p) & (trt == dmso)].astype(np.float64)
                t = Xd[(pat == p) & (trt == drug)].astype(np.float64)
                if len(s) < MIN_CELLS or len(t) < MIN_CELLS: continue
                qs = np.percentile(s, QS, 0); true = np.percentile(t, QS, 0) - qs
                ps, pt = rng.permutation(len(s)), rng.permutation(len(t))
                hA = np.percentile(t[pt[: len(t) // 2]], QS, 0) - np.percentile(s[ps[: len(s) // 2]], QS, 0)
                hB = np.percentile(t[pt[len(t) // 2:]], QS, 0) - np.percentile(s[ps[len(s) // 2:]], QS, 0)
                pred = {"cellot": np.percentile(cellot_transport(g, s), QS, 0) - qs,
                        "quantile_map": np.percentile(quantile_map_apply(s, qp), QS, 0) - qs,
                        "median_shift": np.percentile(s + shift, QS, 0) - qs,
                        "gauss_ot": (np.percentile(gauss_ot_apply(s, gp), QS, 0) - qs) if gp is not None
                                    else np.full_like(true, np.nan)}
                for qi, q in enumerate(QS):
                    for mi, m in enumerate(feats):
                        rows.append((a.celltype, stim, drug, p, m, q, len(s), len(t), true[qi, mi],
                                     abs(hA[qi, mi] - hB[qi, mi]), *(pred[k][qi, mi] for k in pred)))
            print(f"[DONE ] {a.celltype}/{stim}/{drug}: {len(pts)} patients ({time.time() - t0:.0f}s)", flush=True)
            del Xd, obsd, is_test, g; gc.collect()
        del X, obs; gc.collect()
    cols = ["celltype", "stim", "drug", "patient", "marker", "q", "n_src", "n_tgt", "true", "noise",
            "cellot", "quantile_map", "median_shift", "gauss_ot"]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp"); pd.DataFrame(rows, columns=cols).to_csv(tmp, index=False); tmp.rename(out)
    print(f"[DONE ] {a.celltype}: {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
