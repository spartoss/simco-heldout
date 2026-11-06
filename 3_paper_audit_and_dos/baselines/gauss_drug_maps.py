"""Fit gauss_ot (default) or quantile_map DRUG maps (control DMSO -> drug, within a stim) for one celltype.

The no-neural-network counterpart of the paper's CellOT drug maps (results_drug: 1512 stim x drug,
results_unstim: 252 Unstim x drug). Same data path as CellOT's drug training, via baselines/drug_data:
the drug's plate control from ``use_DMSO`` (dmso_for), the runtime batch correction from
batchcorrection_2.csv (pair_frame -> batch_correct), and the drug-grouped 80/20 cell split with
random_state 0 (fit on the train side only, as simco_features.py does for the stim maps).
``weighting="cell"``: the Gaussian of the pooled cells, the arm the benchmark quotes.

Writes <outdir>/<celltype>.npz with keys "<drug>|<stim>|{mu_s,mu_t,A}" for every stim of the
168-pair grid in that celltype plus Unstim. Stims are read one at a time (Granulocytes is 30 GB).
    python -m baselines.gauss_drug_maps --celltype pDCs --outdir <dir>
With ``--method quantile_map`` the keys are "<drug>|<stim>|{u,q_src,q_tgt}" (pooled-cell quantile functions of
the control and the drug on a 1001-point grid, baselines.quantile_map_fit, weighting="cell").
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

from baselines.baselines import gauss_ot_fit, quantile_map_fit
from baselines.drug_data import dmso_for, load_bc, load_drug_stims, pair_frame
from baselines.simco_features import PERCOND, read_list, valid_pairs

CR = Path("/home/groups/gbrice/ptb-drugscreen/ot/cellot")
DATA = CR / "datasets/ptb_concatenated_batchcorrected"
BC = CR / "cross_validation/batchcorrection_2.csv"
DRUGS = ["SA", "RIF", "SALPZ", "CHT", "THF", "LPZ", "MAP", "PRA", "MF"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--celltype", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--stims", nargs="*", default=None, help="smoke tests")
    ap.add_argument("--drugs", nargs="*", default=DRUGS)
    ap.add_argument("--method", choices=["gauss_ot", "quantile_map"], default="gauss_ot")
    a = ap.parse_args(argv)
    out = Path(a.outdir) / f"{a.celltype}.npz"
    if out.exists() and a.stims is None:
        print(f"[SKIP ] {out} exists"); return
    features = read_list(PERCOND / "13features.txt")
    stims = a.stims or sorted({s for s, c in valid_pairs() if c == a.celltype}) + ["Unstim"]
    t0 = time.time(); blob = {}
    for stim in stims:
        X, obs = load_drug_stims(DATA, a.celltype, features, [stim])
        bc = load_bc(BC, pd.unique(obs["cell_type"].astype(str)), [stim])
        for drug in a.drugs:
            dmso = dmso_for(BC, drug)
            Xd, obsd, is_test = pair_frame(X, obs, bc, drug, dmso, stim, features, 0.2, 0)
            if Xd is None:
                print(f"[WARN ] {a.celltype}/{stim}/{drug}: no cells"); continue
            tr = ~is_test
            pat = obsd["patient"].astype(str).to_numpy(); trt = obsd["drug"].astype(str).to_numpy()
            both = [p for p in pd.unique(pat) if ((pat == p) & (trt == dmso) & tr).any()
                    and ((pat == p) & (trt == drug) & tr).any()]
            src = [Xd[(pat == p) & (trt == dmso) & tr].astype(np.float64) for p in both]
            tgt = [Xd[(pat == p) & (trt == drug) & tr].astype(np.float64) for p in both]
            fit = gauss_ot_fit if a.method == "gauss_ot" else quantile_map_fit
            params = fit(src, tgt, weighting="cell")
            for k, v in params.items():
                blob[f"{drug}|{stim}|{k}"] = np.asarray(v)
            print(f"[FIT  ] {a.celltype}/{stim}/{drug} vs {dmso}: {sum(map(len, src))} src, "
                  f"{sum(map(len, tgt))} tgt, {len(both)} patients ({time.time() - t0:.0f}s)", flush=True)
        del X, obs; gc.collect()
    blob["features"] = np.array(features); blob["stims"] = np.array(stims); blob["drugs"] = np.array(a.drugs)
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp.npz"); np.savez(tmp, **blob); os.replace(tmp, out)
    print(f"[DONE ] {a.celltype}: {sum(k.endswith(('|A', '|q_src')) for k in blob)} maps -> {out}")


if __name__ == "__main__":
    main()
