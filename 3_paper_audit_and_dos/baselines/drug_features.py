"""Drug-treated feature tables for the OOL cohort under three DOS-model feature sets, with the
drug perturbation held fixed (the paper's CellOT drug maps). Question: does the Fig 5d/5e drug
ranking depend on which features the DOS model was built on?

For one drug, per OOL woman and celltype (13 functional markers):
  Unstim block   x -> CellOT Unstim->drug map (results_unstim/.../model-Unstim_<ct>)
  stim blocks    x -> stim map -> CellOT drug map (results_drug/.../<drug>/<stim>/<ct>)
                 with the stim map = CellOT (paper; reproduces cellot_results/ina_13OG_<drug>_pred_unstim.csv)
                               or   = gauss_ot (baselines/gaussot_maps)
  measured       night2/distrib_all_v2.csv with the functional `_med` and `_qtl` columns recomputed on
                 the Unstim->drug-transported cells; phenotypic medians and cell frequencies unchanged
                 (the drug maps output only the 13 functional markers; 2 h of drug exposure is assumed
                 not to move phenotype or frequencies).

Writes <out>/<drug>_{cellot,gaussot}.csv (1216-col schema of the paper table) and <out>/<drug>_distrib.csv.
    python -m baselines.drug_features --drug THF --out <dir> [--patients-limit 2 --celltypes pDCs]
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")  # group-home NFS, errno 121

import numpy as np
import pandas as pd
import torch

from baselines.baselines import gauss_ot_apply
from baselines.null_map_features import patient_cells, transport
from baselines.simco_features import CELLTYPE_NAME_MAP, PERCOND, read_list, valid_pairs

CR = Path("/home/groups/gbrice/ptb-drugscreen/ot/cellot")
STIM = CR / "results/cross_validation_original_13_HVPV"
DRUG = CR / "results_drug/cross_validation_drug_OG13"
UNST = CR / "results_unstim/cross_validation_drug_OG13"
GMAPS = Path("/scratch/users/bmerwan/simco_work/baselines/gaussot_maps")
ONSET = Path("/scratch/users/bmerwan/simco_work/onset_test")
QS = (10, 25, 75, 90)


def load_net(d):
    from cellot.models.cellot import load_networks
    from cellot.utils.helpers import load_config
    _, g = load_networks(load_config(d / "config.yaml"), input_dim=13)
    g.load_state_dict(torch.load(d / "cache/model.pt", map_location="cpu")["g_state"])
    g.eval()
    for p in g.parameters():
        p.requires_grad_(False)
    return g


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drug", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--patients-limit", type=int, default=None)
    ap.add_argument("--celltypes", nargs="*", default=None)
    a = ap.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))

    func = read_list(PERCOND / "13features.txt")
    markers = read_list(CR / "datasets/ool_by_patient/features_final.txt")
    fidx = [markers.index(m) for m in func]
    pairs = [(s, c) for s, c in valid_pairs() if a.celltypes is None or c in a.celltypes]
    cts = sorted({c for _, c in pairs})
    t0 = time.time()
    stim_net = {(s, c): load_net(STIM / s / c / f"model-{s}_{c}") for s, c in pairs}
    drug_net = {(s, c): load_net(DRUG / a.drug / s / c / f"model-{s}_{c}") for s, c in pairs}
    unst_net = {c: load_net(UNST / a.drug / c / f"model-Unstim_{c}") for c in cts}
    gmap = {}
    for c in cts:
        z = np.load(GMAPS / f"{c}.npz", allow_pickle=True)
        gmap[c] = {s: {k: z[f"{s}|{k}"] for k in ("mu_s", "mu_t", "A")} for s in map(str, z["stims"])}
    print(f"[INFO ] {a.drug}: {len(pairs)} pairs, nets loaded ({time.time() - t0:.0f}s)", flush=True)

    patients = read_list(PERCOND / "patients_ina.txt")[:a.patients_limit]
    rows = {"cellot": {}, "gaussot": {}, "distrib": {}}
    for n, pt in enumerate(patients, 1):
        X, ct_obs = patient_cells(pt, markers)
        r_c, r_g, r_d = {}, {}, {}
        for c in cts:
            x = X[ct_obs == CELLTYPE_NAME_MAP.get(c, c)][:, fidx].astype(np.float32)
            if not len(x):
                continue
            u = transport(unst_net[c], x)                              # Unstim -> drug
            for m, v in zip(func, np.median(u, 0)):
                r_c[f"{c}_{m}_Unstim"] = r_g[f"{c}_{m}_Unstim"] = v
                r_d[f"{c}_{m}_med"] = v
            if len(x) >= 20:
                q = np.percentile(u, QS, axis=0)
                for k, qk in enumerate(QS):
                    for m, v in zip(func, q[k]):
                        r_d[f"{c}_{m}_q{qk}_qtl"] = v
            for s in [s for s, cc in pairs if cc == c]:
                yc = transport(stim_net[(s, c)], x)                    # CellOT stim
                yg = gauss_ot_apply(x.astype(np.float64), gmap[c][s]).astype(np.float32)  # gauss_ot stim
                for tag, ys, r in (("c", yc, r_c), ("g", yg, r_g)):
                    dd = transport(drug_net[(s, c)], ys.astype(np.float32))   # stim -> stim + drug
                    for m, v in zip(func, np.median(dd, 0)):
                        r[f"{c}_{m}_{s}"] = v
        rows["cellot"][pt], rows["gaussot"][pt], rows["distrib"][pt] = r_c, r_g, r_d
        print(f"[INFO ] {a.drug} {n}/{len(patients)} {pt} ({time.time() - t0:.0f}s)", flush=True)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(ONSET / "ina_13OG_final_long_allstims_filtered.csv", index_col=0, nrows=1).columns
    for tag in ("cellot", "gaussot"):
        t = pd.DataFrame.from_dict(rows[tag], orient="index")
        t.index.name = "Individual"
        if a.celltypes is None:
            t = t.reindex(columns=ref)
        tmp = out / f"{a.drug}_{tag}.{os.getpid()}.tmp"; t.to_csv(tmp); tmp.rename(out / f"{a.drug}_{tag}.csv")
    base = pd.read_csv(ONSET / "night2/distrib_all_v2.csv", index_col=0)
    base = base.loc[[p for p in base.index if p in rows["distrib"]]]
    upd = pd.DataFrame.from_dict(rows["distrib"], orient="index").reindex(index=base.index)
    cols = [c for c in upd.columns if c in base.columns]
    base[cols] = upd[cols]
    tmp = out / f"{a.drug}_distrib.{os.getpid()}.tmp"; base.to_csv(tmp); tmp.rename(out / f"{a.drug}_distrib.csv")
    print(f"[DONE ] {a.drug}: {len(patients)} patients, distrib cols replaced {len(cols)} -> {out}")


if __name__ == "__main__":
    main()
