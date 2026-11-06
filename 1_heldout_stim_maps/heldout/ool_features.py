"""Simulated stim features for the 177 OOL women: q10/q25/q50/q75/q90 of every (stim, celltype, marker), 196 pairs.

Maps: CellOT stim maps trained on all 19 donors for all 196 pairs (work/heldout/stims_all, lc/train.py, last iterate),
and gauss_ot fitted on the same cells (cache_stim_all, all cells of all donors). Each woman's unstimulated cells of a
celltype (the OOL per-patient h5ad files, read-only) are transported and summarised. Columns:
`<ct>_<marker>_<stim>` (median, the paper's naming) and `<ct>_<marker>_q<k>_<stim>q` (k = 10, 25, 75, 90).
Loader and celltype-label map copied from simco/baselines/{null_map_features,simco_features}.py; the cells are read
from the per-woman cache written by heldout.ool_cache (CPU env), because the GPU env has no anndata / h5py.
    python -m heldout.ool_features  -> WORK/features/ool_{cellot,gauss_ot}_allq.csv
"""
import os
import time

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from heldout.common import CACHE_STIM_ALL, OOL, WORK  # noqa: E402
from lc.paths import MARKERS  # noqa: E402

QS = (10, 25, 50, 75, 90)
PATIENTS = [l.strip() for l in open("/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_per_condition_celltype/patients_ina.txt") if l.strip()]
OOL_LABEL = {"CD4Tcm_CCR2pos": "CD4Tcm CCR2+", "CD4Tem_CCR2pos": "CD4Tem CCR2+", "CD8Tcells_Th1": "CD8Tcells Th1",
             "CD8Tcm_CCR2pos": "CD8Tcm CCR2+", "CD8Tem_CCR2pos": "CD8Tem CCR2+", "NK_cells_CD11cpos": "NK cells CD11c+",
             "NK_cells_CD11cneg": "NK cells CD11c-"}      # every other celltype keeps its name


def patient_cells(p, tries=5):
    import anndata as ad       # CPU env only (heldout.ool_cache); the GPU env reads the npz cache instead
    for k in range(tries):
        try:
            a = ad.read_h5ad(OOL / f"{p}_Unstim.h5ad"); break
        except OSError:
            if k == tries - 1: raise
            time.sleep(10 * (k + 1))
    X = np.asarray(a.X)[:, [a.var_names.get_loc(m) for m in MARKERS]].astype(np.float32)
    return X, a.obs["cell_type"].astype(str).to_numpy()


def cols(ct, stim):
    return {q: [f"{ct}_{m}_{stim}" if q == 50 else f"{ct}_{m}_q{q}_{stim}q" for m in MARKERS] for q in QS}


def main():
    import torch
    from lc.batched_icnn import BatchedICNN
    from lc.baselines import apply as cf_apply, fit as cf_fit
    from lc.train import sub
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = WORK / "stims_all"; maps = pd.read_csv(d / "maps.csv")
    st = torch.load(d / "final_maps.pt", map_location=dev)["g_last"]
    M = len(maps); g = BatchedICNN(M, len(MARKERS), (64, 64, 64, 64), fnorm_penalty=1).to(dev); g.load_state_dict(st)
    gauss = {}
    for ct in maps.celltype.unique():
        z = np.load(CACHE_STIM_ALL / f"{ct}.npz"); donors = sorted({k.split("|")[1] for k in z.files})
        for s in maps[maps.celltype == ct].stim:
            ds = [p for p in donors if f"Unstim|{p}" in z.files and f"{s}|{p}" in z.files]
            gauss[(s, ct)] = cf_fit(np.concatenate([z[f"Unstim|{p}"] for p in ds]), np.concatenate([z[f"{s}|{p}"] for p in ds]))
    by_ct = {ct: g_.index.tolist() for ct, g_ in maps.groupby("celltype")}
    rows = {"cellot": {}, "gauss_ot": {}}; t0 = time.time(); qt = torch.tensor([q / 100 for q in QS], device=dev)
    for n, p in enumerate(PATIENTS, 1):
        zc = np.load(WORK / f"cache_ool/{p}.npz"); r = {"cellot": {}, "gauss_ot": {}}
        for ct, idx in by_ct.items():
            x = zc[ct] if ct in zc.files else np.zeros((0, len(MARKERS)), np.float32)
            if len(x) < 20: continue
            xt = torch.from_numpy(x).to(dev)
            with torch.enable_grad():
                xb = xt.unsqueeze(0).expand(len(idx), -1, -1).clone().requires_grad_(True)
                T = sub(g, torch.tensor(idx, device=dev)).transport(xb).detach()
            Q = torch.quantile(T, qt, dim=1).cpu().numpy() if T.numel() < 15_000_000 else \
                np.stack([np.percentile(T[i].cpu().numpy(), QS, axis=0) for i in range(len(idx))], 1)
            for ii, mi in enumerate(idx):
                s = maps.stim[mi]; c = cols(ct, s)
                for qi, q in enumerate(QS): r["cellot"].update(dict(zip(c[q], Q[qi, ii])))
                qg = np.percentile(cf_apply("gauss_ot", x, gauss[(s, ct)]), QS, axis=0)
                for qi, q in enumerate(QS): r["gauss_ot"].update(dict(zip(c[q], qg[qi])))
        for k in rows: rows[k][p] = r[k]
        print(f"[INFO ] {n}/{len(PATIENTS)} {p} ({time.time() - t0:.0f}s)", flush=True)
    out = WORK / "features"; out.mkdir(parents=True, exist_ok=True)
    for k, v in rows.items():
        t = pd.DataFrame.from_dict(v, orient="index"); t.index.name = "Individual"
        tmp = out / f"ool_{k}_allq.{os.getpid()}.tmp"; t.to_csv(tmp); tmp.rename(out / f"ool_{k}_allq.csv")
        print(f"[DONE ] {k}: {t.shape}, NaN {int(t.isna().sum().sum())}")


if __name__ == "__main__":
    main()
