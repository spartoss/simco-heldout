"""Irreducible-noise floor of every scored case (stim, celltype, donor, quantile, marker), from the per-well quantiles of
the well_ceiling project (/scratch/users/bmerwan/well_ceiling/work/quant/<ct>.npz, read-only; built from exactly the cells
of the learning-curve cache, so "pool" equals our truth).

The truth is the quantile of the donor's cells pooled over its three DMSO plate-control wells. Even a perfect model of
the donor's response misses it by the measurement noise of that pooled value. Three estimates of that noise variance:
- floor_samp: finite cells only, (h1 - h2)^2 / 4 from the two random halves of the same cells;
- floor_dxw:  donor x well noise, the realistic floor. The well values minus the well effect shared by all donors (the
  mean over donors of well - pool for that stim / celltype / well; a model trained on the same wells can learn it),
  spread around their count-weighted mean (unbiased weighted variance s^2), scaled to the pooled value: s^2 * sum w_j^2;
- floor_all:  the same without removing the shared well effect (upper bound).
- floor_resp: donor x well noise of the RESPONSE (stim - Unstim, well by well, shared effect removed). Every method
  reads the donor's measured Unstim values, so their noise reaches the prediction too; this is the realistic floor
  (the correction noted in work/lc/STATUS.md, 2026-10-14). Wells with < 20 cells in either condition are skipped.
MAE floors assume normal noise: E|e| = sigma * sqrt(2 / pi).
Writes WORK/tab/final/noise_floor.pkl. (Separate from the earlier k15 noise analysis, whose outputs are in
work/lc/noise_floor/; its script, also named lc/noise_floor.py, was overwritten by mistake on 2026-10-14.)
    python -m lc.noise_floor_lodo
"""
from pathlib import Path

import numpy as np
import pandas as pd

from lc.paths import MARKERS, PAIRS, QS, WORK

QUANT = Path("/scratch/users/bmerwan/well_ceiling/work/quant")
WELLS = ("DMSO1", "DMSO2", "DMSO3")
OUT = WORK / "tab" / "final"


def wvar(vals, w):
    """Unbiased weighted variance of the rows of vals (n_wells, 5, 13) with weights w (sum 1), and sum w^2."""
    m = np.tensordot(w, vals, 1); s2w = (w ** 2).sum()
    return np.tensordot(w, (vals - m) ** 2, 1) / (1 - s2w), s2w


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ct in sorted({c for _, c in PAIRS}):
        z = np.load(QUANT / f"{ct}.npz")
        keys = set(z.files)
        stims = sorted({k.split("|")[0] for k in keys} - {"Unstim"})
        for s in stims:
            donors = sorted({k.split("|")[1] for k in keys if k.startswith(s + "|")})
            # shared well effect: mean over donors of (well - pool)
            eff = {w: np.nanmean([z[f"{s}|{d}|w:{w}"] - z[f"{s}|{d}|pool"] for d in donors if f"{s}|{d}|w:{w}" in keys], 0)
                   for w in WELLS if any(f"{s}|{d}|w:{w}" in keys for d in donors)}
            ok_r = lambda d, w: all(f"{c}|{d}|w:{w}" in keys and float(z[f"{c}|{d}|w:{w}|n"]) >= 20 for c in (s, "Unstim"))
            resp = lambda d, w: z[f"{s}|{d}|w:{w}"] - z[f"Unstim|{d}|w:{w}"]
            effr = {w: np.nanmean([resp(d, w) - (z[f"{s}|{d}|pool"] - z[f"Unstim|{d}|pool"]) for d in donors if ok_r(d, w)], 0)
                    for w in WELLS if any(ok_r(d, w) for d in donors)}
            for d in donors:
                ws = [w for w in WELLS if f"{s}|{d}|w:{w}" in keys and float(z[f"{s}|{d}|w:{w}|n"]) >= 20]
                h1, h2 = z[f"{s}|{d}|h1"], z[f"{s}|{d}|h2"]
                samp = (h1 - h2) ** 2 / 4
                if len(ws) >= 2:
                    n = np.array([float(z[f"{s}|{d}|w:{w}|n"]) for w in ws]); wt = n / n.sum()
                    V = np.stack([z[f"{s}|{d}|w:{w}"] for w in ws])
                    va, s2w = wvar(V, wt); vd, _ = wvar(V - np.stack([eff[w] for w in ws]), wt)
                    fa, fd = va * s2w, vd * s2w
                else:
                    fa = fd = np.full((5, 13), np.nan)
                wr = [w for w in WELLS if ok_r(d, w)]
                if len(wr) >= 2:
                    n = np.array([min(float(z[f"{s}|{d}|w:{w}|n"]), float(z[f"Unstim|{d}|w:{w}|n"])) for w in wr]); wt = n / n.sum()
                    vr, s2w = wvar(np.stack([resp(d, w) - effr[w] for w in wr]), wt); fr = vr * s2w
                else:
                    fr = np.full((5, 13), np.nan)
                for qi, q in enumerate(QS):
                    for mi, m in enumerate(MARKERS):
                        rows.append((s, ct, d, q, m, samp[qi, mi], fd[qi, mi], fa[qi, mi], fr[qi, mi], len(ws)))
        print(f"[NOISE] {ct}", flush=True)
    F = pd.DataFrame(rows, columns=["stim", "celltype", "donor", "q", "marker", "floor_samp", "floor_dxw", "floor_all",
                                   "floor_resp", "n_wells"])
    F.to_pickle(OUT / "noise_floor.pkl")
    print(F.groupby("stim")[["floor_samp", "floor_dxw", "floor_all", "floor_resp"]].mean().round(5).to_string())
    print(f"[DONE ] {len(F)} cases -> {OUT / 'noise_floor.pkl'}")


if __name__ == "__main__":
    main()
