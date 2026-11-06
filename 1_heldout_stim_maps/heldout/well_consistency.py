"""Consistency of each donor's cell counts and population frequencies across the atlas wells. Used by
notebooks/well_consistency.ipynb (builder heldout/build_well_consistency_notebook.py).

Input: work/heldout/pheno/<celltype>.csv.gz (heldout/pheno_summary.py), which has the exact cell count of every
(stim, treatment, donor, celltype). A well = (stim, treatment) and holds all 19 donors, barcoded. The plate of a
treatment is its control in batchcorrection_2.csv (use_DMSO): plate 1 = DMSO1 SA LPZ, plate 2 = DMSO2 CHT MAP MF PRA,
plate 3 = DMSO3 RIF SALPZ THF.

Decomposition (balanced grid: every donor in every well). For y = log(count) or log(frequency) of one celltype:
    y[d, s, t] = mean + well[s, t] + donor[d] + donor x stim[d, s] + donor x treatment[d, t] + residual[d, s, t]
well = shared by every donor in the well (stim effect, plate staining, drug); donor = what we want for a frequency;
the last three are the donor's wells disagreeing with each other. Each term is reported as an RMS on the log scale
(0.05 = about 5 %). The sampling floor comes from redrawing every count from Poisson around the donor + well model
(nothing donor-specific per well) and decomposing the redraw the same way.
    python -m heldout.well_consistency   -> WORK/well_consistency/counts.csv.gz
"""
import glob

import numpy as np
import pandas as pd

from heldout.common import WORK

OUT = WORK / "well_consistency"
PLATE = {"DMSO1": 1, "SA": 1, "LPZ": 1, "DMSO2": 2, "CHT": 2, "MAP": 2, "MF": 2, "PRA": 2,
         "DMSO3": 3, "RIF": 3, "SALPZ": 3, "THF": 3}
MAIN = ["Unstim", "GMCSF", "IFNa", "IL246", "IL33", "TNFa"]     # stims that leave the gating markers alone (PI apart)
TERMS = ["well", "donor", "donor_x_stim", "donor_x_treatment", "residual"]


def counts():
    """Long table (stim, treatment, donor, celltype, n_cells), cached."""
    f = OUT / "counts.csv.gz"
    if f.exists(): return pd.read_csv(f)
    P = pd.concat([pd.read_csv(p, usecols=["celltype", "stim", "treatment", "donor", "n_cells", "marker"])
                   .query("marker == 'CD45'").drop(columns="marker") for p in sorted(glob.glob(str(WORK / "pheno/*.csv.gz")))],
                  ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True); P.to_csv(f, index=False); return P


def grid(P, stims):
    """Wide count matrix, rows (stim, treatment, donor), columns celltypes; a missing population counts 0."""
    return P[P.stim.isin(stims)].pivot_table(index=["stim", "treatment", "donor"], columns="celltype",
                                              values="n_cells", fill_value=0)


def decompose(y):
    """y: Series indexed (stim, treatment, donor). Returns a frame with one column per term (same index)."""
    df = y.rename("y").reset_index(); g = df.y.mean()
    df["well"] = df.groupby(["stim", "treatment"]).y.transform("mean") - g
    don = df.groupby("donor").y.transform("mean") - g
    r = df.y - g - df.well - don
    df["donor_x_stim"] = r.groupby([df.donor, df.stim]).transform("mean")
    df["donor_x_treatment"] = r.groupby([df.donor, df.treatment]).transform("mean")
    df["residual"] = r - df.donor_x_stim - df.donor_x_treatment
    out = df.set_index(["stim", "treatment", "donor"])
    out["donor"] = don.values
    return out[TERMS]


def rms(D):
    return np.sqrt((D ** 2).mean())


def freq_table(N, n_null=5, seed=0):
    """Per celltype: RMS of each term of log frequency, the same under Poisson redraws, and the median count."""
    tot = N.sum(1); rng = np.random.default_rng(seed); rows = []
    for c in N.columns:
        y = np.log((N[c] + 0.5) / tot); D = decompose(y)
        mu = np.exp(y.mean() + D.well + D.donor) * tot
        null = pd.concat([rms(decompose(pd.Series(np.log((rng.poisson(mu.values) + 0.5) / tot.values), index=y.index)))
                          for _ in range(n_null)], axis=1).mean(1)
        rows.append({"celltype": c, "median_cells": N[c].median(), **rms(D).to_dict(),
                     **{f"null_{k}": null[k] for k in TERMS[2:]}})
    T = pd.DataFrame(rows).set_index("celltype")
    T["within"] = np.sqrt(T.donor_x_stim ** 2 + T.donor_x_treatment ** 2 + T.residual ** 2)
    T["null_within"] = np.sqrt(T.null_donor_x_stim ** 2 + T.null_donor_x_treatment ** 2 + T.null_residual ** 2)
    T["excess"] = T.within / T.null_within
    T["between_over_within"] = T.donor / T.within
    return T


if __name__ == "__main__":
    P = counts(); print(P.shape, "->", OUT / "counts.csv.gz")
