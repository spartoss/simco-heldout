"""Per-well table for notebooks/stim_cross_correlations.ipynb.

Input: atlas_heldout/work/heldout/pheno/<celltype>.csv.gz (heldout/pheno_summary.py): one row per
(celltype, stim, treatment, donor, marker), median (q50) of <= 20k cells, raw arcsinh, no batch correction,
plus the exact cell count of that celltype in that well. Only the plate controls are kept
(treatment DMSO1/2/3 = no drug), Unstim included, so a "well" = (stim, plate, donor).
Output: simco_work/stim_correlations/wells_q50.csv.gz (long) and wells_counts.csv.gz.
    python baselines/stim_corr_wells.py
"""
from pathlib import Path

import pandas as pd

PHENO = Path("/scratch/users/bmerwan/atlas_heldout/work/heldout/pheno")
OUT = Path("/scratch/users/bmerwan/simco_work/stim_correlations")
OUT.mkdir(parents=True, exist_ok=True)

parts, counts = [], []
for f in sorted(PHENO.glob("*.csv.gz")):
    d = pd.read_csv(f, usecols=["celltype", "stim", "treatment", "donor", "n_cells", "marker", "q50"])
    d = d[d.treatment.isin(["DMSO1", "DMSO2", "DMSO3"])].rename(columns={"treatment": "plate"})
    counts.append(d.drop_duplicates(["celltype", "stim", "plate", "donor"])[["celltype", "stim", "plate", "donor", "n_cells"]])
    parts.append(d.drop(columns="n_cells"))
    print(f.name, len(d), flush=True)
pd.concat(parts).to_csv(OUT / "wells_q50.csv.gz", index=False)
pd.concat(counts).to_csv(OUT / "wells_counts.csv.gz", index=False)
