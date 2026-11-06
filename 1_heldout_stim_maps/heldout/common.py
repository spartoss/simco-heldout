"""Shared constants for the held-out experiments (drugs, quantile features). The learning-curve package `lc` holds the
atlas cache, the batched CellOT trainer and the closed-form maps; this package reuses them."""
from pathlib import Path

from lc.paths import DONORS, MARKERS, PAIRS, REPO  # noqa: F401

WORK = Path("/scratch/users/bmerwan/atlas_heldout/work/heldout")
LC = Path("/scratch/users/bmerwan/atlas_heldout/work/lc")
CACHE_STIM_ALL = WORK / "cache_stim_all"      # DMSO wells pooled, uncorrected, Unstim + all 7 stims, <= 10k cells
CACHE_DRUG = WORK / "cache_drug"              # per (stim, treatment, donor), batch-corrected, <= 5k cells
BC = Path("/home/groups/gbrice/ptb-drugscreen/ot/cellot/cross_validation/batchcorrection_2.csv")   # read-only
OOL = Path("/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ool_by_patient/unstim_final")    # read-only
SIMCO_WORK = Path("/scratch/users/bmerwan/simco_work")                                            # read-only inputs
STIMS = ["GMCSF", "IFNa", "IL246", "IL33", "LPS", "PI", "TNFa"]
ALL_PAIRS = [(s, c) for c in sorted({c for _, c in PAIRS}) for s in STIMS]            # 196
DRUGS = ["SA", "RIF", "SALPZ", "CHT", "THF", "LPZ", "MAP", "PRA", "MF"]
CONTROLS = ["DMSO1", "DMSO2", "DMSO3"]
FLOOR = 0.0273
