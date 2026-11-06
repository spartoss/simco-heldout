"""Paths and constants. The atlas is read-only (group home); everything written goes under WORK.

Definitions mirror /scratch/users/bmerwan/atlas_heldout (read, not imported):
- lc/paths.py: CAP_PER_DONOR = 10_000 cells per (celltype, condition, donor), QS = (10, 25, 50, 75, 90);
- lc/score.py: FLOOR = 0.0273 (atlas technical floor), MIN_CELLS = 20, skill = 1 - err / effect.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "work"
QUANT = WORK / "quant"
SUMMARY = WORK / "summary"
DATA = Path("/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets")                 # read-only
ATLAS = DATA / "ptb_concatenated_batchcorrected"                                       # <celltype>_HVPV.h5ad
GRID = DATA / "ptb_concatenated_per_condition_celltype"
LC_DESIGN = Path("/scratch/users/bmerwan/atlas_heldout/work/lc/design.json")           # read-only: the 5 repeats
LC_CURVE = Path("/scratch/users/bmerwan/atlas_heldout/work/lc/summary/curve.csv")      # read-only: the plateau

CAP_LC = 10_000      # = atlas_heldout lc CAP_PER_DONOR (pooled over the three wells)
CAP_WELL = 30_000    # per-well cap for the "large-n" variant
QS = (10, 25, 50, 75, 90)
FLOOR = 0.0273
MIN_CELLS = 20
WELLS = ("DMSO1", "DMSO2", "DMSO3")


def read_list(p):
    return [l.strip() for l in open(p) if l.strip()]


MARKERS = read_list(GRID / "13features.txt")
PAIRS = [tuple(l.split(",")[:2]) for l in read_list(GRID / "valid_jobs.txt")]   # (stim, celltype), 168
CELLTYPES = sorted({c for _, c in PAIRS})
