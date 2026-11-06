"""Paths and constants. The atlas is read-only (group home); everything written goes to WORK."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORK = Path("/scratch/users/bmerwan/atlas_heldout/work/lc")
ATLAS = Path("/home/groups/gbrice/ptb-drugscreen/ot/cellot/datasets/ptb_concatenated_batchcorrected")  # read-only
CACHE = WORK / "cache"

CAP_PER_DONOR = 10_000      # cells kept per (celltype, condition, donor)
VAL_FRAC = 0.2              # held out within each TRAINING donor, for paper-style checkpoint selection
QS = (10, 25, 50, 75, 90)   # quantiles scored on the test donors


def read_list(p):
    return [l.strip() for l in open(p) if l.strip()]


MARKERS = read_list(REPO / "meta/markers.txt")
DONORS = read_list(REPO / "meta/donors.txt")
PAIRS = [tuple(l.split(",")[:2]) for l in read_list(REPO / "meta/pairs.txt")]   # (stim, celltype)
CELLTYPES = sorted({c for _, c in PAIRS})
