"""Panel cells (arcsinh(x/5), batch-corrected, fm cache columns) -> ImmuneFM token rows (n, 204).

`paper` reproduces ImmuneFM's pre-training preprocessing (code/pretrain.py): the atlas is already arcsinh(x/5)
(simco convert_fcs_to_h5, cofactor 5, the same as pre-training); per cell, the non-zero values of the panel are
MinMax-scaled to [1, 10] across that cell's markers and truncated to int (SCDataset's .long()). Zero stays 0 = the
token of absent markers too. The scaling is per CELL, so a marker's token is relative to the cell's other markers.
`mask_cols` sets the given panel columns to MASK (e.g. the functional markers, for a stim-blind phenotype context).
"""
from pathlib import Path

import numpy as np

from fm.immunefm import MASK, N_BINS, SEQ_LEN, markers

REPO = Path(__file__).resolve().parents[1]


def marker_map():
    rows = [l.split("\t") for l in (REPO / "meta/fm_marker_map.tsv").read_text().splitlines()[1:] if l]
    return {p: f for p, f in rows}


def columns(panel, shuffle_seed=None):
    """For the cache's panel order: (indices into the panel of the mapped markers, their ImmuneFM columns).
    shuffle_seed permutes the ImmuneFM columns among themselves (control: does the model use marker identity?)."""
    mp, vocab = marker_map(), markers()
    src = np.array([i for i, m in enumerate(panel) if m in mp])
    dst = np.array([vocab.index(mp[panel[i]]) for i in src])
    if shuffle_seed is not None:
        dst = np.random.default_rng(shuffle_seed).permutation(dst)
    return src, dst


def paper_bins(x):
    """x (n, p) arcsinh values -> int64 tokens 0..N_BINS, per-cell MinMax of non-zero values to [1, N_BINS]."""
    x = np.asarray(x, np.float64)
    nz = x != 0
    lo = np.where(nz, x, np.inf).min(1, keepdims=True)
    hi = np.where(nz, x, -np.inf).max(1, keepdims=True)
    lo = np.where(np.isfinite(lo), lo, 0.0); hi = np.where(np.isfinite(hi), hi, 0.0)
    rng = hi - lo
    # sklearn MinMaxScaler arithmetic exactly (scale_, min_; zero range -> scale 1), so the cell max sometimes lands
    # at 9.999.. and truncates to 9, as it did in pre-training
    scale = (N_BINS - 1) / np.where(rng == 0, 1.0, rng)
    v = x * scale + (1 - lo * scale)
    return np.where(nz, v, 0).astype(np.int64)


def tokens(x, panel, mask_cols=(), shuffle_seed=None):
    """x (n, len(panel)) -> (n, SEQ_LEN) int64 token rows; unmapped vocabulary columns and the trailing one are 0."""
    src, dst = columns(panel, shuffle_seed)
    t = np.zeros((len(x), SEQ_LEN), np.int64)
    t[:, dst] = paper_bins(np.asarray(x)[:, src])
    for m in mask_cols:
        if m in panel and panel.index(m) in src:
            t[:, dst[list(src).index(panel.index(m))]] = MASK
    return t


def bin_width(x, panel):
    """Per cell, the arcsinh width of one token bin: (max - min of non-zero mapped values) / (N_BINS - 1)."""
    src, _ = columns(panel)
    v = np.asarray(x, np.float64)[:, src]
    nz = v != 0
    hi = np.where(nz, v, -np.inf).max(1); lo = np.where(nz, v, np.inf).min(1)
    return (hi - lo) / (N_BINS - 1)


# ---- per-marker quantile tokens (fm/finetune.py): absolute levels, finer bins, all 38 panel markers ----
Q_BINS = 64                    # value tokens 1..Q_BINS; 0 = exact zero; Q_BINS + 1 = MASK
Q_MASK = Q_BINS + 1            # 64 bins: 2.5 % of live stim responses leave the median token unmoved (32: 9 %, paper tokens: 63 %)
# the 5 panel markers absent from the vocabulary take columns of markers our panel never measures:
# CD45RA -> the CD45 isoform column, the others -> pathogen antigens (no prior meaning for blood phenotype)
EXTRA = {"CD45RA": "B220/CD45R", "CD44": "NP", "CD36": "P24", "GLUT1": "4G2", "pSTAT4": "NS3"}


def panel_pos(panel):
    """Vocabulary column of every panel marker (mapped ones via meta/fm_marker_map.tsv, the rest via EXTRA)."""
    mp, vocab = {**marker_map(), **EXTRA}, markers()
    return np.array([vocab.index(mp[m]) for m in panel])


def q_edges(x, n_bins=Q_BINS):
    """x (n, p) cells (training donors only) -> (p, n_bins - 1) interior quantile edges of each marker's non-zero values."""
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return np.stack([np.quantile(v[v != 0], qs) if (v != 0).any() else np.zeros(len(qs))
                     for v in np.asarray(x, np.float64).T])


def q_tokens(x, edges):
    """x (n, p) -> uint8 tokens: 0 for zero, else 1 + bin index over that marker's edges (1..Q_BINS)."""
    x = np.asarray(x)
    t = np.empty(x.shape, np.uint8)
    for j in range(x.shape[1]):
        t[:, j] = np.where(x[:, j] != 0, 1 + np.searchsorted(edges[j], x[:, j], side="right"), 0)
    return t
