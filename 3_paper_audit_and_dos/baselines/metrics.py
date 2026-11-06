"""Scoring for predicted-vs-measured stimulated cell distributions.

``mmd`` comes straight from cellot.losses so the baselines and CellOT are judged by
the same code path that trained CellOT (cellot/train/train.py:76-86).

Note the repo's training-time metric is named MedianError but aggregates with a
mean over markers; ``median_error`` here reproduces it exactly, and ``mean_error``
is the same quantity with the per-marker summary switched from median to mean.
"""

import numpy as np
from cellot.losses import compute_scalar_mmd

# MMD is O(n^2) in cells via a dense RBF kernel, so it is computed on a fixed-size
# subsample. 2000 is ~60x the 256-cell batch cellot/train/train.py:57 evaluates on,
# and the seed is fixed so the CellOT and baseline arms see the same cells.
MMD_SUBSAMPLE = 2000
MMD_SEED = 0
QUANTILES = (10, 25, 75, 90)


def _subsample(x, n, rng):
    if len(x) <= n:
        return x
    return x[rng.choice(len(x), size=n, replace=False)]


def marker_errors(pred, true):
    """Per-marker absolute error of the distribution summary, averaged over markers."""
    mean_err = np.abs(pred.mean(axis=0) - true.mean(axis=0))
    median_err = np.abs(np.median(pred, axis=0) - np.median(true, axis=0))
    return mean_err, median_err


def wasserstein1(pred, true, n_grid=1001):
    """Per-marker 1-D Wasserstein-1 distance, averaged over markers.

    The quantile_map baseline minimises exactly this on the training patients, so it
    is reported as the metric that baseline is built to win -- any CellOT advantage
    has to come from somewhere else.
    """
    u = np.linspace(0.0, 1.0, n_grid)
    qp = np.quantile(pred, u, axis=0)
    qt = np.quantile(true, u, axis=0)
    return np.abs(qp - qt).mean(axis=0)


def correlation_error(pred, true):
    """Frobenius norm of the difference in marker-marker correlation matrices.

    median_shift and quantile_map are diagonal maps: they cannot change the copula at
    all, so on those this only measures how wrong the source correlations already were.
    gauss_ot and CellOT can both move it, which is what makes this the metric that
    separates them from the rest -- and from each other, since gauss_ot matches second
    moments exactly by construction.

    linear_median* sit in between and only in a pooled evaluation: each is a
    translation within a patient, but a different one per patient, so concatenating the
    transported blocks shifts the pooled correlations via the between-patient spread.
    Scored per patient (--eval-mode lopo) they are pinned to identity's value.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        cp = np.corrcoef(pred, rowvar=False)
        ct = np.corrcoef(true, rowvar=False)
    if np.isnan(cp).any() or np.isnan(ct).any():
        return np.nan
    return float(np.linalg.norm(cp - ct, ord="fro"))


def score(pred, true, markers):
    """All metrics for one (method, stim, celltype, held-out patient) cell set."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)

    mean_err, median_err = marker_errors(pred, true)
    w1 = wasserstein1(pred, true)

    rng = np.random.default_rng(MMD_SEED)
    mmd = compute_scalar_mmd(
        _subsample(true, MMD_SUBSAMPLE, rng), _subsample(pred, MMD_SUBSAMPLE, rng)
    )

    row = {
        "mean_error": float(mean_err.mean()),
        "median_error": float(median_err.mean()),
        "w1": float(w1.mean()),
        "mmd": float(mmd),
        "corr_error": correlation_error(pred, true),
    }
    # Per-marker quantile errors |q_pred - q_true| at the quantiles the measured-feature
    # tables use (distrib_features.QS) -- added 2026-09-24; extra columns only, every
    # pre-existing column is computed exactly as before.
    qerr = np.abs(np.percentile(pred, QUANTILES, axis=0) - np.percentile(true, QUANTILES, axis=0))
    for j, q in enumerate(QUANTILES):
        row[f"q{q}_error"] = float(qerr[j].mean())
    # Keep the per-marker breakdown: the aggregate hides that a single marker can
    # carry most of the error, which matters when comparing a diagonal map to CellOT.
    for k, marker in enumerate(markers):
        row[f"mean_error__{marker}"] = float(mean_err[k])
        row[f"median_error__{marker}"] = float(median_err[k])
        row[f"w1__{marker}"] = float(w1[k])
        for j, q in enumerate(QUANTILES):
            row[f"q{q}_error__{marker}"] = float(qerr[j, k])
    return row
