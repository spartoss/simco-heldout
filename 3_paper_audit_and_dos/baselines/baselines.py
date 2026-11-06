"""Closed-form perturbation-response baselines to benchmark CellOT against.

Every baseline is fit on a set of training patients and applied to a held-out
patient's control (Unstim) cells, exactly the regime the CellOT maps are used in:
per (stim, celltype), transport Unstim cells to the stimulated distribution.

None of them has anything to optimise iteratively -- a fit is order statistics, a
13x13 eigendecomposition, or a least-squares solve.

    identity             T(x) = x                  (floor: is there any signal at all)
    median_shift         T(x) = x + d              d constant per marker
    quantile_map         T_j(x) = F_k^-1(F_c(x))   1-D optimal transport, per marker
    gauss_ot             T(x) = m_t + A(x - m_s)   affine; the W2-optimal map between
                                                   Gaussians (Bures-Wasserstein)
    linear_median        T(x) = x + (a + Bu - u)   u = this patient's own Unstim median;
    linear_median_ridge                            a + Bu is a regression, across
                                                   patients, of the stimulated median
                                                   on the unstimulated one

Three groups, by what structure each can express:

  * ``median_shift`` / ``quantile_map`` are **diagonal**: they never mix markers, so
    they cannot change the copula of the source cells at all.
  * ``gauss_ot`` is **affine and full-rank**: it does mix markers, and it is the first
    baseline that can move a marker-marker correlation. It is what CellOT has to beat
    before any claimed advantage can be attributed to *non-linear* structure rather
    than to second-order structure alone.
  * ``linear_median*`` are **patient-adaptive**: the map applied to cells is only a
    translation, but its size is predicted from the held-out patient's own Unstim
    profile using all 13 markers. Every other baseline here (and CellOT itself)
    applies one fixed map to every patient; these ask whether knowing a patient's
    baseline state tells you how they will respond.

Every fit takes a ``weighting``:

    unweighted  per-patient statistics, then an equal-weight average over patients.
                The better estimator -- a patient contributing ten times as many
                cells does not dominate.
    cell        pool every training patient's cells and fit one statistic on the
                pooled array. Worse as an estimator, but it is what CellOT sees:
                its dataloader samples cells, not patients, so a large patient
                does dominate its training.

The ``cell`` arm exists so the comparison against CellOT can be made like for like;
the gap between the two arms measures what the weighting choice alone is worth.
"""

import numpy as np

# Probability grid on which averaged quantile functions are tabulated. 1001 points
# resolves the tails finely enough that the interpolation error is far below the
# patient-to-patient spread we are trying to measure.
N_GRID = 1001


def _u_grid(n_grid=N_GRID):
    return np.linspace(0.0, 1.0, n_grid)


def _strictly_increasing(q):
    """Nudge a non-decreasing tabulated quantile function to be strictly increasing.

    ``np.interp`` requires increasing ``xp``; flat runs (repeated values, common at
    the floor of a marker with many zero-ish cells) otherwise make the inverse
    lookup pick an arbitrary point of the plateau. Adding a ramp of order 1e-9
    per grid point makes the inverse well defined without moving any value
    perceptibly.
    """
    ramp = np.arange(q.shape[0], dtype=np.float64) * 1e-9
    return np.maximum.accumulate(q) + ramp


# --------------------------------------------------------------------------- #
# per-patient summaries
# --------------------------------------------------------------------------- #


WEIGHTINGS = ("unweighted", "cell")


def patient_stats(x):
    """One patient's cells reduced to what the moment-based fits need.

    Computed once per (patient, stim) by the caller and threaded through the ``stats``
    argument, because the leave-one-out loop refits 19 times on overlapping patient
    sets and these summaries do not change between those refits. Covariance uses the
    population convention (divide by n, not n-1) so that per-patient covariances pool
    exactly -- see _pool_gaussian.
    """
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    centred = x - mean
    return {
        "n": len(x),
        "median": np.median(x, axis=0),
        "mean": mean,
        "cov": (centred.T @ centred) / len(x),
    }


def _stats(src_by_patient, tgt_by_patient, stats):
    """Use the caller's precomputed summaries, or derive them on the spot."""
    if stats is not None:
        return stats["src"], stats["tgt"]
    return (
        [patient_stats(x) for x in src_by_patient],
        [patient_stats(x) for x in tgt_by_patient],
    )


# --------------------------------------------------------------------------- #
# identity
# --------------------------------------------------------------------------- #


def identity_fit(src_by_patient, tgt_by_patient, weighting="unweighted", stats=None):
    """Invariant to both weighting and which patients are in the fit -- nothing is fit.

    Useful as a consistency check: its scores must be identical across every arm.
    """
    return {}


def identity_apply(x, params):
    return np.asarray(x, dtype=np.float64)


# --------------------------------------------------------------------------- #
# median shift
# --------------------------------------------------------------------------- #


def median_shift_fit(src_by_patient, tgt_by_patient, weighting="unweighted", stats=None):
    """d_j = median_j(target) - median_j(source), per marker.

    ``unweighted``: per-patient medians first, then an equal-weight mean over patients.
    ``cell``: one median over the pooled cells -- note this is the median of the pool,
    not a cell-count-weighted mean of the per-patient medians; the two differ, and the
    pooled median is the one that matches what CellOT's cell-level sampling sees. The
    pooled median is not recoverable from per-patient summaries, so this arm always
    touches the cells even when ``stats`` is supplied.
    """
    if weighting == "cell":
        src_med = np.median(np.concatenate(src_by_patient), axis=0)
        tgt_med = np.median(np.concatenate(tgt_by_patient), axis=0)
        return {"delta": tgt_med - src_med}

    src_stats, tgt_stats = _stats(src_by_patient, tgt_by_patient, stats)
    src_med = np.stack([s["median"] for s in src_stats])
    tgt_med = np.stack([s["median"] for s in tgt_stats])
    return {"delta": tgt_med.mean(axis=0) - src_med.mean(axis=0)}


def median_shift_apply(x, params):
    return np.asarray(x, dtype=np.float64) + params["delta"]


# --------------------------------------------------------------------------- #
# per-marker 1-D optimal transport (quantile mapping)
# --------------------------------------------------------------------------- #


def quantile_map_fit(src_by_patient, tgt_by_patient, weighting="unweighted", stats=None,
                     n_grid=N_GRID):
    """Tabulate source and target quantile functions on a shared probability grid.

    ``unweighted``: the 1-D Wasserstein-2 barycentre of the per-patient quantile
    functions. In one dimension that barycentre is just their pointwise average, so
    averaging ``np.quantile(patient, u)`` *is* the barycentre -- no iteration needed --
    and it keeps patients equally weighted, matching median_shift_fit.

    ``cell``: the quantile function of the pooled cells. This is a mixture of the
    per-patient distributions rather than their barycentre -- a genuinely different
    object, wider whenever patients differ, and the one CellOT's cell sampling sees.

    The transport map is the composition of the two, applied in quantile_map_apply.

    ``stats`` is accepted for a uniform call signature and ignored: a quantile function
    is not recoverable from the moments patient_stats keeps.
    """
    u = _u_grid(n_grid)
    if weighting == "cell":
        q_src = np.quantile(np.concatenate(src_by_patient), u, axis=0)
        q_tgt = np.quantile(np.concatenate(tgt_by_patient), u, axis=0)
        return {"u": u, "q_src": q_src, "q_tgt": q_tgt}

    qs = np.stack([np.quantile(x, u, axis=0) for x in src_by_patient])  # (P, G, M)
    qt = np.stack([np.quantile(x, u, axis=0) for x in tgt_by_patient])
    return {"u": u, "q_src": qs.mean(axis=0), "q_tgt": qt.mean(axis=0)}


def quantile_map_apply(x, params):
    """T_j(x) = F_k^-1(F_c(x)), marker by marker.

    F_c(x) is evaluated by interpolating x into the tabulated source quantile
    function; F_k^-1 by interpolating the resulting probability into the target
    one. Values outside the training source support clamp to the extreme training
    target values -- the standard quantile-mapping convention, and the conservative
    choice here (extrapolating a tail slope off ~8-18 patients would invent signal).
    """
    x = np.asarray(x, dtype=np.float64)
    u, q_src, q_tgt = params["u"], params["q_src"], params["q_tgt"]
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        probs = np.interp(x[:, j], _strictly_increasing(q_src[:, j]), u)
        out[:, j] = np.interp(probs, u, q_tgt[:, j])
    return out


# --------------------------------------------------------------------------- #
# affine Gaussian optimal transport (Bures-Wasserstein)
# --------------------------------------------------------------------------- #

# Relative floor on the eigenvalues of a covariance before it is raised to a negative
# power. With 13 markers and >=50 cells the Gram matrix is generically full rank, but
# markers in this panel are strongly correlated and a near-zero direction would
# otherwise be amplified by 1/sqrt(eig) into pure noise.
EIG_REL_FLOOR = 1e-8


def _sym_pow(S, power, rel_floor=EIG_REL_FLOOR):
    """S^power for a symmetric PSD S, via its eigendecomposition, with a floor."""
    w, V = np.linalg.eigh(0.5 * (S + S.T))
    floor = max(w.max(), 0.0) * rel_floor
    w = np.clip(w, floor if floor > 0 else rel_floor, None)
    return (V * w**power) @ V.T


def _pool_gaussian(stats):
    """Mean and covariance of the pooled cells, from per-patient summaries alone.

    Exact for the population covariance: E[xx'] is the cell-count-weighted average of
    each patient's E[xx'] = cov_p + m_p m_p', and the pooled covariance is that minus
    the outer square of the pooled mean. Note this is the covariance of the *mixture*,
    so it picks up between-patient spread -- which is the point of the ``cell`` arm.
    """
    n = np.array([s["n"] for s in stats], dtype=np.float64)
    w = n / n.sum()
    mean = sum(wp * s["mean"] for wp, s in zip(w, stats))
    second = sum(wp * (s["cov"] + np.outer(s["mean"], s["mean"])) for wp, s in zip(w, stats))
    return mean, second - np.outer(mean, mean)


def _bw_barycentre(covs, tol=1e-10, max_iter=100):
    """Equal-weight Bures-Wasserstein barycentre of a list of covariances.

    The covariance analogue of what quantile_map's ``unweighted`` arm does in 1-D:
    average the *distributions* under W2 rather than averaging their parameters. The
    arithmetic mean of covariances is a different (and wider) object -- it is the
    covariance of the mixture, i.e. what the ``cell`` arm computes.

    Standard fixed point S <- S^-1/2 (mean_p (S^1/2 C_p S^1/2)^1/2)^2 S^-1/2, started
    at the arithmetic mean. It is a strict contraction for PD inputs and converges in a
    handful of iterations at 13x13; the iteration cap is a guard, not a budget.
    """
    covs = list(covs)
    if len(covs) == 1:
        return 0.5 * (covs[0] + covs[0].T)

    S = np.mean(covs, axis=0)
    for _ in range(max_iter):
        S_half = _sym_pow(S, 0.5)
        M = np.mean([_sym_pow(S_half @ C @ S_half, 0.5) for C in covs], axis=0)
        S_inv_half = _sym_pow(S, -0.5)
        S_next = S_inv_half @ (M @ M) @ S_inv_half
        S_next = 0.5 * (S_next + S_next.T)
        if np.linalg.norm(S_next - S) <= tol * max(np.linalg.norm(S), 1.0):
            return S_next
        S = S_next
    return S


def gauss_ot_fit(src_by_patient, tgt_by_patient, weighting="unweighted", stats=None):
    """The W2-optimal map between two Gaussians fitted to source and target.

    T(x) = m_t + A (x - m_s) with

        A = S_s^-1/2 (S_s^1/2 S_t S_s^1/2)^1/2 S_s^-1/2

    the unique symmetric PSD matrix with A S_s A = S_t. This is exactly optimal if both
    distributions are Gaussian and, whatever they are, it is the best map available
    from their first two moments -- so the gap between it and CellOT is the part of
    CellOT's advantage that needs more than means and covariances to explain.

    ``unweighted``: the Bures-Wasserstein barycentre of the per-patient Gaussians
    (mean of means, barycentric covariance). ``cell``: the Gaussian fitted to the
    pooled cells, which inflates the covariance by the between-patient spread in the
    same way the other ``cell`` arms do.
    """
    src_stats, tgt_stats = _stats(src_by_patient, tgt_by_patient, stats)

    if weighting == "cell":
        mu_s, S_s = _pool_gaussian(src_stats)
        mu_t, S_t = _pool_gaussian(tgt_stats)
    else:
        mu_s = np.mean([s["mean"] for s in src_stats], axis=0)
        mu_t = np.mean([s["mean"] for s in tgt_stats], axis=0)
        S_s = _bw_barycentre([s["cov"] for s in src_stats])
        S_t = _bw_barycentre([s["cov"] for s in tgt_stats])

    S_s_half = _sym_pow(S_s, 0.5)
    S_s_inv_half = _sym_pow(S_s, -0.5)
    middle = _sym_pow(S_s_half @ S_t @ S_s_half, 0.5)
    A = S_s_inv_half @ middle @ S_s_inv_half
    return {"mu_s": mu_s, "mu_t": mu_t, "A": 0.5 * (A + A.T)}


def gauss_ot_apply(x, params):
    x = np.asarray(x, dtype=np.float64)
    return params["mu_t"] + (x - params["mu_s"]) @ params["A"].T


# --------------------------------------------------------------------------- #
# linear model of the response median on the baseline median
# --------------------------------------------------------------------------- #

# Ridge path for the penalised variant. Predictors are standardised before the penalty
# is applied, so alpha is scale-free and this range spans "no shrinkage worth the name"
# to "B is crushed to zero", where the model degenerates to a constant median shift.
RIDGE_ALPHAS = np.logspace(-4.0, 3.0, 22)


def _median_design(src_stats, tgt_stats, weighting):
    """One row per patient: U = Unstim medians, Y = stimulated medians, w = row weight.

    ``cell`` weights a patient by min(n_src, n_tgt) -- the smaller of the two cell
    counts is what limits the precision of that patient's (u, y) pair, and it keeps the
    arm's meaning ("a big patient counts more", as in CellOT's sampling) intact.
    """
    U = np.stack([s["median"] for s in src_stats])
    Y = np.stack([s["median"] for s in tgt_stats])
    if weighting == "cell":
        w = np.array([min(a["n"], b["n"]) for a, b in zip(src_stats, tgt_stats)], dtype=np.float64)
    else:
        w = np.ones(len(U), dtype=np.float64)
    return U, Y, w


def _fit_linear_median(U, Y, w, alpha):
    """Weighted fit of Y = a + B U across patients. ``alpha=None`` is plain OLS.

    Predictors are centred and scaled to unit weighted variance before solving, so the
    ridge penalty treats markers on equal terms regardless of their arcsinh scale, and
    the intercept is never penalised (it is recovered afterwards from the means).

    OLS is solved with lstsq rather than a normal-equation inverse: with 13 predictors
    and 18 patients the design is only just overdetermined, and rank-deficient cases do
    occur (a stim with few usable patients), where lstsq returns the minimum-norm
    solution instead of raising.
    """
    sw = w / w.sum()
    u_bar = sw @ U
    y_bar = sw @ Y
    Uc, Yc = U - u_bar, Y - y_bar

    scale = np.sqrt(sw @ (Uc**2))
    scale = np.where(scale > 1e-12, scale, 1.0)
    Z = Uc / scale

    if alpha is None:
        root = np.sqrt(sw)[:, None]
        coef = np.linalg.lstsq(root * Z, root * Yc, rcond=None)[0]
    else:
        gram = Z.T @ (sw[:, None] * Z)
        rhs = Z.T @ (sw[:, None] * Yc)
        coef = np.linalg.solve(gram + alpha * np.eye(Z.shape[1]), rhs)

    B = (coef / scale[:, None]).T
    return y_bar - B @ u_bar, B


def _select_alpha(U, Y, w, alphas=RIDGE_ALPHAS):
    """Pick the ridge penalty by leave-one-patient-out error, inside the fit set.

    Nested inside the outer leave-one-patient-out loop, so the held-out patient of the
    benchmark is never involved in choosing alpha. Brute force: <=19 patients x 22
    alphas of a 13x13 solve.
    """
    n = len(U)
    if n < 4:
        return alphas[-1]
    idx = np.arange(n)
    errors = []
    for alpha in alphas:
        total = 0.0
        for i in idx:
            keep = idx != i
            a, B = _fit_linear_median(U[keep], Y[keep], w[keep], alpha)
            total += float(np.sum((a + B @ U[i] - Y[i]) ** 2))
        errors.append(total)
    return float(alphas[int(np.argmin(errors))])


def linear_median_fit(src_by_patient, tgt_by_patient, weighting="unweighted", stats=None):
    """Unpenalised Y = a + B U, one (u, y) row per training patient.

    Expect this to overfit: B has 169 free parameters (13x13) estimated from 18 rows.
    It is here as the unregularised control for linear_median_ridge -- the pair
    measures how much of any patient-adaptive gain survives honest shrinkage.
    """
    src_stats, tgt_stats = _stats(src_by_patient, tgt_by_patient, stats)
    U, Y, w = _median_design(src_stats, tgt_stats, weighting)
    a, B = _fit_linear_median(U, Y, w, alpha=None)
    return {"a": a, "B": B, "alpha": 0.0, "n_patients": len(U)}


def linear_median_ridge_fit(src_by_patient, tgt_by_patient, weighting="unweighted", stats=None):
    """Y = a + B U with an L2 penalty on B, alpha chosen by inner LOO over patients.

    At large alpha B -> 0 and the prediction collapses to a + 0 = mean(Y), making the
    applied shift mean(Y) - u: still patient-adaptive through u, but no longer using
    cross-marker structure. So the alpha this picks is itself the readout -- if it sits
    at the top of the path, the patient's baseline profile carries no usable signal
    beyond its own level.
    """
    src_stats, tgt_stats = _stats(src_by_patient, tgt_by_patient, stats)
    U, Y, w = _median_design(src_stats, tgt_stats, weighting)
    alpha = _select_alpha(U, Y, w)
    a, B = _fit_linear_median(U, Y, w, alpha)
    return {"a": a, "B": B, "alpha": alpha, "n_patients": len(U)}


def linear_median_apply(x, params):
    """Shift the cells by the predicted change in median for *this* patient.

    u is read off the cells being transported, so the map is a pure function of its
    input like every other one here. Under --eval-split test that means u comes from
    the held-out 20% of the patient's Unstim cells, the same cells being scored, while
    the fit patients' u came from their 80% -- a small estimation asymmetry, and the
    conservative direction: it adds noise to this baseline, not to CellOT.

    Within one patient the map is a translation, so it cannot change that patient's
    spread or copula. Across patients it can: each gets a different shift, so a pooled
    evaluation (--eval-mode pooled, where blocks are transported separately and then
    concatenated) does see the second moments move, through the between-patient spread
    rather than the within-patient one. Under --eval-mode lopo, where each patient is
    scored alone, corr_error is identical to identity's by construction.
    """
    x = np.asarray(x, dtype=np.float64)
    u = np.median(x, axis=0)
    return x + (params["a"] + params["B"] @ u - u)


# Methods whose fit is a regression *across patients* rather than a statistic over
# pooled cells. Two consequences for the caller: the patient list handed to the fit
# must be one whose per-patient medians are actually estimable (a patient with 5 cells
# is a garbage row, not a weak one), and the map must be applied patient by patient,
# since its whole content is that different patients get different shifts. Applying any
# other method here block-by-block and concatenating is a no-op -- they are all row-wise
# functions -- so the caller can use one code path for everything.
PATIENT_LEVEL = frozenset({"linear_median", "linear_median_ridge"})


METHODS = {
    "identity": (identity_fit, identity_apply),
    "median_shift": (median_shift_fit, median_shift_apply),
    "quantile_map": (quantile_map_fit, quantile_map_apply),
    "gauss_ot": (gauss_ot_fit, gauss_ot_apply),
    "linear_median": (linear_median_fit, linear_median_apply),
    "linear_median_ridge": (linear_median_ridge_fit, linear_median_apply),
}
