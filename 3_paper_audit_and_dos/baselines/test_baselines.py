"""Correctness checks for the closed-form baselines, on distributions with known maps."""

import numpy as np

from baselines.baselines import (
    _bw_barycentre,
    gauss_ot_apply,
    gauss_ot_fit,
    linear_median_apply,
    linear_median_fit,
    linear_median_ridge_fit,
    median_shift_fit,
    median_shift_apply,
    patient_stats,
    quantile_map_fit,
    quantile_map_apply,
)


def _patients(rng, n_patients, n_cells, loc, scale, n_markers=3):
    return [
        rng.normal(loc, scale, size=(n_cells, n_markers)) for _ in range(n_patients)
    ]


def _correlated(rng, n_cells, mean, chol):
    """Cells with a known covariance chol @ chol.T."""
    return rng.normal(0, 1, size=(n_cells, chol.shape[0])) @ chol.T + mean


def test_median_shift_recovers_known_shift():
    rng = np.random.default_rng(0)
    src = _patients(rng, 8, 20000, 0.0, 1.0)
    tgt = _patients(rng, 8, 20000, 2.0, 1.0)
    params = median_shift_fit(src, tgt)
    assert np.allclose(params["delta"], 2.0, atol=0.05), params["delta"]

    x = rng.normal(0, 1, size=(100, 3))
    assert np.allclose(median_shift_apply(x, params), x + params["delta"])
    print("ok: median_shift recovers a pure location shift")


def test_quantile_map_is_identity_when_distributions_match():
    rng = np.random.default_rng(1)
    src = _patients(rng, 8, 20000, 0.0, 1.0)
    tgt = _patients(rng, 8, 20000, 0.0, 1.0)
    params = quantile_map_fit(src, tgt)
    x = rng.normal(0, 1, size=(2000, 3))
    out = quantile_map_apply(x, params)
    # Tails are the noisiest part of an averaged quantile function; compare on the
    # bulk where the grid is well determined.
    bulk = np.abs(x) < 2
    assert np.abs(out - x)[bulk].max() < 0.1, np.abs(out - x)[bulk].max()
    print("ok: quantile_map is ~identity when source and target agree")


def test_quantile_map_recovers_shift_and_scale():
    rng = np.random.default_rng(2)
    src = _patients(rng, 8, 40000, 0.0, 1.0)
    tgt = _patients(rng, 8, 40000, 2.0, 3.0)
    params = quantile_map_fit(src, tgt)
    x = rng.normal(0, 1, size=(4000, 3))
    out = quantile_map_apply(x, params)
    # For Gaussians the 1-D OT map is exactly affine: T(x) = 2 + 3x.
    expected = 2.0 + 3.0 * x
    bulk = np.abs(x) < 2
    err = np.abs(out - expected)[bulk].max()
    assert err < 0.2, err
    print("ok: quantile_map recovers the affine Gaussian OT map")


def test_quantile_map_beats_shift_on_a_variance_change():
    """The case that separates the two baselines: same median, different spread."""
    rng = np.random.default_rng(3)
    src = _patients(rng, 8, 20000, 0.0, 1.0)
    tgt = _patients(rng, 8, 20000, 0.0, 4.0)
    x = rng.normal(0, 1, size=(20000, 3))
    truth = rng.normal(0, 4, size=(20000, 3))

    shift = median_shift_apply(x, median_shift_fit(src, tgt))
    qmap = quantile_map_apply(x, quantile_map_fit(src, tgt))

    # median_shift cannot change spread, so its std stays at ~1 while the target is 4.
    assert abs(shift.std() - 1.0) < 0.1, shift.std()
    assert abs(qmap.std() - 4.0) < 0.2, qmap.std()

    u = np.linspace(0, 1, 501)
    w1 = lambda a: np.abs(np.quantile(a, u, axis=0) - np.quantile(truth, u, axis=0)).mean()
    assert w1(qmap) < w1(shift) / 5, (w1(qmap), w1(shift))
    print(f"ok: on a pure variance change, W1 shift={w1(shift):.3f} qmap={w1(qmap):.3f}")


def test_quantile_map_is_monotone_and_clamps():
    rng = np.random.default_rng(4)
    src = _patients(rng, 5, 5000, 0.0, 1.0)
    tgt = _patients(rng, 5, 5000, 1.0, 2.0)
    params = quantile_map_fit(src, tgt)

    x = np.sort(rng.normal(0, 1, size=(1000, 1)), axis=0)
    out = quantile_map_apply(x, params)
    assert np.all(np.diff(out[:, 0]) >= -1e-9), "map must be non-decreasing"

    far = np.array([[-50.0], [50.0]])
    out_far = quantile_map_apply(far, params)
    assert out_far[0, 0] >= params["q_tgt"][:, 0].min() - 1e-6
    assert out_far[1, 0] <= params["q_tgt"][:, 0].max() + 1e-6
    print("ok: quantile_map is monotone and clamps outside the training support")


def test_bw_barycentre_matches_the_closed_form_for_diagonals():
    """Commuting case: the barycentre is diag(((sqrt(a)+sqrt(b))/2)^2), not the mean."""
    a = np.array([1.0, 4.0, 9.0])
    b = np.array([4.0, 1.0, 16.0])
    got = _bw_barycentre([np.diag(a), np.diag(b)])
    want = np.diag(((np.sqrt(a) + np.sqrt(b)) / 2) ** 2)
    assert np.allclose(got, want, atol=1e-8), got

    # The distinction that makes the unweighted arm worth having: averaging the
    # distributions is not averaging their parameters.
    assert not np.allclose(got, np.diag((a + b) / 2), atol=0.1)
    assert np.allclose(_bw_barycentre([np.diag(a)]), np.diag(a))
    print("ok: BW barycentre matches the closed form and differs from the mean")


def test_gauss_ot_recovers_a_known_affine_map():
    rng = np.random.default_rng(10)
    M = np.array([[1.5, 0.4, 0.0], [0.4, 2.0, 0.3], [0.0, 0.3, 1.2]])  # symmetric PD
    shift = np.array([1.0, -0.5, 2.0])
    src = [rng.normal(0, 1, size=(20000, 3)) for _ in range(6)]
    tgt = [x @ M.T + shift for x in src]

    params = gauss_ot_fit(src, tgt)
    # The OT map between Gaussians is the unique *symmetric* PSD A with A S_s A = S_t,
    # so a symmetric ground-truth M is recovered as itself.
    assert np.allclose(params["A"], M, atol=0.05), params["A"]
    assert np.allclose(params["mu_t"], shift, atol=0.05), params["mu_t"]
    print("ok: gauss_ot recovers a known symmetric affine map")


def test_gauss_ot_matches_target_moments_out_of_sample():
    rng = np.random.default_rng(11)
    chol_s = np.array([[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.2, 0.1, 1.3]])
    chol_t = np.array([[2.0, 0.0, 0.0], [-0.5, 1.1, 0.0], [0.9, 0.4, 0.7]])
    mean_s, mean_t = np.array([0.0, 1.0, -1.0]), np.array([2.0, 0.0, 0.5])

    src = [_correlated(rng, 20000, mean_s, chol_s) for _ in range(6)]
    tgt = [_correlated(rng, 20000, mean_t, chol_t) for _ in range(6)]
    params = gauss_ot_fit(src, tgt)

    held_out = _correlated(rng, 30000, mean_s, chol_s)
    out = gauss_ot_apply(held_out, params)
    assert np.allclose(out.mean(axis=0), mean_t, atol=0.05), out.mean(axis=0)
    assert np.allclose(np.cov(out, rowvar=False), chol_t @ chol_t.T, atol=0.1)
    print("ok: gauss_ot matches target mean and covariance on held-out cells")


def test_gauss_ot_fixes_a_correlation_change_no_diagonal_map_can():
    """Identical marginals, different copula -- invisible to every per-marker map."""
    rng = np.random.default_rng(12)
    rho = 0.8
    corr = np.full((3, 3), rho)
    np.fill_diagonal(corr, 1.0)
    chol_t = np.linalg.cholesky(corr)

    src = [rng.normal(0, 1, size=(20000, 3)) for _ in range(6)]
    tgt = [_correlated(rng, 20000, np.zeros(3), chol_t) for _ in range(6)]
    x = rng.normal(0, 1, size=(20000, 3))

    gauss = gauss_ot_apply(x, gauss_ot_fit(src, tgt))
    qmap = quantile_map_apply(x, quantile_map_fit(src, tgt))
    err = lambda a: np.linalg.norm(np.corrcoef(a, rowvar=False) - corr, ord="fro")

    # quantile_map sees identical marginals, so it is ~the identity and keeps corr ~0.
    assert err(qmap) > 1.5, err(qmap)
    assert err(gauss) < 0.1, err(gauss)
    print(f"ok: on a pure correlation change, corr_error qmap={err(qmap):.3f} "
          f"gauss={err(gauss):.3f}")


def test_linear_median_recovers_a_known_rule():
    rng = np.random.default_rng(20)
    B_true = np.array([[1.0, 0.5, 0.0], [0.0, 0.8, 0.2], [0.3, 0.0, 1.1]])
    a_true = np.array([0.5, -1.0, 0.25])

    src, tgt = [], []
    for _ in range(12):
        u = rng.normal(0, 1, size=3)
        src.append(rng.normal(u, 0.5, size=(4000, 3)))
        tgt.append(rng.normal(a_true + B_true @ u, 0.5, size=(4000, 3)))

    params = linear_median_fit(src, tgt)
    assert np.allclose(params["B"], B_true, atol=0.1), params["B"]
    assert np.allclose(params["a"], a_true, atol=0.1), params["a"]

    # Applying it moves the held-out patient's median to a + B u exactly, since a
    # translation moves the median by itself.
    x = rng.normal(np.array([0.7, -0.2, 1.3]), 0.5, size=(5000, 3))
    u = np.median(x, axis=0)
    out = linear_median_apply(x, params)
    assert np.allclose(np.median(out, axis=0), a_true + B_true @ u, atol=0.05)
    print("ok: linear_median recovers a known median-to-median rule")


def test_linear_median_is_patient_adaptive():
    """Two patients with different baselines get different shifts -- median_shift's
    defining limitation, and the reason this baseline exists."""
    rng = np.random.default_rng(22)
    B_true = np.diag([2.0, 2.0, 2.0])
    src, tgt = [], []
    for _ in range(12):
        u = rng.normal(0, 1, size=3)
        src.append(rng.normal(u, 0.5, size=(4000, 3)))
        tgt.append(rng.normal(B_true @ u, 0.5, size=(4000, 3)))
    params = linear_median_fit(src, tgt)

    low = rng.normal(np.full(3, -1.0), 0.5, size=(4000, 3))
    high = rng.normal(np.full(3, 1.0), 0.5, size=(4000, 3))
    shift = lambda x: np.median(linear_median_apply(x, params) - x, axis=0)
    # y = 2u so the shift is +u: negative for the low patient, positive for the high one.
    assert (shift(low) < -0.8).all(), shift(low)
    assert (shift(high) > 0.8).all(), shift(high)
    print("ok: linear_median adapts the shift to the patient's own baseline")


def test_ridge_generalises_better_when_there_is_no_rule():
    """Target median independent of baseline: B_true = 0, so anything OLS finds in 8
    patients is noise. This is the regime the real fits are in -- 13 predictors, 18
    patients -- and the reason the penalised variant is worth carrying."""
    rng = np.random.default_rng(21)
    c = np.array([1.0, -0.5, 0.75])

    def patient():
        u = rng.normal(0, 1, size=3)
        return rng.normal(u, 1.0, size=(300, 3)), rng.normal(c, 1.0, size=(300, 3))

    pairs = [patient() for _ in range(8)]
    src = [s for s, _ in pairs]
    tgt = [t for _, t in pairs]

    ols = linear_median_fit(src, tgt)
    ridge = linear_median_ridge_fit(src, tgt)
    assert np.abs(ridge["B"]).max() < np.abs(ols["B"]).max(), (ridge["alpha"])

    err_ols = err_ridge = 0.0
    for _ in range(50):
        x, y_cells = patient()
        truth = np.median(y_cells, axis=0)
        err_ols += float(np.sum((np.median(linear_median_apply(x, ols), axis=0) - truth) ** 2))
        err_ridge += float(np.sum((np.median(linear_median_apply(x, ridge), axis=0) - truth) ** 2))
    assert err_ridge < err_ols, (err_ridge, err_ols)
    print(f"ok: with no rule to learn, held-out SSE ols={err_ols:.2f} "
          f"ridge={err_ridge:.2f} (alpha={ridge['alpha']:.3g})")


def test_precomputed_stats_match_recomputing_them():
    """Guard on the caching in run_baselines: passing stats must change nothing."""
    from baselines.baselines import METHODS

    rng = np.random.default_rng(30)
    src = _patients(rng, 6, 900, 0.0, 1.0)
    tgt = _patients(rng, 6, 900, 1.0, 1.5)
    stats = {
        "src": [patient_stats(x) for x in src],
        "tgt": [patient_stats(x) for x in tgt],
    }
    x = rng.normal(0, 1, size=(500, 3))

    for name, (fit, apply_) in METHODS.items():
        for weighting in ("unweighted", "cell"):
            plain = apply_(x, fit(src, tgt, weighting=weighting))
            cached = apply_(x, fit(src, tgt, weighting=weighting, stats=stats))
            assert plain.shape == x.shape and np.isfinite(plain).all(), name
            assert np.allclose(plain, cached), (name, weighting)
    print("ok: every method runs in both weightings and ignores nothing in stats")


if __name__ == "__main__":
    test_median_shift_recovers_known_shift()
    test_quantile_map_is_identity_when_distributions_match()
    test_quantile_map_recovers_shift_and_scale()
    test_quantile_map_beats_shift_on_a_variance_change()
    test_quantile_map_is_monotone_and_clamps()
    test_bw_barycentre_matches_the_closed_form_for_diagonals()
    test_gauss_ot_recovers_a_known_affine_map()
    test_gauss_ot_matches_target_moments_out_of_sample()
    test_gauss_ot_fixes_a_correlation_change_no_diagonal_map_can()
    test_linear_median_recovers_a_known_rule()
    test_linear_median_is_patient_adaptive()
    test_ridge_generalises_better_when_there_is_no_rule()
    test_precomputed_stats_match_recomputing_them()
    print("\nall baseline tests passed")
