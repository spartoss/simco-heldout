"""Donor-conditioned closed-form maps on the learning-curve design (next experiment, step 1).

Question: does a map that knows what kind of donor a cell comes from predict a new donor better than the pooled map?
Covariate of a donor (per celltype) = its Unstim cells summarised: the 13 marker medians (--cov med) or the
q10/q50/q90 of each marker (--cov qtl), z-scored and reduced to N_PC principal components fitted on the TRAINING
donors only, each rescaled to unit spread (2026-10-05; the first run, tag med, was unscaled); the test donor is
projected. Optional HV/PV indicator appended (--hvpv).

Methods (all gauss_ot-shaped, built from per-donor moments of the train side, so pooling is analytic):
- gauss_ot_pool    : the plain pooled gauss_ot (cell-weighted), equal to lc/baselines.py's gauss_ot (checked).
- gauss_ot_knn     : training donors weighted by exp(-dist^2 / (2 h^2)) in covariate space; h picked by inner
                     leave-one-donor-out (LODO) over the training donors from H_GRID x median distance (inf = pool).
- gauss_ot_cond    : pooled map + a correction: each training donor's residual = its stim mean / covariance minus
                     the pooled map pushed through its own Unstim Gaussian; residuals ridge-regressed on the
                     covariate; the test donor's map goes from its own Unstim moments to pushed + predicted residual.
                     alpha by inner LODO from ALPHAS; alpha = inf is exactly the pooled map. The ridge shrinks the
                     slopes only: a large FINITE alpha leaves the intercept (mean residual over training donors).
- gauss_ot_icpt    : that intercept alone, no covariate: the same correction for every donor. Separates "a better
                     donor-independent correction" from "a donor-specific one" (cond - icpt).
                     (Not "own source -> pooled target": that OT map outputs the pooled distribution itself and
                     erases the donor -- skill -0.9 on pDCs in the smoke test.)
Inner LODO loss = squared error of predicted vs the held-out training donor's q10..q90 per marker (the score's
quantity), averaged over donors. k < 3 has no inner LODO: knn/cond fall back to the pool.
Output WORK/cond_baselines/<tag>/repeat_<r>.npz, same layout as lc/baselines.py (+ "chosen" per map) so lc/score.py
reads it via --baseline-dir after renaming methods, or lc/cond_score.py.
    python -m lc.cond_baselines --repeat 0 [--celltypes pDCs] [--cov med] [--hvpv]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from lc.baselines import MIN_TEST_CELLS, sym_pow
from lc.paths import CACHE, PAIRS, QS, VAL_FRAC, WORK

METHODS = ["gauss_ot_pool", "gauss_ot_knn", "gauss_ot_cond", "gauss_ot_icpt"]
N_PC = 3
H_GRID = [0.25, 0.5, 1.0, 2.0, np.inf]
ALPHAS = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, np.inf]
INNER_CELLS = 5_000          # cells per held-out training donor used for the inner LODO quantiles
MIN_CELLS = 50               # per donor and condition, to enter the per-donor moments


def moments(x):
    x = x.astype(np.float64)
    return len(x), x.mean(0), np.cov(x, rowvar=False)


def pool(ms, w=None):
    """Cell-weighted pooled mean / covariance from per-donor (n, mean, cov); w = extra donor weights."""
    n = np.array([m[0] for m in ms], float) * (1 if w is None else np.asarray(w, float))
    n = n / n.sum()
    mu = sum(ni * m[1] for ni, m in zip(n, ms))
    S = sum(ni * (m[2] + np.outer(m[1] - mu, m[1] - mu)) for ni, m in zip(n, ms))
    return mu, S


def ot_map(mu_s, S_s, mu_t, S_t):
    h, ih = sym_pow(S_s, 0.5), sym_pow(S_s, -0.5)
    A = ih @ sym_pow(h @ S_t @ h, 0.5) @ ih
    return mu_s, mu_t, 0.5 * (A + A.T)


def transport_q(x, mp):
    mu_s, mu_t, A = mp
    return np.percentile(mu_t + (x.astype(np.float64) - mu_s) @ A.T, QS, axis=0)


def covariate(x, kind):
    if kind == "med": return np.median(x, 0)
    return np.percentile(x, [10, 50, 90], axis=0).ravel()


def embed(C_train, C_other, n_pc=N_PC):
    """z-score + PCA on training donors; returns (train scores, other scores). n_pc capped at n_train - 1."""
    mu, sd = C_train.mean(0), C_train.std(0) + 1e-9
    Z = (C_train - mu) / sd
    k = max(1, min(n_pc, len(Z) - 1))
    _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    P = Vt[:k].T
    zc = Z.mean(0)
    s_sd = ((Z - zc) @ P).std(0) + 1e-9          # each component rescaled to unit spread (as lc/cond_train.py)
    f = lambda C: (((C - mu) / sd - zc) @ P) / s_sd
    return f(C_train), f(C_other)


# ---- the three predictors, each: (train donor dicts, covariates, test-donor source moments/cov) -> map
def m_pool(D, *_):
    return ot_map(*pool([d["src"] for d in D]), *pool([d["tgt"] for d in D]))


def m_knn(D, ctr, cte, _own, h):
    if not np.isfinite(h): return m_pool(D)
    dist2 = ((ctr - cte) ** 2).sum(1)
    scale = np.median(np.sqrt(((ctr[:, None] - ctr[None]) ** 2).sum(-1))[np.triu_indices(len(ctr), 1)]) or 1.0
    w = np.exp(-dist2 / (2 * (h * scale) ** 2)) + 1e-12
    return ot_map(*pool([d["src"] for d in D], w), *pool([d["tgt"] for d in D], w))


def pushed(mp, src):
    """Moments of the pooled map applied to a donor's source Gaussian: (mu_t + A(mu - mu_s), A S A)."""
    mu_s, mu_t, A = mp
    return mu_t + A @ (src[1] - mu_s), A @ src[2] @ A


def m_cond(D, ctr, cte, own, alpha, slopes=True):
    """Pooled map + a covariate-predicted correction of the target moments. alpha = inf is exactly the pooled map.
    slopes=False keeps only the intercept (the training donors' mean residual, no covariate): gauss_ot_icpt."""
    mp = m_pool(D)
    if not np.isfinite(alpha): return mp
    iu = np.triu_indices(len(mp[0]))
    res = []
    for d in D:
        m0, S0 = pushed(mp, d["src"])
        res.append(np.concatenate([d["tgt"][1] - m0, (d["tgt"][2] - S0)[iu]]))
    Y = np.stack(res)
    X = ctr - ctr.mean(0)
    B = np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ (Y - Y.mean(0)))
    y = Y.mean(0) + ((cte - ctr.mean(0)) @ B if slopes else 0)
    m0, S0 = pushed(mp, own)
    n = len(m0)
    R = np.zeros_like(S0); R[iu] = y[n:]; R = R + R.T - np.diag(np.diag(R))
    w, V = np.linalg.eigh(S0 + R); S = (V * np.clip(w, 1e-6, None)) @ V.T      # keep it a covariance
    return ot_map(own[1], own[2], m0 + y[:n], S)


def inner_select(D, C, fn, grid, cov_kind, hvpv):
    """Pick a hyperparameter by leave-one-donor-out over the training donors."""
    loss = []
    for g in grid:
        e = []
        for i in range(len(D)):
            rest = [d for j, d in enumerate(D) if j != i]
            ctr, cte = embed(np.stack([C[j] for j in range(len(D)) if j != i]), C[i][None])
            ctr, cte = add_hvpv(ctr, cte, [d["name"] for d in rest], [D[i]["name"]], hvpv)
            mp = fn(rest, ctr, cte[0], D[i]["src"], g)
            e.append(np.mean((transport_q(D[i]["x_in"], mp) - D[i]["q_tgt"]) ** 2))
        loss.append(np.mean(e))
    return grid[int(np.argmin(loss))], loss


def add_hvpv(ctr, cte, names_tr, names_te, on):
    if not on: return ctr, cte
    f = lambda ns: np.array([[1.0 if n.startswith("PV") else -1.0] for n in ns])
    return np.hstack([ctr, f(names_tr)]), np.hstack([cte, f(names_te)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, required=True)
    ap.add_argument("--celltypes", nargs="*", default=None)
    ap.add_argument("--ks", nargs="*", type=int, default=None)
    ap.add_argument("--cov", choices=["med", "qtl"], default="med")
    ap.add_argument("--hvpv", action="store_true")
    ap.add_argument("--cache-dir", default=str(CACHE))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tag = a.cov + "_z" + ("_hvpv" if a.hvpv else "")
    out = Path(a.out or WORK / "cond_baselines" / tag); out.mkdir(parents=True, exist_ok=True)
    dz = json.load(open(WORK / "design.json"))[a.repeat]
    tests, ks = dz["test"], a.ks or dz["ks"]
    maps, chosen, preds = [], [], {m: [] for m in METHODS}
    z, zct = None, None
    for stim, ct in PAIRS:
        if a.celltypes and ct not in a.celltypes: continue
        if ct != zct: z, zct = dict(np.load(Path(a.cache_dir) / f"{ct}.npz")), ct
        trn = lambda cond, d: (lambda x: x[:max(1, int(round(len(x) * (1 - VAL_FRAC))))])(z[f"{cond}|{d}"])
        donor = {}
        for d in dz["order"]:
            if f"Unstim|{d}" not in z or f"{stim}|{d}" not in z: continue
            s, t = trn("Unstim", d), trn(stim, d)
            if len(s) < MIN_CELLS or len(t) < MIN_CELLS: continue
            donor[d] = {"name": d, "src": moments(s), "tgt": moments(t), "cov": covariate(s, a.cov),
                        "x_in": s[:INNER_CELLS], "q_tgt": np.percentile(t, QS, axis=0)}
        test = {}
        for d in tests:
            if f"Unstim|{d}" in z and len(z[f"Unstim|{d}"]) >= MIN_TEST_CELLS:
                x = z[f"Unstim|{d}"][:50_000]
                test[d] = {"x": x, "src": moments(x), "cov": covariate(x, a.cov)}
        for k in ks:
            D = [donor[d] for d in dz["order"][:k] if d in donor]
            if not D: continue
            C = np.stack([d["cov"] for d in D])
            if len(D) >= 3:
                h, _ = inner_select(D, C, m_knn, H_GRID, a.cov, a.hvpv)
                al, _ = inner_select(D, C, m_cond, ALPHAS, a.cov, a.hvpv)
            else:
                h, al = np.inf, np.inf
            arrs = {m: np.full((len(tests), len(QS), 13), np.nan, np.float32) for m in METHODS}
            mp_pool = m_pool(D)
            for j, d in enumerate(tests):
                if d not in test: continue
                T = test[d]
                if len(D) >= 2:
                    ctr, cte = embed(C, T["cov"][None])
                    ctr, cte = add_hvpv(ctr, cte, [x["name"] for x in D], [d], a.hvpv)
                    cte = cte[0]
                else:
                    ctr = cte = None
                arrs["gauss_ot_pool"][j] = transport_q(T["x"], mp_pool)
                arrs["gauss_ot_knn"][j] = transport_q(T["x"], mp_pool if ctr is None else m_knn(D, ctr, cte, None, h))
                arrs["gauss_ot_icpt"][j] = transport_q(T["x"], mp_pool if ctr is None
                                                       else m_cond(D, ctr, cte, T["src"], 1.0, slopes=False))
                arrs["gauss_ot_cond"][j] = transport_q(T["x"], mp_pool if ctr is None else m_cond(D, ctr, cte, T["src"], al))
            maps.append((stim, ct, k)); chosen.append((h, al))
            for m in METHODS: preds[m].append(arrs[m])
        print(f"[FIT  ] {stim}:{ct}  last k: h={h} alpha={al}", flush=True)
    np.savez(out / f"repeat_{a.repeat}.npz", maps=np.array(maps, dtype=object), chosen=np.array(chosen, float),
             **{m: np.stack(v) for m, v in preds.items()})
    print(f"[DONE ] repeat {a.repeat}: {len(maps)} maps -> {out}")


if __name__ == "__main__":
    main()
