"""dnorm shape + a fitted between-donor slope on the donor's offset (2026-10-13, k = 15, closed form, no training).

dnorm predicts  m_d + T'(cells - m_d): the donor's Unstim median m_d passes with slope exactly 1. Here the offset gets
its own per-feature slope instead:
    pred = dnorm + (b - 1) (m_d - mbar),   b = 1 + w (b_ols - 1)
b_ols = OLS slope, across the 15 training donors, of their stimulated quantile on their Unstim median (same marker);
mbar = the training donors' mean Unstim median. w shrinks towards 1 (w = 0 is dnorm). w is chosen per repeat by
leave-one-training-donor-out on the offset part alone (S_i - Sbar_-i vs b_-i (m_i - mbar_-i), every feature pooled),
"loo" pooled over stims, "loo_stim" per stim; the test donors are never used. A fixed-w grid is also written
(test-set sensitivity, not a selection).
Writes WORK/dnorm_slope/cases.pkl (one row per repeat, stim, celltype, test donor, q, marker; truth, u, mu_s, every arm)
and w.csv.
    python -m lc.dnorm_slope
"""
import json

import numpy as np
import pandas as pd

from lc.cond_train_score import load_train
from lc.paths import DONORS, MARKERS, PAIRS, QS, WORK
from lc.score import truth_tables

K = 15
WGRID = [0, 0.25, 0.5, 0.75, 1.0]
WLOO = np.linspace(0, 1, 21)
OUT = WORK / "dnorm_slope"
QI50 = list(QS).index(50)


def ols(m, S):
    """m (n,), S (n, ...) -> slope of S on m per trailing entry."""
    mc = m - m.mean(0)
    return np.einsum("i...,i...->...", mc, S - S.mean(0)) / np.maximum((mc ** 2).sum(0), 1e-12)


def loo_terms(m, S):
    """Per left-out training donor: (target S_i - Sbar_-i, b_-i - 1 times (m_i - mbar_-i)) -> err(w) = (tgt - x - w y)^2."""
    n = len(m); tg, x, y = [], [], []
    for i in range(n):
        o = np.arange(n) != i
        b = ols(m[o], S[o]); dm = m[i] - m[o].mean(0)
        tg.append(S[i] - S[o].mean(0)); x.append(dm); y.append((b - 1) * dm)
    return np.stack(tg), np.stack(x), np.stack(y)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    T = truth_tables(sorted({c for _, c in PAIRS}), set(DONORS))
    rows, wrows = [], []
    for dz in json.load(open(WORK / "design.json")):
        r, tests = dz["repeat"], dz["test"]
        maps = [(s, c, K) for s, c in PAIRS]
        DN = load_train(WORK / f"cond_train/repeat_{r}/dnorm", maps)
        NO = load_train(WORK / f"cond_train/repeat_{r}/none", maps)
        fits, loo = {}, {}
        for (s, c, k) in maps:
            trd = [d for d in dz["order"][:k] if (c, s, d) in T and (c, "Unstim", d) in T]
            if len(trd) < 4 or (s, c, k) not in DN or (s, c, k) not in NO: continue
            U = np.stack([T[(c, "Unstim", d)][0] for d in trd]); S = np.stack([T[(c, s, d)][0] for d in trd])
            m = np.broadcast_to(U[:, QI50][:, None, :], S.shape)          # donor's Unstim median, same marker
            fits[(s, c)] = (ols(m, S), m.mean(0), U.mean(0), S.mean(0))
            loo[(s, c)] = loo_terms(m, S)
        # w by leave-one-training-donor-out, offset part only (S_i - Sbar_-i - dm - w (b_-i - 1) dm)^2
        def best_w(keys):
            e = [sum(((loo[q][0] - loo[q][1] - w * loo[q][2]) ** 2).sum() for q in keys) for w in WLOO]
            return WLOO[int(np.argmin(e))]
        w_all = best_w(list(loo)); w_stim = {st: best_w([q for q in loo if q[0] == st]) for st in {q[0] for q in loo}}
        wrows += [{"repeat": r, "stim": "all", "w": w_all}] + [{"repeat": r, "stim": st, "w": w} for st, w in w_stim.items()]
        print(f"repeat {r}: w_loo {w_all:.2f}  per stim {dict(sorted((k, round(v, 2)) for k, v in w_stim.items()))}", flush=True)
        for (s, c), (b, mbar, mu_u, mu_s) in fits.items():
            for j, d in enumerate(tests):
                if (c, s, d) not in T or (c, "Unstim", d) not in T: continue
                u, t = T[(c, "Unstim", d)][0], T[(c, s, d)][0]
                md = np.broadcast_to(u[QI50][None, :], u.shape); dm = md - mbar
                dn = DN[(s, c, K)][j] + md                                  # back from the donor-normalised space
                no = NO[(s, c, K)][j]
                if np.isnan(dn).all() or np.isnan(no).all(): continue
                shift = mu_s + u - mu_u
                arm = {"shift": shift, "paired": no, "dnorm": dn, "ens": 0.5 * no + 0.5 * shift,
                       "dslope_loo": dn + w_all * (b - 1) * dm, "dslope_loo_stim": dn + w_stim[s] * (b - 1) * dm,
                       "slope_shift": shift + w_all * (b - 1) * dm}
                arm.update({f"dslope_w{w:g}": dn + w * (b - 1) * dm for w in WGRID[1:]})
                arm["ens_dslope_paired"] = 0.5 * arm["dslope_loo"] + 0.5 * no
                arm["ens_dslope_shift_paired"] = (arm["dslope_loo"] + shift + no) / 3
                row = {"repeat": r, "stim": s, "celltype": c, "donor": d, "q": np.repeat(QS, len(MARKERS)),
                       "marker": np.tile(MARKERS, len(QS)), "u": u.ravel(), "t": t.ravel(), "mu_s": mu_s.ravel(),
                       "b_ols": b.ravel()}
                row.update({n: v.ravel() for n, v in arm.items()})
                rows.append(pd.DataFrame(row))
    D = pd.concat(rows, ignore_index=True)
    D.to_pickle(OUT / "cases.pkl"); pd.DataFrame(wrows).to_csv(OUT / "w.csv", index=False)
    print(D.shape, "->", OUT)


if __name__ == "__main__":
    main()
