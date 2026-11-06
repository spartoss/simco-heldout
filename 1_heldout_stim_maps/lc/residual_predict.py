"""Can a new donor's stim response BEYOND the shift be predicted from its Unstim data? (patient-covariate, step 1b)

Feature = (stim, celltype, marker, quantile); per donor, quantiles of its cached cells (lc/score.truth_tables).
Shift prediction: train stim mean + (test Unstim - train Unstim mean). Residual = what it misses:
    r = (S - mean_train S) - (U - mean_train U)            (U = the same feature's Unstim value)
Predictors of r for a held-out donor (all fitted on training donors only; ridge alpha by leave-one-out on them):
- shift        r = 0
- own_lin      r = b * u_dev                   per feature (a slope != 1; after standardisation = the Unstim feature)
- own_quad     r = b1 * u_dev + b2 * u_dev^2   per feature (non-monotone allowed: survives Stabl's standardisation)
- celltype     ridge on the same celltype's 65 Unstim features (13 markers x 5 quantiles)
- blood        ridge on all celltypes' Unstim features (28 x 65)
- blood_perm   same, training donors' inputs shuffled against their targets (null; should be <= 0)
- freq         ridge on the donor's 28 Unstim celltype frequencies (WORK/fm/counts.csv, fm/counts.py)
With --fm WORK/fm/donor_emb.npz (fm/embed_donors.py), per ImmuneFM arm A (pretrained / random / funcmasked / ...):
- fm_ct_A      ridge on the same celltype's mean Unstim cell embedding (128)
- fm_wb_A      ridge on the frequency-weighted whole-blood mean embedding (128)
- fm_blood_A   ridge on all celltypes' mean embeddings (28 x 128); fm_blood_perm_A = its shuffled null
Readouts over live features (test donor's |S - U| > FLOOR): res_R2 = 1 - sum (r_hat - r)^2 / sum r^2 (share of the
residual predicted; 0 = the shift) and dev_R2 = 1 - sum (r_hat - r)^2 / sum (S - mean_train S)^2 (as
lc/cond_train_score.py, shift = 0.75 there). Protocols: lodo (19 folds) and lc15 (5 repeats x 4 test donors, k = 15).
Output WORK/residual_predict/: summary.csv, by_stim.csv.
    python -m lc.residual_predict [--celltypes pDCs ...] [--fm WORK/fm/donor_emb.npz --out WORK/fm/residual_predict]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

from lc.paths import DONORS, PAIRS, WORK
from lc.score import FLOOR, truth_tables

ALPHAS = np.logspace(-2, 4, 13)
OUT = WORK / "residual_predict"


def own_fit(u, r, quad, alphas=ALPHAS):
    """Per-feature ridge (no intercept: both centred on training means) of r on [u] or [u, u^2]; one alpha per
    (stim, celltype) by pooled closed-form LOO. u, r: (n, F). Returns a predictor of test u_dev (F,) -> r_hat."""
    n, F = u.shape
    Xs = [np.stack([u[:, f], u[:, f] ** 2 - (u[:, f] ** 2).mean()], 1) if quad else u[:, f:f + 1] for f in range(F)]
    best, coefs = None, None
    for a in alphas:
        loo, cs = 0.0, []
        for f in range(F):
            X = Xs[f]; G = X.T @ X + a * np.eye(X.shape[1])
            c = np.linalg.solve(G, X.T @ r[:, f]); h = np.einsum("ij,jk,ik->i", X, np.linalg.inv(G), X)
            loo += (((r[:, f] - X @ c) / (1 - h)) ** 2).sum(); cs.append(c)
        if best is None or loo < best: best, coefs = loo, cs
    m2 = [(u[:, f] ** 2).mean() for f in range(F)]
    return lambda ut: np.array([c @ (np.array([ut[f], ut[f] ** 2 - m2[f]]) if quad else ut[f:f + 1]) for f, c in enumerate(coefs)])


def ridge(Xtr, Ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    m = RidgeCV(alphas=ALPHAS).fit((Xtr - mu) / sd, Ytr)
    return m.predict(((Xte - mu) / sd)[None])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--celltypes", nargs="*", default=None)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--merge", nargs="*", default=None, help="merge the cases.parquet of these dirs into --out")
    ap.add_argument("--fm", default=None, help="donor_emb.npz from fm/embed_donors.py: adds the fm_* predictors")
    ap.add_argument("--base", nargs="*", default=None, help="only these non-fm methods (default: all)")
    ap.add_argument("--fm-repeat", type=int, default=None,
                    help="only the lc15 fold of this repeat (fine-tuned covariates are valid for their own repeat only)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.merge:
        return report(pd.concat([pd.read_pickle(Path(m) / "cases.pkl") for m in a.merge]), out)
    cts = sorted({c for _, c in PAIRS})
    T = truth_tables(cts, set(DONORS))
    # whole-blood Unstim profile: donors x (celltype, q, marker); missing -> NaN, imputed with the training mean
    blood = np.full((len(DONORS), len(cts) * 65), np.nan)
    for i, d in enumerate(DONORS):
        for j, c in enumerate(cts):
            if (c, "Unstim", d) in T: blood[i, j * 65:(j + 1) * 65] = T[(c, "Unstim", d)][0].ravel()
    cnt = pd.read_csv(WORK / "fm" / "counts.csv").pivot(index="donor", columns="celltype", values="freq")
    freq = cnt.loc[DONORS, cts].to_numpy()
    fmX = {}                                   # name -> (per-celltype (n_donors, n_ct, dim) or None, whole (n_donors, dim))
    if a.fm:
        z = np.load(a.fm); assert list(z["donors"]) == DONORS
        jj = [list(z["celltypes"]).index(c) for c in cts]
        for arm in [k for k in z.files if k not in ("donors", "celltypes")]:
            E = z[arm][:, jj]                  # (n_donors, n_ct, 128), NaN where missing
            W = np.where(np.isnan(E[..., 0]), 0, freq); W = W / W.sum(1, keepdims=True)
            fmX[f"fm_ct_{arm}"] = (E, None)
            fmX[f"fm_wb_{arm}"] = (None, np.einsum("dc,dck->dk", W, np.nan_to_num(E)))
            fmX[f"fm_blood_{arm}"] = (None, E.reshape(len(DONORS), -1))
    folds = [("lodo", f"d{i}", [x for x in DONORS if x != d], [d]) for i, d in enumerate(DONORS)]
    for dz in json.load(open(WORK / "design.json")):
        folds.append(("lc15", f"r{dz['repeat']}", dz["order"][:15], dz["test"]))
    if a.fm_repeat is not None: folds = [f for f in folds if f[:2] == ("lc15", f"r{a.fm_repeat}")]
    rng = np.random.default_rng(0); rng_fm = np.random.default_rng(1)   # separate stream: fm nulls leave blood_perm as before
    rows = []
    for proto, fid, tr, te in folds:
        itr = [DONORS.index(d) for d in tr]
        B = blood.copy(); colmu = np.nanmean(B[itr], 0)
        B = np.where(np.isnan(B), colmu, B)
        imp = lambda M: np.where(np.isnan(M), np.nanmean(M[itr], 0), M)       # training-donor mean imputation
        perm = rng.permutation(len(itr))       # unused; kept so the rng stream (blood_perm) matches earlier runs
        FM = {k: (imp(E) if E is not None else None, imp(G) if G is not None else None) for k, (E, G) in fmX.items()}
        for stim, ct in PAIRS:
            if a.celltypes and ct not in a.celltypes: continue
            trd = [d for d in tr if (ct, stim, d) in T and (ct, "Unstim", d) in T]
            if len(trd) < 5: continue
            U = np.stack([T[(ct, "Unstim", d)][0].ravel() for d in trd]); S = np.stack([T[(ct, stim, d)][0].ravel() for d in trd])
            mu_u, mu_s = U.mean(0), S.mean(0)
            R = (S - mu_s) - (U - mu_u)
            ib = [DONORS.index(d) for d in trd]; j = cts.index(ct)
            f_lin = own_fit(U - mu_u, R, False); f_quad = own_fit(U - mu_u, R, True)
            for d in te:
                if (ct, stim, d) not in T or (ct, "Unstim", d) not in T: continue
                u = T[(ct, "Unstim", d)][0].ravel(); s = T[(ct, stim, d)][0].ravel()
                r = (s - mu_s) - (u - mu_u); ud = u - mu_u
                idb = DONORS.index(d)
                preds = {"shift": np.zeros_like(r), "own_lin": f_lin(ud), "own_quad": f_quad(ud),
                         "celltype": ridge(U, R, u),
                         "blood": ridge(B[ib], R, B[idb]),
                         "blood_perm": ridge(B[ib][rng.permutation(len(ib))], R, B[idb]),
                         "freq": ridge(freq[ib], R, freq[idb])}
                if a.base is not None: preds = {k: v for k, v in preds.items() if k in a.base}
                for k, (E, G) in FM.items():
                    M = E[:, j] if E is not None else G
                    preds[k] = ridge(M[ib], R, M[idb])
                    if k.startswith("fm_blood_"):
                        preds[k.replace("fm_blood_", "fm_blood_perm_")] = ridge(M[ib][rng_fm.permutation(len(ib))], R, M[idb])
                live = np.abs(s - u) > FLOOR
                for m, p in preds.items():
                    rows.append(pd.DataFrame({"proto": proto, "fold": fid, "method": m, "stim": stim, "celltype": ct,
                                              "donor": d, "err2": ((p - r) ** 2)[live], "r2": (r ** 2)[live],
                                              "dev2": ((s - mu_s) ** 2)[live]}))
        print(f"[FOLD ] {proto} {fid}", flush=True)
    D = pd.concat(rows); D.to_pickle(out / "cases.pkl")
    report(D, out)


def report(D, out):
    agg = lambda g: pd.Series({"res_R2": 1 - g.err2.sum() / g.r2.sum(), "dev_R2": 1 - g.err2.sum() / g.dev2.sum(),
                               "n": len(g)})
    S_ = D.groupby(["proto", "method"]).apply(agg, include_groups=False).reset_index()
    S_.round(4).to_csv(out / "summary.csv", index=False)
    Bs = D.groupby(["proto", "method", "stim"]).apply(agg, include_groups=False).reset_index()
    Bs.round(4).to_csv(out / "by_stim.csv", index=False)
    pd.set_option("display.width", 200)
    print(S_.round(3).to_string(index=False))
    print(Bs[Bs.proto == "lodo"].pivot(index="method", columns="stim", values="res_R2").round(3).to_string())
    print(f"[DONE ] -> {out}")


if __name__ == "__main__":
    main()
