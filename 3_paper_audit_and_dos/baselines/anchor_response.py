"""Anchor-response test: does knowing part of a donor's measured stim response predict the rest of it?

Same data as stim_cross_correlations.py (plate-control wells, per-celltype marker medians, response delta = stim -
Unstim of the same plate). The target is measured on one plate and every predictor on the other (DMSO2 <-> DMSO3,
both directions), so target and predictors share no well. Leave-one-donor-out over the 19 donors. For each held-out donor and each target feature (functional marker x celltype that responds, i.e. training
donors' mean |delta| > 0.1), predict her delta with a ridge fitted on the 18 training donors
(one alpha per fit by efficient LOO on the training donors; inputs standardised on the training donors and
clipped to +-4 SD for the held-out donor). The shift baseline predicts the
training mean delta. Score = residual R^2 = 1 - SSE(arm) / SSE(shift), pooled over cases: the share of what the
shift misses that the arm recovers (0 = shift, the quantity residual_predict / where_cellot_skill call res_R2).

Arms (predictors seen for the held-out donor):
  group_HVPV      HV/PV indicator
  freq            28 Unstim log frequencies
  own_unstim      the target feature's own Unstim value
  unstim_all      all 364 Unstim functional medians + 28 log frequencies
  replicate       the same feature measured on the other plate (how much of the target is predictable at all)
  anchor_<ct>     her measured delta of the 13 functional markers in one anchor celltype, same stim
  all_other_ct    her measured delta in every other celltype, same stim (upper bound of a same-stim anchor)
  other_stims     her measured delta in the OOL-measured stims (LPS, IL246, IFNa, GMCSF) other than the target stim
  + combinations with unstim_all.
Anchor arms are scored on targets outside the anchor celltype; every arm is scored on the same cases per comparison.

Second part: the 28 Unstim log frequencies vs donor responsiveness (PC1 of the functional delta matrix, as in
stim_cross_correlations.py), pooled, adjusted for HV/PV, and within HV (n=8) / PV (n=11), with a permutation
null that shuffles donors within group.

    python baselines/anchor_response.py   -> simco_work/stim_correlations/anchor/
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import RidgeCV

from stim_cross_correlations import CT, FUNC, PLATES, STIMS, W, pc1, ranks

OUT = W / "anchor"; OUT.mkdir(exist_ok=True)
ANCHORS = ["Granulocytes", "CD4Tnaive", "Bcells", "CD56loCD16posNK", "cMCs"]
OOL_STIMS = ["LPS", "IL246", "IFNa", "GMCSF"]
ALPHAS = np.logspace(0, 5, 21)   # one shared alpha per fit; per-target alphas from 0.01 extrapolated wildly
RESP = 0.1
SPLITS = [("DMSO2", "DMSO3"), ("DMSO3", "DMSO2")]   # (target plate, predictor plate); DMSO1 left out (PI outlier plate)
rng = np.random.default_rng(0)

q = pd.read_csv(W / "wells_q50.csv.gz"); cnt = pd.read_csv(W / "wells_counts.csv.gz")
wide = q.pivot_table(index=["stim", "plate", "donor"], columns=["celltype", "marker"], values="q50")
donors = sorted(q.donor.unique()); nd = len(donors)
preg = np.array([d.startswith("PV") for d in donors], float)
n = cnt.pivot_table(index=["stim", "plate", "donor"], columns="celltype", values="n_cells").fillna(0)
lf = np.log(n.div(n.sum(1), axis=0).clip(lower=1e-5))
dmean = lambda df, s, pl: df.loc[s].loc[pl].groupby(level="donor").mean().reindex(donors)
fcols = pd.MultiIndex.from_tuples([(c, m) for c in CT for m in FUNC])
U = dmean(wide, "Unstim", ["DMSO1", "DMSO2", "DMSO3"])[fcols]
LF = dmean(lf, "Unstim", ["DMSO1", "DMSO2", "DMSO3"])[CT]
D = {s: (dmean(wide, s, PLATES[s]) - dmean(wide, "Unstim", PLATES[s]))[fcols] for s in STIMS}


def ridge_pred(Xtr, Ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    m = RidgeCV(alphas=ALPHAS).fit((Xtr - mu) / sd, Ytr)
    return m.predict(np.clip((Xte - mu) / sd, -4, 4))


# Target and predictors come from different plates (= different wells), so no well-level technical offset is
# shared: a same-plate version let the anchors, the other stims (same Unstim well subtracted) and the Unstim
# block predict the target's own well noise.
plate = lambda df, s, p: df.loc[(s, p)].reindex(donors)
ct_of = np.array([c for c, _ in fcols]); mk_of = np.array([m for _, m in fcols])
rows = []
for pt, pp in SPLITS:
    Dt = {s: (plate(wide, s, pt) - plate(wide, "Unstim", pt))[fcols].values for s in STIMS}
    Dp = {s: (plate(wide, s, pp) - plate(wide, "Unstim", pp))[fcols] for s in STIMS}
    Up = plate(wide, "Unstim", pp)[fcols].values
    LFp = plate(lf, "Unstim", pp)[CT].values
    Uall = np.column_stack([Up, LFp])
    for s in STIMS:
        Y = Dt[s]
        Xother = np.column_stack([Dp[t].values for t in OOL_STIMS if t != s])
        for i in range(nd):
            tr = np.arange(nd) != i
            ok = np.isfinite(Y).all(0) & (np.abs(Y[tr].mean(0)) > RESP)   # responding targets, chosen on training donors
            if not ok.any():
                continue
            Ytr, yte = Y[tr][:, ok], Y[i, ok]
            tct, tmk = ct_of[ok], mk_of[ok]
            pred = {"shift": Ytr.mean(0)}

            def fit(X):
                X = np.asarray(X, float); g = np.isfinite(X).all(0) & (X[tr].std(0) > 0)
                return ridge_pred(X[tr][:, g], Ytr, X[[i]][:, g])[0]

            def per_target(X):                                        # one univariate ridge per target column
                X = np.asarray(X, float)[:, ok]
                return np.array([ridge_pred(X[tr][:, [j]], Ytr[:, j], X[[i]][:, [j]])[0]
                                 if np.isfinite(X[:, j]).all() and X[tr, j].std() > 0 else Ytr[:, j].mean()
                                 for j in range(ok.sum())])
            pred["group_HVPV"] = fit(preg[:, None])
            pred["freq"] = fit(LFp)
            pred["own_unstim"] = per_target(Up)
            pred["unstim_all"] = fit(Uall)
            pred["replicate"] = per_target(Dp[s].values)              # the same feature on the other plate
            pred["other_stims"] = fit(Xother)
            pred["other_stims+unstim_all"] = fit(np.column_stack([Xother, Uall]))
            for a in ANCHORS:
                Xa = Dp[s][a].values
                pred[f"anchor_{a}"] = fit(Xa)
                pred[f"anchor_{a}+unstim_all"] = fit(np.column_stack([Xa, Uall]))
            pa = np.full(ok.sum(), np.nan)
            Yp = Dp[s].values
            for c in np.unique(tct):
                sel = tct == c
                Xc = Yp[:, ct_of != c]; g = np.isfinite(Xc).all(0) & (Xc[tr].std(0) > 0)
                pa[sel] = ridge_pred(Xc[tr][:, g], Ytr[:, sel], Xc[[i]][:, g])[0]
            pred["all_other_ct"] = pa
            rows.append(pd.DataFrame({"split": f"{pt}->target,{pp}->predictors", "stim": s, "donor": donors[i],
                                      "is_PV": preg[i], "celltype": tct, "marker": tmk, "truth": yte, **pred}))
        print(f"[{pt}/{pp} {s}] done", flush=True)

P = pd.concat(rows, ignore_index=True)
P.to_csv(OUT / "predictions.csv.gz", index=False, float_format="%.5g")

# ---- frequencies vs responsiveness (PC1), pooled / group-adjusted / within group ----
fr = []
for s in STIMS:
    X = D[s].values; X = X[:, np.isfinite(X).all(0) & (X.std(0) > 0)]
    g1, ve = pc1(ranks(X)); g1 = g1[:, 0]
    if np.corrcoef(g1, X.mean(1))[0, 1] < 0:                         # sign: + = responds more on average
        g1 = -g1
    for c in CT:
        f = LF[c].values
        r_pool = spearmanr(f, g1)[0]
        # adjusted: ranks residualised on the group indicator
        rf = rankdata(f) - np.array([rankdata(f)[preg == p].mean() for p in preg])
        rg = rankdata(g1) - np.array([rankdata(g1)[preg == p].mean() for p in preg])
        r_adj = np.corrcoef(rf, rg)[0, 1]
        r_hv = spearmanr(f[preg == 0], g1[preg == 0])[0]; r_pv = spearmanr(f[preg == 1], g1[preg == 1])[0]
        null = []
        for _ in range(2000):
            perm = np.arange(nd)
            for p in (0, 1):
                idx = np.where(preg == p)[0]; perm[idx] = rng.permutation(idx)
            null.append(np.corrcoef(rf[perm], rg)[0, 1])
        p_adj = (np.sum(np.abs(null) >= abs(r_adj)) + 1) / (len(null) + 1)
        fr.append((s, c, ve, r_pool, r_adj, p_adj, r_hv, r_pv, spearmanr(f, preg)[0]))
F = pd.DataFrame(fr, columns=["stim", "celltype", "pc1_var", "r_pooled", "r_group_adjusted", "p_adjusted_within_group_perm",
                              "r_within_HV", "r_within_PV", "r_freq_vs_pregnant"])
F.to_csv(OUT / "freq_vs_responsiveness.csv", index=False)
print("written", OUT)
