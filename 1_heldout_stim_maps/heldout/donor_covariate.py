"""A donor covariate from Unstim features that maximises between-donor variance and minimises within-donor variance
(a donor's Unstim wells vs its stim wells). Used by notebooks/donor_covariate.ipynb.

Features per donor-well (atlas, work/heldout/pheno): log frequency of the 28 populations ("freq"), optionally plus the
median of the 31 markers that the OOL table also has, per population ("freq+q50", 896 features). Each well's mean over
the TRAINING donors is subtracted (shared stim / drug / plate offset, see well_consistency.ipynb), so "within donor" =
the donor's wells disagreeing with each other once the shared offsets are removed.

Method: regularised Fisher discriminant with donors as classes. With features standardised by the SD of the training
donors' means, B = covariance of donor means, W = pooled covariance of a donor's wells around its mean; axes solve
B v = lambda (W + gamma * mean(diag W) * I) v. gamma -> 0 is pure between/within (overfits with 15 donors), gamma -> inf
is a PCA of the donor means (ignores within). Baseline "pca_unstim" = ordinary PCA on the Unstim wells.
Fit on the main set (Unstim, GMCSF, IFNa, IL246, IL33, TNFa; all 12 treatments). PI and LPS are never fitted.

Held-out check: the 5 learning-curve repeats (15 training donors, 4 test donors). Scores of every well are compared
with Mahalanobis distance under the training within-donor covariance of the scores. Metrics on the 4 test donors:
- ident_1well: a test donor's stim well is assigned to the nearest of 19 donor signatures, each signature = that donor's
  single Unstim DMSO2 well (what an OOL woman has). Chance 1/19.
- ident_12well: same with the mean of the donor's 12 Unstim wells as signature.
- icc: per axis, between-test-donor variance / (between + within), over the test donors' Unstim + evaluated stim wells,
  averaged over the first K axes.
Evaluated stim sets: "main" (the 5 main-set stims), "LPS", "PI" (never seen in the fit).
    python -m heldout.donor_covariate  -> WORK/donor_covariate/{eval.csv, final_*.csv, ool_scores.csv}
"""
import glob
import json

import numpy as np
import pandas as pd
from scipy.linalg import eigh

from heldout.common import WORK
from lc.paths import WORK as LCW

OUT = WORK / "donor_covariate"
OOL_TABLE = "/scratch/users/bmerwan/simco_work/onset_test/ml_dos/measured_unstim_full.csv"
MAIN = ["Unstim", "GMCSF", "IFNa", "IL246", "IL33", "TNFa"]
GAMMAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e4]
KS = [1, 2, 3, 5]


def atlas_features():
    """Wide table, rows (stim, treatment, donor), columns '<celltype>__freq' (natural log) and '<celltype>__<marker>__q50'."""
    f = OUT / "atlas_features.csv.gz"
    if f.exists(): return pd.read_csv(f, index_col=[0, 1, 2])
    P = pd.concat([pd.read_csv(p, usecols=["celltype", "stim", "treatment", "donor", "n_cells", "marker", "q50"])
                   for p in sorted(glob.glob(str(WORK / "pheno/*.csv.gz")))], ignore_index=True)
    key = ["stim", "treatment", "donor"]
    n = P[P.marker == "CD45"].pivot_table(index=key, columns="celltype", values="n_cells", fill_value=0)
    fr = np.log((n + 0.5).div(n.sum(1), axis=0)); fr.columns = [f"{c}__freq" for c in fr.columns]
    q = P.pivot_table(index=key, columns=["celltype", "marker"], values="q50")
    q.columns = [f"{c}__{m}__q50" for c, m in q.columns]
    X = pd.concat([fr, q], axis=1)
    OUT.mkdir(parents=True, exist_ok=True); X.to_csv(f); return X


def ool_features(cols):
    """OOL women's Unstim features on the atlas columns. The OOL table stores log10 frequencies -> natural log."""
    head = pd.read_csv(OOL_TABLE, nrows=0).columns
    O = pd.read_csv(OOL_TABLE, usecols=["Individual"] + [c for c in cols if c in head]).set_index("Individual")
    fq = [c for c in O.columns if c.endswith("__freq")]
    O[fq] = O[fq] * np.log(10)
    return O


def feature_set(X, name, ool_cols):
    freq = [c for c in X.columns if c.endswith("__freq")]
    if name == "freq": return freq
    return freq + [c for c in X.columns if c.endswith("__q50") and c in ool_cols]


def well_center(X, train):
    m = X[X.index.get_level_values("donor").isin(train)].groupby(level=["stim", "treatment"]).mean()
    return X - m.reindex(X.index.droplevel("donor")).values


def fit(Z, method, gamma=None):
    """Z: well-centred training wells. Returns (mu, sd, V) with scores = ((x - mu) / sd) @ V."""
    d = Z.index.get_level_values("donor")
    means = Z.groupby(d).mean(); mu = means.mean(); sd = means.std().replace(0, 1).fillna(1)
    S = (Z - mu) / sd
    if method == "pca_unstim":
        U = S[S.index.get_level_values("stim") == "Unstim"]; C = np.cov(U.values.T)
        w, V = np.linalg.eigh(C); return mu, sd, V[:, ::-1][:, :max(KS)]
    M = S.groupby(d).mean(); B = np.cov(M.values.T)
    R = S - M.reindex(d).values; W = R.values.T @ R.values / (len(R) - len(M))
    Wr = W + gamma * np.mean(np.diag(W)) * np.eye(len(W))
    w, V = eigh(B, Wr); return mu, sd, V[:, ::-1][:, :max(KS)]


def scores(X, mu, sd, V):
    return pd.DataFrame(((X - mu) / sd).values @ V, index=X.index)


def evaluate(Sc, train, test, K):
    """Sc: scores of all wells (main + LPS + PI). Mahalanobis under the training within-donor covariance."""
    Sk = Sc.iloc[:, :K]; d = Sk.index.get_level_values("donor"); st = Sk.index.get_level_values("stim")
    tr = Sk[d.isin(train) & st.isin(MAIN)]
    R = tr - tr.groupby(level="donor").transform("mean"); Wk = np.atleast_2d(np.cov(R.values.T)) + 1e-9 * np.eye(K)
    L = np.linalg.cholesky(np.linalg.inv(Wk)); Y = pd.DataFrame(Sk.values @ L, index=Sk.index)
    un = Y[st == "Unstim"]
    sig1 = un.xs("DMSO2", level="treatment").groupby(level="donor").mean(); sig12 = un.groupby(level="donor").mean()
    out = []
    for ev, stims in [("main", MAIN[1:]), ("LPS", ["LPS"]), ("PI", ["PI"])]:
        E = Y[d.isin(test) & st.isin(stims)]; ed = E.index.get_level_values("donor")
        row = {"evalset": ev}
        for nm, sig in [("ident_1well", sig1), ("ident_12well", sig12)]:
            D = ((E.values[:, None, :] - sig.values[None]) ** 2).sum(-1)
            row[nm] = np.mean(sig.index.values[D.argmin(1)] == ed.values)
        T = pd.concat([Y[d.isin(test) & (st == "Unstim")], E]); td = T.index.get_level_values("donor")
        mean = T.groupby(td).transform("mean"); bv = T.groupby(td).mean().var(ddof=1); wv = (T - mean).var(ddof=1)
        row["icc"] = float(np.mean(bv / (bv + wv)))
        out.append(row)
    return out


def main():
    X = atlas_features()
    ool_cols = set(pd.read_csv(OOL_TABLE, nrows=0).columns)
    design = json.load(open(LCW / "design.json"))
    donors = sorted(X.index.get_level_values("donor").unique())
    rows = []
    for fs in ["freq", "freq+q50"]:
        cols = feature_set(X, fs, ool_cols); Xf = X[cols]
        for dz in design:
            test = dz["test"]; train = [x for x in donors if x not in test]
            Z = well_center(Xf, train)
            fitset = Z[Z.index.get_level_values("donor").isin(train) & Z.index.get_level_values("stim").isin(MAIN)]
            for method, g in [("pca_unstim", None)] + [("lda", g) for g in GAMMAS]:
                mu, sd, V = fit(fitset, method, g); Sc = scores(Z, mu, sd, V)
                for K in KS:
                    for r in evaluate(Sc, train, test, K):
                        rows.append({"features": fs, "method": method, "gamma": g, "K": K, "repeat": dz["repeat"], **r})
            # reference: no reduction, all standardised features (Mahalanobis on the diagonal only)
            mu, sd, _ = fit(fitset, "lda", 1.0); S = (Z - mu) / sd
            Sc = pd.DataFrame(S.values, index=S.index)
            for r in evaluate_diag(Sc, train, test):
                rows.append({"features": fs, "method": "all_features", "gamma": None, "K": S.shape[1], "repeat": dz["repeat"], **r})
        print(f"[DONE ] {fs}: {len(cols)} features", flush=True)
    E = pd.DataFrame(rows); OUT.mkdir(parents=True, exist_ok=True); E.to_csv(OUT / "eval.csv", index=False)
    print(E.groupby(["features", "method", "gamma", "K", "evalset"], dropna=False)[["ident_1well", "ident_12well", "icc"]]
          .mean().round(3).to_string())


def evaluate_diag(Sc, train, test):
    """All standardised features, Euclidean after scaling each by its training within-donor SD."""
    d = Sc.index.get_level_values("donor"); st = Sc.index.get_level_values("stim")
    tr = Sc[d.isin(train) & st.isin(MAIN)]; wsd = (tr - tr.groupby(level="donor").transform("mean")).std().replace(0, 1)
    Y = Sc / wsd; un = Y[st == "Unstim"]
    sig1 = un.xs("DMSO2", level="treatment").groupby(level="donor").mean(); sig12 = un.groupby(level="donor").mean()
    out = []
    for ev, stims in [("main", MAIN[1:]), ("LPS", ["LPS"]), ("PI", ["PI"])]:
        E = Y[d.isin(test) & st.isin(stims)]; ed = E.index.get_level_values("donor"); row = {"evalset": ev}
        for nm, sig in [("ident_1well", sig1), ("ident_12well", sig12)]:
            D = ((E.values[:, None, :] - sig.values[None]) ** 2).sum(-1)
            row[nm] = np.mean(sig.index.values[D.argmin(1)] == ed.values)
        row["icc"] = np.nan; out.append(row)
    return out



def final(fs="freq", gamma=0.01, K=3):
    """Fit on all 19 donors; write loadings (structure = correlation of each axis with each feature over donor means),
    atlas donor scores (mean of the 12 Unstim wells), OOL scores (OOL table centred on the OOL cohort mean, atlas SDs),
    and the same for the pca_unstim baseline."""
    X = atlas_features(); ool_cols = set(pd.read_csv(OOL_TABLE, nrows=0).columns)
    cols = feature_set(X, fs, ool_cols); donors = sorted(X.index.get_level_values("donor").unique())
    Z = well_center(X[cols], donors)
    fitset = Z[Z.index.get_level_values("stim").isin(MAIN)]
    O = ool_features(cols)[cols]; Oc = O - O.mean() + fitset.groupby(level="donor").mean().mean()   # cohort-centred
    tag = f"{fs}_g{gamma:g}"
    for method, g, name in [("lda", gamma, tag), ("pca_unstim", None, f"{fs}_pca")]:
        mu, sd, V = fit(fitset, method, g); V = V[:, :K]
        S = scores(Z, mu, sd, V); un = S[S.index.get_level_values("stim") == "Unstim"].groupby(level="donor").mean()
        dm = Z[Z.index.get_level_values("stim") == "Unstim"].groupby(level="donor").mean()
        sgn = np.where(un.corrwith(dm["Granulocytes__freq"]) < 0, -1, 1)                # axis sign: + = more granulocytes
        S = S * sgn; un = un * sgn
        within_sd = (S[S.index.get_level_values("stim").isin(MAIN)] - S[S.index.get_level_values("stim").isin(MAIN)]
                     .groupby(level="donor").transform("mean")).std()
        scale = within_sd.values                                                        # report in within-donor SD units
        un = un / scale; un.columns = [f"axis{i + 1}" for i in range(K)]
        so = pd.DataFrame(((Oc - mu) / sd).values @ V * sgn / scale, index=Oc.index, columns=un.columns)
        load = pd.DataFrame({a: dm.corrwith(un[a]) for a in un.columns})
        un.to_csv(OUT / f"final_{name}_atlas_scores.csv"); so.to_csv(OUT / f"final_{name}_ool_scores.csv")
        load.to_csv(OUT / f"final_{name}_loadings.csv")
        print(f"[DONE ] {name}: atlas {un.shape}, OOL {so.shape}")


def lc_contexts(gamma=0.01, K=3):
    """Context file for lc/cond_train.py --ctx lda: for every learning-curve repeat and training size k, the covariate
    refitted on that repeat's first k training donors only (well-centred on them, frequencies, regularised LDA), and every
    donor's score = mean over its 12 Unstim wells, divided by the training donors' score SD. min(K, k - 1) axes, zero-padded.
    -> WORK/donor_covariate/lc_ctx.json  {"<repeat>_<k>": {donor: [K floats]}}"""
    X = atlas_features(); fr = [c for c in X.columns if c.endswith("__freq")]; Xf = X[fr]
    design = json.load(open(LCW / "design.json")); out = {}
    for dz in design:
        for k in [3, 5, 8, 15]:
            train = dz["order"][:k]
            Z = well_center(Xf, train)
            fitset = Z[Z.index.get_level_values("donor").isin(train) & Z.index.get_level_values("stim").isin(MAIN)]
            mu, sd, V = fit(fitset, "lda", gamma)
            n = min(K, k - 1); V = V[:, :n]
            un = scores(Z[Z.index.get_level_values("stim") == "Unstim"], mu, sd, V).groupby(level="donor").mean()
            un = un / un.loc[train].std(ddof=1)
            vec = np.zeros((len(un), K)); vec[:, :n] = un.values
            out[f"{dz['repeat']}_{k}"] = {d: [round(float(x), 6) for x in v] for d, v in zip(un.index, vec)}
    json.dump(out, open(OUT / "lc_ctx.json", "w"), indent=0)
    print(f"[DONE ] {len(out)} (repeat, k) contexts -> {OUT / 'lc_ctx.json'}")


if __name__ == "__main__":
    import sys
    if "--final" in sys.argv: final()
    elif "--lc-ctx" in sys.argv: lc_contexts()
    else: main()
