"""Cross-celltype coupling of stimulation responses, across donors and across wells.

Input: simco_work/stim_correlations/wells_{q50,counts}.csv.gz (baselines/stim_corr_wells.py).
Output: CSVs in simco_work/stim_correlations/, read by notebooks/stim_cross_correlations.ipynb.
    python baselines/stim_cross_correlations.py [--nperm 200]

Units and features
- well = (stim, plate, donor), plate = DMSO1/2/3 (the three plate controls, no drug); 19 donors.
- feature = median (arcsinh) of one marker in one celltype in one well. Functional set = 13 markers x 28 celltypes.
- response delta = stim well - Unstim well, same donor, same plate. Donor-level delta = mean over plates
  (PI: DMSO2/3 only, the DMSO1 PI wells are the known outlier plate). Unstim donor-level = mean over the 3 plates.
- technical reference ("tref") = phenotypic (non-functional) markers that are expressed in that celltype
  (Unstim median > 1) and NOT moved by that stim (|mean delta| < 0.1, the paper's own threshold). Their
  delta should be ~0 biologically, so whatever cross-celltype correlation they show is technical / donor-level
  coupling that has nothing to do with signalling.

Correlations are Spearman (Pearson on ranks across donors). Variants:
- raw;
- pc1: ranks residualised on the first PC of the functional delta matrix (a "global donor responsiveness" factor);
- tech: residualised on the first PC of the tref delta matrix (a technical / well factor);
- pc1tech: both.
Null: donors permuted independently per celltype block (all features of a celltype keep their joint structure;
only the link between celltypes is broken). Empirical p for cross-celltype pairs from the pooled null; BH per
(stim, variant) over cross-celltype pairs.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

W = Path("/scratch/users/bmerwan/simco_work/stim_correlations")
FUNC = ["CD25", "HLADR", "IkB", "pCREB", "pERK", "pMK2", "pNFkB", "pS6", "pSTAT1", "pSTAT3", "pSTAT5", "pSTAT6", "pp38"]
STIMS = ["IFNa", "IL246", "IL33", "LPS", "PI", "TNFa", "GMCSF"]
CT = ["Granulocytes", "MDSCs", "cMCs", "intMCs", "ncMCs", "mDCs", "pDCs", "Bcells",
      "CD56hiCD16negNK", "CD56loCD16posNK", "NK_cells_CD11cneg", "NK_cells_CD11cpos", "NKT", "CD4negCD8negTcells",
      "CD4Tnaive", "CD4Tcm", "CD4Tcm_CCR2pos", "CD4Tem", "CD4Tem_CCR2pos", "CD4Teff", "CD4Tregs",
      "CD8Tnaive", "CD8Tcm", "CD8Tcm_CCR2pos", "CD8Tem", "CD8Tem_CCR2pos", "CD8Teff", "CD8Tcells_Th1"]
PLATES = {s: (["DMSO2", "DMSO3"] if s == "PI" else ["DMSO1", "DMSO2", "DMSO3"]) for s in STIMS}
RESP = 0.1        # paper's selection threshold on |mean delta|
EXPR = 1.0        # "expressed" for a tref marker: Unstim median > 1 arcsinh
BINS = np.linspace(0, 1, 1001)
GR = "Granulocytes"


def ranks(X):
    return np.apply_along_axis(rankdata, 0, X)


def zcols(X):
    X = X - X.mean(0)
    s = X.std(0)
    s[s == 0] = np.nan
    return X / s


def resid(Z, C):
    if C is None or C.shape[1] == 0:
        return Z
    A = np.column_stack([np.ones(len(Z)), C])
    beta, *_ = np.linalg.lstsq(A, Z, rcond=None)
    return Z - A @ beta


def corr(Z):
    Z = zcols(Z)
    return Z.T @ Z / len(Z)


def pc1(Z):
    Z = zcols(Z)
    Z = Z[:, np.isfinite(Z).all(0)]
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    return U[:, :1] * S[0], S[0] ** 2 / (S ** 2).sum()


def usable(X, min_unique=10):
    return np.array([len(np.unique(np.round(c, 6))) >= min_unique for c in X.T])


def cat_matrix(cts, mks):
    same_ct = cts[:, None] == cts[None, :]
    same_mk = mks[:, None] == mks[None, :]
    cat = np.where(same_ct, "within", np.where(same_mk, "cross_same_marker", "cross_diff_marker"))
    return cat, same_ct


def block_mean(A, cts, blocks):
    """mean |A| per (celltype, celltype) block, diagonal of within blocks excluded."""
    G = (cts[:, None] == np.array(blocks)[None, :]).astype(float)
    F = np.isfinite(A).astype(float); np.fill_diagonal(F, 0)
    Aa = np.where(F > 0, np.abs(np.nan_to_num(A)), 0.0)
    S = G.T @ Aa @ G; Cn = G.T @ F @ G
    with np.errstate(invalid="ignore"):
        return S / Cn


def hub(Bm):
    """per source celltype: mean of its block mean |r| over all other celltypes."""
    M = Bm.copy(); np.fill_diagonal(M, np.nan)
    return np.nanmean(M, 1)


def perm_rows(Z, cts, rng):
    P = np.empty_like(Z)
    for c in np.unique(cts):
        k = np.where(cts == c)[0]
        P[:, k] = Z[rng.permutation(len(Z))][:, k]
    return P


def analyse(name, Z, cts, mks, covs, nperm, rng, store_mat=None):
    """Z: n x p rank matrix. Returns dicts of tables for every variant."""
    blocks = [c for c in CT if c in set(cts)]
    cat, same_ct = cat_matrix(cts, mks)
    iu = np.triu_indices(len(cts), 1)
    cross = ~same_ct[iu]
    res = {"pairs": [], "blocks": [], "dist": [], "hist": [], "hubs": []}
    for var, C in covs.items():
        R = resid(Z, C)
        A = corr(R)
        if store_mat is not None:
            store_mat[var] = A
        obs_b = block_mean(A, cts, blocks)
        nb = np.zeros((nperm,) + obs_b.shape)
        hist_null = {k: np.zeros(len(BINS) - 1) for k in ("within", "cross_same_marker", "cross_diff_marker")}
        for b in range(nperm):
            Ap = corr(perm_rows(R, cts, rng))
            nb[b] = block_mean(Ap, cts, blocks)
            v, cv = np.abs(Ap[iu]), cat[iu]
            for k in hist_null:
                hist_null[k] += np.histogram(v[(cv == k) & np.isfinite(v)], BINS)[0]
        # pair table, cross-celltype only
        r = A[iu]; a_i, b_i = iu
        null_cross = hist_null["cross_same_marker"] + hist_null["cross_diff_marker"]
        tail = np.cumsum(null_cross[::-1])[::-1]  # count of null |r| >= bin lower edge
        bi = np.clip(np.searchsorted(BINS, np.abs(r), side="right") - 1, 0, len(BINS) - 2)
        p = (tail[bi] + 1) / (null_cross.sum() + 1)
        m = cross & np.isfinite(r)
        pp = p[m]
        order = np.argsort(pp); q = np.empty_like(pp)
        q[order] = np.minimum.accumulate((pp[order] * len(pp) / np.arange(1, len(pp) + 1))[::-1])[::-1]
        res["pairs"].append(pd.DataFrame({"set": name, "variant": var, "ct_a": cts[a_i][m], "mk_a": mks[a_i][m],
                                          "ct_b": cts[b_i][m], "mk_b": mks[b_i][m], "category": cat[iu][m],
                                          "r": r[m], "p_perm": pp, "q_bh": np.minimum(q, 1)}))
        mu, sd = nb.mean(0), nb.std(0)
        ho = hub(obs_b); hn = np.array([hub(x) for x in nb])
        for i, x in enumerate(blocks):
            res["hubs"].append((name, var, x, ho[i], hn[:, i].mean(), hn[:, i].std(), (hn[:, i] >= ho[i]).mean()))
        for i, x in enumerate(blocks):
            for j, y in enumerate(blocks):
                res["blocks"].append((name, var, x, y, obs_b[i, j], mu[i, j], sd[i, j], (nb[:, i, j] >= obs_b[i, j]).mean()))
        for k in hist_null:
            v = np.abs(A[iu][(cat[iu] == k) & np.isfinite(A[iu])])
            res["hist"].append(pd.DataFrame({"set": name, "variant": var, "category": k, "bin_lo": BINS[:-1],
                                             "obs": np.histogram(v, BINS)[0], "null": hist_null[k] / nperm}))
            nv = hist_null[k] / nperm
            cdf = np.cumsum(nv) / nv.sum()
            res["dist"].append((name, var, k, len(v), np.median(v), np.mean(v), np.quantile(v, 0.95),
                                BINS[1:][np.searchsorted(cdf, 0.5)], BINS[1:][np.searchsorted(cdf, 0.95)],
                                np.mean(v > BINS[1:][np.searchsorted(cdf, 0.95)])))
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nperm", type=int, default=200); a = ap.parse_args()
    rng = np.random.default_rng(0)
    q = pd.read_csv(W / "wells_q50.csv.gz")
    cnt = pd.read_csv(W / "wells_counts.csv.gz")
    wide = q.pivot_table(index=["stim", "plate", "donor"], columns=["celltype", "marker"], values="q50")
    donors = sorted(q.donor.unique())
    markers = sorted(q.marker.unique()); PHENO = [m for m in markers if m not in FUNC]
    # frequencies: share of the 28 celltypes' cells in each well
    n = cnt.pivot_table(index=["stim", "plate", "donor"], columns="celltype", values="n_cells").fillna(0)
    freq = n.div(n.sum(1), axis=0)
    lfreq = np.log(freq.clip(lower=1e-5))

    def dmean(df, stim, plates):
        return df.loc[stim].loc[plates].groupby(level="donor").mean().reindex(donors)

    U = dmean(wide, "Unstim", ["DMSO1", "DMSO2", "DMSO3"])
    U_lf = dmean(lfreq, "Unstim", ["DMSO1", "DMSO2", "DMSO3"])
    fcols = pd.MultiIndex.from_tuples([(c, m) for c in CT for m in FUNC])

    meta, pairs, blocks, dist, hists, hubs, gf, simil, base_pairs, base_blocks, well_rows = ([] for _ in range(11))
    mats = {}
    for s in STIMS:
        pl = PLATES[s]
        D = dmean(wide, s, pl) - dmean(wide, "Unstim", pl)
        # plate-wise deltas for reliability
        Dp = {p: (wide.loc[(s, p)] - wide.loc[("Unstim", p)]).reindex(donors) for p in pl}
        mean_d = D.mean(); Umed = U.median()
        rel = {}
        for col in D.columns:
            rs = []
            for i in range(len(pl)):
                for j in range(i + 1, len(pl)):
                    x, y = Dp[pl[i]][col].values, Dp[pl[j]][col].values
                    rs.append(np.corrcoef(rankdata(x), rankdata(y))[0, 1] if np.std(x) > 0 and np.std(y) > 0 else np.nan)
            r1 = np.nanmean(rs) if np.isfinite(rs).any() else np.nan
            k = len(pl)
            rel[col] = (r1, k * r1 / (1 + (k - 1) * r1) if np.isfinite(r1) and r1 > -1 / (k - 1) else np.nan)
        tref = [c for c in D.columns if c[1] in PHENO and Umed[c] > EXPR and abs(mean_d[c]) < RESP]
        for col in list(fcols) + tref:
            meta.append((s, col[0], col[1], "func" if col[1] in FUNC else "tref", mean_d[col], D[col].std(),
                         Umed[col], rel[col][0], rel[col][1]))
        # functional set
        Xf = D[fcols].values; okf = usable(Xf)
        Zf = ranks(Xf[:, okf]); cf = np.array([c for c, _ in fcols])[okf]; mf = np.array([m for _, m in fcols])[okf]
        Xt = D[tref].values; okt = usable(Xt)
        Zt = ranks(Xt[:, okt]); ct_t = np.array([c for c, _ in tref])[okt]; mt = np.array([m for _, m in tref])[okt]
        g1, ve_f = pc1(Zf); t1, ve_t = pc1(Zt)
        covs = {"raw": None, "pc1": g1, "tech": t1, "pc1tech": np.column_stack([g1, t1])}
        simil.append((s, "pc1_var_explained_func", ve_f)); simil.append((s, "pc1_var_explained_tref", ve_t))
        simil.append((s, "corr_pc1func_pc1tref", abs(np.corrcoef(g1[:, 0], t1[:, 0])[0, 1])))
        simil.append((s, "n_func_features", okf.sum())); simil.append((s, "n_tref_features", okt.sum()))
        gscore = zcols(Zf[:, cf == GR]).mean(1) if (cf == GR).any() else None
        simil.append((s, "corr_pc1func_granulocyte_score", abs(np.corrcoef(g1[:, 0], gscore)[0, 1])))
        mats[s] = {}
        rf = analyse("func", Zf, cf, mf, covs, a.nperm, rng, store_mat=mats[s])
        rt = analyse("tref", Zt, ct_t, mt, {"raw": None, "pc1": g1}, a.nperm, rng)
        for rr in (rf, rt):
            pairs += [d.assign(stim=s) for d in rr["pairs"]]
            blocks += [(s,) + t for t in rr["blocks"]]
            hubs += [(s,) + t for t in rr["hubs"]]
            dist += [(s,) + t for t in rr["dist"]]
            hists += [d.assign(stim=s) for d in rr["hist"]]
        np.save(W / f"mat_func_{s}_cols.npy", np.array([f"{c}|{m}" for c, m in zip(cf, mf)]))
        for var, A in mats[s].items():
            pd.DataFrame(A, index=[f"{c}|{m}" for c, m in zip(cf, mf)],
                         columns=[f"{c}|{m}" for c, m in zip(cf, mf)]).to_csv(W / f"mat_func_{s}_{var}.csv.gz", float_format="%.4f")

        # ---- granulocyte frequency vs other celltypes' responses ----
        lf_u = U_lf[GR].values; lf_s = dmean(lfreq, s, pl)[GR].values
        for fname, fv in (("gran_logfreq_unstim", lf_u), ("gran_dlogfreq_stim", lf_s - dmean(lfreq, "Unstim", pl)[GR].values)):
            for var, C in (("raw", None), ("pc1", g1)):
                Rz = zcols(resid(Zf, C)); fz = zcols(resid(rankdata(fv)[:, None], C))[:, 0]
                obs = np.nanmean(np.abs((Rz[:, cf != GR] * fz[:, None]).mean(0)))
                rr = (Rz * fz[:, None]).mean(0)
                null = np.array([np.nanmean(np.abs((Rz[:, cf != GR] * fz[rng.permutation(len(fz))][:, None]).mean(0)))
                                 for _ in range(a.nperm)])
                gf.append((s, fname, var, obs, null.mean(), null.std(), (null >= obs).mean(),
                           int(np.sum(np.abs(rr[cf != GR]) > 0.6))))

        # ---- baseline (Unstim, other celltype) -> response, controlling own baseline ----
        Ub = U[fcols].values[:, okf]; Zu = ranks(Ub)
        lfu = ranks(U_lf[CT].values)
        Rd = np.column_stack([zcols(resid(Zf[:, [j]], Zu[:, [j]]))[:, 0] for j in range(Zf.shape[1])])
        P = np.column_stack([zcols(Zu), zcols(lfu)])
        pct = np.r_[cf, np.array(CT)]; pmk = np.r_[mf, np.array(["freq"] * len(CT))]
        B = P.T @ np.nan_to_num(Rd) / len(Rd)          # predictors x responses
        crossm = pct[:, None] != cf[None, :]
        nullB = []
        for _ in range(a.nperm):
            Pp = P[rng.permutation(len(P))]
            nullB.append(np.abs(Pp.T @ np.nan_to_num(Rd) / len(Rd))[crossm])
        nullB = np.sort(np.concatenate(nullB))
        ii, jj = np.where(crossm)
        p = (len(nullB) - np.searchsorted(nullB, np.abs(B[ii, jj]), side="left") + 1) / (len(nullB) + 1)
        order = np.argsort(p); qv = np.empty_like(p)
        qv[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
        base_pairs.append(pd.DataFrame({"stim": s, "pred_ct": pct[ii], "pred_mk": pmk[ii], "resp_ct": cf[jj],
                                        "resp_mk": mf[jj], "r_partial": B[ii, jj], "p_perm": p, "q_bh": np.minimum(qv, 1)}))
        for x in CT:
            for y in CT:
                if x == y:
                    continue
                M = np.abs(B[np.ix_((pct == x) & (pmk != "freq"), cf == y)])
                base_blocks.append((s, x, y, M.mean() if M.size else np.nan))

        # ---- Unstim-Unstim reference vs delta structure ----
        Au = corr(Zu)
        iu = np.triu_indices(len(cf), 1); cm = cf[iu[0]] != cf[iu[1]]
        for var in ("raw", "pc1"):
            Ad = mats[s][var]
            x, y = Au[iu][cm], Ad[iu][cm]; ok = np.isfinite(x) & np.isfinite(y)
            simil.append((s, f"corr_unstim_vs_delta_matrix_{var}", np.corrcoef(rankdata(x[ok]), rankdata(y[ok]))[0, 1]))

        # ---- well-level (within-donor) coupling: stim wells, two-way (donor, plate) residuals ----
        for wstim in (s, "Unstim"):
            Wd = wide.loc[wstim].loc[pl]
            Wd = Wd.reindex(pd.MultiIndex.from_product([pl, donors], names=["plate", "donor"]))
            for kind, cols in (("func", list(fcols)), ("tref", tref)):
                X = Wd[cols].values
                good = np.isfinite(X).all(0) & usable(np.nan_to_num(X), 15)
                X = X[:, good]; cc = np.array([c for c, _ in cols])[good]
                dm = pd.get_dummies(pd.DataFrame({"d": Wd.index.get_level_values("donor"), "p": Wd.index.get_level_values("plate")}),
                                    drop_first=True).values.astype(float)
                Rw = resid(X, dm)
                Zw = ranks(Rw)
                A = corr(Zw)
                iu2 = np.triu_indices(len(cc), 1); crs = cc[iu2[0]] != cc[iu2[1]]
                v = np.abs(A[iu2][crs])
                nv = []
                for _ in range(min(a.nperm, 50)):
                    Ap = corr(perm_rows(Zw, cc, rng)); nv.append(np.abs(Ap[iu2][crs]))
                nv = np.concatenate(nv)
                gmask = ((cc[iu2[0]] == GR) | (cc[iu2[1]] == GR))[crs]
                well_rows.append((s, wstim, kind, len(cc), X.shape[0], np.median(v), np.median(nv),
                                  np.mean(v > np.quantile(nv, 0.95)), np.median(v[gmask]) if gmask.any() else np.nan))
        print(f"[{s}] func {okf.sum()} tref {okt.sum()} pc1 {ve_f:.2f}", flush=True)

    pd.DataFrame(meta, columns=["stim", "celltype", "marker", "kind", "mean_delta", "sd_delta", "unstim_median",
                                "rel_plate_pair", "rel_plate_mean"]).to_csv(W / "features_meta.csv", index=False)
    pd.concat(pairs).to_csv(W / "pairs_cross.csv.gz", index=False, float_format="%.5g")
    pd.DataFrame(blocks, columns=["stim", "set", "variant", "ct_a", "ct_b", "mean_abs_r", "null_mean", "null_sd",
                                  "p_perm"]).to_csv(W / "blocks.csv", index=False)
    pd.DataFrame(dist, columns=["stim", "set", "variant", "category", "n_pairs", "median_abs_r", "mean_abs_r",
                                "q95_abs_r", "null_median", "null_q95", "frac_above_null_q95"]).to_csv(W / "dist_summary.csv", index=False)
    pd.DataFrame(hubs, columns=["stim", "set", "variant", "source", "hub", "null_mean", "null_sd",
                                "p_perm"]).to_csv(W / "hubs.csv", index=False)
    pd.concat(hists).to_csv(W / "abs_r_hist.csv.gz", index=False)
    pd.DataFrame(gf, columns=["stim", "feature", "variant", "mean_abs_r_obs", "null_mean", "null_sd", "p_perm",
                              "n_abs_r_gt_0.6"]).to_csv(W / "granulocyte_freq.csv", index=False)
    pd.DataFrame(simil, columns=["stim", "quantity", "value"]).to_csv(W / "stim_summary.csv", index=False)
    pd.concat(base_pairs).to_csv(W / "baseline_to_response_pairs.csv.gz", index=False, float_format="%.5g")
    pd.DataFrame(base_blocks, columns=["stim", "pred_ct", "resp_ct", "mean_abs_r_partial"]).to_csv(W / "baseline_to_response_blocks.csv", index=False)
    pd.DataFrame(well_rows, columns=["stim", "wells_of", "set", "n_celltypes", "n_wells", "median_abs_r_cross",
                                     "null_median", "frac_above_null_q95", "median_abs_r_cross_granulocyte"]).to_csv(W / "well_level.csv", index=False)


if __name__ == "__main__":
    main()
