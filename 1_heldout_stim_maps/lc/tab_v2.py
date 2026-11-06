"""TabPFN v2 feature sets, built up step by step; per-celltype tables and lineage-group tables (2026-10-12 request).

Same target and protocol as lc/tab_features.py + lc/tab_predict.py (residual beyond the shift, leave-that-donor-out
means for training rows, learning-curve design), with these changes:
- stims PI and IL33 left out; only q50 / q75 / q90 rows are predicted (q10 / q25 stay available as inputs of the PCs);
- `marker` (and `celltype` in group tables) passed to TabPFN as categorical; log_n dropped.
Donor-level blocks, all fitted on the training donors only (PCA on standardised columns; min(n, #train - 1) axes,
the rest zero, hence dropped as constant):
- l1pc: PCs of the donor's own-celltype Unstim profile (13 markers x 5 quantiles = 65), per celltype
- fpc:  3 PCs of the 28 log celltype frequencies (WORK/fm/counts.csv)
- hvpv: 1 for PV donors, 0 for HV
- bpc:  PCs of the whole-blood Unstim profile (28 celltypes x 65, missing celltypes imputed with the training mean)
Layers (incremental):  S0 = u_dev, mu_u, eff, sd_u, noise_u, q, marker
  L1cat = S0 + the raw 65 own-celltype columns (old L1 with marker categorical, no log_n)
  S1_5 / S1_7 = S0 + 5 / 7 l1pc;  S2 = S1_5 + fpc;  S3 = S2 + hvpv;  S4_3 / S4_5 = S3 + 3 / 5 bpc
  S4_7 = S1_7 + fpc + hvpv + 5 bpc;  S4perm = S4_5 with every donor-level block shuffled across training donors (null)
Second round (2026-10-13), raw own profile + whole-donor blocks, and data-driven marker axes:
  mx0-mx2: 3 MDS coordinates per marker of the marker-marker correlation of mean stim effects (S - U, over the 5 kept
    stims x 5 quantiles, per celltype, Fisher-z averaged over celltypes), fitted on the training donors of (repeat, k)
  L1cat_mx = L1cat with mx0-2 instead of the categorical marker; L1cat_mxc = L1cat + mx0-2 (marker kept)
  L1F = L1cat + 28 raw log frequencies (f_<ct>);  L1fpc = L1cat + fpc;  L1all_pc = L1cat + fpc + hvpv + bpc0-2
  L1cat_q3f = L1cat with only the q50 / q75 / q90 own-profile columns (39 instead of 65)
  --qkeep 10 25 50 75 90: build and predict all five quantile rows (default q50-q90)
  L1all_raw = L1cat + 28 raw log freq + hvpv + bpc0-2;  L1all_raw_perm = L1all_raw with the own profile and every
    donor block taken from another training donor (null)
LightGBM (--model lgbm): the same rows and features, one model per (repeat, k, stim) over all celltypes with `celltype`
  categorical (as lc/tab_predict.py), rounds by donor-grouped CV; output WORK/tab/v2lgb_<layer>/.
Lineage round 2 (2026-10-14):
  row-aligned other-celltype features: mk_<ct> (28) = the donor's deviation at the row's quantile x marker in every
  celltype (training-donor means, leave-that-donor-out for training rows, NaN if the donor lacks the celltype);
  dw_mean / dw_sd = their mean / sd over celltypes; REF4 = mk_ of the 4 dominant celltypes.
  ct layers: L1X = L1cat + dw + REF4; L1M = L1cat + mk28 + dw; L1allM = L1all_raw + mk28 + dw.
  group layers (raw own profile + categorical celltype): G_L1cat, G_L1F, G_L1all_raw, G_L1M, G_L1allM.
  --groupset cur | big | T (GROUPSETS: stim-specific membership; groups of one celltype are skipped).
Column reduction (2026-10-15), base + mk28 + dw + hvpv always kept, whole-blood PCs dropped: own profile as q50-q90 (R_ownq3),
  medians (R_ownq50), per-marker mean + q90-q10 spread (R_ownMS), 5 marker-pathway means x 5 quantiles (R_ownPath), 5 / 10 / 15
  PCs (R_ownPC<n>), 3 / 5 / 10 PLS components (R_ownPLS<n>: supervised on the table's residual target, fitted without the
  donor for training rows); frequencies as 3 / 5 / 10 PCs (R_frPC<n>), 3 / 5 PLS (R_frPLS<n>), 8 lineage log-frequencies
  (R_frLin); L1M_q3; compact combinations C_q3_lin, C_ms_lin, C_path_lin, C_pls5_pls3.
Group tables (--mode group): one table per (stim, lineage group) with `celltype` as a categorical column; layers
  G_S0, G_S1_5, G_S4_5 (l1pc fitted per celltype).
Writes WORK/tab/v2_<layer>/repeat_<r>/ (maps.csv, rhat.npy (M, n_test, 5, 13), NaN at q10/q25 and outside the run),
checkpoint after every (k, stim); a rerun of the same command resumes.
    python -m lc.tab_v2 --mode ct --repeat 0 [--ks 15 --stims IFNa --layers S0 S1_5 --tag _smoke]
"""
import argparse
import json
import time
from functools import lru_cache

import numpy as np
import pandas as pd

from lc.paths import DONORS, MARKERS, PAIRS, QS, WORK
from lc.tab_features import CTS, donor_static, tables

OUT = WORK / "tab"
KS = [3, 5, 8, 15]
STIMS = ["GMCSF", "IFNa", "IL246", "LPS", "TNFa"]                       # PI and IL33 left out
QKEEP = (50, 75, 90)
GROUPS = {"CD4mem": ["CD4Tcm", "CD4Tcm_CCR2pos", "CD4Tem", "CD4Tem_CCR2pos"],
          "CD8eff": ["CD8Tem", "CD8Tem_CCR2pos", "CD8Teff", "CD8Tcells_Th1"],
          "NK16": ["CD56loCD16posNK", "NK_cells_CD11cneg", "NK_cells_CD11cpos"],
          "Mono": ["cMCs", "intMCs", "ncMCs"]}
BASE = ["u_dev", "mu_u", "eff", "sd_u", "noise_u", "q", "marker"]
L1 = [f"c_q{q}_{m}" for q in QS for m in MARKERS]
L1PC = [f"l1pc{i}" for i in range(7)]; FPC = [f"fpc{i}" for i in range(3)]; BPC = [f"bpc{i}" for i in range(5)]
L1PC15 = [f"l1pc{i}" for i in range(15)]; FPC10 = [f"fpc{i}" for i in range(10)]          # first 7 / 3 = L1PC / FPC
PLSO = [f"plso{i}" for i in range(10)]; PLSF = [f"plsf{i}" for i in range(5)]
LINEAGES = {"CD4T": ["CD4Tnaive", "CD4Tcm", "CD4Tcm_CCR2pos", "CD4Tem", "CD4Tem_CCR2pos", "CD4Teff", "CD4Tregs"],
            "CD8T": ["CD8Tnaive", "CD8Tcm", "CD8Tcm_CCR2pos", "CD8Tem", "CD8Tem_CCR2pos", "CD8Teff", "CD8Tcells_Th1"],
            "otherT": ["CD4negCD8negTcells", "NKT"], "NK": ["CD56hiCD16negNK", "CD56loCD16posNK", "NK_cells_CD11cneg", "NK_cells_CD11cpos"],
            "B": ["Bcells"], "Mono": ["cMCs", "intMCs", "ncMCs"], "DC": ["mDCs", "pDCs"], "GranMDSC": ["Granulocytes", "MDSCs"]}
FLIN = [f"flin_{g}" for g in LINEAGES]
PATHWAYS = {"STAT": ["pSTAT1", "pSTAT3", "pSTAT5", "pSTAT6"], "MAPK": ["pERK", "pp38", "pMK2"], "NFKB": ["pNFkB", "IkB"],
            "S6CREB": ["pS6", "pCREB"], "SURF": ["CD25", "HLADR"]}
LAYERS_CT = {"S0": BASE, "L1cat": BASE + L1, "S1_5": BASE + L1PC[:5], "S1_7": BASE + L1PC,
             "S2": BASE + L1PC[:5] + FPC, "S3": BASE + L1PC[:5] + FPC + ["hvpv"],
             "S4_3": BASE + L1PC[:5] + FPC + ["hvpv"] + BPC[:3], "S4_5": BASE + L1PC[:5] + FPC + ["hvpv"] + BPC,
             "S4_7": BASE + L1PC + FPC + ["hvpv"] + BPC, "S4perm": BASE + [f"p_{x}" for x in L1PC[:5] + FPC + ["hvpv"] + BPC]}
LAYERS_GROUP = {"G_S0": BASE + ["celltype"], "G_S1_5": BASE + ["celltype"] + L1PC[:5],
                "G_S4_5": BASE + ["celltype"] + L1PC[:5] + FPC + ["hvpv"] + BPC}
F28 = [f"f_{c}" for c in CTS]; MX = ["mx0", "mx1", "mx2"]
NOM = [c for c in BASE if c != "marker"]
LAYERS_CT.update({"L1cat_mx": NOM + MX + L1, "L1cat_mxc": BASE + MX + L1, "L1F": BASE + L1 + F28,
                  "L1fpc": BASE + L1 + FPC, "L1all_pc": BASE + L1 + FPC + ["hvpv"] + BPC[:3],
                  "L1all_raw": BASE + L1 + F28 + ["hvpv"] + BPC[:3],
                  "L1all_raw_perm": BASE + [f"p_{x}" for x in L1 + F28 + ["hvpv"] + BPC[:3]],
                  "L1cat_q3f": BASE + [x for x in L1 if x.startswith(("c_q50_", "c_q75_", "c_q90_"))]})
BLOOD4 = ["Granulocytes", "CD4Tnaive", "cMCs", "CD8Tnaive"]
MK = [f"mk_{c}" for c in CTS]; DW = ["dw_mean", "dw_sd"]; REF4 = [f"mk_{c}" for c in BLOOD4]
LAYERS_CT.update({"L1X": BASE + L1 + DW + REF4, "L1M": BASE + L1 + MK + DW,
                  "L1allM": BASE + L1 + F28 + ["hvpv"] + BPC[:3] + MK + DW})
LAYERS_CT.update({   # L1allM ablations (2026-10-15): base, mk28 and dw always kept
    "L1allM_noOwn": BASE + F28 + ["hvpv"] + BPC[:3] + MK + DW, "L1allM_noFreq": BASE + L1 + ["hvpv"] + BPC[:3] + MK + DW,
    "L1allM_noBPC": BASE + L1 + F28 + ["hvpv"] + MK + DW, "Mmin": BASE + MK + DW})
OM = [f"om_mean_{m}" for m in MARKERS] + [f"om_spr_{m}" for m in MARKERS]
OP = [f"op_{g}_q{q}" for g in PATHWAYS for q in QS]
L1Q3 = [x for x in L1 if x.startswith(("c_q50_", "c_q75_", "c_q90_"))]; L1Q50 = [x for x in L1 if x.startswith("c_q50_")]
_K = BASE + MK + DW + ["hvpv"]           # kept in every reduction variant (2026-10-15)
LAYERS_CT.update({"R_ownq3": _K + L1Q3 + F28, "R_ownq50": _K + L1Q50 + F28, "R_ownMS": _K + OM + F28, "R_ownPath": _K + OP + F28,
                  **{f"R_ownPC{n}": _K + L1PC15[:n] + F28 for n in (5, 10, 15)},
                  **{f"R_ownPLS{n}": _K + PLSO[:n] + F28 for n in (3, 5, 10)},
                  **{f"R_frPC{n}": _K + L1 + FPC10[:n] for n in (3, 5, 10)},
                  **{f"R_frPLS{n}": _K + L1 + PLSF[:n] for n in (3, 5)},
                  "R_frLin": _K + L1 + FLIN, "L1M_q3": BASE + L1Q3 + MK + DW,
                  "C_q3_lin": _K + L1Q3 + FLIN, "C_ms_lin": _K + OM + FLIN, "C_path_lin": _K + OP + FLIN,
                  "C_pls5_pls3": _K + PLSO[:5] + PLSF[:3]})
GB = BASE + ["celltype"]
LAYERS_GROUP.update({"G_L1cat": GB + L1, "G_L1F": GB + L1 + F28, "G_L1all_raw": GB + L1 + F28 + ["hvpv"] + BPC[:3],
                     "G_L1M": GB + L1 + MK + DW, "G_L1allM": GB + L1 + F28 + ["hvpv"] + BPC[:3] + MK + DW})
_CD4M = ["CD4Tcm", "CD4Tcm_CCR2pos", "CD4Tem", "CD4Tem_CCR2pos"]; _CD8E = ["CD8Tem", "CD8Tem_CCR2pos", "CD8Teff", "CD8Tcells_Th1"]
_NK = ["CD56loCD16posNK", "NK_cells_CD11cneg", "NK_cells_CD11cpos"]; _MO = ["cMCs", "intMCs", "ncMCs"]
_CD4A = ["CD4Tnaive", "CD4Tcm", "CD4Tcm_CCR2pos", "CD4Tem", "CD4Tem_CCR2pos", "CD4Teff", "CD4Tregs"]
_CD8A = ["CD8Tnaive", "CD8Tcm", "CD8Tcm_CCR2pos", "CD8Tem", "CD8Tem_CCR2pos", "CD8Teff", "CD8Tcells_Th1"]
_NKA = _NK + ["CD56hiCD16negNK", "NKT"]; _MYE = _MO + ["MDSCs", "mDCs"]
GROUPSETS = {   # lessons of the first lineage run: no CD4mem for TNFa, no CD56lo NK for IL246, no monocytes for LPS / GMCSF
    "cur": {"IFNa": [_CD4M, _CD8E, _NK, _MO], "IL246": [_CD4M, _CD8E, _NK[1:], _MO], "LPS": [_CD4M, _CD8E, _NK],
            "TNFa": [_NK, _MO], "GMCSF": []},
    "big": {s: [_CD4A, _CD8A, _NKA, _MYE] for s in STIMS},
    "T": {s: [_CD4A + _CD8A + ["CD4negCD8negTcells", "NKT"]] for s in STIMS}}
DONOR_COLS = L1PC + FPC + ["hvpv"] + BPC


def pca_scores(M, train_idx, n):
    """PCs of the rows of M (donors x features), fitted on train_idx (standardised); returns (n_donors, n), zero-padded."""
    mu = np.nanmean(M[train_idx], 0); M = np.where(np.isnan(M), mu, M)
    Z = M[train_idx]; sd = Z.std(0); ok = sd > 1e-9
    Zs = (Z[:, ok] - Z[:, ok].mean(0)) / sd[ok]
    _, _, Vt = np.linalg.svd(Zs, full_matrices=False)
    m = min(n, len(train_idx) - 1, Vt.shape[0])
    S = ((M[:, ok] - Z[:, ok].mean(0)) / sd[ok]) @ Vt[:m].T
    return np.pad(S, ((0, 0), (0, n - m)))


@lru_cache(1)
def blood_profile():
    T = tables(); B = np.full((len(DONORS), len(CTS) * 65), np.nan)
    for i, d in enumerate(DONORS):
        for j, c in enumerate(CTS):
            if (c, "Unstim", d) in T: B[i, j * 65:(j + 1) * 65] = T[(c, "Unstim", d)][0].ravel()
    return B


def donor_level(train):
    """Whole-donor blocks fpc / hvpv / bpc for all donors, fitted on `train`. DataFrame indexed by donor."""
    freq, _, _ = donor_static()
    it = [DONORS.index(d) for d in train]
    D = pd.DataFrame(index=DONORS)
    D[FPC10] = pca_scores(freq.to_numpy(), it, 10)
    fr = np.exp(freq)                                                    # log -> share; a missing celltype counts 0
    for g, cs in LINEAGES.items(): D[f"flin_{g}"] = np.log(fr[cs].fillna(0).sum(1).clip(lower=1e-6)).to_numpy()
    D["hvpv"] = [1.0 if d.startswith("PV") else 0.0 for d in DONORS]
    D[BPC] = pca_scores(blood_profile(), it, 5)
    F = freq.to_numpy(); D[F28] = np.where(np.isnan(F), np.nanmean(F[it], 0), F)
    return D


def l1_pcs(ct, trd):
    """Own-celltype profile PCs (7) for every donor with Unstim data in ct, fitted on trd."""
    T = tables(); have = [d for d in DONORS if (ct, "Unstim", d) in T]
    M = np.stack([T[(ct, "Unstim", d)][0].ravel() for d in have])
    S = pca_scores(M, [have.index(d) for d in trd], 15)
    return pd.DataFrame(S, index=have, columns=L1PC15)


@lru_cache(8)
def marker_axes(train):
    """3 MDS coordinates per marker of the trajectory correlation (training donors only). train: tuple of donors."""
    T = tables(); Ms = []
    for c in CTS:
        E = []
        for s in STIMS:
            ds = [d for d in train if (c, s, d) in T and (c, "Unstim", d) in T]
            if (s, c) in set(PAIRS) and len(ds) >= 2:
                E.append(np.mean([T[(c, s, d)][0] - T[(c, "Unstim", d)][0] for d in ds], 0))
        if len(E) < 1: continue
        X = np.concatenate(E, 0); ok = X.std(0) > 1e-6
        M = np.full((13, 13), np.nan); M[np.ix_(ok, ok)] = np.corrcoef(X[:, ok].T); Ms.append(M)
    R = np.tanh(np.nanmean(np.arctanh(np.clip(np.stack(Ms), -0.999, 0.999)), 0))
    R = np.where(np.isnan(R), 0, R); np.fill_diagonal(R, 1)
    Dm = np.sqrt(np.maximum(2 - 2 * R, 0)); J = np.eye(13) - 1 / 13; B = -0.5 * J @ (Dm ** 2) @ J
    w, V = np.linalg.eigh(B); i = np.argsort(w)[::-1][:3]
    return V[:, i] * np.sqrt(np.maximum(w[i], 0))                     # (13, 3)


def pls_scores(X, Y, trd, donors, n):
    """Supervised PLS scores (n components) of X[d] (donor -> vector) for Y[d] (training donors). A training donor's
    scores come from a fit without it (no own target), a test donor's from the fit on all of trd. -> {donor: (n,)}"""
    from sklearn.cross_decomposition import PLSRegression
    out = {}
    for d in donors:
        fit = [x for x in trd if x != d]
        Xf = np.stack([X[x] for x in fit]); Yf = np.stack([Y[x] for x in fit])
        kx = Xf.std(0) > 1e-9; ky = Yf.std(0) > 1e-9
        m = min(n, len(fit) - 1, int(kx.sum()))
        v = np.zeros(n)
        if m >= 1 and ky.any():
            with np.errstate(all="ignore"):
                v[:m] = PLSRegression(n_components=m, scale=True).fit(Xf[:, kx], Yf[:, ky]).transform(X[d][kx][None])[0]
        out[d] = np.nan_to_num(v)
    return out


_XCT = {}


def xct_dev(train, d):
    """(28, 5, 13) deviations of donor d's Unstim quantiles from the training donors' mean, every celltype (NaN if d
    lacks it); for a training donor the mean leaves d out."""
    key = (train, d)
    if key not in _XCT:
        T = tables(); ref = [x for x in train if x != d]; out = np.full((len(CTS), 5, 13), np.nan)
        for j, c in enumerate(CTS):
            if (c, "Unstim", d) not in T: continue
            R = [T[(c, "Unstim", x)][0] for x in ref if (c, "Unstim", x) in T]
            if R: out[j] = T[(c, "Unstim", d)][0] - np.mean(R, 0)
        if len(_XCT) > 4000: _XCT.clear()
        _XCT[key] = out
    return _XCT[key]


def build(repeat, k, stim, cts, train, tests, qkeep=QKEEP):
    """Rows (donor, celltype, q in qkeep, marker) for the given celltypes of one (repeat, k, stim)."""
    T = tables(); DL = donor_level(train); MXc = marker_axes(tuple(train))
    rng = np.random.default_rng(1000 * repeat + k)
    perm = dict(zip(train, rng.permutation(train)))
    out = []
    for c in cts:
        if (stim, c) not in set(PAIRS): continue
        trd = [d for d in train if (c, stim, d) in T and (c, "Unstim", d) in T]
        if len(trd) < 3: continue
        P1 = l1_pcs(c, trd)
        U = {d: T[(c, "Unstim", d)][0] for d in trd}; S = {d: T[(c, stim, d)][0] for d in trd}
        allds = trd + [d for d in tests if (c, "Unstim", d) in T]
        mU, mS = np.mean([U[x] for x in trd], 0), np.mean([S[x] for x in trd], 0)
        qk = [QS.index(q) for q in qkeep]
        Yp = {x: ((S[x] - mS) - (U[x] - mU))[qk].ravel() for x in trd}
        Xo = {x: T[(c, "Unstim", x)][0].ravel() for x in allds}; Xf = {x: DL.loc[x, F28].to_numpy(float) for x in allds}
        PO = pls_scores(Xo, Yp, trd, allds, 10); PF = pls_scores(Xf, Yp, trd, allds, 5)
        qi = [QS.index(q) for q in qkeep]; nq = len(qkeep)
        for d, is_test in [(d, False) for d in trd] + [(d, True) for d in tests if (c, "Unstim", d) in T]:
            ref = trd if is_test else [x for x in trd if x != d]
            mu_u = np.mean([U[x] for x in ref], 0); mu_s = np.mean([S[x] for x in ref], 0); sd_u = np.std([U[x] for x in ref], 0)
            u, noise_u = T[(c, "Unstim", d)]
            t = T[(c, stim, d)][0] if (c, stim, d) in T else np.full_like(u, np.nan)
            ud = u - mu_u
            sl = lambda A: A[qi].ravel()
            f = {"donor": d, "is_test": is_test, "celltype": c, "q": np.repeat(qkeep, 13), "marker": np.tile(np.arange(13), nq),
                 "qi": np.repeat(qi, 13), "mi": np.tile(np.arange(13), nq), "target": sl((t - mu_s) - ud),
                 "u_dev": sl(ud), "mu_u": sl(mu_u), "eff": sl(mu_s - mu_u), "sd_u": sl(sd_u), "noise_u": sl(noise_u)}
            f.update(dict(zip(L1, ud.ravel())))
            f.update(dict(zip(PLSO, PO[d]))); f.update(dict(zip(PLSF, PF[d])))
            f.update({f"om_mean_{m}": ud[:, j].mean() for j, m in enumerate(MARKERS)})
            f.update({f"om_spr_{m}": ud[QS.index(90), j] - ud[QS.index(10), j] for j, m in enumerate(MARKERS)})
            f.update({f"op_{g}_q{q}": ud[i, [MARKERS.index(m) for m in ms]].mean() for g, ms in PATHWAYS.items()
                      for i, q in enumerate(QS)})
            mi = np.tile(np.arange(13), nq)
            f.update({x: MXc[mi, j] for j, x in enumerate(MX)})
            Xd = xct_dev(tuple(train), d)[:, np.repeat(qi, 13), mi].T             # rows x 28
            f.update(dict(zip(MK, Xd.T)))
            with np.errstate(all="ignore"):
                f["dw_mean"] = np.nanmean(Xd, 1); f["dw_sd"] = np.nanstd(Xd, 1)
            dv = {**P1.loc[d].to_dict(), **DL.loc[d].to_dict()}
            pd_ = perm.get(d, d) if not is_test else d                    # null: training donors get another donor's blocks
            pv = {**(P1.loc[pd_] if pd_ in P1.index else P1.loc[d]).to_dict(), **DL.loc[pd_].to_dict()}
            f.update(dv); f.update({f"p_{x}": v for x, v in pv.items()})
            up = U[pd_] if (pd_ in U and not is_test) else u                # null own profile: partner's, same centring
            f.update({f"p_{x}": v for x, v in zip(L1, (up - mu_u).ravel())})
            out.append(pd.DataFrame(f))
    if not out: return None
    D = pd.concat(out, ignore_index=True)
    D["celltype_code"] = D.celltype.map({c: i for i, c in enumerate(CTS)}).astype(float)
    return D


_PFN = {}
CKPT = {"path": "auto"}       # --ckpt: a TabPFN v2 regressor checkpoint (file name in $TABPFN_MODEL_CACHE_DIR, or a path)


def fit_tabpfn(X, y, Xte, cat_cols):
    from tabpfn import TabPFNRegressor
    if "m" not in _PFN:
        import torch
        _PFN["m"] = TabPFNRegressor(device="cuda" if torch.cuda.is_available() else "cpu", random_state=0,
                                    ignore_pretraining_limits=True, model_path=CKPT["path"])
    m = _PFN["m"]
    keep = [c for c in X.columns if X[c].std() > 0]
    m.categorical_features_indices = [keep.index(c) for c in cat_cols if c in keep]
    return m.fit(X[keep].to_numpy(float), y).predict(Xte[keep].to_numpy(float))


def fit_lgbm(X, y, groups, Xte, cat_cols):
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    from lc.tab_predict import LGB
    X = X.copy(); Xte = Xte.copy()
    for c in cat_cols: X[c] = X[c].astype(int); Xte[c] = Xte[c].astype(int)
    ds = lgb.Dataset(X, y, categorical_feature=list(cat_cols), free_raw_data=False)
    cv = lgb.cv(LGB, ds, num_boost_round=1000, folds=GroupKFold(min(5, len(np.unique(groups)))).split(X, y, groups),
                callbacks=[lgb.early_stopping(50, verbose=False)])
    return lgb.train(LGB, ds, num_boost_round=len(cv["valid l2-mean"])).predict(Xte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ct", "group"], required=True)
    ap.add_argument("--repeat", type=int, required=True)
    ap.add_argument("--ks", nargs="*", type=int, default=KS)
    ap.add_argument("--stims", nargs="*", default=STIMS)
    ap.add_argument("--layers", nargs="*", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--model", choices=["tabpfn", "lgbm"], default="tabpfn")
    ap.add_argument("--design", default=str(WORK / "design.json"))
    ap.add_argument("--qkeep", nargs="*", type=int, default=list(QKEEP), help="quantile rows built and predicted")
    ap.add_argument("--ckpt", default=None, help="TabPFN checkpoint file (default: the package's default regressor)")
    ap.add_argument("--groupset", default=None, help="group mode: a GROUPSETS name (stim-specific groups)")
    a = ap.parse_args()
    if a.model == "lgbm": a.tag = "lgb" + a.tag
    if a.ckpt:
        import os
        CKPT["path"] = a.ckpt if os.path.isabs(a.ckpt) else os.path.join(os.environ.get("TABPFN_MODEL_CACHE_DIR", "."), a.ckpt)
    LAY = LAYERS_CT if a.mode == "ct" else LAYERS_GROUP
    layers = a.layers or list(LAY)
    dz = [x for x in json.load(open(a.design)) if x["repeat"] == a.repeat][0]
    tests = dz["test"]
    maps = [(s, c, k) for s, c in PAIRS for k in a.ks if s in a.stims]
    pos = {m: i for i, m in enumerate(maps)}
    units = [[c] for c in CTS] if a.mode == "ct" else list(GROUPS.values())
    if a.model == "lgbm": units = [CTS]                               # one pooled table per (k, stim)
    R = {L: np.full((len(maps), len(tests), 5, 13), np.nan) for L in layers}
    done = set(); ck = OUT / f"partial_v2{a.mode}{a.tag}_r{a.repeat}.npz"
    if ck.exists():
        z = np.load(ck, allow_pickle=True)
        if list(z["layers"]) == layers and [tuple(m) for m in z["maps"]] == [tuple(map(str, m)) for m in maps]:
            R = {L: z[f"R_{L}"] for L in layers}; done = {tuple(x) for x in z["done"]}
            print(f"[RESUM] {len(done)} (k, stim) from {ck}", flush=True)
    for k in a.ks:
        for stim in a.stims:
            if (str(k), stim) in done: continue
            t0 = time.time(); n_rows = 0
            if a.mode == "group" and a.groupset:
                units = [[c for c in g if (stim, c) in set(PAIRS)] for g in GROUPSETS[a.groupset][stim]]
                units = [u for u in units if len(u) >= 2]
            for unit in units:
                D = build(a.repeat, k, stim, unit, dz["order"][:k], tests, tuple(a.qkeep))
                if D is None: continue
                tr, te = D[~D.is_test], D[D.is_test]
                if not len(te): continue
                n_rows += len(tr)
                for L in layers:
                    cols = [c if c != "celltype" else "celltype_code" for c in LAY[L]]
                    if a.model == "lgbm":
                        cols = cols + ["celltype_code"]; cat = [c for c in ("marker", "celltype_code") if c in cols]
                        p = fit_lgbm(tr[cols], tr.target.to_numpy(), tr.donor.to_numpy(), te[cols], cat)
                    else:
                        cat = [c for c in ("marker", "celltype_code") if c in cols]
                        p = fit_tabpfn(tr[cols], tr.target.to_numpy(), te[cols], cat)
                    te_ = te.assign(p=p)
                    for (c, d), g in te_.groupby(["celltype", "donor"]):
                        R[L][pos[(stim, c, k)], tests.index(d), g.qi.to_numpy(), g.mi.to_numpy()] = g.p.to_numpy()
            done.add((str(k), stim)); tmp = ck.with_suffix(".tmp.npz")
            np.savez(tmp, layers=np.array(layers), maps=np.array(maps, dtype=str), done=np.array(sorted(done)),
                     **{f"R_{L}": R[L] for L in layers})
            tmp.replace(ck)
            print(f"[MAP  ] r{a.repeat} k{k} {stim} {n_rows} train rows {time.time() - t0:.0f}s", flush=True)
    for L in layers:
        o = OUT / f"v2{a.tag}_{L}" / f"repeat_{a.repeat}"; o.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(maps, columns=["stim", "celltype", "k"]).to_csv(o / "maps.csv", index=False)
        np.save(o / "rhat.npy", R[L])
    ck.unlink(missing_ok=True)
    print(f"[DONE ] -> {OUT}/v2{a.tag}_*")


if __name__ == "__main__":
    main()
