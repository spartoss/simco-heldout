"""Long tables for tabular residual models (GBDT, TabPFN): predict a new donor's stim quantiles from its Unstim profile.

Unit = one map (repeat, k, stim); one row per (donor, celltype, quantile, marker) of the stim's celltypes.
Target = the residual beyond the feature-level shift, as lc/residual_predict.py:
    r = (S - mu_s) - (U - mu_u),   prediction = shift + lambda * r_hat,   shift = mu_s + (U - mu_u)
mu_u / mu_s are the training-donor means; for a TRAINING donor's own rows they are leave-that-donor-out means, so the
training rows look like a test donor (whose own value is never in the mean). The test rows use all k training donors,
so shift equals lc.ctx_score's shift exactly.
Feature layers (each adds to the previous):
- L0 own feature: u_dev = U - mu_u, mu_u, eff = mu_s - mu_u, sd_u (training donors), noise_u (donor's split-half
  noise), log cell count of the celltype; categorical q, marker, celltype.
- L1 + the donor's 65 Unstim values of the same celltype, centred like u_dev (columns c_q<q>_<marker>).
- L2 + whole donor: 28 log frequencies (WORK/fm/counts.csv), PCs of all celltypes' Unstim medians (fitted on the
  training donors), the 3 LDA donor axes refitted per (repeat, k) (work/heldout/donor_covariate/lc_ctx.json).
- L2perm = L2 with the L2 block shuffled across training donors (null for the whole-donor block).
- Lperm = L2 with the L1 and L2 blocks shuffled across training donors (same permutation; null for all donor inputs).
- Ablation layers: F / P / A = L0 + frequencies / PCs / LDA axes only; L1F / L1P / L1A = L1 + that one block.
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd

from lc.paths import DONORS, MARKERS, PAIRS, QS, WORK
from lc.score import truth_tables

CTS = sorted({c for _, c in PAIRS})
LDA_CTX = WORK.parent / "heldout" / "donor_covariate" / "lc_ctx.json"
LAYERS = ("L0", "L1", "L2", "L2perm", "Lperm")
N_PC = 5
CAT = ["q", "marker", "celltype"]
L0 = ["u_dev", "mu_u", "eff", "sd_u", "noise_u", "log_n"]
L1 = [f"c_q{q}_{m}" for q in QS for m in MARKERS]


@lru_cache(1)
def tables():
    return truth_tables(CTS, set(DONORS))


@lru_cache(1)
def donor_static():
    cnt = pd.read_csv(WORK / "fm" / "counts.csv")
    freq = np.log(cnt.pivot(index="donor", columns="celltype", values="freq").reindex(index=DONORS, columns=CTS))
    n = cnt.pivot(index="donor", columns="celltype", values="n").reindex(index=DONORS, columns=CTS)
    T = tables()
    med = np.full((len(DONORS), len(CTS) * 13), np.nan)          # Unstim q50 of every celltype
    for i, d in enumerate(DONORS):
        for j, c in enumerate(CTS):
            if (c, "Unstim", d) in T: med[i, j * 13:(j + 1) * 13] = T[(c, "Unstim", d)][0][QS.index(50)]
    return freq, np.log(n), pd.DataFrame(med, index=DONORS)


def donor_block(train, k, repeat):
    """L2 block (DataFrame indexed by donor, all donors) fitted on the training donors only."""
    freq, _, med = donor_static()
    M = med.to_numpy(); mu = np.nanmean(M[[DONORS.index(d) for d in train]], 0)
    M = np.where(np.isnan(M), mu, M)
    Z = M[[DONORS.index(d) for d in train]]; sd = Z.std(0) + 1e-6
    Zs = (Z - Z.mean(0)) / sd
    _, _, Vt = np.linalg.svd(Zs, full_matrices=False)
    npc = min(N_PC, len(train) - 1)
    pcs = ((M - Z.mean(0)) / sd) @ Vt[:npc].T
    pcs = np.pad(pcs, ((0, 0), (0, N_PC - npc)))
    lda = json.load(open(LDA_CTX))[f"{repeat}_{k}"]
    B = pd.DataFrame(freq.fillna(freq.loc[train].mean()).to_numpy(), index=DONORS, columns=[f"freq_{c}" for c in CTS])
    for i in range(N_PC): B[f"pc{i}"] = pcs[:, i]
    for i in range(3): B[f"lda{i}"] = [lda[d][i] for d in DONORS]
    return B


def build(repeat, k, stim, train, tests, seed=0):
    """Rows for one map. Returns (DataFrame, layer -> feature columns). Columns: donor, is_test, celltype, q, marker,
    qi, mi, target (r; NaN for test donors without stim truth), shift, u, true (NaN if missing), features."""
    T = tables(); _, logn, _ = donor_static()
    B = donor_block(train, k, repeat)
    rng = np.random.default_rng(seed + 1000 * repeat + k)
    perm = dict(zip(train, rng.permutation(train)))
    Bp = B.copy(); Bp.loc[train] = B.loc[[perm[d] for d in train]].to_numpy()
    out = []
    for s, c in PAIRS:
        if s != stim: continue
        trd = [d for d in train if (c, s, d) in T and (c, "Unstim", d) in T]
        if len(trd) < 2: continue
        U = {d: T[(c, "Unstim", d)][0] for d in trd}; S = {d: T[(c, s, d)][0] for d in trd}
        donors = [(d, False) for d in trd if len(trd) >= 3] + [(d, True) for d in tests if (c, "Unstim", d) in T]
        for d, is_test in donors:
            ref = trd if is_test else [x for x in trd if x != d]
            mu_u = np.mean([U[x] for x in ref], 0); mu_s = np.mean([S[x] for x in ref], 0)
            sd_u = np.std([U[x] for x in ref], 0) if len(ref) > 1 else np.zeros_like(mu_u)
            u, noise_u = T[(c, "Unstim", d)]
            t = T[(c, s, d)][0] if (c, s, d) in T else np.full_like(u, np.nan)
            ud = u - mu_u
            f = {"donor": d, "is_test": is_test, "celltype": c,
                 "q": np.repeat(QS, 13), "marker": np.tile(np.arange(13), len(QS)),
                 "qi": np.repeat(np.arange(len(QS)), 13), "mi": np.tile(np.arange(13), len(QS)),
                 "target": ((t - mu_s) - ud).ravel(), "shift": (mu_s + ud).ravel(), "u": u.ravel(), "true": t.ravel(),
                 "u_dev": ud.ravel(), "mu_u": mu_u.ravel(), "eff": (mu_s - mu_u).ravel(), "sd_u": sd_u.ravel(),
                 "noise_u": noise_u.ravel(), "log_n": logn.loc[d, c]}
            f.update(dict(zip(L1, ud.ravel())))
            f.update(B.loc[d].to_dict())
            f.update({f"p_{x}": v for x, v in Bp.loc[d].items()})
            out.append(pd.DataFrame(f))
    D = pd.concat(out, ignore_index=True)
    # L1 block of the permuted donor (same celltype); stays the donor's own if the partner lacks that celltype
    tr = ~D.is_test
    src = D.loc[tr, ["celltype", "donor", "qi", "mi"]].copy(); src["donor"] = src.donor.map(perm)
    lut = D[tr].drop_duplicates(["celltype", "donor"]).set_index(["celltype", "donor"])[L1]
    pp = lut.reindex(pd.MultiIndex.from_frame(src[["celltype", "donor"]])).to_numpy()
    own = D.loc[tr, L1].to_numpy(); pp = np.where(np.isnan(pp), own, pp)
    P1 = D[L1].to_numpy().copy(); P1[tr.to_numpy()] = pp
    D = pd.concat([D, pd.DataFrame(P1, columns=[f"p_{x}" for x in L1], index=D.index)], axis=1)
    D["celltype"] = pd.Categorical(D.celltype, categories=CTS)
    L2 = list(B.columns)
    cols = {"L0": L0 + CAT, "L1": L0 + L1 + CAT, "L2": L0 + L1 + L2 + CAT, "L2perm": L0 + L1 + [f"p_{x}" for x in L2] + CAT,
            "Lperm": L0 + [f"p_{x}" for x in L1 + L2] + CAT}
    # ablation: one donor block at a time, alone (L0 + block) and on top of L1
    blocks = {"F": [x for x in L2 if x.startswith("freq_")], "P": [x for x in L2 if x.startswith("pc")],
              "A": [x for x in L2 if x.startswith("lda")]}
    for b, bc in blocks.items():
        cols[b] = L0 + bc + CAT
        cols[f"L1{b}"] = L0 + L1 + bc + CAT
    return D, cols
