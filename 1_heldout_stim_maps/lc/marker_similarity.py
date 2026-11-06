"""Does ESM2 protein similarity of the 13 markers (lc/marker_embed.py) match how similarly they respond to stims?

Two response similarities between markers m1, m2 (Pearson, Fisher-z averaged), from the pooled truth tables:
- "trajectory": per celltype, correlation over (stim, quantile) of the mean-over-donors stim effect S - U
  (does m1 move like m2 across stims and quantiles in that celltype?);
- "residual": per (celltype, stim, quantile), correlation across the 19 donors of the residual beyond the shift,
  r = (S - mean S) - (U - mean U) (do the donors that deviate on m1 deviate on m2? = what the tabular models predict).
Each 13 x 13 matrix is compared with the ESM2 cosine similarity over the 78 marker pairs: Spearman rho, and a Mantel
test (10,000 permutations of the marker labels). Also per celltype for the trajectory, and the same with 1-3D MDS
coordinates of the ESM2 distances (how much a 1D marker axis keeps).
Writes WORK/marker_embed/: pairs.csv (one row per pair), summary.csv; prints both.
    python -m lc.marker_similarity [--stims GMCSF IFNa IL246 LPS TNFa] [--qs 50 75 90]
"""
import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from lc.paths import DONORS, MARKERS, PAIRS, QS, WORK
from lc.score import truth_tables

OUT = WORK / "marker_embed"
IU = np.triu_indices(13, 1)


def zmean(Ms):
    Z = np.arctanh(np.clip(np.stack(Ms), -0.999, 0.999))
    return np.tanh(np.nanmean(Z, 0))


def mantel(A, B, n=10000, seed=0):
    rng = np.random.default_rng(seed); x = A[IU]; obs = spearmanr(x, B[IU])[0]; cnt = 0
    for _ in range(n):
        p = rng.permutation(13); cnt += spearmanr(x, B[np.ix_(p, p)][IU])[0] >= obs
    return obs, (cnt + 1) / (n + 1)


def mds(Dm, d):
    n = len(Dm); J = np.eye(n) - 1 / n; Bm = -0.5 * J @ (Dm ** 2) @ J
    w, V = np.linalg.eigh(Bm); i = np.argsort(w)[::-1][:d]
    return V[:, i] * np.sqrt(np.maximum(w[i], 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stims", nargs="*", default=["GMCSF", "IFNa", "IL246", "LPS", "TNFa"])
    ap.add_argument("--qs", nargs="*", type=int, default=[50, 75, 90])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    z = np.load(OUT / "esm2_650M.npz"); assert list(z["markers"]) == MARKERS
    C = z["cos"]
    cts = sorted({c for _, c in PAIRS}); T = truth_tables(cts, set(DONORS))
    qi = [QS.index(q) for q in a.qs]
    traj, per_ct, resid = [], {}, []
    for c in cts:
        stims = [s for s, cc in PAIRS if cc == c and s in a.stims]
        E = []                                                       # (stim x q) x marker mean effects
        for s in stims:
            ds = [d for d in DONORS if (c, s, d) in T and (c, "Unstim", d) in T]
            if len(ds) < 5: continue
            S = np.stack([T[(c, s, d)][0][qi] for d in ds]); U = np.stack([T[(c, "Unstim", d)][0][qi] for d in ds])
            E.append((S - U).mean(0))
            R = (S - S.mean(0)) - (U - U.mean(0))                    # donors x q x marker
            for j in range(len(qi)):
                X = R[:, j]; ok = X.std(0) > 1e-6
                M = np.full((13, 13), np.nan); M[np.ix_(ok, ok)] = np.corrcoef(X[:, ok].T); resid.append(M)
        if len(E) * len(qi) >= 4:
            X = np.concatenate(E, 0); ok = X.std(0) > 1e-6
            M = np.full((13, 13), np.nan); M[np.ix_(ok, ok)] = np.corrcoef(X[:, ok].T); traj.append(M); per_ct[c] = M
    Rt, Rr = zmean(traj), zmean(resid)
    Dm = np.sqrt(np.maximum(2 - 2 * C, 0))
    rows = []
    for name, M in [("trajectory", Rt), ("residual", Rr)]:
        rho, p = mantel(C, M)
        row = {"response": name, "spearman_vs_esm2": rho, "mantel_p": p}
        for d in (1, 2, 3):
            Y = mds(Dm, d); Dd = np.sqrt(((Y[:, None] - Y[None]) ** 2).sum(-1))
            row[f"spearman_vs_mds{d}"] = spearmanr(-Dd[IU], M[IU])[0]
        rows.append(row)
    for d in (1, 2, 3):
        Y = mds(Dm, d); Dd = np.sqrt(((Y[:, None] - Y[None]) ** 2).sum(-1))
        rows.append({"response": f"esm2 full vs mds{d} distances", "spearman_vs_esm2": spearmanr(Dd[IU], Dm[IU])[0]})
    rows.append({"response": "trajectory vs residual", "spearman_vs_esm2": spearmanr(Rt[IU], Rr[IU])[0]})
    S = pd.DataFrame(rows)
    pc = pd.Series({c: spearmanr(C[IU], M[IU], nan_policy="omit")[0] for c, M in per_ct.items()}).sort_values()
    P = pd.DataFrame([{"m1": MARKERS[i], "m2": MARKERS[j], "esm2_cos": C[i, j], "trajectory_r": Rt[i, j],
                       "residual_r": Rr[i, j]} for i, j in zip(*IU)]).sort_values("esm2_cos", ascending=False)
    S.to_csv(OUT / f"summary{a.tag}.csv", index=False); P.to_csv(OUT / f"pairs{a.tag}.csv", index=False)
    pc.to_csv(OUT / f"per_celltype{a.tag}.csv")
    pd.set_option("display.width", 200)
    print(f"stims {a.stims}, quantiles {a.qs}; {len(traj)} celltypes, {len(resid)} residual blocks")
    print(S.round(3).to_string(index=False))
    print("--- trajectory vs ESM2 per celltype (Spearman):", pc.round(2).to_dict())
    print("--- pairs by ESM2 similarity"); print(P.round(3).to_string(index=False))
    print("--- ESM2 cosine matrix"); print(pd.DataFrame(C, index=MARKERS, columns=MARKERS).round(2).to_string())
    print("--- 1D MDS coordinate:", dict(zip(MARKERS, mds(Dm, 1)[:, 0].round(3))))


if __name__ == "__main__":
    main()
