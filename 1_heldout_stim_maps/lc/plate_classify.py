"""Can the DMSO plate (DMSO1 / DMSO2 / DMSO3) be recognised from the cells? (2026-10-07, notebooks/dmso_plate.ipynb §3)

Every well holds all donors (barcoded), one (stim, plate) per well, so a learnable plate signature = a WELL effect
shared by all donors. Tested per (condition, celltype), PI excluded (plate 1 is a known PI outlier), Unstim included,
on the 13 functional markers of the single-plate caches (work/lc/dmso/cache_DMSO{1,2,3}), always on HELD-OUT donors
(GroupKFold over donors, 5 folds), chance = 1/3:
  cell       one cell -> plate; HistGradientBoosting; <= 300 cells per (donor, plate)
  bag        a batch of 100 cells -> plate; features = 5 quantiles x 13 markers; logistic regression (standardised);
             up to 10 bags per (donor, plate)
  bag_dc     the same, each donor's bags centred on her own mean bag in that block (removes the donor's own level,
             uses no plate label): "given a donor's batches, which one is plate k?"
Output work/lc/dmso/classify/accuracy.csv (one row per condition x celltype x readout), confusion.csv (bag_dc).
    python -m lc.plate_classify [--celltypes ...] [--jobs 3]
"""
import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lc.paths import PAIRS, WORK

D_ = WORK / "dmso"
OUT = D_ / "classify"
PLATES = ["DMSO1", "DMSO2", "DMSO3"]
QS = (10, 25, 50, 75, 90)
MAX_CELLS, BAG, NBAG = 300, 100, 10


def cv(X, y, g, model):
    pred = np.empty_like(y)
    for tr, te in GroupKFold(n_splits=min(5, len(np.unique(g)))).split(X, y, g):
        if len(np.unique(y[tr])) < 3: return None
        pred[te] = model().fit(X[tr], y[tr]).predict(X[te])
    return pred


def one(ct, cond):
    rng = np.random.default_rng(0)
    Z = [np.load(D_ / f"cache_{p}/{ct}.npz") for p in PLATES]
    donors = sorted({k.split("|")[1] for k in Z[0].files if k.startswith(cond + "|")})
    cx, cy, cg, bx, by, bg = [], [], [], [], [], []
    for d in donors:
        key = f"{cond}|{d}"
        if not all(key in z.files and len(z[key]) >= BAG for z in Z): continue   # need all three plates for this donor
        for i, z in enumerate(Z):
            x = z[key]
            s = x[rng.permutation(len(x))[:MAX_CELLS]]; cx.append(s); cy += [i] * len(s); cg += [d] * len(s)
            for _ in range(min(NBAG, max(1, len(x) // BAG))):
                b = x[rng.choice(len(x), BAG, replace=False)]
                bx.append(np.percentile(b, QS, axis=0).ravel()); by.append(i); bg.append(d)
    if len(set(cg)) < 5: return [], None
    cx, cy, cg = np.concatenate(cx), np.array(cy), np.array(cg)
    bx, by, bg = np.array(bx), np.array(by), np.array(bg)
    bdc = bx - pd.DataFrame(bx).groupby(bg).transform("mean").to_numpy()
    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
    rows, conf = [], None
    for name, X, y, g, m in [("cell", cx, cy, cg, lambda: HistGradientBoostingClassifier(max_iter=150)),
                             ("bag", bx, by, bg, lr), ("bag_dc", bdc, by, bg, lr)]:
        p = cv(X, y, g, m)
        if p is None: continue
        acc_d = pd.Series(p == y).groupby(g).mean()
        rows.append({"condition": cond, "celltype": ct, "readout": name, "n": len(y), "donors": len(set(g)),
                     "accuracy": float((p == y).mean()), "acc_donor_min": acc_d.min(), "acc_donor_max": acc_d.max(),
                     **{f"recall_{PLATES[i]}": float((p[y == i] == i).mean()) for i in range(3)}})
        if name == "bag_dc":
            conf = pd.crosstab(pd.Series(y, name="true"), pd.Series(p, name="pred")).assign(condition=cond, celltype=ct)
    return rows, conf


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--celltypes", nargs="*", default=None)
    ap.add_argument("--jobs", type=int, default=3); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cts = sorted({c for _, c in PAIRS if not a.celltypes or c in a.celltypes})
    jobs = [(c, s) for c in cts for s in ["Unstim"] + sorted({s for s, cc in PAIRS if cc == c and s != "PI"})]
    res = Parallel(n_jobs=a.jobs)(delayed(one)(c, s) for c, s in jobs)
    A = pd.DataFrame([r for rows, _ in res for r in rows])
    A.to_csv(OUT / "accuracy.csv", index=False)
    C = [c for _, c in res if c is not None]
    if C: pd.concat(C).to_csv(OUT / "confusion.csv")
    print(A.groupby("readout").accuracy.describe().round(3))
    print(A.pivot_table(index="condition", columns="readout", values="accuracy", aggfunc="median").round(3))


if __name__ == "__main__":
    main()
