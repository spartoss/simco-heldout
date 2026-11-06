"""Plate / well variance components on one footing (2026-10-07, dmso_plate.ipynb Part 3b). Every number is a standard
deviation in arcsinh units: per feature (percentile x marker), then root-mean-square over the 65 features, then the
median over celltypes. Model for one feature, donor d, plate p (every well holds all donors):
    x[d, p] = mu + a[d] + w[p] + e[d, p]
    a[d]  donor's own value (her biology)       SD_donor
    w[p]  well effect, shared by all donors     SD_well     (learnable: the same for every donor in the well)
    e     donor x well (+ cell sampling)        SD_dxw      (new for every donor in every well)
Estimates (n donors, 3 plates), from the two-way table: residual mean square -> SD_dxw; plate means -> SD_well with
the e contribution removed; donor means (over 3 plates) -> SD_donor with e/3 removed. For the response, x = stim well
- Unstim well of the same plate. mean_response = RMS over features of the average response (how strongly the stim
moves cells), for scale.
    python -m lc.plate_variance -> work/lc/dmso/classify/plate_variance.csv
"""
import numpy as np
import pandas as pd

from lc.paths import PAIRS, WORK

D_ = WORK / "dmso"; P = ["DMSO1", "DMSO2", "DMSO3"]; QS = (10, 25, 50, 75, 90)


def comps(A):
    """A: (n donors, 3 plates, F features) -> per-feature SDs."""
    n, k, _ = A.shape
    gm = A.mean((0, 1)); dm = A.mean(1); pm = A.mean(0)
    res = A - dm[:, None] - pm[None] + gm
    s2_e = (res ** 2).sum((0, 1)) / ((n - 1) * (k - 1))
    s2_w = np.clip(((pm - gm) ** 2).sum(0) / (k - 1) - s2_e / n, 0, None)
    s2_a = np.clip(((dm - gm) ** 2).sum(0) / (n - 1) - s2_e / k, 0, None)
    return s2_a, s2_w, s2_e, gm


rows = []
for ct in sorted({c for _, c in PAIRS}):
    Z = [np.load(D_ / f"cache_{p}/{ct}.npz") for p in P]
    conds = ["Unstim"] + sorted({s for s, c in PAIRS if c == ct})
    Q = {}
    for cond in conds:
        for d in sorted({k.split("|")[1] for k in Z[0].files if k.startswith(cond + "|")}):
            key = f"{cond}|{d}"
            if all(key in z.files and len(z[key]) >= 50 for z in Z):
                Q[(cond, d)] = np.stack([np.percentile(z[key], QS, axis=0).ravel() for z in Z])
    for cond in conds:
        for kind in ["level", "response"]:
            if kind == "response" and cond == "Unstim": continue
            ds = [d for (c, d) in Q if c == cond and ("Unstim", d) in Q]
            if len(ds) < 5: continue
            A = np.stack([Q[(cond, d)] - (Q[("Unstim", d)] if kind == "response" else 0) for d in ds])
            s2_a, s2_w, s2_e, gm = comps(A)
            rms = lambda v: float(np.sqrt(np.mean(v)))
            rows.append({"celltype": ct, "condition": cond, "kind": kind, "donors": len(ds),
                         "SD_donor": rms(s2_a), "SD_well": rms(s2_w), "SD_dxw": rms(s2_e),
                         "mean_response": rms(gm ** 2) if kind == "response" else np.nan})
R = pd.DataFrame(rows); R.to_csv(D_ / "classify/plate_variance.csv", index=False)
pd.set_option("display.width", 200)
print(R.groupby(["kind", "condition"])[["mean_response", "SD_donor", "SD_well", "SD_dxw"]].median().round(3))
