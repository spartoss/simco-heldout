"""Put the models and the replicate wells on the same targets (k = 15 learning-curve cases).

The plateau (0.60) scores model predictions against the *pooled* stim measurement (3 wells pooled), while a
replicate-well ceiling scores one well against *another single well*: two different targets. This script reads the
k = 15 model predictions from atlas_heldout's work/lc/where_skill/cases.pkl (read-only; columns t = pooled truth,
u = pooled Unstim identity, CellOT / gauss_ot / quantile_map = predicted quantiles, noise = |h1 - h2|) and the
per-well quantiles of wc.extract, and scores, on the same live cases (pooled effect > FLOOR):

  target = pooled (lc):  model; split-half replicate (lc ceiling); ideal model under independent noise:
                         sampling only = 1 - |h1-h2|/2/effect, sampling + well = 1 - |wA-wB|/sqrt(6)/effect
                         (the pooled target averages three wells' errors: var = var_well_diff / 6)
  target = one well B:   model; replicate well A; mean of the other wells; ideal model = 1 - |wA-wB|/sqrt(2)/effect

The "ideal model" lines assume each well = donor truth + independent noise of equal variance (Gaussian scaling), so
they are estimates, not measurements. Under PI DMSO1 is left out of the well targets.
    python -m wc.compare
"""
import json

import numpy as np
import pandas as pd

from wc.paths import FLOOR, LC_DESIGN, MARKERS, MIN_CELLS, QS, QUANT, SUMMARY, WELLS

CASES = "/scratch/users/bmerwan/atlas_heldout/work/lc/where_skill/cases.pkl"
MODELS = ["CellOT", "gauss_ot", "quantile_map"]


def well_table(celltypes, drop_pi_dmso1=True):
    rows = []
    for ct in celltypes:
        z = np.load(QUANT / f"{ct}.npz"); keys = set(z.files)
        for k in z.files:
            p = k.split("|")
            if len(p) != 3 or not p[2].startswith("w:") or p[0] == "Unstim": continue
            if int(z[k + "|n"]) < MIN_CELLS: continue
            stim, d, w = p[0], p[1], p[2][2:]
            if drop_pi_dmso1 and stim == "PI" and w == "DMSO1": continue
            v = z[k]
            for qi, q in enumerate(QS):
                for mi, m in enumerate(MARKERS):
                    rows.append((ct, stim, d, w, q, m, v[qi, mi]))
    return pd.DataFrame(rows, columns=["celltype", "stim", "donor", "well", "q", "marker", "value"])


def main():
    C = pd.read_pickle(CASES)
    C = C[C.k == 15].copy()
    C["eff"] = (C.t - C.u).abs()
    C = C[C.eff > FLOOR]
    C = C[[ (QUANT / f"{c}.npz").exists() for c in C.celltype]]
    W = well_table(sorted(C.celltype.unique()))
    key = ["celltype", "stim", "donor", "q", "marker"]
    Wp = W.pivot_table(index=key, columns="well", values="value")
    out = []
    # ---- target = pooled
    J = C.merge(Wp.reset_index(), on=key, how="left")
    for m in MODELS:
        out.append(("pooled", f"model: {m}", J.repeat, 1 - (J[m] - J.t).abs() / J.eff))
    out.append(("pooled", "replicate: split half (lc ceiling)", J.repeat, 1 - J.noise / J.eff))
    out.append(("pooled", "ideal model, sampling noise only", J.repeat, 1 - J.noise / 2 / J.eff))
    pairs = [(a, b) for a in WELLS for b in WELLS if a < b]
    # the pooled target holds all three wells (DMSO1 included under PI), so its noise uses all of them
    Jall = C.merge(well_table(sorted(C.celltype.unique()), drop_pi_dmso1=False)
                   .pivot_table(index=key, columns="well", values="value").reset_index(), on=key, how="left")
    dd = pd.concat([(Jall[a] - Jall[b]).abs().rename(f"{a}{b}") for a, b in pairs], axis=1)
    d_rms = np.sqrt((dd ** 2).mean(1))      # rms over the available well pairs of this case
    out.append(("pooled", "ideal model, sampling + well noise", J.repeat, 1 - d_rms / np.sqrt(6) / J.eff))
    # ---- target = one well B
    for B in WELLS:
        ok = J[B].notna()
        x = J[ok]; effB = (x[B] - x.u).abs()
        for m in MODELS:
            out.append(("one well", f"model: {m}", x.repeat, 1 - (x[m] - x[B]).abs() / effB))
        others = [A for A in WELLS if A != B]
        for A in others:
            y = x[x[A].notna()]
            out.append(("one well", "replicate: another well", y.repeat, 1 - (y[A] - y[B]).abs() / (y[B] - y.u).abs()))
            out.append(("one well", "ideal model, sampling + well noise", y.repeat,
                        1 - (y[A] - y[B]).abs() / np.sqrt(2) / (y[B] - y.u).abs()))
        mo = x[others].mean(1, skipna=True); y = x[x[others].notna().all(1)]
        out.append(("one well", "mean of the other two wells", y.repeat, 1 - (mo[y.index] - y[B]).abs() / (y[B] - y.u).abs()))
    L = pd.concat([pd.DataFrame({"target": tgt, "predictor": lab, "repeat": rep.values,
                                 "stim": J.loc[sk.index, "stim"].values, "celltype": J.loc[sk.index, "celltype"].values,
                                 "q": J.loc[sk.index, "q"].values, "skill": sk.values})
                   for tgt, lab, rep, sk in out]).dropna(subset=["skill"])
    for c in ["target", "predictor", "stim", "celltype"]: L[c] = L[c].astype("category")
    L.to_pickle(SUMMARY / "same_target_cases.pkl")
    g = L.groupby(["target", "predictor", "repeat"], observed=True, sort=False).skill.median()
    R = g.groupby(["target", "predictor"], observed=True, sort=False).agg(skill="mean", skill_sd="std").reset_index()
    R["n_cases_per_repeat"] = L.groupby(["target", "predictor", "repeat"], observed=True, sort=False).size() \
        .groupby(["target", "predictor"], observed=True, sort=False).mean().values.astype(int)
    R.to_csv(SUMMARY / "same_target_comparison.csv", index=False)
    print(R.round(4).to_string(index=False))
    for by in ["stim", "celltype", "q"]:
        T = L.groupby(["target", "predictor", by], observed=True).skill.median().unstack(["target", "predictor"])
        T.to_csv(SUMMARY / f"same_target_by_{by}.csv")
    print(L.groupby(["target", "predictor", "stim"], observed=True).skill.median().unstack(["target", "predictor"]).round(3).T)


if __name__ == "__main__":
    main()
