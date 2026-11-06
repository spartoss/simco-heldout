"""Score every held-out-donor stim predictor tried so far on the SAME cases (input of notebooks/model_zoo.ipynb).

One case = (repeat, stim, celltype, k, test donor, quantile, marker); truth = the test donor's stim quantile (lc.score
truth tables), effect = |truth - Unstim|, cases with effect > FLOOR only (as lc.score / lc.ctx_score).
Two frames, each on the cases where EVERY method of the frame has a prediction:
- lc  : k = 3 5 8 15, methods that exist for every k (learning-curve design);
- k15 : k = 15 only, + the arms run at k = 15 only (dnorm / clip, CondOT).
Repeats 1-4 by default (repeat 0 tuned the context weight decay); --repeats 0 1 2 3 4 for all.
Readouts per (method, k), the same formulas as lc.ctx_score: skill (median per repeat, mean over repeats), dev_R2
(error vs the test donor's deviation from the training stim mean), between_R2 (predictions and truth centred across
the repeat's test donors), and paired vs REFS: % of cases closer, Wilcoxon over (repeat, map, donor) median errors.
Ensembles are formed here: ens = 0.5 CellOT_paired + 0.5 shift; ens_orig = 0.5 CellOT + 0.5 shift.
Outputs WORK/model_zoo/: <frame>_curve.csv, <frame>_paired.csv, <frame>_by_stim.csv (k = 15), cases_<frame>.pkl.
    python -m lc.model_zoo            (env_cpu.sh, ~10 min)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from lc.cond_train_score import load_npz, load_train
from lc.paths import DONORS, MARKERS, PAIRS, QS, WORK
from lc.score import FLOOR, truth_tables

OUT = WORK / "model_zoo"
MFM = Path("/scratch/users/bmerwan/mfm_heldout/work/runs")
# name -> (kind, location, family). kind: dir (lc.train pred_q layout), dnorm (dir, donor-normalised: add the test
# donor's Unstim median offset back), npz:<key> (closed-form npz). {r} = repeat.
METHODS = {
    "CellOT":         ("dir", "train/repeat_{r}", "CellOT"),
    "CellOT_paired":  ("dir", "cond_train/repeat_{r}/none", "CellOT"),
    "gauss_ot":       ("npz:gauss_ot", "baselines/repeat_{r}.npz", "closed form"),
    "quantile_map":   ("npz:quantile_map", "baselines/repeat_{r}.npz", "closed form"),
    "median_shift":   ("npz:median_shift", "baselines/repeat_{r}.npz", "closed form"),
    "gauss_ot_icpt":  ("npz:gauss_ot_icpt", "cond_baselines/med_z/repeat_{r}.npz", "closed form + donor"),
    "gauss_ot_knn":   ("npz:gauss_ot_knn", "cond_baselines/med_z/repeat_{r}.npz", "closed form + donor"),
    "gauss_ot_cond":  ("npz:gauss_ot_cond", "cond_baselines/med_z/repeat_{r}.npz", "closed form + donor"),
    "ctx_pc":         ("dir", "cond_train/repeat_{r}/pc", "CellOT + donor context"),
    "ctx_pc_wd1":     ("dir", "cond_train/repeat_{r}/pc_wd1", "CellOT + donor context"),
    "ctx_pcblood":    ("dir", "cond_train/repeat_{r}/pcblood_wd1", "CellOT + donor context"),
    "ctx_ft_immunefm": ("dir", "cond_train/repeat_{r}/ft_pre_wd1", "CellOT + donor context"),
    "ctx_ft_scratch": ("dir", "cond_train/repeat_{r}/ft_rand_wd1", "CellOT + donor context"),
    "ctx_cluster":    ("dir", "cond_train/repeat_{r}/cluster_K4_stable", "CellOT + cell context"),
    "fm":             ("dir", str(MFM / "fm_sel/repeat_{r}"), "flow matching"),
    "mfm":            ("dir", str(MFM / "mfm_sel/repeat_{r}"), "flow matching"),
    "mfm_blood":      ("dir", str(MFM / "mfm_blood_sel/repeat_{r}"), "flow matching"),
    "mfm_lastiter":   ("dir", str(MFM / "mfm/repeat_{r}"), "flow matching"),
    "dnorm":          ("dnorm", "cond_train/repeat_{r}/dnorm", "CellOT preprocessing (k15)"),
    "dnorm_clip":     ("dnorm", "cond_train/repeat_{r}/dnorm_clip", "CellOT preprocessing (k15)"),
    "clip":           ("dir", "cond_train/repeat_{r}/clip", "CellOT preprocessing (k15)"),
    "condot_ct":      ("dir", "condot/celltype_merged/repeat_{r}", "CondOT (k15)"),
    "condot_global":  ("dir", "condot/global/repeat_{r}", "CondOT (k15)"),
}
K15_ONLY = ["dnorm", "dnorm_clip", "clip", "condot_ct", "condot_global"]
BUILTIN = {"identity": "no change", "shift": "feature level", "ens": "ensemble", "ens_orig": "ensemble"}
REFS = ["CellOT_paired", "shift", "ens"]
KS = [3, 5, 8, 15]


def family(m):
    return METHODS[m][2] if m in METHODS else BUILTIN[m]


def load(name, r, maps):
    kind, loc, _ = METHODS[name]
    p = Path(loc.format(r=r)); p = p if p.is_absolute() else WORK / p
    if kind.startswith("npz:"):
        return load_npz(p, [kind[4:]], maps)[kind[4:]] if p.exists() else {}
    return load_train(p, maps) if (p / "pred_q.npy").exists() else {}


def build(repeats):
    off = np.load(WORK / "cache_dnorm/offsets.npz")
    T = truth_tables(sorted({c for _, c in PAIRS}), set(DONORS))
    rows = []
    for dz in json.load(open(WORK / "design.json")):
        r, tests = dz["repeat"], dz["test"]
        if r not in repeats: continue
        maps = [(s, c, k) for s, c in PAIRS for k in KS]
        P = {n: load(n, r, maps) for n in METHODS}
        for (s, c, k) in maps:
            trd = [d for d in dz["order"][:k] if (c, s, d) in T and (c, "Unstim", d) in T]
            if len(trd) < 2: continue
            mu_u = np.mean([T[(c, "Unstim", d)][0] for d in trd], 0); mu_s = np.mean([T[(c, s, d)][0] for d in trd], 0)
            for j, d in enumerate(tests):
                if (c, s, d) not in T or (c, "Unstim", d) not in T: continue
                u, t = T[(c, "Unstim", d)][0], T[(c, s, d)][0]
                row = {"repeat": r, "stim": s, "celltype": c, "k": k, "donor": d,
                       "q": np.repeat(QS, len(MARKERS)), "marker": np.tile(MARKERS, len(QS)),
                       "true": t.ravel(), "mu_s": mu_s.ravel(), "eff": np.abs(t - u).ravel(),
                       "identity": u.ravel(), "shift": (mu_s + u - mu_u).ravel()}
                for n, D in P.items():
                    p = D.get((s, c, k))
                    p = np.full(u.shape, np.nan) if p is None else np.asarray(p[j], float)
                    if METHODS[n][0] == "dnorm" and not np.isnan(p).all(): p = p + off[f"{c}|{d}"][None, :]
                    row[n] = p.ravel()
                rows.append(pd.DataFrame(row))
        print(f"[LOAD ] repeat {r}", flush=True)
    D = pd.concat(rows, ignore_index=True)
    D["ens"] = 0.5 * D.CellOT_paired + 0.5 * D["shift"]
    D["ens_orig"] = 0.5 * D.CellOT + 0.5 * D["shift"]
    D = D[D.eff > FLOOR]
    for col in D.select_dtypes("float64"): D[col] = D[col].astype("float32")
    return D


def summarise(D, methods, tag):
    key = ["repeat", "stim", "celltype", "k", "donor", "q", "marker"]
    D = D.dropna(subset=methods).copy()
    L = D.melt(id_vars=key + ["true", "mu_s", "eff"], value_vars=methods, var_name="method", value_name="pred")
    L["err"] = (L.pred - L.true).abs(); L["skill"] = 1 - L.err / L.eff
    g = ["repeat", "stim", "celltype", "k", "q", "marker", "method"]
    L["pc"] = L.pred - L.groupby(g).pred.transform("mean"); L["tc"] = L.true - L.groupby(g).true.transform("mean")
    L["ntest"] = L.groupby(g).pred.transform("size")
    res = []
    for (m, k), x in L.groupby(["method", "k"]):
        xb = x[x.ntest >= 3]
        res.append({"method": m, "family": family(m), "k": k, "skill": x.groupby("repeat").skill.median().mean(),
                    "dev_R2": 1 - ((x.pred - x.true) ** 2).sum() / ((x.true - x.mu_s) ** 2).sum(),
                    "between_R2": 1 - ((xb.pc - xb.tc) ** 2).sum() / (xb.tc ** 2).sum(), "n_cases": len(x)})
    C = pd.DataFrame(res)
    E = D.set_index(key)[methods].sub(D.set_index(key)["true"], axis=0).abs()
    pr = []
    for m in methods:
        for b in REFS:
            if b == m or b not in methods: continue
            for k, x in E.groupby(level="k"):
                gm = x.groupby(level=["repeat", "stim", "celltype", "donor"]).median()
                pr.append({"a": m, "b": b, "k": k, "pct_a_closer": 100 * (x[m] < x[b]).mean(),
                           "p_wilcoxon": wilcoxon(gm[m], gm[b]).pvalue})
    Pd = pd.DataFrame(pr)
    kmax = L.k.max()
    B = L[L.k == kmax].groupby(["method", "stim", "repeat"]).skill.median().groupby(["method", "stim"]).mean().unstack()
    OUT.mkdir(parents=True, exist_ok=True)
    C.to_csv(OUT / f"{tag}_curve.csv", index=False); Pd.to_csv(OUT / f"{tag}_paired.csv", index=False)
    B.to_csv(OUT / f"{tag}_by_stim.csv")
    print(f"--- {tag}: {len(D)} cases"); print(C.pivot(index="method", columns="k", values="skill").round(3).to_string())
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", nargs="*", type=int, default=[1, 2, 3, 4])
    a = ap.parse_args()
    D = build(a.repeats)
    lc_methods = ["identity", "shift", "ens", "ens_orig"] + [m for m in METHODS if m not in K15_ONLY]
    summarise(D, lc_methods, "lc")
    summarise(D[D.k == 15], lc_methods + K15_ONLY, "k15")
    D.to_pickle(OUT / "cases.pkl")
    print(f"[DONE ] -> {OUT}")


if __name__ == "__main__":
    main()
