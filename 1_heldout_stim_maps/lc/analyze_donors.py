"""Analyses of the donor experiments (lc/design_donors.py): HV vs PV, good / hard donors, cells vs donors.

For every (map, test donor): skill = median over (quantile, marker) of 1 - |pred - true| / |true effect|, over the cases
whose true effect exceeds FLOOR, i.e. one number per training set x test donor x (stim, celltype). Methods: CellOT (last
iterate, lc/train.py outputs under donors/) and the closed-form maps (donors_base/). Whatever has finished is used.
Outputs in WORK/donors_summary/<exp>/ (CSVs, PNGs).
    python -m lc.analyze_donors [--exp hvpv small pairwise]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

from lc.paths import CACHE, DONORS, QS, WORK  # noqa: E402

FLOOR = 0.0273
MIN_CELLS = 20
METHODS = ["CellOT", "gauss_ot", "quantile_map", "median_shift"]
OUT = WORK / "donors_summary"
BASE, CELLOT = WORK / "donors_base", WORK / "donors"      # overridable for smoke tests


def load_truth(celltypes):
    """(ct, cond, donor) -> quantiles (5, 13) of all cached cells; also per-donor medians for descriptors."""
    T = {}
    for c in celltypes:
        z = np.load(CACHE / f"{c}.npz")
        for k in z.files:
            x = z[k]
            if len(x) >= MIN_CELLS:
                cond, d = k.split("|"); T[(c, cond, d)] = (np.percentile(x, QS, axis=0), len(x))
    return T


def runs(exp):
    """Yield (method, maps DataFrame [stim, celltype, k, cap, set, n_train], preds (M, n_test, 5, 13), tests)."""
    D = WORK / "designs"
    specs = {"hvpv": sorted((D / "hvpv").glob("r*.json")), "small": sorted((D / "small").glob("r*.json")),
             "pairwise": [D / "pairwise.json"]}[exp]
    for sp in specs:
        tests = json.load(open(sp))["tests"]; tag = sp.stem if exp != "pairwise" else "r0"
        b = BASE / f"{exp}/repeat_{tag[1:]}.npz"
        if b.exists():
            z = np.load(b, allow_pickle=True)
            m = pd.DataFrame([tuple(x) for x in z["maps"]], columns=["stim", "celltype", "k", "cap", "set", "n_train"])
            for meth in ["gauss_ot", "quantile_map", "median_shift"]:
                yield meth, tag, m, z[meth], tests
        cdirs = [CELLOT / f"{exp}/{tag}"] if exp != "pairwise" else sorted((CELLOT / "pairwise").glob("g*"))
        for cd in cdirs:
            if (cd / "pred_q.npy").exists():
                m = pd.read_csv(cd / "maps.csv").rename(columns={"n_train_src": "n_train"})
                P = np.load(cd / "pred_q.npy")
                yield "CellOT", tag, m, P[P.shape[0] - 2], tests          # last iterate


def skills(exp):
    rows = []
    cts = set()
    for meth, tag, m, P, tests in runs(exp):
        cts |= set(m.celltype)
    T = load_truth(sorted(cts))
    for meth, tag, m, P, tests in runs(exp):
        for i, r in enumerate(m.itertuples()):
            for j, d in enumerate(tests):
                if exp == "pairwise" and d == r.set: continue
                if (r.celltype, r.stim, d) not in T or (r.celltype, "Unstim", d) not in T or np.isnan(P[i, j]).all(): continue
                tr = T[(r.celltype, r.stim, d)][0]; eff = np.abs(tr - T[(r.celltype, "Unstim", d)][0])
                ok = eff > FLOOR
                if not ok.any(): continue
                sk = 1 - np.abs(P[i, j] - tr)[ok] / eff[ok]
                rows.append({"method": meth, "repeat": tag, "stim": r.stim, "celltype": r.celltype, "set": r.set,
                             "k": r.k, "cap": r.cap, "n_train": r.n_train, "test": d, "skill": np.median(sk), "n_cases": int(ok.sum())})
    return pd.DataFrame(rows), T


def hvpv():
    S, _ = skills("hvpv"); o = OUT / "hvpv"; o.mkdir(parents=True, exist_ok=True)
    if S.empty: print("[hvpv] nothing yet"); return
    S["train_group"] = S["set"].str.extract(r"^(HV|PV|MIX)")[0]; S["test_group"] = S.test.str[:2]
    S.to_csv(o / "skills.csv", index=False)
    per_donor = S.groupby(["method", "repeat", "k", "train_group", "test"]).skill.median().reset_index()
    per_donor["test_group"] = per_donor.test.str[:2]
    tab = per_donor.groupby(["method", "k", "train_group", "test_group"]).skill.mean().unstack(["test_group"]).round(3)
    tab.to_csv(o / "table.csv"); print(tab.to_string())
    tests = []
    for (meth, k), g in per_donor.groupby(["method", "k"]):
        w = g.pivot_table(index=["repeat", "test"], columns="train_group", values="skill")
        for tgrp in ["HV", "PV"]:
            ww = w[[t.startswith(tgrp) for _, t in w.index]]
            for a, b in [("HV", "PV"), ("MIX", "HV"), ("MIX", "PV")]:
                d = (ww[a] - ww[b]).dropna()
                if len(d) > 5:
                    tests.append({"method": meth, "k": k, "test_group": tgrp, "comparison": f"{a} - {b} training",
                                  "mean_diff": d.mean(), "n_test_donor_repeats": len(d), "p": wilcoxon(d).pvalue})
    T = pd.DataFrame(tests); T.to_csv(o / "paired_tests.csv", index=False); print(T.round(4).to_string(index=False))
    # figure: CellOT only; colour = test donors' group; mean +- SE over test donors
    from matplotlib.lines import Line2D
    col = {"HV": "#2a78d6", "PV": "#eb6834"}; sty = {"CellOT": "-", "gauss_ot": "--"}
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    for ax, k in zip(axs, [2, 6]):
        for meth in ["CellOT"]:
            for tg in ["HV", "PV"]:
                h = per_donor[(per_donor.k == k) & (per_donor.method == meth) & (per_donor.test_group == tg)]
                h = h.groupby("train_group").skill.agg(["mean", "sem"]).reindex(["HV", "MIX", "PV"])
                ax.errorbar(np.arange(3), h["mean"], yerr=h["sem"],
                            color=col[tg], ls=sty[meth], lw=2, marker="o", ms=6, capsize=3)
        ax.set_xticks(range(3)); ax.set_xticklabels(["HV only", "half HV + half PV", "PV only"]); ax.set_xlim(-0.35, 2.35)
        ax.set_title(f"{k} training donors, drawn from:", fontsize=10); ax.grid(alpha=.3)
    axs[0].set_ylabel("skill on held-out donors (mean ± SE)")
    handles = [Line2D([], [], color=col["HV"], lw=2, label="tested on HV donors"),
               Line2D([], [], color=col["PV"], lw=2, label="tested on PV (pregnant) donors")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Does the donors' group matter? CellOT maps trained on HV, PV or mixed donors, by test group")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(o / "hvpv.png", dpi=150)


def donor_descriptors(T, celltypes, stims_by_ct):
    """Per donor: HV/PV, cells, response size and atypicality; per donor pair: baseline and response distances."""
    med = lambda key: T[key][0][2] if key in T else None                # q50 vector (13 markers)
    base, resp = {}, {}
    for c in celltypes:
        for d in DONORS:
            u = med((c, "Unstim", d))
            if u is None: continue
            base[(c, d)] = u
            for s in stims_by_ct[c]:
                v = med((c, s, d))
                if v is not None: resp[(s, c, d)] = v - u
    pairs = sorted({(s, c) for s, c, _ in resp})
    desc = []
    for d in DONORS:
        eff = [resp[(s, c, d)] for s, c in pairs if (s, c, d) in resp]
        aty = []
        for s, c in pairs:
            if (s, c, d) not in resp: continue
            others = [resp[(s, c, e)] for e in DONORS if e != d and (s, c, e) in resp]
            if others: aty.append(np.linalg.norm(resp[(s, c, d)] - np.mean(others, 0)))
        baty = []
        for c in celltypes:
            if (c, d) not in base: continue
            others = [base[(c, e)] for e in DONORS if e != d and (c, e) in base]
            baty.append(np.linalg.norm(base[(c, d)] - np.mean(others, 0)))
        cells = [T[(c, "Unstim", d)][1] for c in celltypes if (c, "Unstim", d) in T]
        desc.append({"donor": d, "group": d[:2], "response_size": np.mean([np.linalg.norm(e) for e in eff]),
                     "response_atypicality": np.mean(aty), "baseline_atypicality": np.mean(baty),
                     "median_cells": np.median(cells)})
    pair = []
    for a in DONORS:
        for b in DONORS:
            if a == b: continue
            dr = [np.linalg.norm(resp[(s, c, a)] - resp[(s, c, b)]) for s, c in pairs if (s, c, a) in resp and (s, c, b) in resp]
            db = [np.linalg.norm(base[(c, a)] - base[(c, b)]) for c in celltypes if (c, a) in base and (c, b) in base]
            pair.append({"train": a, "test": b, "response_distance": np.mean(dr), "baseline_distance": np.mean(db),
                         "same_group": a[:2] == b[:2]})
    return pd.DataFrame(desc).set_index("donor"), pd.DataFrame(pair)


def pairwise():
    S, T = skills("pairwise"); o = OUT / "pairwise"; o.mkdir(parents=True, exist_ok=True)
    if S.empty: print("[pairwise] nothing yet"); return
    S = S.rename(columns={"set": "train"}); S.to_csv(o / "skills.csv", index=False)
    cts = sorted(S.celltype.unique()); stims_by_ct = S.groupby("celltype").stim.unique().to_dict()
    desc, pair = donor_descriptors(T, cts, stims_by_ct); desc.to_csv(o / "donor_descriptors.csv"); pair.to_csv(o / "pair_distances.csv", index=False)
    res = []
    for meth, g in S.groupby("method"):
        M = g.groupby(["train", "test"]).skill.median().unstack("test").reindex(index=DONORS, columns=DONORS)
        M.to_csv(o / f"matrix_{meth}.csv")
        long = M.stack().rename("skill").reset_index()
        grand = long.skill.mean(); ra = long.groupby("train").skill.transform("mean") - grand
        ca = long.groupby("test").skill.transform("mean") - grand
        tot = ((long.skill - grand) ** 2).sum()
        share = {"train donor": (ra ** 2).sum() / tot, "test donor": (ca ** 2).sum() / tot,
                 "interaction (the pair)": ((long.skill - grand - ra - ca) ** 2).sum() / tot}
        L = long.merge(pair, on=["train", "test"])
        resid = L.skill - (L.groupby("train").skill.transform("mean") + L.groupby("test").skill.transform("mean") - grand)
        r = {"method": meth, **{f"variance share: {k}": v for k, v in share.items()},
             "rho(skill, response distance)": spearmanr(L.skill, L.response_distance)[0],
             "rho(skill, baseline distance)": spearmanr(L.skill, L.baseline_distance)[0],
             "rho(pair residual, response distance)": spearmanr(resid, L.response_distance)[0],
             "mean skill same group": L[L.same_group].skill.mean(), "mean skill other group": L[~L.same_group].skill.mean()}
        row_mean, col_mean = M.mean(1), M.mean(0)
        for c in ["response_atypicality", "baseline_atypicality", "response_size", "median_cells"]:
            r[f"rho(train goodness, {c})"] = spearmanr(row_mean, desc.loc[row_mean.index, c])[0]
            r[f"rho(test ease, {c})"] = spearmanr(col_mean, desc.loc[col_mean.index, c])[0]
        res.append(r)
        pd.DataFrame({"as training donor": row_mean, "as test donor": col_mean}).join(desc).sort_values(
            "as test donor").to_csv(o / f"donor_scores_{meth}.csv")
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
        im = axs[0].imshow(M.values, cmap="viridis"); axs[0].set_xticks(range(19)); axs[0].set_xticklabels(DONORS, rotation=90, fontsize=7)
        axs[0].set_yticks(range(19)); axs[0].set_yticklabels(DONORS, fontsize=7); axs[0].set_xlabel("test donor"); axs[0].set_ylabel("training donor (alone)")
        plt.colorbar(im, ax=axs[0]); axs[0].set_title(f"{meth}: skill, one donor -> another")
        axs[1].scatter(L.response_distance, L.skill, s=6, c=np.where(L.same_group, "C0", "C1"))
        axs[1].set_xlabel("distance between the two donors' stim responses"); axs[1].set_ylabel("skill"); axs[1].set_title("blue: same group (HV/PV), orange: other")
        axs[2].scatter(desc.loc[col_mean.index, "response_atypicality"], col_mean, c=["C0" if d.startswith("HV") else "C1" for d in col_mean.index])
        for d in col_mean.index: axs[2].annotate(d, (desc.loc[d, "response_atypicality"], col_mean[d]), fontsize=6)
        axs[2].set_xlabel("how atypical the donor's stim response is"); axs[2].set_ylabel("mean skill when tested (ease)")
        fig.tight_layout(); fig.savefig(o / f"pairwise_{meth}.png", dpi=150)
    R = pd.DataFrame(res).set_index("method").T; R.to_csv(o / "summary.csv"); print(R.round(3).to_string())


def small():
    S, _ = skills("small"); o = OUT / "small"; o.mkdir(parents=True, exist_ok=True)
    if S.empty: print("[small] nothing yet"); return
    S.to_csv(o / "skills.csv", index=False)
    cfg = S.groupby(["method", "repeat", "k", "cap"]).agg(skill=("skill", "median"), n_train=("n_train", "median")).reset_index()
    agg = cfg.groupby(["method", "k", "cap"]).agg(skill=("skill", "mean"), sd=("skill", "std"), n_train=("n_train", "median")).reset_index()
    agg.to_csv(o / "grid.csv", index=False)
    fits = []
    for meth, g in S.groupby("method"):
        g = g[g.n_train > 0].copy()
        g["log2_k"] = np.log2(g.k); g["log2_cells_per_donor"] = np.log2(g.n_train / g.k)
        # skill ~ a + b log2(k) + c log2(cells per donor), per (stim, celltype, test donor) cell-centred
        g["cell"] = g.stim + g.celltype + g.test + g.repeat
        X = g[["log2_k", "log2_cells_per_donor"]] - g.groupby("cell")[["log2_k", "log2_cells_per_donor"]].transform("mean")
        y = g.skill - g.groupby("cell").skill.transform("mean")
        b, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
        fits.append({"method": meth, "doubling the donors (same cells per donor)": b[0],
                     "doubling the cells per donor (same donors)": b[1], "n": len(g)})
    F = pd.DataFrame(fits); F.to_csv(o / "doubling_effects.csv", index=False); print(F.round(4).to_string(index=False))
    for meth in [m for m in METHODS if m in set(agg.method)]:
        g = agg[agg.method == meth]
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        for k, h in g.groupby("k"):
            axs[0].plot(h.n_train, h.skill, marker="o", label=f"k = {k} donors")
        for cap, h in g.groupby("cap"):
            axs[1].plot(h.n_train, h.skill, marker="s", label=f"{cap or 'all'} cells/donor")
        for ax, t in zip(axs, ["lines = same number of donors, more cells each", "lines = same cells per donor, more donors"]):
            ax.set_xscale("log"); ax.set_xlabel("total training cells (median over maps)"); ax.set_ylabel("skill on held-out donors")
            ax.set_title(t, fontsize=9); ax.grid(alpha=.3); ax.legend(fontsize=7)
        fig.suptitle(f"{meth}, 8 smallest celltypes"); fig.tight_layout(); fig.savefig(o / f"small_{meth}.png", dpi=150)
    print(agg[agg.method.isin(["CellOT", "gauss_ot"])].pivot_table(index=["method", "k"], columns="cap", values="skill").round(3).to_string())


def main():
    global BASE, CELLOT, OUT
    ap = argparse.ArgumentParser(); ap.add_argument("--exp", nargs="*", default=["hvpv", "pairwise", "small"])
    ap.add_argument("--base-dir"); ap.add_argument("--cellot-dir"); ap.add_argument("--out")
    a = ap.parse_args()
    BASE = Path(a.base_dir) if a.base_dir else BASE; CELLOT = Path(a.cellot_dir) if a.cellot_dir else CELLOT
    OUT = Path(a.out) if a.out else OUT
    for e in a.exp:
        print(f"===== {e}"); {"hvpv": hvpv, "pairwise": pairwise, "small": small}[e]()


if __name__ == "__main__":
    main()
