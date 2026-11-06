"""HV vs PV: which features differ between the groups, and is that where HV-trained maps fail on PV donors?

Part A (all 19 donors, atlas cache): for every (celltype, marker, quantile) three feature families,
  unstim    Q(Unstim)                          the baseline, which every map receives as input
  stim      Q(stim)                            the stimulated level (what a map must output)
  response  Q(stim) - Q(Unstim)                the perturbation itself
group difference = PV mean - HV mean (arcsinh) and Hedges' g; Mann-Whitney p with Benjamini-Hochberg q over each family.
Also with PV01 and HV05 removed (the two baseline outliers of the pairwise experiment).
Part B (hvpv experiment, k = 6): per case (repeat, test donor, stim, celltype, quantile, marker) with a real effect
(|true response| > FLOOR), err = |pred - true stimulated quantile| for the HV-only and the PV-only training set.
excess = err(HV-trained) - err(PV-trained) on PV test donors (and the mirror on HV test donors). Per feature (stim,
celltype, marker, quantile), averaged over test donors and repeats: is the excess everywhere or concentrated, and does
it follow the Part A group differences? pred(HV) - pred(PV) is compared with the HV - PV difference of the truth.
    python -m lc.hvpv_features  -> WORK/donors_summary/hvpv_features/
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu, spearmanr  # noqa: E402

from lc.analyze_donors import FLOOR, OUT, load_truth, runs  # noqa: E402
from lc.paths import DONORS, MARKERS, PAIRS, QS  # noqa: E402

O = OUT / "hvpv_features"
OUTLIERS = ["PV01", "HV05"]
BLUE, ORANGE = "#2a78d6", "#eb6834"


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def group_diff(T, donors):
    rows = []
    for s, c in PAIRS + [("Unstim", c) for c in sorted({c for _, c in PAIRS})]:
        for d in donors:
            if (c, "Unstim", d) not in T or (s != "Unstim" and (c, s, d) not in T): continue
            u = T[(c, "Unstim", d)][0]
            fams = {"unstim": u} if s == "Unstim" else {"stim": T[(c, s, d)][0], "response": T[(c, s, d)][0] - u}
            for fam, v in fams.items():
                for qi, q in enumerate(QS):
                    for mi, m in enumerate(MARKERS):
                        rows.append((fam, s, c, m, q, d[:2], v[qi, mi]))
    L = pd.DataFrame(rows, columns=["family", "stim", "celltype", "marker", "q", "group", "value"])
    out = []
    for key, g in L.groupby(["family", "stim", "celltype", "marker", "q"]):
        hv, pv = g.value[g.group == "HV"].to_numpy(), g.value[g.group == "PV"].to_numpy()
        if len(hv) < 3 or len(pv) < 3: continue
        sp = np.sqrt(((len(hv) - 1) * hv.var(ddof=1) + (len(pv) - 1) * pv.var(ddof=1)) / (len(hv) + len(pv) - 2))
        j = 1 - 3 / (4 * (len(hv) + len(pv)) - 9)
        out.append((*key, pv.mean() - hv.mean(), j * (pv.mean() - hv.mean()) / sp if sp > 0 else 0.0,
                    mannwhitneyu(pv, hv).pvalue if np.ptp(np.r_[hv, pv]) > 0 else 1.0))
    R = pd.DataFrame(out, columns=["family", "stim", "celltype", "marker", "q", "diff_PV_minus_HV", "hedges_g", "p"])
    R["q_bh"] = R.groupby("family").p.transform(bh)
    return R


def case_errors():
    """Per (method, repeat, test donor, stim, celltype, q, marker): errors of HV6 / PV6 / MIX6 maps (real effects only)."""
    cts = set(); items = []
    for meth, tag, m, P, tests in runs("hvpv"):
        if meth in ("CellOT", "gauss_ot"): cts |= set(m.celltype); items.append((meth, tag, m, P, tests))
    T = load_truth(sorted(cts)); rows = []
    for meth, tag, m, P, tests in items:
        m = m.reset_index(drop=True); m["i"] = m.index
        for (s, c), g in m[m.k == 6].groupby(["stim", "celltype"]):
            idx = dict(zip(g.set, g.i))
            if not {"HV6", "PV6", "MIX6"} <= set(idx): continue
            for j, d in enumerate(tests):
                if (c, s, d) not in T or (c, "Unstim", d) not in T: continue
                tr = T[(c, s, d)][0]; eff = tr - T[(c, "Unstim", d)][0]
                pr = {k: P[idx[k], j] for k in ("HV6", "PV6", "MIX6")}
                if any(np.isnan(v).all() for v in pr.values()): continue
                for qi, q in enumerate(QS):
                    for mi, mk in enumerate(MARKERS):
                        if abs(eff[qi, mi]) <= FLOOR: continue
                        rows.append((meth, tag, d, d[:2], s, c, mk, q, tr[qi, mi], eff[qi, mi],
                                     pr["HV6"][qi, mi], pr["PV6"][qi, mi], pr["MIX6"][qi, mi]))
    E = pd.DataFrame(rows, columns=["method", "repeat", "test", "test_group", "stim", "celltype", "marker", "q", "true",
                                    "effect", "pred_HV", "pred_PV", "pred_MIX"])
    for k in ("HV", "PV", "MIX"): E[f"err_{k}"] = (E[f"pred_{k}"] - E["true"]).abs()
    return E


def concentration(x):
    """Share of the total positive excess carried by the top 10 % / 20 % of features, and share of features > 0."""
    x = np.sort(np.clip(x, 0, None))[::-1]; tot = x.sum()
    return {f"top {p} % features carry": x[: max(1, int(len(x) * p / 100))].sum() / tot for p in (10, 20)}


def main():
    O.mkdir(parents=True, exist_ok=True)
    T = load_truth(sorted({c for _, c in PAIRS}))
    # ---------- Part A
    A = group_diff(T, DONORS); A.to_csv(O / "group_diff.csv", index=False)
    A2 = group_diff(T, [d for d in DONORS if d not in OUTLIERS]); A2.to_csv(O / "group_diff_no_outliers.csv", index=False)
    summ = []
    for name, X in [("all 19 donors", A), ("without PV01, HV05", A2)]:
        for fam, g in X.groupby("family"):
            summ.append({"donors": name, "family": fam, "n features": len(g), "median |g|": g.hedges_g.abs().median(),
                         "share |g| > 0.8": (g.hedges_g.abs() > 0.8).mean(), "share q_BH < 0.1": (g.q_bh < 0.1).mean(),
                         "median |PV - HV| (arcsinh)": g.diff_PV_minus_HV.abs().median()})
    S = pd.DataFrame(summ); S.to_csv(O / "family_summary.csv", index=False); print(S.round(3).to_string(index=False))
    for fam in ("unstim", "response", "stim"):
        top = A[A.family == fam].reindex(A[A.family == fam].hedges_g.abs().sort_values(ascending=False).index).head(15)
        top.to_csv(O / f"top_{fam}.csv", index=False)
    bym = A.assign(ag=A.hedges_g.abs()).pivot_table(index="marker", columns="family", values="ag", aggfunc="median")
    bym.to_csv(O / "by_marker.csv")
    byc = A.assign(ag=A.hedges_g.abs()).pivot_table(index="celltype", columns="family", values="ag", aggfunc="median")
    byc.to_csv(O / "by_celltype.csv")
    print(bym.round(2).to_string())

    # ---------- Part B
    E = case_errors()
    feat = ["method", "test_group", "stim", "celltype", "marker", "q"]
    F = E.assign(excess=E.err_HV - E.err_PV, excess_mix=E.err_MIX - E.err_PV, shift=E.pred_HV - E.pred_PV,
                 skill_HV=1 - E.err_HV / E.effect.abs(), skill_PV=1 - E.err_PV / E.effect.abs()) \
        .groupby(feat).agg(excess=("excess", "mean"), excess_mix=("excess_mix", "mean"), shift=("shift", "mean"),
                           err_HV=("err_HV", "mean"), err_PV=("err_PV", "mean"), skill_HV=("skill_HV", "median"),
                           skill_PV=("skill_PV", "median"), n=("true", "size")).reset_index()
    resp = A[A.family == "response"].set_index(["stim", "celltype", "marker", "q"])
    stim = A[A.family == "stim"].set_index(["stim", "celltype", "marker", "q"])
    un = A[A.family == "unstim"].set_index(["celltype", "marker", "q"])
    k4 = list(zip(F.stim, F.celltype, F.marker, F.q))
    F["g_response"] = resp.hedges_g.reindex(k4).to_numpy(); F["d_response"] = resp.diff_PV_minus_HV.reindex(k4).to_numpy()
    F["d_stim"] = stim.diff_PV_minus_HV.reindex(k4).to_numpy()
    F["g_unstim"] = un.hedges_g.reindex(list(zip(F.celltype, F.marker, F.q))).to_numpy()
    F.to_csv(O / "excess_by_feature.csv", index=False)
    B = []
    for (meth, tg), g in F.groupby(["method", "test_group"]):
        B.append({"method": meth, "test donors": tg, "features": len(g),
                  "mean excess error HV- vs PV-trained": g.excess.mean(), "mean error PV-trained": g.err_PV.mean(),
                  "share of features where HV-trained is worse": (g.excess > 0).mean(), **concentration(g.excess.to_numpy()),
                  "rho(excess, |g response|)": spearmanr(g.excess, g.g_response.abs(), nan_policy="omit")[0],
                  "rho(excess, |g unstim|)": spearmanr(g.excess, g.g_unstim.abs(), nan_policy="omit")[0],
                  "rho(pred_HV - pred_PV, HV - PV truth)": spearmanr(g["shift"], -g.d_stim, nan_policy="omit")[0]})
    B = pd.DataFrame(B); B.to_csv(O / "excess_summary.csv", index=False); print(B.round(3).T.to_string())
    pv = F[F.test_group == "PV"]
    for by in ("marker", "stim", "celltype", "q"):
        t = pv.groupby(["method", by]).agg(excess=("excess", "mean"), share_worse=("excess", lambda x: (x > 0).mean()),
                                           err_PV=("err_PV", "mean")).reset_index()
        t.to_csv(O / f"excess_PVtest_by_{by}.csv", index=False)
    pv[pv.method == "CellOT"].sort_values("excess", ascending=False).head(25).to_csv(O / "top_excess_CellOT.csv", index=False)

    # ---------- figures
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (name, X) in zip(axs, [("all 19 donors", A), ("without PV01, HV05", A2)]):
        for fam, col, ls in [("unstim", "#555555", "-"), ("stim", BLUE, "-"), ("response", ORANGE, "-")]:
            v = np.sort(X[X.family == fam].hedges_g.abs().to_numpy())
            ax.plot(v, 1 - np.arange(len(v)) / len(v), color=col, lw=2, ls=ls, label=f"{fam} ({len(v)} features)")
        from scipy.stats import t as tdist
        n1 = sum(d.startswith("HV") for d in DONORS if name.startswith("all") or d not in OUTLIERS); n2 = (19 if name.startswith("all") else 17) - n1
        xs = np.linspace(0, 3, 200); J = 1 - 3 / (4 * (n1 + n2) - 9); sc = J * np.sqrt(1 / n1 + 1 / n2)
        ax.plot(xs, 2 * tdist.sf(xs / sc, n1 + n2 - 2), color="#999999", lw=1.5, ls="--", label="chance (no real difference)")
        ax.axvline(0.8, color="#999999", lw=1, ls=":"); ax.text(0.82, 0.93, "large effect (|g| = 0.8)", fontsize=8, color="#555555")
        ax.set_xlabel("|Hedges' g|, PV vs HV"); ax.set_ylabel("share of features with a larger difference")
        ax.set_title(name, fontsize=10); ax.grid(alpha=.3); ax.set_xlim(0, 3)
    axs[0].legend(title="feature family", fontsize=8, title_fontsize=8, frameon=False)
    fig.suptitle("How different are HV and PV donors? (features: celltype × marker × quantile; stim families also × stim)")
    fig.tight_layout(); fig.savefig(O / "group_diff_ecdf.png", dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    g = pv[pv.method == "CellOT"].copy()
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.2, 1.6, 10]; lab = ["<0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.2", "1.2-1.6", ">1.6"]
    for colname, c, name in [("g_response", ORANGE, "difference in the stim RESPONSE"),
                             ("g_unstim", "#555555", "difference in the Unstim BASELINE")]:
        b = pd.cut(g[colname].abs(), bins, labels=lab, include_lowest=True)
        t = g.groupby(b, observed=False).excess.agg(["mean", "sem", "size"])
        axs[0].errorbar(range(len(t)), t["mean"], yerr=t["sem"], color=c, lw=2, marker="o", ms=6, capsize=3, label=name)
    axs[0].axhline(0, color="#999999", lw=1); axs[0].set_xticks(range(len(lab))); axs[0].set_xticklabels(lab)
    axs[0].set_xlabel("|Hedges' g| between PV and HV donors for that feature")
    axs[0].set_ylabel("extra error of HV-trained maps (arcsinh, mean ± SE)")
    axs[0].legend(frameon=False, fontsize=8, title="features binned by", title_fontsize=8)
    axs[0].set_title("Extra error follows response differences, not baseline ones", fontsize=10); axs[0].grid(alpha=.3)
    t = pv.groupby(["method", "marker"]).apply(lambda h: h.excess.mean() / h.err_PV.mean()).unstack(0).sort_values("CellOT")
    y = np.arange(len(t))
    axs[1].barh(y + 0.2, t["CellOT"] * 100, height=0.38, color=BLUE, label="CellOT")
    axs[1].barh(y - 0.2, t["gauss_ot"] * 100, height=0.38, color="#86b6ef", label="gauss_ot")
    axs[1].set_yticks(y); axs[1].set_yticklabels(t.index); axs[1].axvline(0, color="#999999", lw=1)
    axs[1].set_xlabel("extra error of HV-trained maps, % of the PV-trained error")
    axs[1].legend(frameon=False, fontsize=8, loc="lower right")
    axs[1].set_title("Where it is: by marker", fontsize=10); axs[1].grid(alpha=.3, axis="x")
    fig.suptitle("PV test donors: HV-only vs PV-only training (6 donors each)")
    fig.tight_layout(); fig.savefig(O / "excess_vs_groupdiff.png", dpi=150); plt.close(fig)
    # where: celltype x marker maps of the HV-PV difference (baseline, response) and of CellOT's extra error on PV donors
    from matplotlib.colors import LinearSegmentedColormap
    div = LinearSegmentedColormap.from_list("div", [ORANGE, "#f2f1ec", BLUE])
    seq = LinearSegmentedColormap.from_list("seq", ["#f2f1ec", "#86b6ef", "#1c5cab"])
    cts = sorted(A.celltype.unique()); mk = list(MARKERS)
    Gu = A[A.family == "unstim"].pivot_table(index="celltype", columns="marker", values="hedges_g", aggfunc="median")
    Gr = A[A.family == "response"].pivot_table(index="celltype", columns="marker", values="hedges_g", aggfunc="median")
    g = pv[pv.method == "CellOT"]
    X = (g.groupby(["celltype", "marker"]).excess.mean() / g.groupby(["celltype", "marker"]).err_PV.mean()).unstack() * 100
    fig, axs = plt.subplots(1, 3, figsize=(17, 8.5), sharey=True)
    for ax, M, cm, lim, ttl, cl in [
            (axs[0], Gu, div, 2, "Unstim baseline: PV vs HV (median Hedges' g)", "g  (blue: PV higher, orange: PV lower)"),
            (axs[1], Gr, div, 2, "Stim response: PV vs HV (median g over stims, quantiles)", "g  (blue: PV higher, orange: PV lower)"),
            (axs[2], X, seq, 40, "CellOT on PV donors: extra error of HV-trained maps", "% of the PV-trained error")]:
        M = M.reindex(index=cts, columns=mk)
        im = ax.imshow(M.to_numpy(), cmap=cm, vmin=0 if cm is seq else -lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(len(mk))); ax.set_xticklabels(mk, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(cts))); ax.set_yticklabels(cts, fontsize=8); ax.set_title(ttl, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label(cl, fontsize=8)
    fig.suptitle("Where HV and PV donors differ (left, middle) and where HV-trained maps fail on PV donors (right; negative values shown as 0)")
    fig.tight_layout(); fig.savefig(O / "where_heatmaps.png", dpi=150); plt.close(fig)
    print(f"[DONE ] {O}")


if __name__ == "__main__":
    main()
