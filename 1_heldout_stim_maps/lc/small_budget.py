"""pDCs and mDCs: same total number of training cells, split between more or fewer donors; and 15 donors at every cap.

Reads the small-celltype experiment (donors_summary/small/skills.csv: k in 1/2/4/7/10/15 donors x 25/50/100/200/400/all
cells per donor, 5 repeats, 4 held-out test donors each). A training set's budget is its actual median number of
training cells (Unstim side), which is below k x cap when donors have fewer cells than the cap. For each nominal
budget B, every k whose nominal k x cap equals B (or 7 x cap within 15 %) is kept if its actual budget is within 30 % of B.
Skill per repeat = median over (map, test donor); shown as mean +- SD over the 5 repeats.
    python -m lc.small_budget -> WORK/donors_summary/small_budget/
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lc.analyze_donors import OUT  # noqa: E402

CTS = ["pDCs", "mDCs"]
BUDGETS = [100, 200, 400, 750, 1500]
COLS = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]      # one-hue ramp: budget is a magnitude
O = OUT / "small_budget"


def main():
    O.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(OUT / "small/skills.csv"); S = S[S.celltype.isin(CTS)]
    R = S.groupby(["celltype", "method", "k", "cap", "repeat"]).agg(skill=("skill", "median"), n_train=("n_train", "median"))
    G = R.groupby(["celltype", "method", "k", "cap"]).agg(skill=("skill", "mean"), sd=("skill", "std"),
                                                          n_train=("n_train", "median")).reset_index()
    rows = []
    for B in BUDGETS:
        for (ct, meth), g in G[G.cap > 0].groupby(["celltype", "method"]):
            for k, h in g.groupby("k"):
                h = h.assign(nominal=h.k * h.cap)
                h = h[(abs(h.nominal - B) / B <= 0.15) & (abs(h.n_train - B) / B <= 0.30)]
                if len(h): r = h.iloc[(h.nominal - B).abs().argmin()]; rows.append({"budget": B, **r.to_dict()})
    E = pd.DataFrame(rows); E.to_csv(O / "equal_budget.csv", index=False)
    K15 = G[G.k == 15].copy(); K15.to_csv(O / "k15_by_cap.csv", index=False)
    pd.set_option("display.width", 200)
    for ct in CTS:
        t = E[(E.celltype == ct) & E.method.isin(["CellOT", "gauss_ot"])]
        t = t.assign(cell=[f"{s:.2f} ({int(k)}×{int(c)}, n={int(n)})" for s, k, c, n in zip(t.skill, t.k, t.cap, t.n_train)])
        print(f"\n== {ct}: equal budget (skill, donors × cells/donor, actual training cells)")
        print(t.pivot_table(index=["method", "budget"], columns="k", values="cell", aggfunc="first").to_string())
        print(f"== {ct}: 15 donors, by cells per donor (0 = all)")
        print(K15[K15.celltype == ct].pivot_table(index="method", columns="cap", values="skill").round(3).to_string())

    fig, axs = plt.subplots(2, 2, figsize=(12, 8.5), gridspec_kw={"width_ratios": [1.35, 1]})
    for row, ct in enumerate(CTS):
        ax = axs[row, 0]
        for B, c in zip(BUDGETS, COLS):
            h = E[(E.celltype == ct) & (E.method == "CellOT") & (E.budget == B)].sort_values("k")
            if len(h) < 2: continue
            ax.errorbar(h.k, h.skill, yerr=h.sd, color=c, lw=2, marker="o", ms=7, capsize=3, label=f"{B} training cells in total")
        ax.axhline(0, color="#999999", lw=1); ax.set_xscale("log"); ax.set_xticks([1, 2, 4, 7, 10, 15])
        ax.set_xticklabels(["1", "2", "4", "7", "10", "15"]); ax.set_ylim(-0.3, 0.65)
        ax.set_xlabel("number of training donors (cells per donor = total / donors)")
        ax.set_ylabel("skill on held-out donors (mean ± SD, 5 repeats)"); ax.grid(alpha=.3)
        ax.set_title(f"{ct}: same number of training cells, more or fewer donors", fontsize=10)
        ax.legend(frameon=False, fontsize=8, loc="lower right", title="budget", title_fontsize=8)
        ax = axs[row, 1]; h = K15[K15.celltype == ct]
        for meth, c, ls in [("CellOT", "#2a78d6", "-"), ("gauss_ot", "#eb6834", "--")]:
            q = h[h.method == meth].assign(o=lambda d: d.cap.replace(0, 10_000)).sort_values("o")
            ax.errorbar(range(len(q)), q.skill, yerr=q.sd, color=c, ls=ls, lw=2, marker="o", ms=7, capsize=3, label=meth)
        caps = h[h.method == "CellOT"].assign(o=lambda d: d.cap.replace(0, 10_000)).sort_values("o")
        ax.set_xticks(range(len(caps))); ax.set_xticklabels([("all\n(~%d)" % (r.n_train / 15)) if r.cap == 0 else str(int(r.cap))
                                                              for _, r in caps.iterrows()])
        ax.set_xlabel("cells per donor"); ax.set_ylim(0.2, 0.7); ax.grid(alpha=.3)
        ax.set_title(f"{ct}: 15 donors, more or fewer cells each", fontsize=10); ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Small celltypes: is it donors or cells that matter? (held-out donors; left CellOT, right CellOT vs gauss_ot)")
    fig.tight_layout(); fig.savefig(O / "small_budget.png", dpi=150); plt.close(fig)
    print(f"[DONE ] {O}")


if __name__ == "__main__":
    main()
