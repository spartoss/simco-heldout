"""Score the held-out-donor drug predictions of lc/drug_tab.py (LODO, all stims / folds) on the selected features.

Selected features: meta/feature_set.csv (stim, celltype, marker, q; q50-q90, no PI / IL33), crossed with the 9 drugs.
Case sets: "selected" (all of them) and "selected, real effect" (|E| > 0.0273 and |E| > 2 x its split-half noise).
For each target and method, per donor (each of the 19 left-out donors once; bootstrap 95 % CI over donors):
- drug effect E = D - C: E0 (mean training effect), E_A (D_hat - C_hat, direct), E_B (two-step)
- drug + stim D: D0 (drug shift), D_hat (direct), D_B (C_hat + E_B), D_C0E (C_hat + mean effect)
- control + stim C: C0 (shift), C_hat
MSE, MAE, share of the mean-effect error removed (1 - MSE / MSE(E0), effect only), between-donor R2 (each feature
centred across donors), the sampling-noise floor of E ((h1 - h2)^2 / 4 for D and C), donors better and Wilcoxon p vs
the reference (E0 / D_hat / C0). Per drug and per stim tables for the effect. Writes WORK/drug_tab/score/*.csv.
    python -m lc.drug_score
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from lc.paths import REPO, WORK

OUT = WORK / "drug_tab" / "score"
TARGETS = {"E": (["E0", "E_A", "E_B"], "E0"), "D": (["D0", "D_hat", "D_B", "D_C0E"], "D_hat"), "C": (["C0", "C_hat"], "C0")}
KEY = ["drug", "stim", "celltype", "q", "marker"]
rng = np.random.default_rng(0)


def metrics(g, tgt, meths, ref):
    rows = []
    tc = g[tgt] - g.groupby(KEY)[tgt].transform("mean")
    per = {m: g.assign(e2=(g[m] - g[tgt]) ** 2, e1=(g[m] - g[tgt]).abs()).groupby("donor")[["e2", "e1"]].mean() for m in meths}
    for m in meths:
        x = per[m].e2; b = [x.values[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        pc = g[m] - g.groupby(KEY)[m].transform("mean")
        r = {"target": tgt, "method": m, "MSE": x.mean(), "MSE_lo": np.percentile(b, 2.5), "MSE_hi": np.percentile(b, 97.5),
             "MAE": per[m].e1.mean(), "between_R2": 1 - ((pc - tc) ** 2).sum() / (tc ** 2).sum(), "n_cases": len(g)}
        if tgt == "E": r["share_mean_effect_err_removed"] = 1 - x.mean() / per["E0"].e2.mean()
        if m != ref:
            y = per[ref].e2.reindex(x.index); r["better_than_ref"] = int((x < y).sum()); r["p_vs_ref"] = wilcoxon(x - y).pvalue
            r["ref"] = ref
        rows.append(r)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = pd.concat([pd.read_pickle(f) for f in sorted((WORK / "drug_tab").glob("lodo_*/fold_*.pkl")) if "smoke" not in str(f)],
                  ignore_index=True)
    fs = pd.read_csv(REPO / "meta" / "feature_set.csv")[["stim", "celltype", "marker", "q"]]
    R = R.merge(fs, on=["stim", "celltype", "marker", "q"]).dropna(subset=["D", "C"])
    R["noise_E"] = np.sqrt(R.noise_D ** 2 + R.noise_C ** 2); R["floor_E"] = (R.noise_D ** 2 + R.noise_C ** 2) / 4
    R["real"] = (R.E.abs() > 0.0273) & (R.E.abs() > 2 * R.noise_E)
    R.to_pickle(OUT / "cases.pkl")
    rows, by = [], []
    for cs, g in [("selected", R), ("selected, real effect", R[R.real])]:
        for tgt, (ms, ref) in TARGETS.items():
            for r in metrics(g, tgt, ms, ref): rows.append({"set": cs, **r})
        for col in ["drug", "stim"]:
            for v, h in g.groupby(col):
                for r in metrics(h, "E", TARGETS["E"][0], "E0"): by.append({"set": cs, "by": col, "value": v, **r})
    M = pd.DataFrame(rows); M["floor_E_MSE"] = np.nan
    for cs, g in [("selected", R), ("selected, real effect", R[R.real])]:
        M.loc[(M.set == cs) & (M.target == "E"), "floor_E_MSE"] = g.floor_E.mean()
    M.round(6).to_csv(OUT / "metrics.csv", index=False); pd.DataFrame(by).round(6).to_csv(OUT / "by_drug_stim.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"{len(R)} selected cases ({R.real.sum()} real effects), {R.donor.nunique()} donors, {R.drug.nunique()} drugs")
    print(M.drop(columns=["MSE_lo", "MSE_hi"]).round(5).to_string(index=False))
    print(f"[DONE ] -> {OUT}")


if __name__ == "__main__":
    main()
