"""Score the "step" drug models (lc/drug_step.py: measured control) against the mean training effect E0 and the two-step
model E_B of lc/drug_tab.py (predicted control), on the same cases: LODO, selected features (meta/feature_set.csv) x 9
drugs; sets "selected" and "selected, real effect". Per donor (19, each once): MSE (95 % CI), MAE, dev_R2 =
1 - MSE / MSE(E0), between_R2 (features centred across donors), donors better than E0 and Wilcoxon p; per drug / stim;
distribution of per-feature dev_R2 (features with >= half the donors with a real effect). -> WORK/drug_tab/step_score/
    python -m lc.drug_step_score
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from lc.paths import REPO, WORK

OUT = WORK / "drug_tab" / "step_score"
K = ["fold", "donor", "drug", "stim", "celltype", "q", "marker"]; F = ["drug", "stim", "celltype", "q", "marker"]
METH = ["E0", "E_B", "E_S", "E_S65"]
rng = np.random.default_rng(0)


def table(g):
    tc = g.E - g.groupby(F).E.transform("mean")
    per = {m: g.assign(e2=(g[m] - g.E) ** 2, e1=(g[m] - g.E).abs()).groupby("donor")[["e2", "e1"]].mean() for m in METH}
    rows = []
    for m in METH:
        x = per[m].e2; b = [x.values[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        pc = g[m] - g.groupby(F)[m].transform("mean"); y = per["E0"].e2
        rows.append({"method": m, "MSE": x.mean(), "MSE_lo": np.percentile(b, 2.5), "MSE_hi": np.percentile(b, 97.5),
                     "MAE": per[m].e1.mean(), "dev_R2": 1 - x.mean() / y.mean(),
                     "between_R2": 1 - ((pc - tc) ** 2).sum() / (tc ** 2).sum(),
                     "better_than_E0": int((x < y).sum()) if m != "E0" else np.nan,
                     "p_vs_E0": wilcoxon(x - y).pvalue if m != "E0" else np.nan, "n_cases": len(g)})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S = pd.concat([pd.read_pickle(f) for f in sorted((WORK / "drug_tab").glob("step_*/fold_*.pkl")) if "smoke" not in str(f)],
                  ignore_index=True)
    L = pd.read_pickle(WORK / "drug_tab" / "score" / "cases.pkl")[K + ["E_B"]]
    fs = pd.read_csv(REPO / "meta" / "feature_set.csv")[["stim", "celltype", "marker", "q"]]
    R = S.merge(fs, on=["stim", "celltype", "marker", "q"]).merge(L, on=K)
    R["noise_E"] = np.sqrt(R.noise_D ** 2 + R.noise_C ** 2); R["real"] = (R.E.abs() > 0.0273) & (R.E.abs() > 2 * R.noise_E)
    R["floor_E"] = (R.noise_D ** 2 + R.noise_C ** 2) / 4; R.to_pickle(OUT / "cases.pkl")
    allt, byt = [], []
    for cs, g in [("selected", R), ("selected, real effect", R[R.real])]:
        t = table(g); t.insert(0, "set", cs); t["floor_E_MSE"] = g.floor_E.mean(); allt.append(t)
        for col in ["drug", "stim"]:
            for v, h in g.groupby(col):
                t = table(h); t.insert(0, "set", cs); t.insert(1, "by", col); t.insert(2, "value", v); byt.append(t)
    A = pd.concat(allt); A.round(6).to_csv(OUT / "metrics.csv", index=False); pd.concat(byt).round(6).to_csv(OUT / "by_drug_stim.csv", index=False)
    # per-feature dev_R2, features with >= half the donors with a real effect, with a shuffled-prediction null
    pf = []
    for k, g in R.groupby(F):
        if len(g) < 15 or g.real.mean() < 0.5: continue
        e0 = ((g.E0 - g.E) ** 2).sum(); r = dict(zip(F, k))
        for m in ["E_B", "E_S", "E_S65"]:
            r[m] = 1 - ((g[m] - g.E) ** 2).sum() / e0
            r[m + "_null"] = 1 - ((rng.permutation(g[m].to_numpy()) - g.E) ** 2).sum() / e0
        pf.append(r)
    PF = pd.DataFrame(pf); PF.to_csv(OUT / "per_feature.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"{len(R)} selected cases ({int(R.real.sum())} real), {R.donor.nunique()} donors")
    print(A.round(5).to_string(index=False))
    print("per-feature dev_R2 (real-effect features):"); print(PF.drop(columns=F).describe(percentiles=[.5, .9]).round(3).to_string())
    print(f"[DONE ] -> {OUT}")


if __name__ == "__main__":
    main()
