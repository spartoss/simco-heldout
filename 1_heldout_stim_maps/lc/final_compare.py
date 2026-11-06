"""Final comparison on leave-one-donor-out (19 folds, 18 training donors; PI / IL33 out; q50-q90), from the case table of
lc/tab_compare.py --dump (work/lc/tab/final/cases.pkl) and the noise floor of lc/noise_floor_lodo.py.

Methods: the columns of the case table (CellOT versions, TabPFN / LightGBM layers, shift) plus, built here:
  <cellot>+shift = 0.5 / 0.5;   w<cellot>+<model> = 0.4 * CellOT + 0.6 * model (MIX_CELLOTS x MIX_MODELS).
Per case set (rule = the selected features, floor = all cases with a visible effect) and method:
- MSE, MAE: per donor (each of the 19 left-out donors once), with a 95 % bootstrap CI over donors;
- share_shift_err_removed = 1 - MSE / MSE(shift);
- share_reducible_removed = (MSE(shift) - MSE) / (MSE(shift) - floor), floor = mean response noise (floor_resp) of
  the same cases: how much of what the shift misses AND is not noise has been predicted;
- between_R2: per feature, prediction and truth centred across the 19 donors; 1 - SSE / SS(truth). Its noise ceiling
  is 1 - floor / between-donor variance of the truth;
- vs CellOT_paper and vs CellOT_paired+shift: donors better (of 19), Wilcoxon p over donors (MSE).
Also: between-donor variance decomposition (baseline = explained by the shift, extra = explained beyond it, unexplained,
noise), per stim; per-stim MSE; per-donor MSE.  Outputs in work/lc/tab/final/: metrics_<set>.csv, by_stim_<set>.csv,
by_donor_<set>.csv, variance_<set>.csv.
    python -m lc.final_compare
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from lc.paths import WORK

FIN = WORK / "tab" / "final"
CELLOTS = ["CellOT_paper", "CellOT_paired", "CellOT_pcblood", "CellOT_film", "CellOT_cluster"]
MIX_CELLOTS = CELLOTS
MIX_MODELS = ["pfn_L1cat", "pfn_L1all_raw", "pfn_L1allM", "lgb_L1cat", "lgb_L1F", "lgb_L1allM"]
KEY = ["stim", "celltype", "q", "marker"]
rng = np.random.default_rng(0)


def main():
    D = pd.read_pickle(FIN / "cases.pkl")
    F = pd.read_pickle(FIN / "noise_floor.pkl").rename(columns={"marker": "marker"})
    D = D.merge(F[["stim", "celltype", "donor", "q", "marker", "floor_resp", "floor_dxw", "floor_samp"]],
                on=["stim", "celltype", "donor", "q", "marker"], how="left")
    cel = [c for c in CELLOTS if c in D]
    for c in cel: D[f"{c}+shift"] = 0.5 * D[c] + 0.5 * D["shift"]
    for c in [x for x in MIX_CELLOTS if x in D]:
        for m in [x for x in MIX_MODELS if x in D]: D[f"0.4 {c} + 0.6 {m}"] = 0.4 * D[c] + 0.6 * D[m]
    skip = {"set", "repeat", "stim", "celltype", "k", "donor", "q", "marker", "true", "eff", "tc", "ntest",
            "floor_resp", "floor_dxw", "floor_samp"}
    meths = [c for c in D.columns if c not in skip]
    for s, g in D.groupby("set"):
        tag = "rule" if s.startswith("rule") else "floor"
        g = g.copy(); tc = g.true - g.groupby(KEY).true.transform("mean")
        floor = g.floor_resp.mean(); var_b = (tc ** 2).mean()
        per = {m: g.assign(e2=(g[m] - g.true) ** 2, e1=(g[m] - g.true).abs()).groupby("donor")[["e2", "e1"]].mean() for m in meths}
        mse_shift = per["shift"].e2.mean(); donors = per["shift"].index
        rows = []
        for m in meths:
            x = per[m].e2; boot = [x.values[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
            pc = g[m] - g.groupby(KEY)[m].transform("mean")
            r = {"method": m, "MSE": x.mean(), "MSE_lo": np.percentile(boot, 2.5), "MSE_hi": np.percentile(boot, 97.5),
                 "MAE": per[m].e1.mean(), "share_shift_err_removed": 1 - x.mean() / mse_shift,
                 "share_reducible_removed": (mse_shift - x.mean()) / (mse_shift - floor),
                 "between_R2": 1 - ((pc - tc) ** 2).sum() / (tc ** 2).sum()}
            for ref in ["CellOT_paper", "CellOT_paired+shift"]:
                if ref in per and m != ref:
                    y = per[ref].e2.reindex(x.index); r[f"better_than_{ref}"] = int((x < y).sum()); r[f"p_vs_{ref}"] = wilcoxon(x - y).pvalue
            rows.append(r)
        M = pd.DataFrame(rows)
        M.attrs = {}; M["noise_floor_MSE"] = floor; M["between_R2_ceiling"] = 1 - floor / var_b; M["n_cases"] = len(g)
        M.round(5).to_csv(FIN / f"metrics_{tag}.csv", index=False)
        # per stim and per donor MSE
        pd.DataFrame({m: g.assign(e2=(g[m] - g.true) ** 2).groupby("stim").e2.mean() for m in meths}).T.round(5).to_csv(FIN / f"by_stim_{tag}.csv")
        pd.DataFrame({m: per[m].e2 for m in meths}).round(5).to_csv(FIN / f"by_donor_{tag}.csv")
        # between-donor variance decomposition, overall and per stim
        vr = []
        for st, h in [("ALL", g)] + list(g.groupby("stim")):
            tch = h.true - h.groupby(KEY).true.transform("mean"); V = (tch ** 2).mean(); nz = h.floor_resp.mean()
            ex = lambda m: 1 - ((h[m] - h.groupby(KEY)[m].transform("mean") - tch) ** 2).mean() / V
            for m in meths:
                vr.append({"stim": st, "method": m, "between_var": V, "explained": ex(m), "baseline_shift": ex("shift"),
                           "noise": nz / V})
        pd.DataFrame(vr).round(5).to_csv(FIN / f"variance_{tag}.csv", index=False)
        pd.set_option("display.width", 250)
        show = M.set_index("method").sort_values("MSE")
        print(f"=== {s}: {len(g)} cases, {len(donors)} donors; noise floor MSE {floor:.4f}; between-donor var {var_b:.4f}; "
              f"between_R2 ceiling {1 - floor / var_b:.3f}")
        print(show[["MSE", "MSE_lo", "MSE_hi", "MAE", "share_shift_err_removed", "share_reducible_removed", "between_R2",
                    "better_than_CellOT_paper", "better_than_CellOT_paired+shift", "p_vs_CellOT_paired+shift"]].round(4).to_string())
    print(f"[DONE ] -> {FIN}")


if __name__ == "__main__":
    main()
