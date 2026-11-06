"""Group decomposition (exact TreeSHAP) of the round-3 LightGBM models (lc/tab_v2.py --model lgbm), e.g. L1all_raw.

Refits LightGBM exactly as lc/tab_v2.py (same rows, features, params, donor-grouped CV rounds) for each (repeat, k, stim)
over all celltypes, checks the refit against the stored rhat, and computes TreeSHAP (pred_contrib) on the TEST rows.
A group's SHAP in a row = the sum of its features' SHAP; readouts are mean |group SHAP| over test rows (arcsinh units
of the predicted correction) and its share of the summed group importances. The own profile (65 columns) is split by
its relation to the row: own (= u_dev again), same marker / other quantile, other marker / same quantile, other marker
/ other quantile. Output WORK/tab/shap_v2_<layer>/: by_group.csv (repeat, stim, group), refit_check.csv.
    python -m lc.tab_v2_shap --layer L1all_raw --repeats 0 1 2 3 4 --k 15
"""
import argparse
import json

import numpy as np
import pandas as pd

from lc.paths import QS, WORK
from lc.tab_v2 import BPC, CTS, F28, FPC, L1, L1PC, LAYERS_CT, MX, STIMS, build

OUT = WORK / "tab"
SINGLE = ["u_dev", "mu_u", "eff", "sd_u", "noise_u", "q", "marker", "celltype_code", "hvpv"]


def fit_booster(X, y, groups, cat_cols):
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    from lc.tab_predict import LGB
    X = X.copy()
    for c in cat_cols: X[c] = X[c].astype(int)
    ds = lgb.Dataset(X, y, categorical_feature=list(cat_cols), free_raw_data=False)
    cv = lgb.cv(LGB, ds, num_boost_round=1000, folds=GroupKFold(min(5, len(np.unique(groups)))).split(X, y, groups),
                callbacks=[lgb.early_stopping(50, verbose=False)])
    return lgb.train(LGB, ds, num_boost_round=len(cv["valid l2-mean"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="L1all_raw")
    ap.add_argument("--repeats", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--k", type=int, default=15)
    a = ap.parse_args()
    out = OUT / f"shap_v2_{a.layer}"; out.mkdir(parents=True, exist_ok=True)
    cols = LAYERS_CT[a.layer] + ["celltype_code"]; cat = [c for c in ("marker", "celltype_code") if c in cols]
    l1q = {c: QS[i // 13] for i, c in enumerate(L1)}; l1m = {c: i % 13 for i, c in enumerate(L1)}
    stored = OUT / f"v2lgb_{a.layer}"
    rows, checks = [], []
    for dz in json.load(open(WORK / "design.json")):
        r, tests = dz["repeat"], dz["test"]
        if r not in a.repeats: continue
        for stim in STIMS:
            D = build(r, a.k, stim, CTS, dz["order"][:a.k], tests)
            tr, te = D[~D.is_test], D[D.is_test]
            bst = fit_booster(tr[cols], tr.target.to_numpy(), tr.donor.to_numpy(), cat)
            Xte = te[cols].copy()
            for c in cat: Xte[c] = Xte[c].astype(int)
            C = bst.predict(Xte, pred_contrib=True); S = pd.DataFrame(C[:, :-1], columns=cols, index=te.index)
            st = stored / f"repeat_{r}"
            if (st / "rhat.npy").exists():
                cm = pd.read_csv(st / "maps.csv"); RH = np.load(st / "rhat.npy")
                pos = {m: i for i, m in enumerate(zip(cm.stim, cm.celltype, cm.k))}
                ii = [pos[(stim, c, a.k)] for c in te.celltype]; jj = [tests.index(d) for d in te.donor]
                checks.append({"repeat": r, "stim": stim, "max_abs_diff": float(np.abs(RH[ii, jj, te.qi, te.mi] - C.sum(1)).max())})
            G = {g: S[g].to_numpy() for g in SINGLE if g in cols}
            for name, blk in [("28 raw log frequencies", F28), ("frequency PCs (3)", FPC), ("whole-blood PCs", BPC),
                              ("own-profile PCs", L1PC), ("marker axes", MX)]:
                b = [c for c in blk if c in cols]
                if b: G[name] = S[b].sum(1).to_numpy()
            l1c = [c for c in L1 if c in cols]
            if l1c:
                qq = te.q.to_numpy()[:, None] == np.array([l1q[c] for c in l1c])[None]
                mm = te.mi.to_numpy()[:, None] == np.array([l1m[c] for c in l1c])[None]
                V = S[l1c].to_numpy()
                for name, msk in [("own profile: same q & marker (= u_dev)", qq & mm), ("own profile: same marker, other q", ~qq & mm),
                                  ("own profile: other marker, same q", qq & ~mm), ("own profile: other marker, other q", ~qq & ~mm)]:
                    G[name] = (V * msk).sum(1)
            G["TOTAL (prediction)"] = C[:, :-1].sum(1)
            for g, v in G.items():
                rows.append({"repeat": r, "stim": stim, "group": g, "mean_abs": np.abs(v).mean(),
                             "target_mean_abs": np.abs(te.target).mean()})
            print(f"[SHAP ] r{r} {stim}", flush=True)
    B = pd.DataFrame(rows); Ck = pd.DataFrame(checks)
    B.to_csv(out / "by_group.csv", index=False); Ck.to_csv(out / "refit_check.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"--- refit vs stored rhat: max |diff| = {Ck.max_abs_diff.max():.1e}" if len(Ck) else "--- no stored rhat to check")
    g = B[B.group != "TOTAL (prediction)"]
    T = g.groupby(["group", "stim"]).mean_abs.mean().unstack(); T["ALL"] = g.groupby("group").mean_abs.mean()
    T["share %"] = 100 * T["ALL"] / T["ALL"].sum()
    print(T.sort_values("ALL", ascending=False).round(4).to_string())
    print(f"[DONE ] -> {out}")


if __name__ == "__main__":
    main()
