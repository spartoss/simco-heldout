"""SHAP values of the LightGBM residual models (lc/tab_predict.py): which features drive the predicted correction.

Refits lgbm exactly as tab_predict (same features, params, donor-grouped CV rounds) for each (repeat, k, stim) and
computes exact TreeSHAP (LightGBM pred_contrib) on the TEST donors' rows; the refit is checked against the stored
rhat.npy. Readouts (mean |SHAP| over test rows, in arcsinh units of the predicted residual):
- by_feature.csv   per (layer, k, stim, feature)
- by_group.csv     per (layer, k, stim, group); a group's SHAP = sum of its features' SHAP in the row, then mean |.|.
  L1 is split by its relation to the row: own (same q and marker = u_dev again), same marker other q,
  other marker same q, other marker other q.
    python -m lc.tab_shap --repeats 0 1 2 3 4 --ks 15 --layers L2 [--out WORK/tab/shap]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lc.paths import MARKERS, PAIRS, QS, WORK
from lc.tab_features import CAT, L0, L1, build
from lc.tab_predict import LGB, OUT


def fit_booster(Xtr, y, groups):
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    nf = min(5, len(np.unique(groups)))
    ds = lgb.Dataset(Xtr, y, categorical_feature=[c for c in CAT if c in Xtr], free_raw_data=False)
    cv = lgb.cv(LGB, ds, num_boost_round=1000, folds=GroupKFold(nf).split(Xtr, y, groups),
                callbacks=[lgb.early_stopping(50, verbose=False)])
    return lgb.train(LGB, ds, num_boost_round=len(cv["valid l2-mean"]))


def group_of(col):
    if col in L0 or col in CAT: return col
    if col.startswith("freq_"): return "freq (28)"
    if col.startswith("pc"): return "PCs (5)"
    if col.startswith("lda"): return "LDA (3)"
    return None                                       # L1: depends on the row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--ks", nargs="*", type=int, default=[15])
    ap.add_argument("--layers", nargs="*", default=["L2"])
    ap.add_argument("--out", default=str(OUT / "shap"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stims = sorted({s for s, _ in PAIRS})
    l1_q = {c: QS[i // 13] for i, c in enumerate(L1)}; l1_m = {c: i % 13 for i, c in enumerate(L1)}
    feat_rows, grp_rows, checks = [], [], []
    for dz in json.load(open(WORK / "design.json")):
        r, tests = dz["repeat"], dz["test"]
        if r not in a.repeats: continue
        for k in a.ks:
            for stim in stims:
                D, cols = build(r, k, stim, dz["order"][:k], tests)
                tr, te = D[~D.is_test], D[D.is_test]
                for L in a.layers:
                    X = D[cols[L]]
                    bst = fit_booster(X[~D.is_test], tr.target.to_numpy(), tr.donor.to_numpy())
                    C = bst.predict(X[D.is_test], pred_contrib=True)        # (n, n_feat + 1), last = bias
                    S = pd.DataFrame(C[:, :-1], columns=cols[L], index=te.index)
                    # check against the stored predictions of tab_predict
                    st = OUT / f"lgbm_{L}" / f"repeat_{r}"
                    if (st / "rhat.npy").exists():
                        cm = pd.read_csv(st / "maps.csv"); RH = np.load(st / "rhat.npy")
                        pos = {m: i for i, m in enumerate(zip(cm.stim, cm.celltype, cm.k))}
                        ii = [pos[(stim, c, k)] for c in te.celltype.astype(str)]
                        jj = [tests.index(d) for d in te.donor]
                        stored = RH[ii, jj, te.qi, te.mi]; pred = C.sum(1)
                        checks.append({"repeat": r, "k": k, "stim": stim, "layer": L,
                                       "max_abs_diff": float(np.nanmax(np.abs(stored - pred)))})
                    base = {"repeat": r, "k": k, "stim": stim, "layer": L}
                    feat_rows.append(pd.DataFrame({**base, "feature": cols[L], "mean_abs": S.abs().mean().to_numpy()}))
                    G = {}
                    for col in cols[L]:
                        g = group_of(col)
                        if g is not None: G.setdefault(g, 0); G[g] = G[g] + S[col].to_numpy()
                    l1c = [c for c in cols[L] if c in l1_q]
                    if l1c:
                        qq = te.q.to_numpy()[:, None] == np.array([l1_q[c] for c in l1c])[None]
                        mm = te.mi.to_numpy()[:, None] == np.array([l1_m[c] for c in l1c])[None]
                        V = S[l1c].to_numpy()
                        for name, msk in [("L1 own (= u_dev)", qq & mm), ("L1 same marker, other q", ~qq & mm),
                                          ("L1 other marker, same q", qq & ~mm), ("L1 other marker, other q", ~qq & ~mm)]:
                            G[name] = (V * msk).sum(1)
                    G["TOTAL (prediction)"] = C[:, :-1].sum(1)
                    grp_rows.append(pd.DataFrame({**base, "group": list(G), "mean_abs": [np.abs(v).mean() for v in G.values()],
                                                  "target_mean_abs": np.abs(te.target).mean()}))
                print(f"[SHAP ] r{r} k{k} {stim}", flush=True)
    F = pd.concat(feat_rows); Gd = pd.concat(grp_rows); Ck = pd.DataFrame(checks)
    F.to_csv(out / "by_feature.csv", index=False); Gd.to_csv(out / "by_group.csv", index=False)
    Ck.to_csv(out / "refit_check.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"--- refit vs stored rhat: max |diff| = {Ck.max_abs_diff.max():.2e}")
    for L in a.layers:
        g = Gd[Gd.layer == L].groupby(["group", "stim"]).mean_abs.mean().unstack()
        g["ALL"] = Gd[Gd.layer == L].groupby("group").mean_abs.mean()
        print(f"--- {L}: mean |SHAP| by group (k = {a.ks})"); print(g.sort_values("ALL", ascending=False).round(4).to_string())
        f = F[F.layer == L].groupby("feature").mean_abs.mean().sort_values(ascending=False)
        print(f"--- {L}: top 30 features"); print(f.head(30).round(4).to_string())
    print(f"[DONE ] -> {out}")


if __name__ == "__main__":
    main()
