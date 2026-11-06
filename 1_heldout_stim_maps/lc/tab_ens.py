"""Turn lc/tab_predict.py residuals into scoreable predictions, and report the residual readouts.

For each method dir WORK/tab/<name>/repeat_<r>/ (maps.csv, rhat.npy):
- res.csv: res_R2 = 1 - sum (rhat - r)^2 / sum r^2 and dev_R2 (as lc/residual_predict.py) over live test cases
  (|S - U| > FLOOR), per (k, stim) and overall per k; r uses the full-training-set means (= the test-time target).
- WORK/tab/pred/<name>_l<lambda>/repeat_<r>/: shift + lambda * rhat, and with --cellot also
  <name>_l<lambda>_ens: 0.5 * CellOT_none (cond_train/repeat_r/none, last iterate) + 0.5 * (shift + lambda * rhat);
  other CellOT weights w (--w-cellot) -> <name>_l<lambda>_c<w> = w * CellOT_none + (1 - w) * (shift + lambda * rhat).
Both are pred_q.npy (1, M, n_test, 5, 13) + maps.csv, read by lc.ctx_score.
    python -m lc.tab_ens --names lgbm_L2 tabpfn_L2 --repeats 0 1 2 3 4 --lambdas 0.5 1 [--cellot]
"""
import argparse
import json

import numpy as np
import pandas as pd

from lc.cond_train_score import load_train
from lc.paths import DONORS, PAIRS, WORK
from lc.score import FLOOR, truth_tables

TAB = WORK / "tab"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--repeats", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--lambdas", nargs="*", type=float, default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--cellot", action="store_true")
    ap.add_argument("--w-cellot", nargs="*", type=float, default=[0.5],
                    help="CellOT weights w: w * CellOT + (1 - w) * prediction; 0.5 -> <name>_l<lam>_ens, else _c<w>")
    ap.add_argument("--no-pred", action="store_true", help="residual readouts only")
    ap.add_argument("--design", default=str(WORK / "design.json"))
    ap.add_argument("--res-tag", default="", help="suffix of the residual readout files (res<tag>_r<repeats>.csv)")
    a = ap.parse_args()
    T = truth_tables(sorted({c for _, c in PAIRS}), set(DONORS))
    rows = []
    for dz in json.load(open(a.design)):
        r, tests = dz["repeat"], dz["test"]
        if r not in a.repeats: continue
        for name in a.names:
            d = TAB / name / f"repeat_{r}"
            if not (d / "rhat.npy").exists(): continue
            cm = pd.read_csv(d / "maps.csv"); RH = np.load(d / "rhat.npy")
            maps = list(zip(cm.stim, cm.celltype, cm.k))
            CO = load_train(WORK / f"cond_train/repeat_{r}/none", maps) if a.cellot else {}
            SH = np.full_like(RH, np.nan)
            for i, (s, c, k) in enumerate(maps):
                trd = [x for x in dz["order"][:k] if (c, s, x) in T and (c, "Unstim", x) in T]
                if len(trd) < 2: continue
                mu_u = np.mean([T[(c, "Unstim", x)][0] for x in trd], 0); mu_s = np.mean([T[(c, s, x)][0] for x in trd], 0)
                for j, x in enumerate(tests):
                    if (c, "Unstim", x) not in T: continue
                    u = T[(c, "Unstim", x)][0]; SH[i, j] = mu_s + u - mu_u
                    if (c, s, x) not in T or np.isnan(RH[i, j]).all(): continue
                    t = T[(c, s, x)][0]; res = (t - mu_s) - (u - mu_u); live = (np.abs(t - u) > FLOOR) & ~np.isnan(RH[i, j])
                    rows.append({"name": name, "repeat": r, "k": k, "stim": s, "celltype": c, "donor": x,
                                 "err2": ((RH[i, j] - res) ** 2)[live].sum(), "r2": (res ** 2)[live].sum(),
                                 "dev2": ((t - mu_s) ** 2)[live].sum(), "n": live.sum()})
            if a.no_pred: continue
            for lam in a.lambdas:
                P = SH + lam * np.nan_to_num(RH)
                P[np.isnan(RH)] = np.nan                       # no prediction (map, donor, or q10/q25 in v2 runs)
                outs = {f"{name}_l{lam:g}": P}
                if a.cellot:
                    for w in a.w_cellot:
                        E = np.full_like(P, np.nan)
                        for i, m in enumerate(maps):
                            if m in CO: E[i] = w * CO[m] + (1 - w) * P[i]
                        outs[f"{name}_l{lam:g}_ens" if w == 0.5 else f"{name}_l{lam:g}_c{w:.2f}"] = E
                for n, X in outs.items():
                    o = TAB / "pred" / n / f"repeat_{r}"; o.mkdir(parents=True, exist_ok=True)
                    cm[["stim", "celltype", "k"]].to_csv(o / "maps.csv", index=False); np.save(o / "pred_q.npy", X[None])
        print(f"[REP  ] {r}", flush=True)
    D = pd.DataFrame(rows)
    agg = lambda g: pd.Series({"res_R2": 1 - g.err2.sum() / g.r2.sum(), "dev_R2": 1 - g.err2.sum() / g.dev2.sum(),
                               "n": g.n.sum()})
    S = D.groupby(["name", "k"]).apply(agg, include_groups=False).reset_index()
    B = D.groupby(["name", "k", "stim"]).apply(agg, include_groups=False).reset_index()
    tag = "_".join(map(str, a.repeats))
    S.round(4).to_csv(TAB / f"res{a.res_tag}_r{tag}.csv", index=False)
    B.round(4).to_csv(TAB / f"res_by_stim{a.res_tag}_r{tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print(S.pivot(index="name", columns="k", values="res_R2").round(4).to_string())
    print(S.pivot(index="name", columns="k", values="dev_R2").round(4).to_string())
    if B.k.nunique():
        kk = B.k.max(); print(f"--- res_R2 by stim, k={kk}")
        print(B[B.k == kk].pivot(index="name", columns="stim", values="res_R2").round(3).to_string())


if __name__ == "__main__":
    main()
