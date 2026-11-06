"""Checkpoint-selection diagnostic (2026-10-12): how much does the rule that picks g matter on NEW donors, and is the
best iteration reproducible across repeats?

Reads cond_train runs (pred_q.npy [n_eval + 1, M, n_test, n_q, 13]: one slice per evaluation step, the last slice = the
checkpoint with the lowest validation median error on the TRAINING donors' held-out cells; val_err.npy [n_eval, M]).
Rules, per (arm, k), scored on the held-out test donors (cases with effect > FLOOR, common to every rule):
    val      : the run's own choice (best training-donor val error) = what every experiment used
    last     : the last iterate
    avg3/avg5: mean of the predicted quantiles of the last 3 / 5 evaluation steps (no selection)
    valtop3  : mean of the 3 steps with the lowest val error
    step_<s> : a fixed step for every map
    oracle_step: the one fixed step best on the test donors (uses test truth: reference only)
    oracle_map : per map, the step best on its test donors (uses test truth: optimistic upper bound)
Reproducibility: for each map (stim, celltype, k), the val-best step and the test-best step in each of the 5 repeats
(different training and test donors). Spread across repeats vs a null where steps are shuffled across maps.
    python -m lc.ckpt_diag --arms none pc_wd1 pcblood_wd1 dnorm:dnorm --out WORK/ckpt_diag
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lc.paths import DONORS, PAIRS, QS, WORK
from lc.score import FLOOR, truth_tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="name[:dnorm], dir = cond_train/repeat_<r>/<name>")
    ap.add_argument("--repeats", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    T = truth_tables(sorted({c for _, c in PAIRS}), set(DONORS))
    off = np.load(WORK / "cache_dnorm/offsets.npz")
    design = {dz["repeat"]: dz for dz in json.load(open(WORK / "design.json"))}
    score_rows, step_rows = [], []
    for arm in a.arms:
        name, _, flag = arm.partition(":")
        for r in a.repeats:
            d = WORK / f"cond_train/repeat_{r}/{name}"
            if not (d / "pred_q.npy").exists(): print(f"[WARN ] missing {d}", flush=True); continue
            meta = json.load(open(d / "meta.json")); ev = meta["eval_every"]
            P = np.load(d / "pred_q.npy"); V = np.load(d / "val_err.npy"); cm = pd.read_csv(d / "maps.csv")
            n_eval = V.shape[0]; steps = ev * np.arange(1, n_eval + 1)
            tests = design[r]["test"]
            for i, m in cm.iterrows():
                s, c, k = m.stim, m.celltype, m.k
                if np.isnan(V[:, i]).all(): continue
                vb = int(np.nanargmin(V[:, i])); order = np.argsort(V[:, i])
                cand = {"val": P[n_eval, i], "last": P[n_eval - 1, i], "avg3": P[n_eval - 3:n_eval, i].mean(0),
                        "avg5": P[n_eval - 5:n_eval, i].mean(0), "valtop3": P[order[:3], i].mean(0)}
                for e in range(n_eval): cand[f"step_{steps[e]}"] = P[e, i]
                terr = np.zeros(n_eval); ok_any = False
                for j, dn in enumerate(tests):
                    if (c, s, dn) not in T or (c, "Unstim", dn) not in T: continue
                    t, u = T[(c, s, dn)][0], T[(c, "Unstim", dn)][0]
                    eff = np.abs(t - u); msk = eff > FLOOR
                    if not msk.any() or np.isnan(P[n_eval, i, j]).all(): continue
                    o = off[f"{c}|{dn}"][None, :] if flag == "dnorm" else 0
                    ok_any = True
                    for e in range(n_eval): terr[e] += np.abs(P[e, i, j] + o - t)[msk].sum()
                    rules = list(cand); nc = int(msk.sum())
                    err = np.stack([np.abs(cand[x][j] + o - t)[msk] for x in rules]).ravel()
                    score_rows.append(pd.DataFrame({"arm": name, "repeat": r, "stim": s, "celltype": c, "k": k,
                                                    "donor": dn, "rule": np.repeat(rules, nc),
                                                    "case": np.tile(np.flatnonzero(msk.ravel()), len(rules)),
                                                    "err": err.astype(np.float32), "eff": np.tile(eff[msk], len(rules)).astype(np.float32)}))
                if ok_any:
                    step_rows.append({"arm": name, "repeat": r, "stim": s, "celltype": c, "k": k, "eval_every": ev,
                                      "val_best": int(steps[vb]), "test_best": int(steps[int(np.argmin(terr))]),
                                      "val_best_is_last": vb == n_eval - 1,
                                      "test_err_valbest": terr[vb], "test_err_testbest": terr.min(),
                                      "test_err_last": terr[-1]})
            print(f"{name} repeat {r} done", flush=True)
    S = pd.DataFrame(step_rows); S.to_csv(out / "steps.csv", index=False)
    D = pd.concat(score_rows, ignore_index=True)
    for col in ("arm", "stim", "celltype", "donor"): D[col] = D[col].astype("category")
    D["skill"] = 1 - D.err / D.eff
    # oracle rules from the fixed-step rows (test truth): one step per (arm, k), or per map
    fx = D[D.rule.str.startswith("step_")]
    best_step = fx.groupby(["arm", "k", "rule"], observed=True).err.mean().groupby(level=["arm", "k"], observed=True).idxmin()
    keep = fx.set_index(["arm", "k", "rule"]).index.isin(list(best_step.values))
    o1 = fx[keep].assign(rule="oracle_step")
    mkey = ["arm", "repeat", "stim", "celltype", "k"]
    mbest = fx.groupby(mkey + ["rule"], observed=True).err.sum().groupby(level=mkey, observed=True).idxmin()
    o2 = fx[fx.set_index(mkey + ["rule"]).index.isin(list(mbest.values))].assign(rule="oracle_map")
    D = pd.concat([D, o1, o2], ignore_index=True)
    res = []
    for (arm, k, rule), g in D.groupby(["arm", "k", "rule"], observed=True):
        res.append({"arm": arm, "k": k, "rule": rule, "skill": g.groupby("repeat").skill.median().mean(),
                    "mean_err": g.err.mean(), "n": len(g)})
    R = pd.DataFrame(res); R.to_csv(out / "rules.csv", index=False)
    print("oracle fixed step per (arm, k):", {kk: v[2] for kk, v in best_step.items()})
    main_rules = ["val", "last", "avg3", "avg5", "valtop3", "oracle_step", "oracle_map"]
    pd.set_option("display.width", 220)
    print("--- skill by rule"); print(R[R.rule.isin(main_rules)].pivot(index=["arm", "k"], columns="rule", values="skill")[main_rules].round(4).to_string())
    print("--- skill by fixed step"); print(R[R.rule.str.startswith("step_")].pivot(index=["arm", "k"], columns="rule", values="skill").round(4).to_string())
    # paired: % of (map, donor) where rule closer than val, per arm/k
    W = D[D.rule.isin(main_rules)].pivot_table(index=["arm", "repeat", "stim", "celltype", "k", "donor", "case"], columns="rule", values="err", observed=True)
    pr = [{"arm": arm, "k": k, **{f"{x}_closer_than_val_%": round(100 * (g[x] < g["val"]).mean(), 1) for x in main_rules if x != "val"}}
          for (arm, k), g in W.groupby(level=["arm", "k"], observed=True)]
    Pd = pd.DataFrame(pr); Pd.to_csv(out / "paired_vs_val.csv", index=False); print("--- % cases closer than val"); print(Pd.to_string(index=False))
    # reproducibility of the best step across repeats
    rep = []
    rng = np.random.default_rng(0)
    for (arm, k), g in S.groupby(["arm", "k"]):
        for col in ("val_best", "test_best"):
            sd = g.groupby(["stim", "celltype"])[col].std().mean()
            null = np.mean([g.assign(x=rng.permutation(g[col].values)).groupby(["stim", "celltype"]).x.std().mean() for _ in range(200)])
            bystim = g.groupby("stim")[col].median().to_dict()
            rep.append({"arm": arm, "k": k, "which": col, "median_step": g[col].median(), "sd_across_repeats": round(sd, 0),
                        "sd_null_shuffled": round(null, 0), "median_by_stim": bystim})
        rep.append({"arm": arm, "k": k, "which": "corr(val_best, test_best)", "median_step": np.nan,
                    "sd_across_repeats": round(g[["val_best", "test_best"]].corr(method="spearman").iloc[0, 1], 3),
                    "sd_null_shuffled": np.nan, "median_by_stim": {}})
        rep.append({"arm": arm, "k": k, "which": "test err ratio valbest/testbest, last/testbest", "median_step": np.nan,
                    "sd_across_repeats": round((g.test_err_valbest / g.test_err_testbest).median(), 3),
                    "sd_null_shuffled": round((g.test_err_last / g.test_err_testbest).median(), 3), "median_by_stim": {}})
    Rp = pd.DataFrame(rep); Rp.to_csv(out / "reproducibility.csv", index=False)
    print("--- best step reproducibility"); print(Rp.to_string(index=False))
    print("--- share of maps by val-best step"); print(S.groupby(["arm", "k"]).val_best.value_counts(normalize=True).unstack().round(3).to_string())
    print("--- share of maps by test-best step"); print(S.groupby(["arm", "k"]).test_best.value_counts(normalize=True).unstack().round(3).to_string())


if __name__ == "__main__":
    main()
