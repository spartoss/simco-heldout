"""Do the maps get the DIFFERENCES between held-out donors right? (the readout that matters for Stabl)

Stabl standardises every feature across women, so a correction that is the same for every donor cannot matter.
Here each method's predictions and the truth are centred across the test donors of a repeat (per stim, celltype, k,
quantile, marker), which removes any common offset; what is left is the predicted vs true between-donor pattern.
- between-donor R2 = 1 - sum (pred_c - true_c)^2 / sum true_c^2 over the included groups (pooled sums);
- r = correlation of centred predictions and centred truths, pooled;
- per-group median r.
Null: identity, which here equals median_shift (a constant shift centres away): "each donor keeps its Unstim
difference". A map with patient-specific information has to beat it.
Noise: the true donor quantiles carry sampling noise; var(full) ~ (split-half diff)^2 / 4. Groups are kept if the
stim moves the cells (mean effect > FLOOR) and, for the "reliable" subset, the true between-donor variance is at
least twice its expected noise. noise_R2_ceiling = share of the true centred variance that is not noise.
Outputs WORK/cond_between/<tag>/: between.csv (method x k x subset), between_by_stim.csv (k = 15).
    python -m lc.cond_between --tag med_z
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lc.paths import QS, WORK
from lc.score import FLOOR, truth_tables


def load(r, tests, indir, train_dirs):
    z = np.load(indir / f"repeat_{r}.npz", allow_pickle=True)
    maps = [tuple(m)[:3] for m in z["maps"]]
    preds = {m: z[m] for m in z.files if m not in ("maps", "chosen")}
    pick = lambda P, idx: np.stack([P[i] if i >= 0 else np.full(P.shape[1:], np.nan) for i in idx])
    b = np.load(WORK / "baselines" / f"repeat_{r}.npz", allow_pickle=True)
    bpos = {tuple(m)[:3]: i for i, m in enumerate(b["maps"])}
    for m in ("identity", "quantile_map"):
        preds[m] = pick(b[m], [bpos.get(x, -1) for x in maps])
    for name, d in train_dirs.items():
        cdir = Path(str(d).format(r=r))
        if not (cdir / "pred_q.npy").exists(): continue
        cm = pd.read_csv(cdir / "maps.csv"); P = np.load(cdir / "pred_q.npy", mmap_mode="r")
        cpos = {(s, c, k): i for i, (s, c, k) in enumerate(zip(cm.stim, cm.celltype, cm.k))}
        preds[name] = pick(P[P.shape[0] - 1], [cpos.get(x, -1) for x in maps])
    return maps, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="med_z")
    ap.add_argument("--train", nargs="*", default=[], help="name=dir with {r}, e.g. CellOT_pc=work/lc/cond_train/repeat_{r}/pc")
    a = ap.parse_args()
    indir = WORK / "cond_baselines" / a.tag
    out = WORK / "cond_between" / a.tag; out.mkdir(parents=True, exist_ok=True)
    train_dirs = {"CellOT": WORK / "train" / "repeat_{r}"}
    train_dirs.update(dict(t.split("=", 1) for t in a.train))
    rows = []
    for dz in json.load(open(WORK / "design.json")):
        r, tests = dz["repeat"], dz["test"]
        if not (indir / f"repeat_{r}.npz").exists(): continue
        maps, preds = load(r, tests, indir, train_dirs)
        T = truth_tables(sorted({m[1] for m in maps}), set(tests))
        for i, (stim, ct, k) in enumerate(maps):
            ok = [j for j, d in enumerate(tests) if (ct, stim, d) in T and (ct, "Unstim", d) in T]
            if len(ok) < 3: continue
            tr = np.stack([T[(ct, stim, tests[j])][0] for j in ok])                 # (n, q, m)
            nz = np.stack([T[(ct, stim, tests[j])][1] for j in ok]) ** 2 / 4         # noise variance per donor
            iden = np.stack([T[(ct, "Unstim", tests[j])][0] for j in ok])
            eff = np.abs(tr - iden).mean(0)
            tc = tr - tr.mean(0); n = len(ok)
            sst = (tc ** 2).sum(0); noise = nz.sum(0) * (1 - 1 / n)
            for m, P in preds.items():
                p = P[i][ok]
                if np.isnan(p).any(): continue
                pc = p - p.mean(0)
                sse = ((pc - tc) ** 2).sum(0); spp = (pc ** 2).sum(0); spt = (pc * tc).sum(0)
                for qi in range(len(QS)):
                    for mi in range(tr.shape[2]):
                        rows.append((m, r, stim, ct, k, QS[qi], mi, n, eff[qi, mi], sst[qi, mi], noise[qi, mi],
                                     sse[qi, mi], spp[qi, mi], spt[qi, mi]))
        print(f"[DONE ] repeat {r}", flush=True)
    D = pd.DataFrame(rows, columns=["method", "repeat", "stim", "celltype", "k", "q", "marker", "n", "effect", "sst",
                                    "noise", "sse", "spp", "spt"])
    D = D[D.effect > FLOOR]
    D["reliable"] = D.sst >= 2 * D.noise

    def summ(g):
        r_g = g.spt / np.sqrt(g.spp * g.sst).replace(0, np.nan)
        return pd.Series({"between_R2": 1 - g.sse.sum() / g.sst.sum(),
                          "r_pooled": g.spt.sum() / np.sqrt(g.spp.sum() * g.sst.sum()),
                          "r_median_group": r_g.median(),
                          "noise_R2_ceiling": 1 - g.noise.sum() / g.sst.sum(), "n_groups": len(g)})
    res = []
    for sub, G in [("all live", D), ("reliable", D[D.reliable])]:
        s = G.groupby(["method", "k"]).apply(summ).reset_index(); s.insert(0, "subset", sub); res.append(s)
    S = pd.concat(res); S.round(4).to_csv(out / "between.csv", index=False)
    kmax = D.k.max()
    B = D[(D.k == kmax) & D.reliable].groupby(["method", "stim"]).apply(summ).reset_index()
    B.round(4).to_csv(out / "between_by_stim.csv", index=False)
    pd.set_option("display.width", 200)
    for sub in ("all live", "reliable"):
        print(f"--- {sub}: between-donor R2 (rows method, cols k)")
        print(S[S.subset == sub].pivot(index="method", columns="k", values="between_R2").round(3).to_string())
        print(f"--- {sub}: pooled r")
        print(S[S.subset == sub].pivot(index="method", columns="k", values="r_pooled").round(3).to_string())
        print("noise ceiling", S[S.subset == sub].groupby("k").noise_R2_ceiling.first().round(3).to_dict())
    print(f"[DONE ] -> {out}")


if __name__ == "__main__":
    main()
