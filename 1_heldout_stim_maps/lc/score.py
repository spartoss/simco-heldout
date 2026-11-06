"""Score every method on the test donors and summarise the learning curves.

For each (map = stim, celltype, k; test donor; quantile q; marker):
- truth = q-th percentile of the donor's stimulated cells; effect = |truth - identity| (what "no effect" misses);
- err = |predicted - truth|; skill = 1 - err / effect;
- noise = |q of first half - q of second half| of the donor's stimulated cells: a split-half sampling floor
  (at half the cells, so it overstates the noise by ~sqrt(2)); ceiling skill = 1 - noise / effect.
Only cases whose effect exceeds FLOOR (the atlas technical floor, 0.0273) are summarised; --thr adds a stricter cut.
Outputs WORK/summary/: curve.csv, curve_by_stim.csv, curve_by_q.csv, trajectory.csv, saturation.csv, *.png.
    python -m lc.score
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lc.paths import CACHE, QS, WORK  # noqa: E402

FLOOR = 0.0273
MIN_CELLS = 20


def truth_tables(celltypes, donors, cache_dir=CACHE):
    T = {}
    for c in celltypes:
        z = np.load(Path(cache_dir) / f"{c}.npz")
        for key in z.files:
            cond, d = key.split("|")
            if d not in donors or len(z[key]) < MIN_CELLS: continue
            x = z[key]; h = len(x) // 2
            T[(c, cond, d)] = (np.percentile(x, QS, axis=0), np.abs(np.percentile(x[:h], QS, 0) - np.percentile(x[h:], QS, 0)))
    return T


def cases(maps, preds, tests, T):
    """Long table for one repeat: maps (list of (stim, ct, k, cap)), preds {method: (M, n_test, n_q, n_m)}."""
    rows = []
    for i, (stim, ct, k, cap) in enumerate(maps):
        for j, d in enumerate(tests):
            if (ct, stim, d) not in T or (ct, "Unstim", d) not in T: continue
            tr, noise = T[(ct, stim, d)]; iden = T[(ct, "Unstim", d)][0]
            eff = np.abs(tr - iden)
            base = {"stim": stim, "celltype": ct, "k": k, "donor": d}
            for m, P in preds.items():
                p = P[i, j]
                if np.isnan(p).all(): continue
                err = np.abs(p - tr)
                for qi, q in enumerate(QS):
                    for mi in range(tr.shape[1]):
                        rows.append((m, stim, ct, k, cap, d, q, mi, err[qi, mi], eff[qi, mi], noise[qi, mi]))
    return pd.DataFrame(rows, columns=["method", "stim", "celltype", "k", "cap", "donor", "q", "marker", "err", "effect",
                                       "noise"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--train-dir", default=str(WORK / "train"))
    ap.add_argument("--baseline-dir", default=str(WORK / "baselines"))
    ap.add_argument("--out", default=str(WORK / "summary"))
    ap.add_argument("--repeats", nargs="*", type=int, default=None)
    ap.add_argument("--part", default="", help="sub-directory of each repeat (train_full: curve / cells)")
    ap.add_argument("--cache-dir", default=str(CACHE), help="for the truth; cache_full for the uncapped run")
    a = ap.parse_args()
    design = json.load(open(WORK / "design.json"))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    allc, traj = [], []
    for dz in design:
        r = dz["repeat"]; tests = dz["test"]
        if a.repeats is not None and r not in a.repeats: continue
        preds, maps = {}, None
        bfile = Path(a.baseline_dir) / f"repeat_{r}.npz"      # baselines_full/<part>/ is passed as --baseline-dir
        if bfile.exists():
            z = np.load(bfile, allow_pickle=True)
            bmaps = [(tuple(x) + (0,))[:4] for x in z["maps"]]     # capped run: 3-tuples (cap 0); spec runs: 6-tuples
            for m in ["identity", "median_shift", "quantile_map", "gauss_ot"]: preds[m] = z[m]
            maps = bmaps
        cdir = Path(a.train_dir) / f"repeat_{r}" / a.part
        if (cdir / "pred_q.npy").exists():
            cm = pd.read_csv(cdir / "maps.csv"); P = np.load(cdir / "pred_q.npy")
            cmaps = list(zip(cm.stim, cm.celltype, cm.k, cm["cap"] if "cap" in cm else [0] * len(cm)))
            if maps is None: maps = cmaps
            pos = {mk: i for i, mk in enumerate(cmaps)}
            idx = [pos.get(mk, -1) for mk in maps]
            take = lambda slot: np.stack([P[slot, i] if i >= 0 else np.full(P.shape[2:], np.nan) for i in idx])
            n_eval = P.shape[0] - 1
            preds["CellOT (last)"] = take(n_eval - 1); preds["CellOT (paper-style best)"] = take(n_eval)
            ev = json.load(open(cdir / "meta.json"))["eval_every"]
            for s in range(n_eval):
                traj.append((r, (s + 1) * ev, take(s)))
        if maps is None:
            print(f"[SKIP ] repeat {r}: nothing yet"); continue
        T = truth_tables(sorted({m[1] for m in maps}), set(tests), a.cache_dir)
        df = cases(maps, preds, tests, T); df["repeat"] = r
        allc.append(df)
        # CellOT trajectory, same cases
        for (rr, it, P) in [t for t in traj if t[0] == r]:
            d2 = cases(maps, {"CellOT": P}, tests, T)
            d2 = d2[d2.effect > FLOOR]
            g = d2.assign(skill=1 - d2.err / d2.effect).groupby(["k", "cap"]).skill.median()
            for (k, cp), v in g.items(): TRAJ.append({"repeat": r, "iteration": it, "k": k, "cap": cp, "median skill": v})
        print(f"[SCORE] repeat {r}: {len(df)} rows, methods {sorted(preds)}", flush=True)
    if not allc:
        print("[DONE ] nothing to score"); return
    D = pd.concat(allc)
    D["skill"] = 1 - D.err / D.effect
    D["ceiling"] = 1 - D.noise / D.effect
    live = D[D.effect > FLOOR]
    summ = lambda x, keys: x.groupby(keys).agg(median_skill=("skill", "median"), median_err=("err", "median"),
                                               ceiling=("ceiling", "median"), n=("skill", "size")).reset_index()
    cur = summ(live, ["method", "repeat", "k", "cap"]); cur.to_csv(out / "curve_per_repeat.csv", index=False)
    agg = cur.groupby(["method", "k", "cap"]).agg(skill=("median_skill", "mean"), skill_sd=("median_skill", "std"),
                                           err=("median_err", "mean"), ceiling=("ceiling", "mean"), n=("n", "mean")).reset_index()
    agg.to_csv(out / "curve.csv", index=False)
    summ(live[live.effect > a.thr], ["method", "repeat", "k", "cap"]).to_csv(out / f"curve_per_repeat_effect_gt_{a.thr}.csv", index=False)
    summ(live, ["method", "stim", "k", "cap"]).to_csv(out / "curve_by_stim.csv", index=False)
    summ(live, ["method", "q", "k", "cap"]).to_csv(out / "curve_by_q.csv", index=False)
    summ(live, ["method", "celltype", "k", "cap"]).to_csv(out / "curve_by_celltype.csv", index=False)
    if TRAJ: pd.DataFrame(TRAJ).to_csv(out / "trajectory.csv", index=False)

    # saturation: first k reaching 90 / 95 % of the k = 15 skill, per repeat
    sat = []
    for (m, r), g in cur[cur.cap == cur.cap.min()].groupby(["method", "repeat"]):
        g = g.set_index("k").median_skill.sort_index(); top = g.iloc[-1]
        if m == "identity" or not np.isfinite(top) or top <= 0: continue
        for frac in (0.9, 0.95):
            ok = g[g >= frac * top]
            sat.append({"method": m, "repeat": r, "fraction": frac, "k": int(ok.index.min()) if len(ok) else np.nan})
    S = pd.DataFrame(sat); S.to_csv(out / "saturation_per_repeat.csv", index=False)
    if len(S):
        S.groupby(["method", "fraction"]).k.describe()[["mean", "min", "50%", "max"]].to_csv(out / "saturation.csv")

    # plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    multi = agg.cap.nunique() > 1
    for (m, cp), g in agg.groupby(["method", "cap"]):
        if m == "identity": continue
        ax.errorbar(g.k, g.skill, yerr=g.skill_sd, marker="o", capsize=3,
                    label=f"{m}, {cp or 'all'} cells/donor" if multi else m)
    cl = agg[agg.method != "identity"].ceiling.mean()
    ax.axhline(cl, ls="--", c="grey", lw=1, label=f"split-half noise ceiling (~{cl:.2f})")
    ax.set_xlabel("training donors (k)"); ax.set_ylabel(f"median skill on held-out donors\n(effects > {FLOOR})")
    ax.set_title("How many donors does a stim map need?"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / "learning_curve.png", dpi=150)
    print(agg.round(3).to_string(index=False))
    print(f"[DONE ] -> {out}")


TRAJ = []

if __name__ == "__main__":
    main()
