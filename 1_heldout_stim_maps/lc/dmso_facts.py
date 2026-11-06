"""Part 1 of notebooks/dmso_plate.ipynb: how much plate-to-plate difference is in the stim data, per stim.

Input (read-only): /scratch/users/bmerwan/well_ceiling/work/quant/<celltype>.npz (wc/extract.py, 2026-10-06): per
(condition, donor) the q10..q90 of the 13 functional markers in
  pool        the learning-curve cache cells (DMSO1/2/3 pooled, <= 10k), with w:<plate>|n = cells of each plate in it;
  W:<plate>   up to 30k cells of ONE plate-control well (DMSO1, DMSO2 or DMSO3);
  PW:<plate>  "pseudo-wells": the same cells re-partitioned at random into groups of the W sizes (sampling noise only).
For each (stim, celltype) of the 168-pair grid, donor, quantile, marker:
  R      = pool stim - pool Unstim                 (the response the learning curve scores)
  r_p    = W stim_p - W Unstim_p                   (plate-matched response on plate p)
  ru_p   = W stim_p - mean_p W Unstim_p            (plate-unmatched: stim of one plate, Unstim of all three;
           across plates its variance is that of W stim_p)
  var_*  = variance across the 3 plates (ddof 1) of r_p / ru_p / W stim_p / W Unstim_p; the same on PW = sampling part;
           plate part = real - pseudo.
  dev_p  = r_p - mean of the two other plates' r   (how far plate p sits from the other two)
Across donors: between-donor variance of R (ddof 1). Live features: |mean over donors of R| > FLOOR.
Outputs WORK/dmso/facts/: by_stim.csv, by_stim_plate.csv (dev_p), shares.csv (plate shares of the pooled cache),
cases.pkl.gz (per feature x donor).
    python -m lc.dmso_facts
"""
import numpy as np
import pandas as pd

from lc.paths import DONORS, MARKERS, PAIRS, QS, WORK
from lc.score import FLOOR

QUANT = "/scratch/users/bmerwan/well_ceiling/work/quant"
OUT = WORK / "dmso" / "facts"
PL = ("DMSO1", "DMSO2", "DMSO3")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, shares = [], []
    for c in sorted({c for _, c in PAIRS}):
        z = np.load(f"{QUANT}/{c}.npz"); have = set(z.files)
        for s in sorted({s for s, cc in PAIRS if cc == c}):
            for d in DONORS:
                if not all(f"{x}|{d}|W:{p}" in have for x in ("Unstim", s) for p in PL): continue
                if f"{s}|{d}|pool" not in have or f"Unstim|{d}|pool" not in have: continue
                R = z[f"{s}|{d}|pool"] - z[f"Unstim|{d}|pool"]
                Sw = np.stack([z[f"{s}|{d}|W:{p}"] for p in PL]); Uw = np.stack([z[f"Unstim|{d}|W:{p}"] for p in PL])
                PS = np.stack([z[f"{s}|{d}|PW:{p}"] for p in PL]); PU = np.stack([z[f"Unstim|{d}|PW:{p}"] for p in PL])
                r, pr = Sw - Uw, PS - PU
                ru, pru = Sw - Uw.mean(0), PS - PU.mean(0)
                v = lambda a: a.var(0, ddof=1).ravel()
                dev = np.stack([r[i] - r[[j for j in range(3) if j != i]].mean(0) for i in range(3)])
                pdev = np.stack([pr[i] - pr[[j for j in range(3) if j != i]].mean(0) for i in range(3)])
                n = {p: int(z[f"{s}|{d}|w:{p}|n"]) if f"{s}|{d}|w:{p}|n" in have else 0 for p in PL}
                nu = {p: int(z[f"Unstim|{d}|w:{p}|n"]) if f"Unstim|{d}|w:{p}|n" in have else 0 for p in PL}
                shares.append({"celltype": c, "stim": s, "donor": d, **{f"stim_{p}": n[p] for p in PL},
                               **{f"Unstim_{p}": nu[p] for p in PL},
                               **{f"W_{p}": int(z[f"{s}|{d}|W:{p}|n"]) for p in PL}})
                sh = np.array([n[p] for p in PL], float); sh = sh / sh.sum()
                row = {"celltype": c, "stim": s, "donor": d, "q": np.repeat(QS, len(MARKERS)),
                       "marker": np.tile(MARKERS, len(QS)), "R": R.ravel(),
                       "var_r": v(r), "var_r_pseudo": v(pr), "var_ru": v(ru), "var_ru_pseudo": v(pru),
                       "var_S": v(Sw), "var_S_pseudo": v(PS), "var_U": v(Uw), "var_U_pseudo": v(PU),
                       "sum_share2": float((sh ** 2).sum())}
                for i, p in enumerate(PL):
                    row[f"r_{p}"] = r[i].ravel(); row[f"dev_{p}"] = dev[i].ravel(); row[f"pdev_{p}"] = pdev[i].ravel()
                rows.append(pd.DataFrame(row))
        print(c, flush=True)
    D = pd.concat(rows, ignore_index=True)
    F = ["stim", "celltype", "q", "marker"]
    g = D.groupby(F).R
    D["R_mean"] = g.transform("mean"); D["var_between"] = g.transform("var")
    D["live"] = D.R_mean.abs() > FLOOR
    for col in D.select_dtypes("float64"): D[col] = D[col].astype("float32")
    D.to_pickle(OUT / "cases.pkl.gz")
    L = D[D.live]
    rms = lambda x: float(np.sqrt(np.mean(x)))

    def summ(x):
        vp = (x.var_r - x.var_r_pseudo).clip(lower=0)
        return pd.Series({
            "features": x.groupby(["celltype", "q", "marker"]).ngroups, "cases": len(x),
            "response |R| median": x.R.abs().median(),
            "between-donor SD of R": rms(x.drop_duplicates(F).var_between),
            "plate SD of r (matched)": rms(x.var_r), "sampling SD of r (pseudo-wells)": rms(x.var_r_pseudo),
            "plate-only SD of r": rms(vp),
            "plate SD of stim level": rms(x.var_S), "plate-only SD of stim level": rms((x.var_S - x.var_S_pseudo).clip(lower=0)),
            "plate SD of Unstim level": rms(x.var_U), "plate-only SD of Unstim level": rms((x.var_U - x.var_U_pseudo).clip(lower=0)),
            "median plate-only SD of r": float(np.median(np.sqrt((x.var_r - x.var_r_pseudo).clip(lower=0)))),
            "median between-donor SD of R": float(np.median(np.sqrt(x.drop_duplicates(F).var_between))),
            "plate-only var / between-donor var": float(vp.mean() / x.drop_duplicates(F).var_between.mean()),
            "plate var in pooled truth / between var": float((vp * x.sum_share2).mean() / x.drop_duplicates(F).var_between.mean()),
            "matched / unmatched plate var": float(x.var_r.mean() / x.var_S.mean()),
            "matched / unmatched plate-only var": float((x.var_r - x.var_r_pseudo).mean() / (x.var_S - x.var_S_pseudo).mean()),
        })
    T = L.groupby("stim")[list(L.columns)].apply(summ); T.loc["all"] = summ(L)
    T.to_csv(OUT / "by_stim.csv")
    pr = []
    for p in PL:
        for s, x in list(L.groupby("stim")) + [("all", L)]:
            sysm = x.groupby(F)[f"dev_{p}"].mean()                  # common to donors (systematic plate offset)
            far = np.stack([x[f"dev_{q}"].abs() for q in PL], 1).argmax(1) == PL.index(p)
            pr.append({"stim": s, "plate": p, "RMS dev from other two": rms(x[f"dev_{p}"] ** 2),
                       "RMS dev, pseudo-wells": rms(x[f"pdev_{p}"] ** 2),
                       "RMS systematic (mean over donors)": rms(sysm ** 2),
                       "mean signed dev x sign(R)": float((x[f"dev_{p}"] * np.sign(x.R_mean)).mean()),
                       "share of cases where this plate is the odd one": float(far.mean())})
    pd.DataFrame(pr).to_csv(OUT / "by_stim_plate.csv", index=False)
    S = pd.DataFrame(shares); S.to_csv(OUT / "shares.csv", index=False)
    print(T.round(4).to_string())


if __name__ == "__main__":
    main()
