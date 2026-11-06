"""Are the OOL women's Unstim cells inside the range the stim maps were trained on?

Atlas = the 19 drug-screen donors' Unstim cells (cache_stim_all: plate controls pooled, uncorrected, <= 10k per donor;
the stim maps' training input). OOL = the 175 women's Unstim cells (cache_ool; 36_G1 / 52_G1 excluded), the input the
maps are applied to. Per (celltype, marker), 13 functional markers:
  atlas_q50 / ool_q50      median over donors (women) of each donor's (woman's) median
  shift_q50                ool_q50 - atlas_q50 (arcsinh); also at q10 and q90
  z                        shift_q50 / SD of the atlas donors' medians (how many donor-to-donor spreads away)
  out_cells                median over women of the share of her cells outside the atlas's pooled 0.5-99.5 % range
  out_low / out_high       the same, split by side
Plus, per simulated OOL column, whether it is implausible (outside [-3, 12]) and the input shift of its (celltype, marker).
    python -m heldout.atlas_vs_ool -> WORK/atlas_vs_ool/
"""
import glob

import numpy as np
import pandas as pd

from heldout.common import CACHE_STIM_ALL, WORK
from lc.paths import CELLTYPES, MARKERS

O = WORK / "atlas_vs_ool"
EXCL = {"36_G1", "52_G1"}
QS = (10, 50, 90)


def main():
    O.mkdir(parents=True, exist_ok=True)
    ool = {f.split("/")[-1][:-4]: np.load(f) for f in sorted(glob.glob(str(WORK / "cache_ool/*.npz")))}
    ool = {w: z for w, z in ool.items() if w not in EXCL}
    rows = []
    for ct in CELLTYPES:
        z = np.load(CACHE_STIM_ALL / f"{ct}.npz")
        A = {k.split("|")[1]: z[k] for k in z.files if k.startswith("Unstim|")}
        pooled = np.concatenate(list(A.values())); lo, hi = np.percentile(pooled, [0.5, 99.5], axis=0)
        aq = np.array([np.percentile(x, QS, axis=0) for x in A.values() if len(x) >= 20])          # donors x 3 x 13
        W = {w: zz[ct] for w, zz in ool.items() if ct in zz.files and len(zz[ct]) >= 20}
        oq = np.array([np.percentile(x, QS, axis=0) for x in W.values()])
        lo_s = np.array([(x < lo).mean(0) for x in W.values()]); hi_s = np.array([(x > hi).mean(0) for x in W.values()])
        n_ool = np.median([len(x) for x in W.values()]); n_atl = np.median([len(x) for x in A.values()])
        for j, m in enumerate(MARKERS):
            r = {"celltype": ct, "marker": m, "n_women": len(W), "cells_per_woman": n_ool, "cells_per_donor": n_atl}
            for qi, q in enumerate(QS):
                r[f"atlas_q{q}"] = np.median(aq[:, qi, j]); r[f"ool_q{q}"] = np.median(oq[:, qi, j])
                r[f"shift_q{q}"] = r[f"ool_q{q}"] - r[f"atlas_q{q}"]
            sd = aq[:, 1, j].std(ddof=1); r["atlas_donor_sd"] = sd; r["z"] = r["shift_q50"] / sd if sd > 0 else np.nan
            r["out_low"] = np.median(lo_s[:, j]); r["out_high"] = np.median(hi_s[:, j]); r["out_cells"] = r["out_low"] + r["out_high"]
            rows.append(r)
        print(f"[DONE ] {ct}: {len(A)} donors, {len(W)} women", flush=True)
    T = pd.DataFrame(rows); T.to_csv(O / "shift_by_celltype_marker.csv", index=False)

    # simulated OOL columns: implausible values vs the input shift of their (celltype, marker)
    sim = pd.read_csv(WORK / "features/ool_cellot_allq.csv", index_col=0).drop(list(EXCL), errors="ignore")
    bad = (sim.min() < -3) | (sim.max() > 12)
    recs = []
    for c in sim.columns:
        parts = c.split("_")
        stim = parts[-1].rstrip("q") if parts[-1].endswith("q") and parts[-1] != "Unstim" else parts[-1]
        q = 50; core = parts[:-1]
        if len(core) >= 2 and core[-1].startswith("q") and core[-1][1:].isdigit(): q = int(core[-1][1:]); core = core[:-1]
        m = core[-1]; ct = "_".join(core[:-1])
        recs.append({"column": c, "celltype": ct, "marker": m, "stim": stim, "q": q, "implausible": bool(bad[c]),
                     "min": sim[c].min(), "max": sim[c].max()})
    S = pd.DataFrame(recs).merge(T[["celltype", "marker", "shift_q50", "z", "out_cells"]], on=["celltype", "marker"], how="left")
    S.to_csv(O / "simulated_columns.csv", index=False)
    print(T[["shift_q50", "z", "out_cells"]].describe().round(3).to_string())
    print("implausible columns:", int(S.implausible.sum()), "| median out_cells, implausible vs plausible:",
          round(S[S.implausible].out_cells.median(), 3), round(S[~S.implausible].out_cells.median(), 3))


if __name__ == "__main__":
    main()
