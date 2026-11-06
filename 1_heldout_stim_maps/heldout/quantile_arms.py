"""Feature sets for the quantile re-test (Stabl, 100 splits), all built from the same retrained CellOT stim maps.

Every set also contains the measured features of the combined model (simco night2/distrib_all_v2.csv: cell
frequencies `_freq`, 31-marker medians `_med`, and — depending on the set — the measured functional quantiles).
36_G1 and 52_G1 are removed. Simulated features use the paper's naming, so Stabl blocks are, by column suffix:
the 7 stims (medians), `<stim>q` (any other simulated feature), `freq`, `med`, `qtl` (any other measured feature).

  ref852      simulated medians of the paper's 852 (stim, celltype, marker) + measured freq / med / raw quantiles
  rule_med    simulated medians of the any-quantile-rule set (1085)        + measured freq / med / raw quantiles
  rule_rawq   rule_med + simulated q10 / q25 / q75 / q90 of the rule set
  rule_selq   only the simulated (stim, celltype, marker, quantile) whose own shift passes 0.1 (q10..q90, the median
              included only if it passes itself; 3911 columns vs 5425 for rule_rawq) + the same measured features
  rule_shape  per rule feature: median, upper spread q90 - q50, lower spread q50 - q10 (simulated), and the same
              recoding for the measured quantiles (spreads instead of raw quantiles)
  rule_rep    one representative per (celltype, marker[, stim]) among q10 / q25 / q50 / q75 / q90: the one with the
              largest variance across women (simulated and measured alike)
  rule_corr   rule_rawq filtered for redundancy: features in order medians > frequencies > spreads/quantiles (then by
              variance); a feature is kept only if |r| < 0.9 with every feature already kept
None of these choices look at DOS, so they are made once on all women without leakage.
    python -m heldout.quantile_arms -> WORK/quantile_arms/tables/<set>.csv and blocks.json
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from heldout.common import SIMCO_WORK, STIMS, WORK
from lc.paths import MARKERS

EXCL = ["36_G1", "52_G1"]
QK = (10, 25, 75, 90)
R_MAX = 0.9


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--sim", default=str(WORK / "features/ool_cellot_allq.csv"))
    ap.add_argument("--out", default=str(WORK / "quantile_arms/tables")); a = ap.parse_args()
    sim = pd.read_csv(a.sim, index_col=0)
    meas = pd.read_csv(SIMCO_WORK / "onset_test/night2/distrib_all_v2.csv", index_col=0)
    inc = pd.read_csv(WORK / "features/inclusion.csv")
    rows = sim.index.intersection(meas.index).drop(EXCL, errors="ignore")
    sim, meas = sim.loc[rows], meas.loc[rows]
    m_freq = [c for c in meas if c.endswith("_freq")]; m_med = [c for c in meas if c.endswith("_med")]
    m_q = [c for c in meas if c.endswith("_qtl")]
    paper = set(inc[inc.in_paper_852].feature); rule = inc[inc.any_quantile_rule]
    rule_feats = [f for f in rule.feature if f in sim]
    base = lambda f: f.rsplit("_", 1)                      # "<ct>_<m>", stim
    simq = lambda f, q: f"{base(f)[0]}_q{q}_{base(f)[1]}q"
    meas_base = pd.concat([meas[m_freq], meas[m_med]], axis=1)

    def mshape():
        out = {}
        for c in [c for c in m_med if c.split("_")[-2] in MARKERS]:
            b = c[:-4]
            if f"{b}_q90_qtl" in meas and f"{b}_q10_qtl" in meas:
                out[f"{b}_up_qtl"] = meas[f"{b}_q90_qtl"] - meas[c]; out[f"{b}_lo_qtl"] = meas[c] - meas[f"{b}_q10_qtl"]
        return pd.DataFrame(out)

    def rep(frame_by_q):
        # frame_by_q: {q: Series} for one group; keep the column with the largest variance across women
        return max(frame_by_q.items(), key=lambda kv: kv[1].var())

    T = {}
    T["ref852"] = pd.concat([sim[[f for f in sim.columns if f in paper]], meas_base, meas[m_q]], axis=1)
    T["rule_med"] = pd.concat([sim[rule_feats], meas_base, meas[m_q]], axis=1)
    qcols = [simq(f, q) for f in rule_feats for q in QK if simq(f, q) in sim]
    T["rule_rawq"] = pd.concat([T["rule_med"], sim[qcols]], axis=1)
    sel = [(r, q) for _, r in rule.iterrows() for q in (10, 25, 50, 75, 90) if r[f"shift_q{q}"] > 0.1]
    selcols = [r.feature if q == 50 else simq(r.feature, q) for r, q in sel]
    T["rule_selq"] = pd.concat([sim[[c for c in selcols if c in sim]], meas_base, meas[m_q]], axis=1)
    shp = {}
    for f in rule_feats:
        b, s = base(f)
        if simq(f, 90) in sim and simq(f, 10) in sim:
            shp[f"{b}_up_{s}q"] = sim[simq(f, 90)] - sim[f]; shp[f"{b}_lo_{s}q"] = sim[f] - sim[simq(f, 10)]
    T["rule_shape"] = pd.concat([sim[rule_feats], pd.DataFrame(shp), meas_base, mshape()], axis=1)
    reps = {}
    for f in rule_feats:
        cand = {50: sim[f], **{q: sim[simq(f, q)] for q in QK if simq(f, q) in sim}}
        q, s_ = rep(cand); reps[f if q == 50 else simq(f, q)] = s_
    mreps = {}
    for c in [c for c in m_med if c.split("_")[-2] in MARKERS]:
        b = c[:-4]; cand = {50: meas[c], **{q: meas[f"{b}_q{q}_qtl"] for q in QK if f"{b}_q{q}_qtl" in meas}}
        q, s_ = rep(cand); mreps[c if q == 50 else f"{b}_q{q}_qtl"] = s_
    phen_med = [c for c in m_med if c.split("_")[-2] not in MARKERS]
    T["rule_rep"] = pd.concat([pd.DataFrame(reps), meas[m_freq], meas[phen_med], pd.DataFrame(mreps)], axis=1)
    full = T["rule_rawq"]
    prio = lambda c: 0 if (c.endswith("_med") or c.split("_")[-1] in STIMS) else (1 if c.endswith("_freq") else 2)
    Z = (full - full.mean()) / full.std(ddof=0).replace(0, np.nan)
    Z = Z.dropna(axis=1).fillna(0)
    order = sorted(Z.columns, key=lambda c: (prio(c), -full[c].var()))
    kept, K = [], np.zeros((len(Z), 0))
    for c in order:
        z = Z[c].values
        if K.shape[1] and np.abs(z @ K / len(z)).max() >= R_MAX: continue
        kept.append(c); K = np.c_[K, z]
    T["rule_corr"] = full[kept]

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    blocks = STIMS + [f"{s}q" for s in STIMS] + ["freq", "med", "qtl"]
    info = {}
    for k, t in T.items():
        t = t.loc[:, ~t.columns.duplicated()]
        t.index.name = "Individual"; t.to_csv(out / f"{k}.csv")
        present = [b for b in blocks if any(c.endswith(b) for c in t.columns)]
        fam = pd.Series(["sim median" if c.split("_")[-1] in STIMS else "sim other" if re.search(r"q$", c) and c.split("_")[-1][:-1] in STIMS
                         else "freq" if c.endswith("_freq") else "meas median" if c.endswith("_med") else "meas other" for c in t.columns])
        info[k] = {"blocks": present, "n_features": t.shape[1], **fam.value_counts().to_dict()}
        print(f"{k:11s} {t.shape}  " + ", ".join(f"{a} {b}" for a, b in fam.value_counts().items()))
    json.dump(info, open(out.parent / "blocks.json", "w"), indent=1)


if __name__ == "__main__":
    main()
