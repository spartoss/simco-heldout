"""What the quantile feature sets actually select: which quantiles, what each stands for, how much it moves.

For each set (rule_rawq, rule_selq, rule_corr, selq_corr, selq_corr_eff), from its 100-split Stabl ALasso selections:
  selection frequency of every column, and for every column selected at least once:
    kind            sim median / sim quantile / measured median / measured quantile / frequency
    stim, celltype, marker, q
    shift_q, shift_q50   atlas stim effect at that quantile and at the median (mean over donors of |per-donor shift|,
                         heldout/inclusion.py); ratio = shift_q / shift_q50 (how much more the quantile moves)
    r_own_median    correlation across the 175 women with its own median (simulated: <ct>_<m>_<stim>; measured:
                    <ct>_<m>_med), i.e. "is it just the median again?"
    group_size, group_r_median, group_members   for the correlation-filtered sets: the columns it absorbed (|r| >= 0.9)
                    and their median |r| with it. rule_corr has no stored record, so its filter is replayed here exactly
                    (same order and threshold as heldout.quantile_arms); selq_corr* read groups_<set>.csv.
    implausible     simulated values outside [-3, 12] arcsinh (heldout.atlas_vs_ool)
Writes WORK/quantile_arms/selection_report/{features_<set>.csv, by_q.csv, by_stim.csv}.
    python -m heldout.quantile_selection_report
"""
import ast
import re

import numpy as np
import pandas as pd

from heldout.common import STIMS, WORK

Q = WORK / "quantile_arms"
O = Q / "selection_report"
SETS = ["rule_rawq", "rule_selq", "rule_corr", "selq_corr", "selq_corr_eff"]
R_MAX = 0.9


def parse(c):
    p = c.split("_"); last = p[-1]
    if c.endswith("_freq"): return {"kind": "frequency", "celltype": c[:-5], "marker": None, "stim": None, "q": None}
    if last in STIMS: return {"kind": "sim median", "celltype": "_".join(p[:-2]), "marker": p[-2], "stim": last, "q": 50}
    if c.endswith("_med"): return {"kind": "measured median", "celltype": "_".join(p[:-2]), "marker": p[-2], "stim": "Unstim", "q": 50}
    m = re.match(r"(.*)_([^_]+)_q(\d+)_(\w+)$", c)
    if m:
        ct, mk, q, tail = m.groups()
        if tail == "qtl": return {"kind": "measured quantile", "celltype": ct, "marker": mk, "stim": "Unstim", "q": int(q)}
        if tail.endswith("q") and tail[:-1] in STIMS:
            return {"kind": "sim quantile", "celltype": ct, "marker": mk, "stim": tail[:-1], "q": int(q)}
    return {"kind": "other", "celltype": None, "marker": None, "stim": None, "q": None}


def own_median(c, info):
    if info["kind"] == "sim quantile": return f"{info['celltype']}_{info['marker']}_{info['stim']}"
    if info["kind"] == "measured quantile": return f"{info['celltype']}_{info['marker']}_med"
    return None


def replay_rule_corr():
    """Exactly heldout.quantile_arms' rule_corr filter, recording which kept column absorbed each dropped one."""
    full = pd.read_csv(Q / "tables/rule_rawq.csv", index_col=0)
    prio = lambda c: 0 if (c.endswith("_med") or c.split("_")[-1] in STIMS) else (1 if c.endswith("_freq") else 2)
    Z = (full - full.mean()) / full.std(ddof=0).replace(0, np.nan); Z = Z.dropna(axis=1).fillna(0)
    order = sorted(Z.columns, key=lambda c: (prio(c), -full[c].var()))
    kept, K, rec = [], np.zeros((len(Z), 0)), []
    for c in order:
        z = Z[c].to_numpy()
        if K.shape[1]:
            r = z @ K / len(z); j = int(np.abs(r).argmax())
            if abs(r[j]) >= R_MAX: rec.append((kept[j], c, float(r[j]))); continue
        kept.append(c); K = np.c_[K, z]
    check = pd.read_csv(Q / "tables/rule_corr.csv", index_col=0, nrows=1).columns
    assert set(kept) == set(check), "replay does not reproduce rule_corr"
    return pd.DataFrame(rec, columns=["kept", "absorbed", "r"])


def main():
    O.mkdir(parents=True, exist_ok=True)
    inc = pd.read_csv(WORK / "features/inclusion.csv")
    shift = {}
    for r in inc.itertuples(index=False):
        for q in (10, 25, 50, 75, 90): shift[(r.celltype, r.marker, r.stim, q)] = getattr(r, f"shift_q{q}")
    sim = pd.read_csv(WORK / "features/ool_cellot_allq.csv", index_col=0).drop(["36_G1", "52_G1"], errors="ignore")
    meas = pd.read_csv("/scratch/users/bmerwan/simco_work/onset_test/night2/distrib_all_v2.csv", index_col=0).reindex(sim.index)
    allcols = pd.concat([sim, meas], axis=1); allcols = allcols.loc[:, ~allcols.columns.duplicated()]
    bad = set(pd.read_csv(WORK / "atlas_vs_ool/simulated_columns.csv").query("implausible").column)
    groups = {"rule_corr": replay_rule_corr()}
    for s in ("selq_corr", "selq_corr_eff"):
        g = pd.read_csv(Q / f"groups_{s}.csv"); groups[s] = g[g.kept != "(constant, dropped)"][["kept", "absorbed", "r"]]
    byq, bys = [], []
    for s in SETS:
        sel = pd.read_csv(Q / f"stabl/{s}/Training CV/Selected Features STABL ALasso.csv", index_col=0)
        L = [ast.literal_eval(x) for x in sel["Fold selected features"]]
        freq = pd.Series([c for l in L for c in l]).value_counts() / len(L)
        rows = []
        for c, f in freq.items():
            info = parse(c); om = own_median(c, info)
            r = {"column": c, "sel_freq": f, **info}
            if info["kind"].startswith("sim") and info["q"] is not None:
                r["shift_q"] = shift.get((info["celltype"], info["marker"], info["stim"], info["q"]))
                r["shift_q50"] = shift.get((info["celltype"], info["marker"], info["stim"], 50))
                r["ratio_vs_median"] = r["shift_q"] / r["shift_q50"] if r["shift_q"] is not None and r["shift_q50"] else np.nan
            if om is not None and om in allcols and c in allcols:
                r["r_own_median"] = allcols[c].corr(allcols[om])      # pairwise, skips women without that celltype
            if s in groups:
                g = groups[s][groups[s].kept == c]
                r["group_size"] = len(g); r["group_r_median"] = g.r.abs().median() if len(g) else np.nan
                r["group_members"] = ", ".join(g.sort_values("r", key=abs, ascending=False).absorbed.head(6))
            r["implausible"] = c in bad
            rows.append(r)
        F = pd.DataFrame(rows); F.to_csv(O / f"features_{s}.csv", index=False)
        w = F.assign(qq=F.q.fillna(-1)).groupby(["kind", "qq"]).sel_freq.sum()
        byq.append((w / F.sel_freq.sum()).rename(s))
        sq = F[F.kind == "sim quantile"]
        bys.append((sq.groupby("stim").sel_freq.sum() / F.sel_freq.sum()).rename(s))
        top = F[F.kind == "sim quantile"].sort_values("sel_freq", ascending=False).head(8)
        print(f"\n{s}: {len(F)} columns ever selected; selections by kind:",
              (F.groupby('kind').sel_freq.sum() / F.sel_freq.sum()).round(2).to_dict())
        print(top[["column", "sel_freq", "ratio_vs_median", "r_own_median"] + (["group_size"] if s in groups else [])].round(2).to_string(index=False))
    pd.concat(byq, axis=1).to_csv(O / "by_q.csv"); pd.concat(bys, axis=1).to_csv(O / "by_stim.csv")
    print(f"[DONE ] {O}")


if __name__ == "__main__":
    main()
