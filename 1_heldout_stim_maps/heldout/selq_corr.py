"""rule_selq + correlation filter, in two orderings, with a record of what each kept column stands for.

Input: quantile_arms/tables/rule_selq.csv (simulated columns whose own quantile shift passes 0.1, plus the measured
freq / 31-marker medians / functional quantiles; 36_G1 and 52_G1 already removed). Constant columns are dropped first.
Greedy filter: columns are visited in a fixed order; a column is kept unless |r| >= 0.9 (across women) with a column
already kept, in which case it is recorded as absorbed by the kept column it correlates with most.
  selq_corr      same order as rule_corr: medians (simulated and measured) > frequencies > quantiles, then variance
  selq_corr_eff  measured medians > frequencies (both by variance) > simulated columns, median or quantile alike, by
                 stim effect size (mean over atlas donors of |per-donor quantile shift|, heldout.inclusion) >
                 measured quantiles (by variance). Inside a group of near-copies, the name kept is then the stim /
                 quantile that moves most, and a simulated column that duplicates a measured one is dropped.
None of this looks at DOS. Outputs: tables/<set>.csv, groups_<set>.csv (kept, absorbed, r, effect sizes), blocks.json.
    python -m heldout.selq_corr
"""
import json
import re

import numpy as np
import pandas as pd

from heldout.common import STIMS, WORK

Q = WORK / "quantile_arms"
R_MAX = 0.9


def kind(c):
    last = c.split("_")[-1]
    if last in STIMS: return "sim median"
    if re.search(r"q$", last) and last[:-1] in STIMS: return "sim quantile"
    if c.endswith("_freq"): return "freq"
    if c.endswith("_med"): return "meas median"
    return "meas quantile"


def effect_sizes(cols):
    inc = pd.read_csv(WORK / "features/inclusion.csv")
    eff = {}
    for r in inc.itertuples(index=False):
        eff[f"{r.celltype}_{r.marker}_{r.stim}"] = r.shift_q50
        for q in (10, 25, 75, 90): eff[f"{r.celltype}_{r.marker}_q{q}_{r.stim}q"] = getattr(r, f"shift_q{q}")
    return pd.Series({c: eff.get(c, np.nan) for c in cols})


def greedy(X, order):
    Z = ((X - X.mean()) / X.std(ddof=0)).fillna(0)
    kept, K, absorbed = [], np.zeros((len(Z), 0)), []
    for c in order:
        z = Z[c].to_numpy()
        if K.shape[1]:
            r = z @ K / len(z); j = int(np.abs(r).argmax())
            if abs(r[j]) >= R_MAX: absorbed.append((kept[j], c, float(r[j]))); continue
        kept.append(c); K = np.c_[K, z]
    return kept, pd.DataFrame(absorbed, columns=["kept", "absorbed", "r"])


def main():
    T = pd.read_csv(Q / "tables/rule_selq.csv", index_col=0)
    const = T.columns[T.std(ddof=0) == 0]; X = T.drop(columns=const)
    k = pd.Series({c: kind(c) for c in X}); var = X.var(); eff = effect_sizes(X.columns)
    orders = {
        "selq_corr": sorted(X, key=lambda c: (0 if k[c] in ("sim median", "meas median") else 1 if k[c] == "freq" else 2,
                                              -var[c])),
        "selq_corr_eff": sorted(X, key=lambda c: ({"meas median": 0, "freq": 1, "sim median": 2, "sim quantile": 2,
                                                   "meas quantile": 3}[k[c]],
                                                  -(eff[c] if k[c].startswith("sim") else var[c]))),
    }
    info = json.load(open(Q / "blocks.json"))
    blocks = STIMS + [f"{s}q" for s in STIMS] + ["freq", "med", "qtl"]
    for name, order in orders.items():
        kept, A = greedy(X, order)
        A["kind_kept"] = A.kept.map(k); A["kind_absorbed"] = A.absorbed.map(k)
        A["effect_kept"] = A.kept.map(eff); A["effect_absorbed"] = A.absorbed.map(eff)
        A = pd.concat([A, pd.DataFrame({"kept": "(constant, dropped)", "absorbed": const})], ignore_index=True)
        A.to_csv(Q / f"groups_{name}.csv", index=False)
        # Stabl's per-block preprocessing drops columns with variance < 0.01 and crashes on a block left empty; drop such
        # blocks here (same model, since Stabl would discard those columns anyway)
        v = T[kept].var(); dead = []
        for b in blocks:
            cols = [c for c in kept if c.endswith(b)]
            if cols and not (v[cols] > 0.01).any():
                dead += cols; print(f"   dropping block {b}: {len(cols)} columns, max variance {v[cols].max():.4f}")
        kept = [c for c in kept if c not in set(dead)]
        t = T[kept]; t.index.name = "Individual"; t.to_csv(Q / f"tables/{name}.csv")
        fam = pd.Series([kind(c) for c in kept]).value_counts().to_dict()
        info[name] = {"blocks": [b for b in blocks if any(c.endswith(b) for c in kept)], "n_features": len(kept), **fam}
        g = A[A.kept != "(constant, dropped)"]
        print(f"{name:14s} {T.shape[1]} columns -> {len(kept)} kept ({len(const)} constant, {len(g)} absorbed); {fam}")
        print("   absorbed (rows) by kept (columns):")
        print(pd.crosstab(g.kind_absorbed, g.kind_kept).to_string().replace("\n", "\n   "))
    json.dump(info, open(Q / "blocks.json", "w"), indent=1)


if __name__ == "__main__":
    main()
