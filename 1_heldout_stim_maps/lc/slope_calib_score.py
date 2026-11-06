"""Score WORK/slope_calib/cases.pkl with lc.dnorm_slope_score.readout (all features, rule set, rule set w/o PI / IL33)."""
import numpy as np
import pandas as pd

import lc.dnorm_slope_score as dss
from heldout.common import WORK as HW
from lc.score import FLOOR
from lc.slope_calib import OUT

dss.ARMS = ["shift", "paired", "ens", "force1", "calib", "force1_ens", "calib_ens"]


def main():
    D = pd.read_pickle(OUT / "cases.pkl"); D["eff"] = (D.t - D.u).abs(); D = D[D.eff > FLOOR].copy()
    inc = pd.read_csv(HW / "features/inclusion.csv")
    L = inc.melt(id_vars=["stim", "celltype", "marker"], value_vars=[f"shift_q{q}" for q in (10, 25, 50, 75, 90)],
                 var_name="q", value_name="rule_eff"); L["q"] = L.q.str[7:].astype(int)
    D = D.merge(L, on=["stim", "celltype", "marker", "q"], how="left")
    rule = D.q.isin([50, 75, 90]) & (D.rule_eff > 0.1)
    pd.set_option("display.width", 250)
    for name, X in [("all", D), ("rule", D[rule]), ("rule_noPI_IL33", D[rule & ~D.stim.isin(["PI", "IL33"])])]:
        R = dss.readout(X); R.round(4).to_csv(OUT / f"score_{name}.csv")
        print(f"\n=== {name}: {len(X)} cases"); print(R.round(4).to_string())
    # specific R2 (departure from the shift, centred across test donors) on the rule set
    X = D[rule].copy(); g = [X[c] for c in dss.G]; X["n"] = X.groupby(dss.G).t.transform("size"); X = X[X.n >= 3]; g = [X[c] for c in dss.G]
    r = X.t - X["shift"]; rc = r - r.groupby(g).transform("mean")
    print("\nspecific R2 (rule set):", {m: round(1 - (((X[m] - X["shift"]) - (X[m] - X["shift"]).groupby(g).transform("mean") - rc) ** 2).sum() / (rc ** 2).sum(), 4)
                                         for m in dss.ARMS if m != "shift"})


if __name__ == "__main__":
    main()
