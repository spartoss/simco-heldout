"""Fig 5b feature tables from the new best model's OOL predictions (lc/ool_tab.py), in the paper's 1216-column schema.

Same rows, columns and order as the paper table (ina_13OG_final_long_allstims_filtered.csv): the 364 measured Unstim
columns are copied unchanged; each of the 852 simulated stim columns <celltype>_<marker>_<stim> (a median) becomes
- tab_both: shift + r_hat at q50 (TabPFN C_ms_lin, full r_hat)
- mix_both: 0.4 x the paper's CellOT column + 0.6 x tab_both's column (the best held-out mix, with the paper's maps,
  the only CellOT maps applied to OOL)
Writes simco_work/onset_test/paper_protocol/tables/{tab_both,mix_both}.csv and prints per-column correlations with CellOT.
    python -m lc.ool_stabl_tables [--raw]   (--raw: lc/ool_tab.py --raw predictions -> tab_raw_both / mix_raw_both)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lc.paths import WORK

O = Path("/scratch/users/bmerwan/simco_work/onset_test")
PAPER = O / "ina_13OG_final_long_allstims_filtered.csv"
OUTD = O / "paper_protocol" / "tables"
W_CELLOT = 0.4


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--raw", action="store_true"); a = ap.parse_args()
    pat, sfx = ("pred_qraw_*.pkl", "_raw") if a.raw else ("pred_q_*.pkl", "")
    C = pd.read_csv(PAPER, index_col=0)
    R = pd.concat([pd.read_pickle(f) for f in sorted((WORK / "ool").glob(pat)) if "smoke" not in f.name],
                  ignore_index=True)
    R = R[R.q == 50].assign(v=lambda d: d["shift"] + d.rhat)
    R["col"] = R.celltype + "_" + R.marker + "_" + R.stim
    P = R.pivot_table(index="sample", columns="col", values="v").reindex(C.index)
    stim_cols = [c for c in C.columns if not c.endswith("_Unstim")]
    miss = [c for c in stim_cols if c not in P.columns]
    assert not miss, f"{len(miss)} stim columns not predicted, e.g. {miss[:5]}"
    tab = C.copy(); tab[stim_cols] = P[stim_cols].to_numpy()
    mix = C.copy(); mix[stim_cols] = W_CELLOT * C[stim_cols].to_numpy() + (1 - W_CELLOT) * P[stim_cols].to_numpy()
    OUTD.mkdir(parents=True, exist_ok=True)
    for name, T in [(f"tab{sfx}_both", tab), (f"mix{sfx}_both", mix)]:
        assert T.shape == C.shape and list(T.columns) == list(C.columns) and list(T.index) == list(C.index)
        T.to_csv(OUTD / f"{name}.csv"); print(f"[WRITE] {name} {T.shape}, NaN {int(T.isna().sum().sum())} "
                                             f"(paper {int(C.isna().sum().sum())})")
    r = pd.Series({c: tab[c].corr(C[c]) for c in stim_cols})
    u = pd.Series({c: tab[c].corr(C[c.rsplit("_", 1)[0] + "_Unstim"]) for c in stim_cols if c.rsplit("_", 1)[0] + "_Unstim" in C})
    print("corr(tab, CellOT) per column:", r.describe().round(3).to_dict())
    print("corr(tab, own Unstim median) per column:", u.describe().round(3).to_dict())
    print(r.groupby(r.index.str.rsplit("_", n=1).str[1]).median().round(3).to_dict())


if __name__ == "__main__":
    main()
