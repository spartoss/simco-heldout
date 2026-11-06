"""OOL visits as a per-woman longitudinal dataset.

One row per visit (175 after removing 36_G1 / 52_G1): woman, visit (G1/G2/G3), EGA (weeks at the draw), and the
per-woman target GA_del = gestational age at delivery in days = 7*EGA - DOS (DOS = days from draw to labour onset,
negative), identical across a woman's visits to within ~0.2 days. Clinical covariates from the OOL clinical table.
Feature sets (measured Unstim, no CellOT map), from simco_work/onset_test/night2/distrib_all_v2.csv:
  freqmed   log cell frequencies + medians of the 31 markers (896 columns)
  freqmedq  + functional-marker quantiles (2352)
  sim       the paper's 852 CellOT-simulated stim medians (its Fig 5b input, no measured features)
  freqmed_sim  freqmed + sim (1748)
All inputs are read-only from simco_work; nothing is written there.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SIMCO_WORK = Path("/scratch/users/bmerwan/simco_work/onset_test")
WORK = Path("/scratch/users/bmerwan/ool_longitudinal/work")
EXCL = ["36_G1", "52_G1"]       # mishandled visits: their Unstim tubes were stimulated (see simco CLAUDE.md)
CLIN = ["Age", "BMI", "Gravida", "Para"]


def visits():
    c = pd.read_csv(SIMCO_WORK / "Preprocessed_OOL_Clinical_new.csv")
    c = c[~c.ID.isin(EXCL)].copy()
    c["woman"] = c.Patient.astype(str); c["visit"] = c.Timepoint
    c["GA_del"] = 7 * c.EGA - c.DOS
    c["GA_del"] = c.groupby("woman").GA_del.transform("mean")          # one target per woman
    c["preterm"] = (c.Delivery == "preterm").astype(int)
    return c.set_index("ID")[["woman", "visit", "EGA", "DOS", "GA_del", "preterm"] + CLIN]


SIM = SIMCO_WORK / "ina_13OG_final_long_allstims_filtered_no_Unstim.csv"   # the paper's 852 CellOT-simulated stim medians


def features(kind="freqmed"):
    m = pd.read_csv(SIMCO_WORK / "night2/distrib_all_v2.csv", index_col=0)
    parts = []
    if kind in ("freqmed", "freqmedq", "freqmed_sim"):
        suf = ("_freq", "_med", "_qtl") if kind == "freqmedq" else ("_freq", "_med")
        parts.append(m[[c for c in m.columns if c.endswith(suf)]])
    if kind in ("sim", "freqmed_sim"):
        parts.append(pd.read_csv(SIM, index_col=0))
    X = pd.concat(parts, axis=1, join="inner")
    return X.drop(index=EXCL, errors="ignore")


def women(V):
    return V.groupby("woman").agg(GA_del=("GA_del", "first"), preterm=("preterm", "first"),
                                  visits=("visit", lambda x: ",".join(sorted(x))), n=("visit", "size"))
