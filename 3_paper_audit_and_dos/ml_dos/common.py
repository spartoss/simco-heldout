"""Shared data, splits and scoring for the supervised DOS benchmark (measured features only).

Splits are the paper protocol's, so every model here is paired split-by-split with the CellOT +
Stabl reference: ``GroupShuffleSplit(100, test_size=0.2, random_state=42)`` grouped by woman, over
the rows in the order of the paper's feature table (the order fixes the draws). 36_G1 / 52_G1 (Unstim
tubes actually stimulated, CLAUDE.md) are then dropped from both sides of every split -- the same
convention as ``aligned_rescore.py --exclude``, which produced the reference's 0.415.

A woman's visits (G1/G2/G3) are never split between train and test, and every inner tuning split is
a GroupKFold on women too: a model can never see another visit of a test woman.

Inputs never include timing: no EGA, no visit/timepoint, no GA at delivery or anything derived from
DOS (EGA_DOS, DOS_diff, Delivery). ``n_cells_total`` is also left out (acquisition, not biology).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

O = Path("/scratch/users/bmerwan/simco_work/onset_test")
OUT = O / "ml_dos"
MEASURED = OUT / "measured_unstim_full.csv"
PAPER_TABLE = O / "ina_13OG_final_long_allstims_filtered.csv"
EXCLUDE = ("36_G1", "52_G1")
N_SPLITS = 100
CLINICAL_OK = ["Age", "BMI", "Gravida", "Para"]          # no EGA / GA / Timepoint / DOS-derived
STIM15 = ["LPS", "IL246", "IFNa", "GMCSF"]                 # OOL measured 15-min stims (filled ~98 %)

QBLOCKS = ["q10", "q25", "q50", "q75", "q90"]
ARMS = {
    # name: (blocks of measured_unstim_full.csv, extras)
    "freqmed": (["freq", "q50"], []),
    "qtl": (["freq"] + QBLOCKS, []),
    "full": (["freq"] + QBLOCKS + ["mean", "sd", "corr"], []),
    "full_plus": (["freq"] + QBLOCKS + ["mean", "sd", "corr"], ["stim15", "clinical"]),
    # ablations of full_plus (2026-10-07): which add-on carries the gain
    "full_stim15": (["freq"] + QBLOCKS + ["mean", "sd", "corr"], ["stim15"]),
    "full_clin": (["freq"] + QBLOCKS + ["mean", "sd", "corr"], ["clinical"]),
    "stim15": ([], ["stim15"]),
    "clin": ([], ["clinical"]),
    # learner-vs-features control (2026-10-07): the same zoo on the paper's CellOT table
    "cellot": ([], ["cellot_all"]),                                            # 1216 = 852 sim stims + 364 Unstim
    "full_cellot": (["freq"] + QBLOCKS + ["mean", "sd", "corr"], ["cellot_stims"]),  # measured + 852 sim stims
}


def rows():
    """Row order of the paper's feature table (fixes the GroupShuffleSplit draws)."""
    return pd.read_csv(PAPER_TABLE, index_col=0, usecols=[0]).index


def outcome(idx):
    return pd.read_csv(O / "outcome_table_all_pre.csv", index_col=0)["DOS"].loc[idx].astype(float)


def groups(idx):
    return idx.to_series().str.split("_").str[0].values


def splits(idx):
    """100 outer splits on all 177 rows, excluded visits then removed from both sides."""
    ex = np.isin(idx, EXCLUDE)
    assert ex.sum() == len(EXCLUDE)
    out = []
    for tr, te in GroupShuffleSplit(N_SPLITS, test_size=0.2, random_state=42).split(
            np.zeros(len(idx)), groups=groups(idx)):
        out.append((tr[~ex[tr]], te[~ex[te]]))
    return out


def load_arm(arm):
    idx = rows()
    blocks, extras = ARMS[arm]
    t = pd.read_csv(MEASURED, index_col=0).loc[idx]
    X = t[[c for c in t.columns if c.rsplit("__", 1)[-1] in blocks]]
    if "stim15" in extras:
        ce = pd.read_csv(O / "cellengine_processed_features.csv", index_col=0).reindex(idx)
        X = X.join(ce[[c for c in ce.columns if c.rsplit("_", 1)[-1] in STIM15]].add_suffix("__stim15"))
    if "clinical" in extras:
        cl = pd.read_csv(O / "Preprocessed_OOL_Clinical_new.csv", index_col=0).reindex(idx)
        X = X.join(cl[CLINICAL_OK].add_suffix("__clinical"))
    if "cellot_all" in extras or "cellot_stims" in extras:
        pt = pd.read_csv(PAPER_TABLE, index_col=0).loc[idx]
        if "cellot_stims" in extras:
            pt = pt[[c for c in pt.columns if not c.endswith("_Unstim")]]
        X = X.join(pt.add_suffix("__cellot"))
    forbidden =("EGA", "Timepoint", "GA__", "DOS", "Delivery", "n_cells")
    assert not [c for c in X.columns if any(f in c for f in forbidden)], "timing leak"
    return X.astype(float), outcome(idx), groups(idx)


def pooled(pred_long):
    """Stabl's pooled out-of-fold score: per sample, median of its predictions over the splits where
    it was in test; then one R2 / RMSE / MAE over those medians (stablVMax/stabl/single_omic.py:295)."""
    med = pred_long.groupby("sample")["pred"].median()
    y = pred_long.groupby("sample")["y"].first().loc[med.index]
    return dict(R2=r2_score(y, med), RMSE=mean_squared_error(y, med) ** .5,
                MAE=mean_absolute_error(y, med), n=len(med)), med


def per_split(pred_long):
    return pd.Series({k: r2_score(d.y, d.pred) for k, d in pred_long.groupby("split")}, name="R2")
