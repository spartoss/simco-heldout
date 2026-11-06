"""Summarise the night2 arms: pooled-OOF scores + paired per-split R2 vs the CellOT reference
of the same target (identical random_state=42 splits).

  DOS         reference = CellOT `both` (cellot_both_n20/both); also listed: Unstim-only,
              gauss_ot, and EGA alone (clinical_baselines_n20.csv, not a Stabl run)
  DOS_offset  reference = o_cellot_both; the 'mean only' floor is R2 ~ 0 by construction

Writes night2/night2_scores.csv and night2/night2_paired.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ONSET = Path("/scratch/users/bmerwan/simco_work/onset_test")
N = ONSET / "night2"
runs = {  # name -> (target, dir)
    "cellot_both": ("DOS", ONSET / "cellot_both_n20/both"),
    "gaussot_both": ("DOS", ONSET / "gaussot_stabl_n20/both"),
    "unstim_only": ("DOS", ONSET / "cmp_unstim_vs_simstim_n20/unstim_only"),
}
for line in (N / "arms.txt").read_text().splitlines():
    name, target, *_ = line.split("|")
    runs[name] = (target, N / "stabl_n20" / name)
REF = {"DOS": "cellot_both", "DOS_offset": "o_cellot_both"}

scores, folds, missing = [], {}, []
for name, (target, d) in runs.items():
    s, f = d / "Summary/Scores training CV.csv", d / "Summary/fold_metrics.csv"
    if not (s.exists() and f.exists()):
        missing.append(name)
        continue
    for model, r in pd.read_csv(s, index_col=0).iterrows():
        scores.append(dict(run=name, target=target, model=model, **r.to_dict()))
    for model, g in pd.read_csv(f).groupby("model"):
        folds[(name, model)] = (target, g.set_index("fold")["r2"])

paired = []
for (name, model), (target, r2) in folds.items():
    ref = REF[target]
    if name == ref or (ref, model) not in folds:
        continue
    rr = folds[(ref, model)][1]
    common = rr.index.intersection(r2.index)
    d = (r2.loc[common] - rr.loc[common]).values
    paired.append(dict(run=name, target=target, model=model, vs=ref, n_splits=len(d),
                       mean_diff_run_minus_ref=d.mean(),
                       wilcoxon_p=stats.wilcoxon(d).pvalue if len(d) > 5 and np.any(d) else np.nan,
                       run_wins=int((d > 0).sum())))

sc = pd.DataFrame(scores)
sc.to_csv(N / "night2_scores.csv", index=False)
pp = pd.DataFrame(paired)
pp.to_csv(N / "night2_paired.csv", index=False)
with pd.option_context("display.width", 220, "display.max_rows", 200):
    if len(sc):
        print(sc[["target", "run", "model", "R2", "RMSE", "N features"]].sort_values(
            ["target", "model", "run"]).to_string(index=False))
    if len(pp):
        print(pp.round(3).to_string(index=False))
ega = ONSET / "clinical_baselines_n20.csv"
if ega.exists():
    print("\nDOS reference, EGA alone (20 splits):")
    print(pd.read_csv(ega).round(3).to_string(index=False))
print("missing:", missing or "none")
