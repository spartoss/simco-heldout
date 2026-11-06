"""Predicted vs observed days-to-onset (DOS), one panel per feature block.

Reads the stored 100-split GroupShuffleSplit runs in group home (read-only) and
writes a PNG under simco_work/. Best surviving model per arm; the model name is in
each panel subtitle.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

B = "/home/groups/gbrice/ptb-drugscreen/ool_stabl/onset_test"
OUT = "/scratch/users/bmerwan/simco_work/onset_test/pred_vs_obs_DOS.png"
MODEL = "STABL ElasticNet"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = "#2a78d6"

# Lasso is the best model on the arms that carry signal, so it is used wherever its
# predictions.csv survives; the `both` dir only kept ALasso and ElasticNet, hence the
# per-panel model name in the subtitle.
ARMS = [
    ("Unstim mesuré seul", f"{B}/results_XGB_KO_long_unstim_only/Training_CV",
     "STABL Lasso"),
    ("stims simulées (sans Unstim)", f"{B}/results_XGB_KO_long_no_unstim/Training CV",
     "STABL Lasso"),
    # identified as the paper's Fig 5b run: RMSE 24.120 [21.707, 26.854], N features
    # 21.000 [18.000, 30.250], Pearson r 0.645 -- matches the preprint's "R = 0.64,
    # RMSE = 24.1 days, 21 features (18-30), 177 samples / 63 patients".
    ("Unstim + stims simulées — papier, Fig 5b",
     f"{B}/GS_cellot_longallstims_foldinfo/Training CV", "STABL ALasso"),
]

y = pd.read_csv(
    "/scratch/users/bmerwan/simco_work/onset_test/outcome_table_all_pre.csv", index_col=0
)["DOS"].astype(float)
ymean = y.mean()

fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.4), sharex=True, sharey=True)
fig.patch.set_facecolor(SURFACE)
lim = (-120, 8)

for ax, (label, path, model) in zip(axes, ARMS):
    f = os.path.join(path, model, f"{model} predictions.csv")
    pred = pd.read_csv(f, index_col=0).iloc[:, 0].astype(float)
    idx = pred.index.intersection(y.index)
    yt, yp = y.loc[idx].values, pred.loc[idx].values

    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # reference: perfect prediction, and the constant "always the cohort mean" model
    ax.plot(lim, lim, color=AXIS, linewidth=2, zorder=1)
    ax.axhline(ymean, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)

    ax.scatter(yt, yp, s=42, color=SERIES, alpha=0.75,
               edgecolor=SURFACE, linewidth=0.9, zorder=3)

    r2 = r2_score(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    r = pearsonr(yt, yp)[0]
    med = np.median(np.abs(yt - yp))

    ax.set_title(f"{label}\n{model.replace('STABL ', '')}",
                 color=INK, fontsize=12, pad=10, loc="left")
    ax.text(0.03, 0.965,
            f"R² {r2:.2f}   r {r:.2f}\nRMSE {rmse:.1f} j\nMAE {mae:.0f} j",
            transform=ax.transAxes, va="top", ha="left",
            color=INK2, fontsize=10.5, linespacing=1.5)

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.tick_params(colors=MUTED, labelsize=9.5, length=0)
    for s in ax.spines.values():
        s.set_visible(False)

axes[0].set_ylabel("DOS prédit (jours)", color=INK2, fontsize=10.5)

# the two reference marks get a figure-level legend rather than in-plot labels,
# which collided with the point cloud
handles = [
    Line2D([], [], color=AXIS, linewidth=2, label="prédiction parfaite"),
    Line2D([], [], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)),
           label=f"modèle constant (moyenne, {ymean:.0f} j)"),
]
leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.945),
                 ncol=2, frameon=False, fontsize=10, handlelength=2.4,
                 labelcolor=INK2, borderaxespad=0)

fig.suptitle("Prédiction du délai avant début du travail — 177 échantillons / 63 patientes, "
             "100 splits",
             color=INK, fontsize=13, x=0.012, ha="left", y=0.985)
fig.supxlabel("DOS observé (jours)", color=INK2, fontsize=10.5, y=0.015)
fig.tight_layout(rect=(0, 0.055, 1, 0.885))
fig.savefig(OUT, dpi=170, facecolor=SURFACE)
print("wrote", OUT)
