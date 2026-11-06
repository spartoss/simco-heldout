"""Write notebooks/atlas_vs_ool.ipynb: are the OOL women's Unstim cells on the same scale as the atlas cells the stim
maps were trained on? Inputs from heldout/atlas_vs_ool.py (WORK/atlas_vs_ool/) and the caches.
    python -m heldout.build_atlas_vs_ool_notebook && jupyter nbconvert --to notebook --execute --inplace notebooks/atlas_vs_ool.ipynb
"""
import nbformat as nbf

from lc.paths import REPO

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Atlas vs OOL: the maps are applied to cells they were not trained on (2026-10-01)

Every simulated feature is `map(OOL woman's Unstim cells)`, with maps trained on the **atlas** donors' Unstim → stim
cells. This notebook compares the two inputs, per celltype × functional marker (the maps' 13 inputs):
- *atlas*: the 19 drug-screen donors' Unstim cells (`cache_stim_all`: plate controls pooled, uncorrected, ≤ 10k per donor);
- *OOL*: the 175 women's Unstim cells (`cache_ool`, 36_G1 and 52_G1 excluded).

Statistics (`heldout/atlas_vs_ool.py` → `work/heldout/atlas_vs_ool/`): median over donors / women of each one's median
(and q10, q90); `shift` = OOL − atlas in arcsinh; `z` = shift / SD of the atlas donors' medians; `out_cells` = median over
women of the share of her cells outside the atlas's pooled 0.5–99.5 % range.

It started from an anomaly: 113 simulated OOL columns from the retrained CellOT maps reach impossible values (e.g.
`CD56hiCD16negNK_pNFkB_q10_TNFaq` down to −50; median over women −3.2), because many of those women's NK cells have
pNFkB far below anything in the atlas, where the network extrapolates linearly.
""")
code(r"""
import sys; sys.path.insert(0, "/scratch/users/bmerwan/atlas_heldout")
import glob, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display
from heldout.common import WORK, CACHE_STIM_ALL
from lc.paths import CELLTYPES, MARKERS
pd.set_option("display.width", 220)
DIV = LinearSegmentedColormap.from_list("div", ["#eb6834", "#f2f1ec", "#2a78d6"])
T = pd.read_csv(WORK / "atlas_vs_ool/shift_by_celltype_marker.csv")
S = pd.read_csv(WORK / "atlas_vs_ool/simulated_columns.csv")
""")
md("## 1. The shift, per celltype and marker")
code(r"""
X = T.pivot_table(index="celltype", columns="marker", values="shift_q50").reindex(index=CELLTYPES, columns=MARKERS)
fig, ax = plt.subplots(figsize=(9, 9))
im = ax.imshow(X.to_numpy(), cmap=DIV, vmin=-2, vmax=2, aspect="auto")
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        v = X.iat[i, j]
        if abs(v) >= 0.5: ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5, color="white" if abs(v) > 1.3 else "#222222")
ax.set_xticks(range(len(MARKERS))); ax.set_xticklabels(MARKERS, rotation=60, ha="right"); ax.set_yticks(range(len(CELLTYPES)))
ax.set_yticklabels(CELLTYPES, fontsize=8); fig.colorbar(im, ax=ax, fraction=0.04).set_label("OOL − atlas median (arcsinh)")
ax.set_title("OOL Unstim vs atlas Unstim (orange: OOL lower)", fontsize=10); plt.show()
display(T.groupby("marker")[["shift_q10", "shift_q50", "shift_q90", "z", "out_low", "out_high"]].median()
        .sort_values("shift_q50").round(2).style.set_caption("Per marker, median over the 28 celltypes"))
""")
md(r"""
## 2. Pregnancy does not explain it

The atlas includes 11 pregnant donors (PV). If the OOL shift were pregnancy, OOL would sit close to the atlas PV donors.
""")
code(r"""
ool = {f.split("/")[-1][:-4]: np.load(f) for f in sorted(glob.glob(str(WORK / "cache_ool/*.npz")))}
ool = {w: z for w, z in ool.items() if w not in ("36_G1", "52_G1")}
rows = []
for ct in CELLTYPES:
    z = np.load(CACHE_STIM_ALL / f"{ct}.npz")
    med = {k.split("|")[1]: np.median(z[k], 0) for k in z.files if k.startswith("Unstim|") and len(z[k]) >= 20}
    hv = np.median([v for d, v in med.items() if d.startswith("HV")], 0); pv = np.median([v for d, v in med.items() if d.startswith("PV")], 0)
    o = np.median([np.median(zz[ct], 0) for zz in ool.values() if ct in zz.files and len(zz[ct]) >= 20], 0)
    for j, m in enumerate(MARKERS): rows.append((ct, m, o[j] - hv[j], o[j] - pv[j], pv[j] - hv[j]))
G = pd.DataFrame(rows, columns=["celltype", "marker", "OOL − atlas HV", "OOL − atlas PV", "atlas PV − HV"])
display(G.groupby("marker").median(numeric_only=True).sort_values("OOL − atlas PV").round(2)
        .style.set_caption("Median over celltypes (arcsinh)"))
""")
md(r"""
## 3. The implausible simulated columns
""")
code(r"""
display(S.groupby(["implausible"]).agg(columns=("column", "size")).T)
display(S[S.implausible].celltype.value_counts().head(8).to_frame("implausible columns"))
display(S[S.implausible].groupby("q").size().to_frame("implausible columns, by quantile"))
""")
md(r"""
## 4. Reading

- **The OOL inputs are systematically lower than the atlas on most phospho-markers, in every celltype.** Median shift
  over celltypes: pNFkB −1.30, pS6 −1.34, pMK2 −0.83, pCREB −0.74, HLADR −0.72, IkB −0.70, pp38 −0.46 arcsinh; pSTAT6 is
  *higher* in OOL T cells (+0.5). pSTAT1/3/5 and pERK match. The median OOL value sits ~2 atlas donor-SDs away
  (median z −2.1; pS6 −5.4). For pS6 and pNFkB, 20–28 % of a typical woman's cells are below the atlas's 0.5 % quantile.
  Myeloid cells and pDCs are shifted most (pNFkB −1.9 to −2.1).
- **It is not pregnancy:** against the atlas's pregnant donors alone the gap is nearly the same (pNFkB −1.18, pS6
  −1.18), 3–6× the pregnant vs non-pregnant difference inside the atlas (−0.21, −0.32).
- **So it is technical or protocol.** A likely cause (to confirm with the lab): the atlas Unstim wells went through the
  drug-screen protocol (incubation at 37 °C with DMSO vehicle before fixation), which raises basal phosphorylation of
  pS6 / pNFkB / pCREB / pMK2, while the OOL samples were processed on the short protocol. The 15-min vs 2-h stimulation
  difference noted earlier is the same issue on the stim side.
- **Consequences.**
  - Every simulated feature, the paper's included, applies the maps to inputs that are off the training distribution on
    several markers. The impossible values (113 columns of the retrained maps; 16 in the paper's table before removing
    36_G1/52_G1) are where the extrapolation becomes visible; everywhere else it is silent.
  - Stabl standardises each column, so a shift common to all women does not matter by itself; what matters is that
    the map's non-linear part is evaluated where it was never fitted, so the *between-women differences* it produces
    are not the ones it learned on atlas donors.
  - The baselines are affected too (gauss_ot: 45 implausible columns; quantile_map clamps at the atlas support).
- **Fixes to compare downstream:** (a) align each OOL woman's inputs to the atlas per (celltype, marker) before mapping
  (median shift, or quantile normalisation onto the atlas distribution); (b) clip simulated outputs to the atlas
  stimulated range; (c) keep as is. Each is one table rebuild plus one Stabl run.
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
nbf.write(nb, REPO / "notebooks/atlas_vs_ool.ipynb"); print("wrote notebooks/atlas_vs_ool.ipynb")
