"""One-off: report obs schema, patient/stim coverage and cell counts for a batchcorrected h5ad."""
import sys
import anndata as ad
import pandas as pd

path = sys.argv[1]
adata = ad.read_h5ad(path, backed="r")

print(f"path: {path}")
print(f"shape: {adata.shape}")
print(f"var_names ({adata.n_vars}): {list(adata.var_names)}")
print(f"obs columns: {list(adata.obs.columns)}")
for col in ["patient", "stim", "drug", "cell_type", "sampleID"]:
    if col in adata.obs.columns:
        vals = adata.obs[col].unique()
        print(f"  {col} ({len(vals)}): {sorted(map(str, vals))[:40]}")

obs = adata.obs
dmso = obs[obs["drug"].astype(str).str.startswith("DMSO")]
print(f"\nDMSO-prefixed cells: {len(dmso)} / {len(obs)}")
print("\ncells per (patient, stim) within DMSO:")
tab = pd.crosstab(dmso["patient"], dmso["stim"])
print(tab.to_string())
