# mfm_heldout — Meta Flow Matching for stim responses of held-out donors

Separate project. It reads `atlas_heldout` (caches, design, `lc` scoring code) **read-only** through `PYTHONPATH` and
writes only to `mfm_heldout/work/`.

Question: atlas_heldout showed that donor covariates fed to a conditional ICNN (raw Unstim PCs, ImmuneFM / atlas
encoder embeddings, phenotype subclusters) do not beat context-free CellOT on new donors. Meta Flow Matching
(Atanackovic et al., ICLR 2025, github.com/lazaratan/meta-flow-matching, MIT) instead **learns** a population
embedding of the donor's Unstim cells, end to end, across many populations, and was built for unseen patients.

## Setup (choices)
- **Model**: `mfm/model.py` re-implements their trellis model in plain torch (no torch_geometric / lightning / hydra):
  - KnnGCN: 2 layers, 128 hidden, k = 100; this is GCNConv on a kNN graph, where every node has degree k + 1.
  - Mean pooling and L2 normalisation.
  - SkipMLP velocity field: 7 × 512, SELU, sinusoidal t embedding, random Fourier features.
  - Loss: ||b||² − 2 b·u with base = source and independent coupling.
  - Alternating GNN / flow updates, Adam 1e-4.
- **One global model per (repeat, k, arm)**, as their trellis model (one model for all patients × treatments).
  - A population is (training donor × celltype): about 15 × 28 populations at k = 15.
  - The flow acts on the 13 functional markers and is conditioned on stim one-hot (7) + celltype one-hot (28).
  - The GCN embeds the population's Unstim cells on all 38 markers + celltype one-hot.
- **Arms**:
  - `fm`: no population embedding. This is their FM baseline, a pooled conditional flow.
  - `mfm`: the donor's own-celltype Unstim population.
  - `mfm_blood`: own population plus the donor's 4 dominant celltypes (Granulocytes, CD4Tnaive, cMCs, CD8Tnaive, ≈ 75 % of cells), each embedded by the same GCN (with its celltype one-hot), embeddings concatenated. This tests whether the response depends on the rest of the blood.
- **Design** = atlas_heldout learning curve: 5 repeats × 4 held-out donors, k = 3 5 8 15 training donors.
  - Same cells, same train-side 80 %.
  - Test donors use their own Unstim cells only: up to 20k cells, midpoint ODE, 100 steps.
  - Scored with `lc.ctx_score` against CellOT_none, shift, etc.
- 60 000 steps, batch = 8 (stim, celltype) pairs × 256 cells; graph = up to 512 source cells.

## Run
    source env.sh
    sbatch --array=0-19 -J mfm_<arm> slurm/train.sbatch <arm>        # task t: repeat t/4, k = (3 5 8 15)[t % 4]
    python3 -m mfm.merge --arms fm mfm mfm_blood                     # -> work/runs/<arm>/repeat_<r>/
    source env_cpu.sh; python -m lc.ctx_score --methods none=cond_train/repeat_{r}/none \
        mfm=/scratch/users/bmerwan/mfm_heldout/work/runs/mfm/repeat_{r} ... --vs none shift fm --out work/score/<tag>

Run state: `work/STATUS.md`.
