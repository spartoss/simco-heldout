# SIMCO interview figures

Every figure is regenerated from the source result tables by one script, so the numbers are consistent across slides.

    module load python/3.9.0; source /scratch/users/bmerwan/venv-cellot/bin/activate
    python make_figures.py        # -> figures/*.png (200 dpi), data/*.csv (the exact numbers plotted)

Metrics: DOS = R² and RMSE in days (paper protocol, 100 splits, params_43, row-order bug fixed).
Stim maps = MSE on new donors (leave-one-donor-out, 19 donors, PI / IL33 out, q50-q90 selected features),
except 04 (RMSE) and 10a (variance components).

| file | slide | message |
|---|---|---|
| 01_dos | 4 | baselines (random ICNN, wrong maps, measured cells) come close to CellOT; measured cells used richly: 0.42 -> 0.55 R², 24.1 -> 21.0 days |
| 03_stim_cellot_vs_shift | 4 | per donor, CellOT ~ the average response (−5 %, 9/19 donors) |
| 04_learning_curve | 5 | RMSE vs number of training donors: paper CellOT vs whole-blood and subcluster covariate CellOT (repeats 1-4) |
| 05_stim_ladder | 5 | paired -> phenotype-subcluster covariate -> + shift -> + TabPFN: −23 % error |
| 06_tabpfn_features | 6 | raw own profile > PCs; the donor's other cell types add signal |
| 07_lineage_gain | 6 | gain in every lineage, −11 % to −26 % |
| 08_lineage_stim_heatmap | 6 (backup) | lineage × stim gains |
| 09_plate_design | 7 | schematic: barcoded donors in each well, 3 plates |
| 10a_noise_components | 7 | per stim: donor (real) / well effect shared by all donors / donor × well noise; real share of the 3-plate mean |
| 10d_noise_floor_mse | 7 | MSE twin of 10b: floor 0.0013 (split-half) -> 0.0036 realistic (donor × well noise, stimulated + resting wells); best model 0.0305, so noise = 12 % of our error |
| 10e_vs_one_well | 7 | same target for everyone (one well): perfect model 0.0116, another well of the same donor 0.0231, ours 0.0381 (1.6×), paper CellOT 0.0475; GMCSF / TNFa ≈ re-measuring the donor |
| 10c_variance_explained | 7 (backup) | best model: share of donor differences explained; noise 4 %, real but unexplained 30 % |
| 12_lineage_grouping | TabPFN | one table per lineage group vs per cell type, per (cell type, stim): helps some pairs, hurts others; no net gain with the final features |
| 11_what_did_not_work | 8 | failed leads with their numbers |

Supersedes the older figures in simco/meeting/figures, simco/presentation and
atlas_heldout/work/lc/tab/final/lineage (some use the bugged DOS script, in-sample benchmarks or skill at k = 15).
