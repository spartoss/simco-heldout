# Summary figures

Every figure is regenerated from the result tables by one script, so the numbers are consistent across figures; the
exact numbers plotted are in `data/`. The scripts read the result tables on Sherlock (no data is included).

    python lc_rmse.py          # -> data/lc_rmse_cases.csv (learning curve in RMSE, from the saved predictions)
    python k15_methods_mse.py  # -> data/k15_methods_cases.csv (methods of figure 11, MSE at 15 training donors)
    python baselines_mse.py    # -> data/baselines_cases.csv (figure 03: CellOT vs closed-form maps)
    python covariates_mse.py   # -> data/covariates_cases.csv (figures 15-17: covariate CellOT variants)
    python drug_maps_mse.py    # -> data/drug_maps_cases.csv (figure 18: drug maps and TabPFN, repeated design)
    python make_figures.py     # -> *.png (200 dpi), data/*.csv

Metrics: labour onset (DOS) = R² and RMSE in days (paper protocol, 100 splits by woman, row-order bug fixed).
Stimulation and drug maps = MSE / RMSE on new donors (leave-one-donor-out, 19 donors, PI / IL33 out, selected features
q50-q90), except 10a (variance components).

| file | topic | message |
|---|---|---|
| 01_dos | labour onset | baselines (random ICNN, wrong maps, measured cells) come close to CellOT; measured cells used richly: R² 0.42 -> 0.55, 24.1 -> 21.0 days |
| 03_baselines | baselines | CellOT vs closed-form transport maps (Gaussian OT, median shift, quantile map): 26 % lower error than the best one, at every training-set size |
| 04_learning_curve | stim maps | RMSE vs number of training donors: paper CellOT vs whole-blood and subcluster covariate CellOT |
| 05_stim_ladder | stim maps | paired batches -> phenotype-subcluster covariate -> TabPFN alone -> ensemble 0.4 CellOT + 0.6 TabPFN: 0.0299, −25 % error |
| 15_covariates | covariates | paired batches, then each donor covariate on top: whole blood −11 %, phenotype subclusters −10 % vs paper CellOT (16/16 donors better than paired) |
| 16_conditional_icnn | covariates | schematic: the covariate shifts biases and gates the W ≥ 0 path (map stays convex); architecture variants for the whole-blood covariate |
| 17_covariate_table | covariates | table of the covariates tried: what each encodes, dimension, result vs paired CellOT |
| 06_tabpfn_features | TabPFN | raw own profile > PCs; the donor's other cell types add signal |
| 19_tabpfn_feature_table | TabPFN | feature-engineering steps: columns and MSE (raw > PCs; other cell types; compact 72 columns) |
| 07_lineage_gain | TabPFN | the final model's gain holds in every lineage, −12 % to −30 % |
| 08_lineage_stim_heatmap | TabPFN | the same, lineage × stim |
| 12_lineage_grouping | TabPFN | one table per lineage group vs per cell type: helps some pairs, hurts others; no net gain with the final features |
| 09_plate_design | noise | schematic: barcoded donors in each well, 3 plates |
| 10a_noise_components | noise | per stim: donor (real) / well effect shared by all donors / donor × well noise |
| 10e_vs_one_well | noise | against one well: perfect model 0.0116, another well of the same donor 0.0231, ours 0.0375 (1.6×), paper CellOT 0.0475 |
| 10d_noise_floor_mse | noise | error of a perfect model against the pooled 3 wells: 0.0036 = 12 % of our error |
| 10c_variance_explained | noise | share of donor differences explained by the best model; noise 4 %, real but unexplained 29 % |
| 13_drugs_not_patient_specific | drugs | a new donor's drug effect is not predictable beyond the mean effect (between-donor R² ≤ 0); 83 % of the error is measurement noise |
| 14_drug_ranking | drugs | paper (CellOT stims + CellOT drug maps) vs ours (CellOT + TabPFN stims + average drug effect): THF and SALPZ delay labour in both; the rest of the ranking is not robust |
| 18_drug_between_r2 | drugs | between-donor R² of the drug effect: every map ≤ 0 (no patient-specific prediction); only TabPFN given the measured stimulated cells +0.04 |
| 11_what_did_not_work | negative results | failed leads with their numbers |
