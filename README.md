# SAFE-IIoT final experiment package (Stage 5)

## Install (Kaggle notebook cell 1)
```
!pip -q install lightgbm pyyaml seaborn scipy scikit-learn   # torch, numpy, pandas, matplotlib are preinstalled on Kaggle
```
Upload the `iiot_pipeline/` folder as a Kaggle Dataset (or paste files into /kaggle/working/iiot_pipeline), attach the three datasets
(Edge-IIoTset `DNN-EdgeIIoT-dataset.csv`, `X-IIoTID dataset.csv`, `CICIoT2023/part-*.csv`). Edit ONLY `config_final.yaml`.

## Order of operations (no trial and error)
1. **Taxonomy (once, 1 min, CPU):** `!cd /kaggle/working/iiot_pipeline && python run_final.py --stage taxonomy --datasets edge_iiotset,x_iiotid,ciciot2023`
   -> writes `taxonomy.csv`. Open it, correct the `group` column by hand (siblings = same attack mechanism), save, run once more to print the final `sha256`,
   paste it into `config_final.yaml -> final.taxonomy_sha256`. After that LOAO refuses to run if the file changes. Do this BEFORE seeing any LOAO result.
2. **One Kaggle version per dataset** (same `--run-id` in all three so provenance matches):
   `!python run_final.py --stage audit,ladder,loao --datasets edge_iiotset --run-id SAFE_FINAL_01`
   (repeat for `x_iiotid`, `ciciot2023`). Use *Save Version -> Save & Run All (Commit)*, accelerator GPU T4, *Always save output*. LOAO is resumable:
   if a session times out, re-run the same command and finished (family, seed) pairs are skipped (PARTIAL csv files).
3. **Merge + report:** put the three `results_final/` folders into one notebook (copy `closed_set/`, `loao/`, `cleaned_features/`, `splits/`, `predictions/`, `leak_audit/`, `dataset_audit/`),
   then `!python run_final.py --stage stats,figures,check --run-id SAFE_FINAL_01` -> tables, figures (PDF+PNG), `numbers.tex`, `consistency_report.csv`.
4. The tables and figures of the article are generated from the result files by `07_stats_tables.py` and `08_make_figures.py`.
If `consistency_report.csv` shows any FAIL the program prints `TRACEABILITY FAILURE -- DO NOT REPORT`.

## Output tree (`results_final/`)
data_audit* (Stage 0 writes `dataset_audit/`, Stage 1 `leak_audit/`), cleaned_features/<ds>/<rung>.json (feature list, removed list, preprocessing),
splits/<ds>/ (closed-set indices per rung/seed; LOAO indices for the first seed; sha256 of every split in loao/<ds>/split_hashes.csv),
closed_set/<ds>/ (ladder_by_seed.csv, per_class_by_seed.csv, cm_*.csv, class_counts.csv, model_config.json), loao/<ds>/ (loao_by_family_seed_method.csv, family_inventory.csv,
proximity.csv, known_class_closed_set_F1.csv, split_hashes.csv), predictions/ (closed-set npz; loao/<ds>/<family>_seed*.npz with raw OOD scores),
statistics/ (loao_per_family.csv + every T_*.csv), tables/ (CSV+MD+TEX, numbers.tex, numbers_manifest.csv), figures/ (PDF, PNG, manifest, captions, audit), logs/.
`open_set/` is intentionally unused: all open-set results are in `loao/` (single source of truth).

## Experiment -> file -> manuscript mapping
| Experiment | Output CSV | Table | Figure |
|---|---|---|---|
| Dataset overview, taxonomy, near/far counts | tables/T_dataset_overview.csv, taxonomy.csv | Table 1 | - |
| Shortcut ladder (A raw, D strict, E3 no-port, E4 network-only) x 4 models x 5 seeds | closed_set/<ds>/ladder_by_seed.csv -> tables/T_ladder_closed_set.csv | Table 2 | Fig 2 |
| Closed-set comparison at cleanest rung (+ECE, EDL control) | tables/T_ladder_closed_set.csv (filter) ; closed_set/<ds>/per_class_by_seed.csv | Table 3 | Fig 3 (supplement) |
| LOAO all families, 8 scores, 2 conditions | loao/<ds>/loao_by_family_seed_method.csv -> tables/T_loao_method_summary.csv | Table 4 | Fig 5 |
| Near vs far, proximity correlation | tables/T_loao_near_vs_far.csv, statistics/loao_per_family.csv | Table 5 | Fig 4, Fig S1 |
| Rank fusion vs components (paired over families, Holm) | tables/T_loao_paired_methods.csv | Table 5 | - |
| Failure cases | tables/T_loao_failure_cases.csv | - | Fig 7 |
| Known-class F1 during LOAO | tables/T_loao_known_class_F1.csv | text | - |
| Framework | - | - | Fig 1 (manual vector, spec in figures/FIG1_SPEC.txt) |

## Input / output schemas
Result CSV keys (all carry `run_id`): `ladder_by_seed.csv`: dataset, rung, model, seed, n_features, n_train, n_val, n_test, MacroF1, MCC, BinaryMCC, Accuracy, ECE, seconds.
`loao_by_family_seed_method.csv`: dataset, variant, family, group, near_far, n_siblings, seed, method, condition(with_normal|attacks_only), AUROC, AUROC_CI_low, AUROC_CI_high, AUPR, FPR@95TPR,
DetRate@val5FPR, KnownFPR@val5, n_known, n_unknown. `loao_per_family.csv`: dataset, condition, method, family, group, near_far, n_siblings, AUROC, AUPR, FPR95, DetRate, KnownFPR, AUROC_seed_sd, n_seeds, n_unknown, n_known, proximity_dist.
`proximity.csv`: dataset, family, seed, proximity_dist, nearest_known. `taxonomy.csv`: dataset, family, group, rule.

## Metric definitions
Positive = held-out (unknown) family; negative = known test rows (with_normal) or known ATTACK test rows (attacks_only). Score higher = more unknown.
AUROC/AUPR: unknown positive. FPR@95TPR: known rows flagged when the threshold keeps 95% of unknowns (oracle threshold). DetRate@val5FPR: share of unknowns flagged at the
threshold fixed on known VALIDATION rows (95th percentile); KnownFPR@val5 is the realised false-alarm rate on known test rows. AUROC CI = Hanley-McNeil (row-level, not training variance);
across-seed intervals are t-intervals over seeds. Statistics unit = family (Wilcoxon needs >=6 families with non-zero differences; Holm across the declared pairs).
Rank fusion = 0.5*ECDF_val(RF 1-maxprob) + 0.5*ECDF_val(MLP Mahalanobis); ECDFs fitted on known validation rows only; weights fixed because validation contains no unknowns. `fusion_z` (log-z-score) is a pre-declared sensitivity row.

## Deliberate decisions
* TabICLv2 is NOT coded: no peer-reviewed journal source (arXiv/ICML only) -> remove RQ4/Fig 6/Table 6.
* EDL is a control only (`edl:vacuity`, `mlp_edl_both`). Hyper-parameters inherited from Stage 2, never tuned on test data.
* Ladder rung A uses raw-spelling encoding; if a rung equals its predecessor (e.g. CICIoT2023 has no identifier/port columns) it is skipped and logged.
* Sampling: CICIoT2023 is class-capped (priors not natural); Edge/X-IIoTID use `model_rows` stratified subsets. State both in the paper.
* Tested here on synthetic data with a torch-free stub for the networks; the PyTorch path reuses the Stage 2/4 functions unchanged.
