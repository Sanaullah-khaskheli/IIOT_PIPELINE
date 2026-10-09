# IIOT_PIPELINE

Code and results for the paper "Beyond Closed-Set Scores: Shortcuts and Unseen Attack Families in IoT and IIoT Intrusion Detection" (submitted to the International Journal of Information Security).

The code runs three experiments on Edge-IIoTset, X-IIoTID and CICIoT2023:

1. A shortcut audit. Columns that identify the capture rather than the attack are removed in steps (raw, strict-clean, no-port, network-only) and closed-set classifiers are retrained at each step.
2. Leave-one-attack-family-out evaluation. Each attack family is held out in turn and scored as unknown.
3. Statistics and tables over the held-out families, including near and far families defined by a fixed taxonomy.

## Folders

- `code/` the scripts, `config_final.yaml`, `taxonomy.csv` and `requirements.txt`
- `cleaned_features/`, `closed_set/`, `dataset_audit/`, `figures/`, `leak_audit/`, `loao/`, `logs/`, `statistics/` and `tables/` result files of the run `SAFE_FINAL_01` (CSV tables, per-family results, feature lists, audit files, figures, logs)

The datasets are not included. Download them from their original sources and put the files where `config_final.yaml` expects them (`datasets:` section, `search:` paths):

- Edge-IIoTset: `DNN-EdgeIIoT-dataset.csv` 
- X-IIoTID: `X-IIoTID dataset.csv` 
- CICIoT2023: `CICIoT2023/part-*.csv` 

## Setup

Python 3 with the packages in `requirements.txt`:

```
pip install -r code/requirements.txt
```

The reported results were produced in Kaggle notebooks (GPU). Package versions were not recorded.

## Running

Run from inside `code/`. All settings (seeds, paths, models) are in `config_final.yaml`.

1. Build the taxonomy file once and review it:
   ```
   python run_final.py --stage taxonomy --datasets edge_iiotset,x_iiotid,ciciot2023
   ```
   The `sha256` of `taxonomy.csv` is stored in `config_final.yaml` (`final.taxonomy_sha256`). The held-out-family stage stops if the file no longer matches.
2. Run audit, ladder and held-out-family stages, once per dataset, with the same run id:
   ```
   python run_final.py --stage audit,ladder,loao --datasets edge_iiotset --run-id SAFE_FINAL_01
   ```
   The held-out-family stage can be restarted; finished family and seed pairs are skipped.
3. Put the three `results_final/` folders together, then:
   ```
   python run_final.py --stage stats,figures,check --run-id SAFE_FINAL_01
   ```

`check` runs `09_consistency_check.py`, which compares the tables with the per-seed result files.

## Results layout

| Folder | Content |
|---|---|
| `cleaned_features/` | feature list and removed columns per rung |
| `closed_set/` | per-seed closed-set results, per-class F1, class counts, confusion matrices (zipped) |
| `loao/` | held-out-family results per family, seed and score, family inventory, split hashes |
| `statistics/`, `tables/` | per-family table and the summary tables used in the paper |
| `dataset_audit/`, `leak_audit/` | dataset audit and single-feature screen outputs |
| `figures/` | figures (PDF, PNG), manifest and data audit |
| `logs/` | run logs |

## Scores and metrics

All scores are oriented so that larger means more unknown: RF max-probability, MLP max-softmax, energy, Mahalanobis distance on penultimate features, rank fusion (0.5 each of the RF score and the Mahalanobis score, using empirical distributions fitted on known validation rows), a z-score fusion variant, and an evidential-MLP vacuity score used as a control. Metrics are AUROC, AUPR, FPR at 95% TPR, and the detection rate at a threshold set on known validation rows (with the false-alarm rate realised on known test rows). Statistics use the attack family as the unit of analysis.

## Notes

- The CICIoT2023 subsample is capped at 40,000 rows per class, so its class proportions are not the natural ones.
- Families with fewer than 200 rows are not evaluated.
- The tree class weights were set after a pilot run in which boosting diverged on CICIoT2023.

## License

This code is released under the MIT License (see the LICENSE file). The datasets are not included and remain under their own licences; see the links above.

