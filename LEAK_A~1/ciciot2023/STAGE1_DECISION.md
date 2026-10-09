### Stage 1 -- ciciot2023

**Does the dataset contain a practically important label leakage/serialisation artifact?** NO practically important single-feature leakage detected by these probes.

- CRITICAL/HIGH single-feature MCC that are artifact-type shortcuts (identifier/time/port/alias): none
- CRITICAL/HIGH but ordinary informative features (NOT counted as leakage): ['IAT']
- Alias-flagged columns (same number, different spelling, label-separating): none
- Macro-F1 original (A) = 0.8143; strict clean (D) = 0.8143; gap = 0.0000
- Exact content overlap test-in-train (random split) = 0.00%
- **Provisional gate: CASE C -- no practically important leakage detected.** Proceed with the original hypothesis; cite this audit as dataset-validity evidence.

| variant | MacroF1 | MacroF1_sd | MCC | BinaryMCC | n_features | Delta_MacroF1_vs_A | Leakage Risk |
|---|---|---|---|---|---|---|---|
| A_original | 0.8143 | 0.0113 | 0.9138 | 0.7872 | 46 | 0.0000 | HIGH (max single-feature MCC=0.707) |
| B_artifact_removed | 0.8143 | 0.0113 | 0.9138 | 0.7872 | 46 | 0.0000 | cleaned reference |
| C_artifact_neutralized | 0.8143 | 0.0113 | 0.9138 | 0.7872 | 46 | 0.0000 | cleaned reference |
| D_strict_clean | 0.8143 | 0.0113 | 0.9138 | 0.7872 | 45 | 0.0000 | cleaned reference |
