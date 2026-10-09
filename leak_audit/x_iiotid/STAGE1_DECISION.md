### Stage 1 -- x_iiotid

**Does the dataset contain a practically important label leakage/serialisation artifact?** YES -- practically important label leakage / serialisation artifact detected.

- CRITICAL/HIGH single-feature MCC that are artifact-type shortcuts (identifier/time/port/alias): ['Scr_IP', 'Timestamp', 'Date']
- CRITICAL/HIGH but ordinary informative features (NOT counted as leakage): ['Scr_ip_bytes', 'Scr_bytes', 'Std_kbmemused', 'Std_ideal_time', 'Std_user_time', 'std_num_cswch/s', 'Std_nice_time', 'Std_system_time', 'total_bytes', 'Avg_kbmemused', 'Avg_ideal_time', 'Avg_num_cswch/s']
- Alias-flagged columns (same number, different spelling, label-separating): none
- Macro-F1 original (A) = 0.9974; strict clean (D) = 0.9428; gap = 0.0546
- Exact content overlap test-in-train (random split) = 2.13%
- **Provisional gate: CASE B -- leakage present; model must be re-tested on the clean variant.** Proceed with variant D/C only. Stage 2 decides whether the proposed method stays strong.

| variant | MacroF1 | MacroF1_sd | MCC | BinaryMCC | n_features | Delta_MacroF1_vs_A | Leakage Risk |
|---|---|---|---|---|---|---|---|
| A_original | 0.9974 | 0.0031 | 0.9999 | 0.9998 | 65 | 0.0000 | CRITICAL (max single-feature MCC=0.993) |
| B_artifact_removed | 0.9974 | 0.0031 | 0.9999 | 0.9998 | 65 | 0.0000 | cleaned reference |
| C_artifact_neutralized | 0.9974 | 0.0031 | 0.9999 | 0.9998 | 65 | 0.0000 | cleaned reference |
| D_strict_clean | 0.9428 | 0.0109 | 0.9919 | 0.9890 | 58 | -0.0546 | cleaned reference |
