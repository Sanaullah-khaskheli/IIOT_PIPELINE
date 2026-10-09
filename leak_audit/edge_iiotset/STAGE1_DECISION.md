### Stage 1 -- edge_iiotset

**Does the dataset contain a practically important label leakage/serialisation artifact?** YES -- practically important label leakage / serialisation artifact detected.

- CRITICAL/HIGH single-feature MCC that are artifact-type shortcuts (identifier/time/port/alias): ['mqtt.conack.flags', 'mqtt.msg', 'mqtt.protoname', 'mqtt.topic', 'dns.qry.name.len', 'tcp.dstport', 'tcp.srcport']
- CRITICAL/HIGH but ordinary informative features (NOT counted as leakage): none
- Alias-flagged columns (same number, different spelling, label-separating): ['dns.qry.name.len', 'mqtt.conack.flags', 'mqtt.msg', 'mqtt.protoname', 'mqtt.topic']
- Macro-F1 original (A) = 1.0000; strict clean (D) = 0.8927; gap = 0.1073
- Exact content overlap test-in-train (random split) = 16.22%
- **Provisional gate: CASE B -- leakage present; model must be re-tested on the clean variant.** Proceed with variant D/C only. Stage 2 decides whether the proposed method stays strong.

| variant | MacroF1 | MacroF1_sd | MCC | BinaryMCC | n_features | Delta_MacroF1_vs_A | Leakage Risk |
|---|---|---|---|---|---|---|---|
| A_original | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 61 | 0.0000 | CRITICAL (max single-feature MCC=1.000) |
| B_artifact_removed | 0.9956 | 0.0015 | 0.9982 | 0.9979 | 56 | -0.0044 | cleaned reference |
| C_artifact_neutralized | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 61 | 0.0000 | cleaned reference |
| D_strict_clean | 0.8927 | 0.0019 | 0.9685 | 0.9894 | 36 | -0.1073 | cleaned reference |
