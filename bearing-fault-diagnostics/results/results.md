Train: 960 records from 30 machines; test: 480 records from 15 **unseen** machines.

## Signal-processing ablation (physics rule, threshold tuned on training machines)

| chain | accuracy | missed faults | false alarms | outer | inner | ball |
|---|---|---|---|---|---|---|
| raw_spectrum | 0.250 | 0.486 | 0.458 | 0.07 | 0.33 | 0.06 |
| envelope_full | 0.260 | 0.819 | 0.175 | 0.03 | 0.10 | 0.08 |
| envelope_sk | 0.823 | 0.164 | 0.083 | 0.69 | 0.92 | 0.77 |
| full | 0.829 | 0.164 | 0.067 | 0.71 | 0.92 | 0.76 |
| full_plus_sidebands | 0.854 | 0.164 | 0.067 | 0.75 | 0.92 | 0.82 |

## Classifiers on unseen machines

| features | accuracy | missed faults | false alarms |
|---|---|---|---|
| statistical_features | 0.431 | 0.369 | 0.033 |
| physics_envelope_features | 0.873 | 0.158 | 0.025 |
| both | 0.871 | 0.158 | 0.017 |

## Fault recall vs severity (full chain + sidebands)

| severity | recall |
|---|---|
| 0.15-0.40 | 0.32 |
| 0.40-0.70 | 0.96 |
| 0.70-1.20 | 1.00 |
