# Validation

```
tcs calibrate data/raw/measurements/drive.csv --preset ofcom_drive      # fits bias, score→RSRP, residuals
tcs run                                                                  # applies calibration.json
tcs validate data/raw/measurements/drive.csv --preset ofcom_drive --section-km 10
```

Metrics reported per operator and 10 km section (`data/processed/<route>/validation_by_section.csv`):

| metric | definition |
|---|---|
| `rsrp_mae`, `rsrp_rmse` | predicted vs observed RSRP (dBm) |
| `class_acc` | agreement of 4-class (strong/usable/poor/outage) labels |
| `outage_precision`, `outage_recall` | predicted outage samples vs observed RSRP < −110 dBm |
| `avail_err_pp` | predicted availability minus observed (percentage points) |
| handover position error | `validate.metrics.handover_position_error` — metres from each observed serving-cell change to the nearest predicted handover |

Inputs accepted through `sources/measurements.py` presets: Ofcom drive tests, Ofcom train-study segment annexes
(segment pass rates), Network Survey exports, modem/router logs, throughput/latency tests. Column names are guessed
by preset and can be overridden with an explicit mapping.

Report validation by section, never as one global score.
