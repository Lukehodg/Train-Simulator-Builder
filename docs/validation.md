# Validation

```
tcs calibrate data/raw/measurements/drive.csv --preset ofcom_drive      # fits bias, score→RSRP, residuals
tcs run                                                                  # applies calibration.json
tcs validate data/raw/measurements/drive.csv --preset ofcom_drive --section-km 10
```

Both commands take `--route <route_id>` (default: `config/route.yaml`). `calibrate` always fits against the
uncalibrated model, so re-running it on the same measurements reproduces the same `calibration.json`.

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

## National calibration (`tcs calibrate-national`)

```
tcs calibrate-national lte-jun18tojun19-yt.csv --preset yellow_train --split 2019-01-01
```

Fits the cellular model to a measurement set that covers many routes, such as Network Rail's Yellow Train LTE logs
(Rail Data Marketplace), and tests it on later measurements the fit never saw. For every built route
(`data/interim/<route>/`; the publish workflow keeps them as its `route-data` artifact):

1. Read the file in chunks, keeping rows near the track (≤ 50 m; legs drawn straight because OSM had no rail path
   are left out). Scanner logs keep the strongest carrier per second, train and network, as a modem would.
2. Reduce to one median RSRP per route point, network and period (before / from `--split`).
3. Choose the rail-environment terms on the fit period, by least absolute error: cutting penalty and full depth, then
   how far signal carries into a tunnel from its portals (`portal_decay_m`) and the deep-tunnel floor, on tunnels
   without in-tunnel coverage. Tunnels measured at −90 dBm or better more than 250 m from either portal can only be
   lit from inside: they are listed under `in_tunnel_coverage` for the route's `das_tunnels`.
4. Fit each network's score bias on open-air points pooled over all routes (no per-section terms). The model's dB
   scale is kept: a least-squares slope shrinks when the predicted score is noisy, squeezing predicted signal into a
   narrow band with the right average but the wrong share of the route in each signal band.
5. Test on the later period, per route, network and setting (open / cutting / tunnel), against the uncalibrated model,
   and again with each route held out of the fit.

It writes `config/calibration.yaml`: the parameters, the provenance (source, periods, routes) and the accuracy figures,
never the measurements. Every route without its own `calibration.json` then uses it
(`cellular.national_calibration` in `config/simulation.yaml`; `null` turns it off). The environment terms it chose go
into `config/simulation.yaml` by hand; a test checks the two files agree. Reports (section 8) and the viewer's Sources
tab show the source, periods and the route's accuracy before and after calibration.

## Checks against later measurements (`tcs check-national`)

```
tcs check-national --four-g Global_View_4G.csv --five-g Global_View_5G.csv --split 2026-04-07
```

Network Rail's Global View trains log 4G and 5G signal on today's networks (Rail Data Marketplace, from 2026; presets
`global_view_4g` / `global_view_5g`). Their scanner logs signal before correcting for its antenna and cable: at the
same route points it reads about 10 dB below the antenna-corrected Yellow Train data, much the same on every network
and train, which coverage changes would not do. So the 2026 data check the calibration rather than replace it:

- **current_check**: one level offset between the set and the calibrated model is fitted on the measurements before
  `--split`; accuracy on those from it is then reported per route, network and setting (open / cutting / tunnel).
- **five_g**: per route and network, the share of the route points passed by a train carrying the 5G scanner (known
  from its 4G log) where that network's 5G reached −110 dBm SS-RSRP, and which NR bands were measured. The scanner
  does not measure 3.4–3.8 GHz, where Three and Vodafone carry most of their 5G, so their shares understate it; the
  model's coverage input (Ofcom) blends 4G and 5G, so these are evidence alongside the predictions, not an input.

Both go into `config/calibration.yaml` (a re-run of `tcs calibrate-national` keeps them) and appear in the reports
(sections 8.2 and 8.3, and the sources table) and in the viewer's Sources tab.
