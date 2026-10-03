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

### With LiDAR cutting depths (model 0.3.0)

In GB, cutting depth now comes from open 2 m LiDAR within 60 m of the track (docs/data-sources.md, 2a) rather than
the 30 m terrain model. LiDAR covers 99–100 % of every route except the Highland Main Line (63 %: no Scottish survey
over parts of the Highlands). LiDAR finds three times as many measured route points in a cutting (56k against 18k),
and deeper ones, so the fit moved from 0.2 of score at 4 m to **0.28 at 8 m** (about 13 dB), and the tunnel terms to
150 m and a floor of 0.08. Against the 2019 test period (Yellow Train, 303k points):

| | before LiDAR | with LiDAR |
|---|---|---|
| mean absolute error | 11.24 dB | 11.14 dB |
| within ±6 dB | 33.5 % | 34.1 % |
| correlation | 0.40 | 0.42 |
| routes held out of the fit, median error | 11.04 dB | 10.89 dB |
| 2026 Global View 4G check (182k points, after the level offset) | 10.29 dB | 10.18 dB |

A small gain: cuttings are one of many things that set mobile signal. The larger effect of LiDAR is on satellite sky
visibility (trees, buildings and bridges beside the line), which no measurement set here can test.

Looked at and left out: lineside clutter. Where trees or buildings stand above the roof antenna within 30 m of the
track (LiDAR surface model), measured signal is 3–4 dB lower relative to the model than on open track, in both the fit
and the test period. A term for it would cut the mean error by only about 0.05 dB, so the model does not carry one.
Embankments are about 1.5 dB better than modelled above 3 m; also too small to add.

### Distance to the serving mast, fitted (October 2026)

The penalty for distance to the serving mast (`cellular.cell_distance`) had been hand-set at 0.06 of score per km beyond
3 km. OpenCellID's mast positions were too rough to fit it on (for a typical mast the train's strongest reading is
about 1 km from where OpenCellID puts it), so the masts were first placed from the Global View logs themselves
(`tcs locate-masts`, docs/data-sources.md 4a). On later trips (masts placed from 16 March – 6 April, 250k readings from
7 April at 2,609 masts both place), predicting each reading from distance to its mast:

| mast positions | distance only | + terrain between mast and train | + each mast's own level |
|---|---|---|---|
| OpenCellID | 10.5 dB | 10.5 dB | 8.6 dB |
| placed from the logs | 8.8 dB | 8.8 dB | 7.2 dB |

(a network-and-band average alone: 11.4 dB). Terrain between mast and train (knife-edge diffraction over a 90 m
elevation model) added nothing, with either set of positions and even where 42 % of the paths cross a ridge.

`tcs calibrate-national` now searches the distance terms with the others. With OpenCellID's positions, with masts placed
before the check's split only, and with masts placed from all the logs, it chose almost no penalty (0.01 per km), and
the three gave the same accuracy to within 0.02 dB: the hand-set penalty had been making stretches far from a mast too
weak. Production uses the masts from all the logs (penalty 0.01 per km from the mast):

| | before (0.06 per km beyond 3 km) | fitted |
|---|---|---|
| 2019 test, mean absolute error | 11.14 dB | 10.66 dB |
| within ±6 dB | 34.1 % | 35.2 % |
| correlation | 0.42 | 0.47 |
| routes held out of the fit, median error | 10.89 dB | 10.62 dB |
| 2026 Global View 4G check (182k points, after the level offset) | 10.18 dB | 9.67 dB |
| 2026 check, correlation | 0.33 | 0.39 |

The placed masts' value is mainly realism: the masts the model hands over between are the ones trains use.

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


### Measured corrections (`tcs correct-routes`)

Signal at a given spot repeats from trip to trip far more closely than any coverage prediction places it (a mast's own
power and aim, a bend behind a hill, a cutting). Where trains have measured a route, the calibrated model's error there,
smoothed along the track, corrects it (tcs/corrections.py; docs/model.md). `tcs check-national` tests this every run:
corrections built from the Global View measurements before the split, plus the Yellow Train logs at a tenth of the
weight, applied through the model, against the Global View measurements from the split (182k route points):

| | model as calibrated | with measured corrections |
|---|---|---|
| mean absolute error | 9.67 dB | 8.30 dB |
| within ±6 dB | 40.1 % | 48.3 % |
| correlation | 0.39 | 0.55 |
| usable / not usable (−110 dBm) agreement | 92.6 % | 92.8 % |

Better on every one of the 15 routes the 2026 logs cover (by 0.7–2.3 dB). Global View's earlier trips alone give
8.66 dB; the 2018–19 Yellow Train logs still help because hills and cuttings have not moved. Tested offline before
building it in: a Gaussian scale of 100 m and a shrink of 0.5 did best (75–150 m and 0.25–1 within 0.03 dB), Yellow
Train at 0.1 better than 0.25 or 0.5. The production corrections (`config/route_corrections.parquet`, 365k route points on all 18
routes, typically 3–5 dB) use every measurement, so on the routes the logs cover the published model sits nearer the
measurements than this test shows; a route no train has measured gets none, and the model alone (9.67 dB).

Looked at and left out:

- **A learned adjustment** (gradient-boosted trees on the score model's error, from what is known everywhere along the
  track: Ofcom level, LiDAR cutting and skyline, distance to the serving mast, urban density, line speed, height above
  the surrounding land). Trained on the other routes' Yellow Train logs, on routes it never saw it cut the error by
  0.8 dB on the 2019 test and 0.5 dB on the 2026 check (better on all 15 routes). With measured corrections it adds
  nothing (8.25 against 8.24 dB), and every route in the catalogue has been measured, so it is not built in; it is the
  next step for a route no train has measured.
- **Snapping placed masts to OpenStreetMap's mapped masts** (11,937 telecom masts and towers in GB, export of
  3 October 2026). Only 17 % of the masts placed from the logs have a mapped mast within 500 m. On the later-trips test
  (masts placed before 7 April, 418k readings from it) moving each to the nearest mapped mast made the error worse
  (8.98 dB as placed; 9.04 dB snapping within 500 m, 9.34 dB within 1 km), and moving it to the mapped mast the earlier
  readings fit best changed nothing (8.97 dB). The positions stay as the logs place them.
- **Ofcom Connected Nations 2025 downloads** (coverage as of July 2025): shares of premises, land and roads covered by
  0–4 networks per local authority and constituency, per network only for the UK and nations. Nothing is located along
  the track, so they cannot sharpen a prediction at a point; the Ofcom API the model queries already returns the
  operators' predictions location by location.

