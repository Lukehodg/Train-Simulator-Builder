# Real data feeds — how each one is integrated and what it buys

The pipeline is a chain of adapters. Every adapter writes a canonical table, records provenance, and has a
fallback, so the app always runs; accuracy is a function of how many adapters are live. The provenance badge in
the app bar and the **Sources** tab show exactly which inputs produced the bundle you are looking at.

| Layer | Source (keyless?) | Adapter | Fallback | Confidence band |
|---|---|---|---|---|
| Route centreline, stations, tunnels/cuttings/bridges, line speed | OpenStreetMap via Overpass (yes) | `tcs/sources/osm_route.py` | synthetic spline | — |
| Terrain + sky visibility | Copernicus DEM GLO-30 COGs on AWS (yes); OS Terrain 50 (OS Data Hub, free) | `tcs/sources/terrain.py` | procedural terrain | — |
| Corridor postcodes / urban density | OS Code-Point Open (yes) | `ofcom_coverage.corridor_postcodes` | station proximity | — |
| Cellular coverage prior | Ofcom Mobile Checker UPRN Coverage API (key, postcode → UPRN rows); Connected Nations open data (yes) | `tcs/sources/ofcom_coverage.py` | synthetic prior | 0.55 / 0.65 with cells |
| Cell sites / handovers | OpenCellID bulk MCC download (token) | `tcs/sources/opencellid.py` | synthetic sites | + cells → 0.65 |
| Measured RF (calibration + validation) | Network Rail Yellow Train LTE logs and Global View 4G/5G logs (Rail Data Marketplace), Ofcom drive-test CSVs, Ofcom Connectivity on Trains study annexes, Network Survey, modem logs | `tcs/sources/measurements.py`, `tcs/model/calibration.py`, `tcs/validate/metrics.py` | none | 0.70–0.92 |
| Satcom | Stage 1: geometry only. Stage 2: Starlink terminal telemetry (`starlink-grpc-tools` export) + train GPS | `tcs/sources/starlink.py` | predictive only | 0.35 → 0.85 |
| Timetable | YAML calling pattern; CIF (Network Rail Open Data / RDG) with CORPUS; GTFS | `tcs/sources/timetable.py` | YAML | — |

## 1. Route geometry — OpenStreetMap

`fetch_route()` asks Overpass for every `railway=rail` way inside an 8 km corridor around the ordered station
chain (stations are resolved by `ref:crs`), builds a graph in the local metric CRS, and routes the shortest path
station-to-station. Parallel tracks and yards are penalised so the main running line wins. Each graph edge keeps
its tags, so every 50 m sample carries `in_tunnel`, `osm_cutting`, `osm_embankment`, `on_bridge` and
`line_maxspeed_kph`. On the ECML this produced 636 km with station distances within ~1 % of the published mileages
and picked up every tunnel (Gasworks/Copenhagen, Hadley Wood, Potters Bar, Welwyn, Stoke, Peascliffe, Calton).

Overpass needs a real `User-Agent` (anonymous clients get HTTP 406). Set `OVERPASS_URL` to a mirror if the public
instance is busy. Responses are cached under `data/raw/<route>/osm_*`.

For infrastructure-manager geometry (Network Rail ELR/track centreline, GTFS shapes, GPX from a cab ride), set
`route.geometry.source: file` and point `file:` at any GeoJSON/GPX/GeoPackage/GeoParquet; the rest is unchanged.

## 2. Terrain and sky visibility — Copernicus DEM (or OS Terrain 50)

`CopernicusDEM.prepare()` reads only the DEM windows the corridor needs (HTTP range requests against the public
COGs; ~40 MB for the ECML, cached as `.npz`). `sky_visibility()` casts 16 rays to 3 km per sample, takes the
terrain horizon per azimuth, and converts it to the **solid-angle fraction of the usable dome** above the
terminal's minimum elevation (20° for the Starlink Performance terminal). Tunnels force 0; station canopies and
cutting depth (DEM minus the smoothed railhead profile, boosted where OSM tags a cutting) subtract from it.

Upgrade path: OS Terrain 50 for GB (`terrain.source: os_terrain50`, drop the tiles in
`data/raw/<route>/os_terrain50/`), and the Environment Agency 1 m LiDAR DSM for England if you want buildings and
tree canopy in the horizon rather than bare earth.

## 3. Cellular coverage prior — Ofcom

Ofcom's checker is built from operator predictions on a 50 m grid; the API exposes it per **postcode → UPRN**
(*Ofcom Mobile Checker UPRN Coverage API*, `GET …/mobilechecker/UPRN/{PostCode}`, June 2025 methodology). The
adapter therefore:

1. takes one nearest Code-Point Open postcode per 50 m sample (within 400 m; ~2,000 unique on the ECML),
2. queries the API once per postcode (responses cached 90 days; the Basic product allows 100 calls/min and
   50,000/month, so a full ECML run is a couple of minutes and well inside the quota),
3. reads `Mc_EE / Mc_TH / Mc_O2 / Mc_VO` from every UPRN row — one **blended 4G+5G** level per operator on the new
   scale (0 poor/none · 1 variable outdoor · 2 good outdoor · 3 variable in-home, good outdoor · 4 good in-home and
   outdoor) — aggregates the UPRNs of the postcode (median by default, `ofcom_uprn_aggregate`) and keeps the spread
   as an uncertainty signal, then maps the level to a score with `ofcom_level_to_score` (a rooftop antenna is an
   outdoor user, so level 2 already scores "good"),
4. never converts a level into a fake RSRP: the score is a prior; RSRP is flagged `synthetic_rsrp` until calibrated.
   Because the level blends 4G and 5G, the throughput prior uses each operator's `capacity_prior_mbps.blend`.

Getting the key: Ofcom API portal → **Products** → *Mobile Coverage UPRN (Basic)* → Subscribe → once approved,
**Profile → Subscriptions → Show primary key** → `.env` as `OFCOM_API_KEY`. `tcs probe-ofcom N1C4TB` prints one raw
payload. The 2G product is not used. Without a key the adapter tries the Connected Nations postcode open data
(`connected_nations.url` — update it to the current year's download; the file is the “mobile postcode” CSV with
`4G_prem_out_<op>` / `5G_…` columns) and finally falls back to the synthetic prior, clearly flagged.

## 4. Cell sites and handovers — OpenCellID

With `OPENCELLID_TOKEN` the adapter downloads the bulk CSV for each MCC in `networks.yaml` (234/235 for the UK),
filters it in DuckDB to the corridor bounding box, then to ±3 km of the line, and maps MNC → operator. Per sample
and operator the five nearest cells within 12 km become **candidates**; the serving cell is chosen sequentially
with hysteresis (switch when a candidate is < 78 % of the serving distance, or the serving cell is > 9 km away), and
each switch emits a handover window (6 samples ≈ 300 m) with a latency spike, packet loss and a capacity factor.

Caveats the app repeats in the UI: cells are logical (several per physical mast), community-contributed, and a
missing cell never means no service. `getInArea` is wrapped for small refreshes only.

The national bulk file is downloaded once into `data/raw/shared/opencellid_bulk/` and shared by every route
(OpenCellID allows two downloads of each file per day; a rate-limit response is detected and the route falls back to
synthetic sites rather than failing). Code-Point Open and OS Terrain 50 are shared the same way.

## 5. Measured data — calibration and validation

`tcs calibrate <csv> --preset network_survey|ofcom_drive|modem_log` attaches each measurement to its nearest
sample (≤ 250 m) and fits, per operator:

- an additive **bias** in score space against the model's nominal score→RSRP map,
- a score→RSRP linear map (so exported RSRP stops being synthetic),
- residual spread (drives confidence bands) and per-10 km section biases.

`tcs run` then applies `data/interim/<route>/calibration.json`, raises confidence to the calibrated band, and
`tcs validate <csv>` reports MAE/RMSE, class accuracy, outage precision/recall and availability error **by route
section**. The Ofcom Connectivity on Trains study annexes load through the `ofcom_train_segments` preset as
segment-level pass rates for the same comparison at the segment scale.

`tcs calibrate-national <file> --preset yellow_train` fits each network's bias (keeping the model's dB scale, and
without per-section terms) to a measurement set covering every route, plus the cutting and tunnel terms, and writes `config/calibration.yaml`, which
routes without their own calibration use (docs/validation.md). The Yellow Train file is Network Rail's measurement
trains' LTE scanner logs: every carrier of every network each second, 2018–19; `cal_rsrp` (corrected for the
measurement antenna) is used and the strongest carrier per network per second kept. The measurements stay outside
the repository; only fitted parameters and accuracy figures are committed.

Highest-value source: logs from the actual rolling stock (Network Survey on a handset, or the onboard router's
per-SIM RSRP/RSRQ/SINR + throughput + GPS). They include the antenna installation and the real RF environment, so
they calibrate the rooftop-vs-handset vehicle profiles directly.

## 6. Satcom — Starlink

There is no public route-level Starlink RF telemetry, so Stage 1 is geometry: sky visibility × terminal type ×
speed × weather scenario → availability, capacity and latency priors, with reason codes
(`TUNNEL`, `STATION_CANOPY`, `DEEP_CUTTING`, `URBAN_OBSTRUCTION`, `TEMPORARY_HANDOVER`, `WEATHER_PENALTY`).
Confidence is capped at 0.35, except tunnels (deterministic, 0.90).

Stage 2 ingests the terminal's own history (`starlink-grpc-tools` `dish_history` CSV: obstruction fraction,
downlink throughput, PoP ping latency and drop rate), joins it to the train GPS log by time, and blends the
empirical per-sample distributions into the priors (`starlink.load_telemetry / join_gps / blend_with_priors`).
An optional orbital layer from CelesTrak TLEs is visualisation only and never claims which satellite serves.

## 7. Timetable

`route.yaml` carries the calling pattern; the movement model stretches physics-based run times to match it. For
real schedules use `timetable.from_cif(path, train_uid, corpus_json)` (Network Rail Open Data SCHEDULE feed + the
CORPUS TIPLOC reference, free registration) or `from_gtfs()`.

## Licences and attribution

OSM (ODbL), Copernicus DEM (free, attribution), OS OpenData (OGL), Ofcom data (OGL/terms of the API portal),
OpenCellID (CC BY-SA 4.0 — ShareAlike applies to redistributed derivatives), CARTO basemaps and AWS Terrain
Tiles (attribution). All are recorded in `data/raw/<route>/*/provenance.json` and surfaced in the app.
