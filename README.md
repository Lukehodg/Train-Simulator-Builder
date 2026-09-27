# Train Route 3D Connectivity Simulation

Route-agnostic, time-aware simulation of onboard train connectivity: configurable cellular operators and satcom
providers, rendered over the real railway geometry and terrain, with confidence/provenance carried on every
estimate and a calibration path against measured data. Reference configuration: **East Coast Main Line,
King's Cross → Edinburgh, EE / O2 / Vodafone / Starlink**.

```
config/*.yaml  ──►  tcs run  ──►  data/processed/<route>/{parquet, geojson}  ──►  web/ (MapLibre + deck.gl viewer)
                       │
      OSM · Copernicus DEM · Code-Point Open · Ofcom · OpenCellID · measurements · Starlink telemetry
```

## Quick start

```bash
# Python pipeline (3.11+)
python -m venv .venv && .venv/Scripts/activate        # or source .venv/bin/activate
pip install -e ".[dev]"
tcs run --offline           # synthetic stand-ins, no network: proves the whole chain (works for any --route,
                            # and `tcs build-all --offline` builds the whole catalogue)
tcs run                     # keyless live sources: OSM centreline + terrain + Code-Point postcodes (default route: ECML)
tcs routes                  # the UK route catalogue (config/routes/*.yaml) and which are built
tcs run --route wcml_eus_glc
tcs build-all               # every route in the catalogue -> web/public/data/<id>/ + index.json (route picker)
tcs report --route ecml_kgx_edb --preset edge_rail_fleet_connect --train examples/azuma-5car.train.json
                            # tender evidence pack: Word report + Excel appendix in data/processed/<route>/reports/
tcs report-all              # packs for every built route (baseline + EDGE Rail + Fleet Connect) for the viewer's Report button
tcs sources                 # which feeds are live, which fall back

# Viewer
cd web && npm install && npm run dev                   # http://localhost:5173  (?route=<route_id>)
```

Copy `.env.example` to `.env` and add `OFCOM_API_KEY` / `OPENCELLID_TOKEN` to turn the coverage prior and cell
sites live. To host the simulator for colleagues with the live feeds refreshed monthly, see
[docs/hosting.md](docs/hosting.md). `docs/data-sources.md` explains every feed, its adapter, its fallback and what it does to confidence.

## What `tcs run` does

1. **Route** – OSM rail network routed station-to-station (or a supplied file), sampled every 50 m with
   cumulative distance, bearing, tunnel / cutting / embankment / bridge / maxspeed flags, station proximity.
2. **Terrain** – DEM at each sample, smoothed railhead profile, cutting depth, 16-ray horizon → sky visibility.
3. **Coverage prior** – Ofcom level per postcode per operator → model score (or Connected Nations, or synthetic).
4. **Cells** – OpenCellID corridor extract → 5 candidates per sample → serving cell with hysteresis → handovers.
5. **Movement** – line-speed caps + accel/decel + dwell, stretched to the timetable → sim clock per sample.
6. **Simulate** – cellular (prior + terrain + cell distance + vehicle loss + handover), satcom (obstruction →
   availability → capacity), link manager (bonding/failover/…), passenger Wi-Fi, confidence.
7. **Export** – `route_samples`, long-form `provider_observation`, `route_connectivity` (Parquet), GeoJSON,
   and the web bundle (`route.arrow` + `meta.json`).

Then `tcs calibrate <measurements.csv>` and `tcs validate <measurements.csv>` close the loop with real data (both take
`--route <route_id>`; the default is `config/route.yaml`).

Two optional inputs shape the onboard side:

- `tcs run --train examples/azuma-5car.train.json` reads a **Train Studio** project (the Train Diagram Builder is
  hosted in the viewer at `/train-builder/index.html`): carriages, EDGE Rail roof units, EDGE Mini / SATCOM
  terminals, switches, access points and Fleet Connect become the vehicle profile, satcom terminal class, AP
  capacity, seat count, cellular aggregation factor and link policy (`simulation.yaml` -> `train:`).
- `tcs run --preset edge_rail_fleet_connect` applies the **EDGE Rail 5G active antenna + Fleet Connect** scenario
  (active-antenna link budget, 4x4 MIMO throughput factor, aggregation of every cellular network and the satcom
  link at once). Both are also available live in the viewer, with KPIs shown against the baseline.

## Giving it to someone else

```bash
tcs package            # -> dist/train-link-simulator-<date>.zip (viewer + Train Builder + every built route)
tcs serve --built      # serve the production build locally without packaging
```

The zip runs with **nothing installed**: on Windows the recipient double-clicks `Start Simulator.bat` (a
PowerShell static file server, no admin rights, no Python or Node); on macOS/Linux `./start-simulator.sh` uses
the preinstalled `python3`. A browser opens automatically. Only the basemap and terrain tiles need the internet -
offline, every ribbon, panel and figure still works against blank map tiles. The evidence packs built by
`tcs report-all` travel inside the site, behind the viewer's **Report** button (`--no-reports` leaves them out for a
smaller zip), and `README.txt` explains how to read the confidence values.

## Routes

`config/routes/` holds the principal UK lines (ECML, WCML to Glasgow and Manchester, GWML to Swansea, West of
England & Cornwall, Midland, Chiltern, Great Eastern, South Western, Brighton, TransPennine, HS1/Kent, CrossCountry,
Marches, North Wales Coast, Settle–Carlisle, Edinburgh–Glasgow, Highland). A route file is just the ordered station
list (CRS codes), calling times and flags; everything else — centreline, tunnels, cuttings, terrain, coverage — is
fetched by the pipeline. Add a route by copying one file. Stations are matched to any track within 90 m, so
multi-track stations route correctly (mileages come out within ~1 % of the published figures).

## Evidence pack (`tcs report`)

A Word report and an Excel appendix per route and scenario: headline KPIs, capacity along the route with stations
and tunnels, service class by station-to-station section, per-link availability/capacity, onboard architecture,
every model assumption, data provenance with live/synthetic status, and the validation status (metrics by section
once measurements are attached). The document states plainly that figures are model predictions until validated.

`tcs report-all` builds the pack for every built route in the two standard scenarios (baseline, and EDGE Rail 5G +
Fleet Connect) and places it beside the route's viewer bundle (`web/public/data/<route>/reports/`, with an
`index.json`). The viewer's **Report** button offers those downloads, marks the one matching the scenario on screen,
and says so when the screen shows something no pack covers (a train design, other overrides, bad weather). It
re-simulates from the cached route data (no API calls; about 2.5 minutes for all 18 routes) and refuses a route whose
viewer bundle comes from a different build. A fresh `tcs run` clears that route's packs, since they describe the
previous build. The monthly publish runs it after the rebuild. The viewer's **Export** button downloads whatever
scenario is on screen as CSV + JSON.

## Viewer

MapLibre GL (CARTO basemap, AWS terrain tiles, no keys) with deck.gl layers: colour-classed ribbons for the
combined WAN and each link, a procedurally modelled multi-carriage train that follows the loaded design (roof
units included) along the real curve, rails and catenary, 3D buildings and woodland trees generated from the
basemap's vector tiles, a sky-visibility window, estimated serving cells, stations; a live train
panel, a per-sample inspector with model reasoning and provenance, a journey timeline with per-link strips, and
scenario controls (link policy, vehicle profile, weather) that re-run the link manager and Wi-Fi model in the
browser (`web/src/sim/model.ts` mirrors `tcs/model/*` and is parity-checked against the Python output at load).

## Checks

CI (`.github/workflows/checks.yml`) runs `ruff check` and `pytest` (Python 3.11 and 3.12), the viewer's input-validation
tests, typecheck + build, Chromium interaction tests (Playwright), and two parity checks between the browser model and
the Python model: pytest's calibrated/uncalibrated cases on a synthetic fixture, and a full-route check that rebuilds
ECML offline and requires the exported bundle to reproduce Python for every link policy × vehicle profile × weather.
Locally:

```bash
ruff check . && pytest                  # the pytest parity cases need web/node_modules (npm ci), otherwise they skip
cd web && npm run test:inputs && npm run build
npx playwright install chromium && npm run test:browser
cd .. && tcs run --offline && python tests/parity_expected.py
cd web && npm run test:parity -- ../data/processed/ecml_kgx_edb/parity_expected.json
```

A change to either model (or to a parameter only one side reads) fails the parity checks until the other side matches.

## Layout

```
config/            route.yaml networks.yaml starlink.yaml simulation.yaml
tcs/               config geo schema cli · sources/ pipeline/ model/ validate/
web/               Vite + TypeScript · src/{data,state,config}.ts src/map src/sim src/ui
docs/              architecture.md data-sources.md model.md validation.md
tests/             hermetic pipeline + unit tests (pytest)
data/              raw/ interim/ processed/ (git-ignored)
```

## Caveats (also shown in the app)

Ofcom coverage is operator-predicted, not measured; OpenCellID is community data and a missing cell is not a
missing site; Starlink has no public route-level RF telemetry; throughput depends on load and spectrum, not just
signal; rooftop antennas and handsets behind coated glass behave very differently. Every synthetic field is
traceable through `source_flags`, `confidence` and `model_version`.
