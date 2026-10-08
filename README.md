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
4. **Cells** – OpenCellID corridor extract, with the 4G masts placed from Network Rail Global View logs
   (`config/masts.csv`, `tcs locate-masts`) standing in for OpenCellID's positions → 5 candidates per sample →
   serving cell with hysteresis → handovers.
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
  capacity, seat count, fitted mobile networks (one per EDGE Rail unit, in the order of `edge_rail_networks`) and
  link policy (`simulation.yaml` -> `train:`).
- `tcs run --preset edge_rail_fleet_connect` applies the **EDGE Rail 5G active antenna + Fleet Connect** scenario
  (active-antenna link budget, 4x4 MIMO throughput factor; three antennas, one each on EE, Vodafone and Three,
  aggregated with the satcom link at once; O2 has no modem on the train). Both are also available live in the viewer, with KPIs shown against the baseline.

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

### United States

A route file with `country: US` (first: `nec_was_bos`, the Acela on the Northeast Corridor, Washington → Boston) runs
on the US country profile in `config/countries/US/` instead of the GB defaults; GB routes are unaffected. In the viewer, the UK / USA switch beside the route list changes region (it opens the route last viewed there);
the route list shows that region's routes, and `?region=us` opens the US directly. UK stays the default. What changes:

| Input | GB | US |
|---|---|---|
| Stations | OSM `ref:crs` | Amtrak codes from the US DOT's NTAD Amtrak Stations layer (keyless) |
| Track | OSM, routed station to station | OSM the same way; the NTAD Amtrak Routes line (`geometry.ntad_route`) when Overpass is unavailable, with tunnels from the route file's `tunnels` |
| Networks | EE, O2, Vodafone, Three | AT&T, Verizon, T-Mobile, matched to OpenCellID by their US PLMNs (several MCCs each) |
| Coverage prior | Ofcom API / Connected Nations by postcode | Operators' FCC National Broadband Map filings (in-vehicle / outdoor, 4G / 5G); distance to OpenCellID sites where a network has none, uncalibrated |
| Terrain | OS Terrain 50 + open LiDAR | Copernicus GLO-30 |
| Network speed | Calibrated to GB measurement trains | Each network's capacity scaled by local Ookla phone-test speeds (`ookla.capacity` in the US profile), damped and clamped |
| Timetable | Route file calls | One real Acela from Amtrak's GTFS feed (`timetable.source: gtfs`, keyless, weekly); the route file's calls if the feed can't be read |
| Calibration | Yellow Train national fit, Global View corrections and masts | None yet: the figures are uncalibrated until US measurements are fitted |

**Every Amtrak service.** Besides the hand-written Acela, `scripts/amtrak_routes.py` writes a route file for each of the
other train services in Amtrak's GTFS feed (42 as of October 2026, `<service>_<from>_<to>.yaml`), each from one real
weekday train (pinned by number): the full run, its stations with their positions (so Toronto, Montreal and Vancouver
work), its calling times, the states it crosses (Census Bureau boundaries, for the FCC files) and a sample spacing that
keeps it to ~20,000 samples (50 m up to 1,000 km, up to 225 m on the Texas Eagle). The track is searched for in OSM
around the feed's shape of that train (`geometry.guide: gtfs_shape`), in 200 km pieces. Times are Amtrak's (US
Eastern), and a train that runs past midnight more than once keeps counting days. Run the script again when Amtrak
changes its timetable. Not in Amtrak's feed, so not here yet: the Gold Runner (California's own feed).

The viewer's region switch (UK / USA) picks the country, the evidence packs drop the Ofcom and Network Rail wording for US
routes, and the publish workflow's live-Ofcom check applies to GB routes only. The publish workflow builds US routes side
by side, one job each, and adds them to the site with the GB routes. Building all of them takes about 2,500 runner
minutes, so bundles built on main are kept in the `us-bundles` release and a publish rebuilds only the US routes whose
route file or US settings changed, or whose bundle is six months old (the "rebuild_us" input rebuilds them all). Add another country by adding
`config/countries/<CC>/profile.yaml` (and its networks file) and route files with that `country`.

## Evidence pack (`tcs report`)

A Word report laid out for a tender submission, and an Excel data appendix, per route and scenario:

- **Front matter:** cover with the key figures, document control and revision history, basis of preparation,
  contents page.
- **Executive summary:** key findings written from the results, the weakest sections, and for any scenario other than
  the baseline a side-by-side comparison with the baseline configuration.
- **Numbered sections:**
  - scope and how to read the figures
  - headline results and throughput along the route
  - service class by section, and route heat maps of signal strength, throughput and latency (five bands each,
    darker = worse, with each band's share of the route) plus a per-network signal strip
  - station-to-station and per-link results, and the onboard configuration
  - methodology, data sources and confidence
  - validation status, assumptions and limitations
  - sensitivity to the main assumptions: the simulation re-run with each uncertain assumption (mobile capacity,
    coverage prediction error, satellite capacity, passengers online, and where they apply the EDGE Rail antenna
    benefit) set pessimistically and optimistically, one at a time and all together. The
    resulting ranges are quoted on the cover, in the headline table and in the executive summary.
- **Appendices:** a landscape table of results by section, and a glossary.

Tables and figures are numbered, every page carries the document reference, classification and "Page X of Y", and
the document states plainly that figures are model predictions until validated. Cover details (your organisation,
the client, the tender reference, the classification, the version) come from `config/report.yaml`; for a one-off
pack, `tcs report` takes them on the command line instead (`--prepared-for "London North Eastern Railway (LNER)"
--tender-ref ... --prepared-by ... --classification ... --doc-version ...`). Word offers to
update the document's fields on first opening. Accept it, and the contents page gets its page numbers.

The Excel appendix has a read-me sheet, the headline measures, every section and link, every 50 m point
(including each network's RSRP), the assumptions and the sources.

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
aa/                active antenna (EDGE Rail) modelling: method, vendor-report reader (its data stays out of git)
scripts/           standalone analysis scripts (e.g. SINR against RSRP from Global View)
tests/             hermetic pipeline + unit tests (pytest)
data/              raw/ interim/ processed/ aa/ (git-ignored)
```

## Caveats (also shown in the app)

Ofcom coverage is operator-predicted, not measured; OpenCellID is community data and a missing cell is not a
missing site; Starlink has no public route-level RF telemetry; throughput depends on load and spectrum, not just
signal; rooftop antennas and handsets behind coated glass behave very differently. Every synthetic field is
traceable through `source_flags`, `confidence` and `model_version`.
