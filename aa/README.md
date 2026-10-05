# Active antenna modelling (EDGE Rail)

Modelling the EDGE Rail 5G active antenna against conventional installs:
- **Hardware:** HUBER+SUHNER SENCITY Rail Active Rooftop, with the modem in the radome.
- **Link management:** Motion Applied's Fleet Connect.

This section holds the code and the method only. The simulator itself is in `tcs/`, and the antenna profiles it uses
today are in `config/simulation.yaml`.

- **[METHOD.md](METHOD.md)**: the formulas for active vs passive (e.g. Icomera X6i) performance, per sample and per
  route, where each input comes from, and how to prove it in a field test.
- **`compare_route.py`**: the comparison along one route (first cut, with stated assumptions): throughput per network and for the train, availability, no-service spells and the active antenna's advantage, with a chart along the route.
- **`field_throughput.py`**: the throughput an active antenna reached in service, from its own periodic download-test log: test cadence, tests
  without a result, rate per test moving and standing and by speed band, with a chart along each trip.
- **`antenna_patterns.py`**: reads vendor antenna reports (PDF charts of gain, pattern cuts, S-parameters) into
  per-band tables: gain towards the horizon per port and with 4-port diversity, and port-to-port isolation.

## Data stays out of git

Vendor reports, data sheets, test logs and **everything derived from them** stay out of the repository.

| What | Where |
|---|---|
| Vendor files and logs | `data/aa/vendor/` (ignored by git) |
| Outputs | `data/aa/<topic>/` (ignored) |
| Originals and outputs, kept safe | the project's Google Drive |

`.gitignore` also refuses PDFs, zips, CSVs and spreadsheets anywhere under `aa/`. Only code, the method, and
conclusions drawn from public or licensed-for-reports data (such as the Global View interference figures in
METHOD.md) come here.

## Steps

| | Step | Status |
|---|---|---|
| 1 | **Antenna:** gain towards the horizon per UK band, 4-port diversity, isolation: `python aa/antenna_patterns.py data/aa/vendor/<report>.zip --out data/aa/antenna/<part>` | done for HUBER+SUHNER 1399.99.0153 (2019 measurements; confirm it is the antenna inside the 1499.00.0005 active unit) |
| 2 | **Signal → SINR:** percentiles of SINR against RSRP per band and network from the Global View 4G logs: `scripts/fit_rsrp_sinr.py` | done (shareable on its own) |
| 3 | **SINR → throughput:** the link efficiency of the active antenna's modem from Trainlab logs (SINR, rank, carriers, throughput per second, from a test that fills the link) | field throughput in: one unit in Germany, 23 May 2022, a download test every 30 s, 440 tests (`python aa/field_throughput.py <log> --out data/aa/field/<name>`). Moving, per test: p10 3.5, median 17, p90 34 Mbit/s, 3 tests without a result; the network it was on is not known. The log has no signal columns, so the SINR → throughput fit still needs RSRP/SINR (and the serving cell) per test |
| 4 | **Passive install:** two profiles with placeholder defaults (METHOD.md section 0): P2, a rack router with 2×2 per modem (Icomera X6/X6i, X5 v1, earlier Nomad), and P4, 4×4 per modem (Icomera X5 v2/X7, Nomad 5G). Each needs the real install's cable runs, connectors, splitters and ports, and its roof antenna's data sheet | profiles set; install details needed |
| 5 | **Per-route comparison:** METHOD.md sections 1–2 on every route, with the Monte Carlo of section 4; then replace the flat +6 dB / ×1.35 in the simulator's EDGE Rail profile. First cut: `python aa/compare_route.py <route> --gv <Global View 4G CSV> --sinr-fit <fit.csv from scripts/fit_rsrp_sinr.py>` (outputs in `data/aa/compare/<route>/`). It uses measured signal and interference where the scanner passed, the simulator elsewhere, and both ends of how antenna combining treats interference; link efficiency, bandwidths, load and cable runs are its stated assumptions | first cut run on TransPennine; final numbers after 3–4 |

Requires the `aa` extras: `pip install -e ".[aa]"` (pdfplumber, scipy).
