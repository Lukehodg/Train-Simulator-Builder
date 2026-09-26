# Code review and optimisation — 25 September 2026

Reviewed the maintained Python pipeline, models, validation/reporting, source caching, packaging and TypeScript viewer, including the Train Studio import boundary. The embedded Train Builder HTML and third-party dependencies were not exhaustively audited. This is a source review with automated verification, not field validation of the connectivity predictions.

## Correctness and reliability fixes

- **High — disconnected Wi-Fi could appear usable.** Latency and loss contributed a positive passenger score even when AP capacity was zero. Python and browser models now classify effective Wi-Fi capacity below 1 Mbps as an outage.
- **High — interrupted downloads poisoned the cache.** HTTP downloads wrote directly to the final path. They now use unique temporary files and atomic replacement, close responses and remove incomplete downloads. Offline mode accepts an existing cache even after its normal freshness period.
- **High — Windows launcher directory boundary.** The path-prefix check admitted sibling directory names beginning with `site`. It now requires the directory separator and uses Windows case-insensitive comparison. The launcher still assumes a trusted local distribution directory.
- **Medium — incorrect baseline comparison.** The viewer reused the initially loaded simulation as its baseline even for bundles built with a design/preset. It now simulates the pristine defaults separately.
- **Medium — satellite weather drift.** The browser reused the original weather's satellite handover flags. It now recalculates handover and stability for the selected weather. New bundle metadata carries provider enabled/default-enabled flags and service countries, which the viewer respects.
- **Medium — unknown measurements became outages.** Validation now excludes absent/non-finite RSRP observations instead of inventing outage measurements. Invalid section lengths fail explicitly.
- **Medium — inconsistent outage distance.** Viewer and report KPIs now use Wi-Fi outage class and actual segment lengths, avoiding an extra final sample's distance and reflecting disconnected APs.
- **Medium — misleading batch success.** `build-all` returns a failing exit code if any route failed. Unknown WAN policies also fail instead of silently using another policy.
- **Medium — packaging portability.** npm is resolved explicitly and invoked without POSIX-incompatible `shell=True` argument handling; clean dependency installs use the lockfile via `npm ci`. Missing builds are rejected before an existing staged package is touched.
- **Viewer reliability.** Restricted local storage no longer aborts startup, routes with fewer than two samples fail clearly, example/reset actions refresh the scenario summary, and tree rendering checks that the map layer exists and enforces its instance cap.

## Performance

Measured against the initial Git commit on this machine using the existing ECML dataset (12,645 samples, 63,225 provider observations). These are individual warm-environment component measurements, not whole-application speed guarantees. Before/after dataframes were compared for equality.

| Component | Before | After | Change |
| --- | ---: | ---: | --- |
| Link manager | 115.9 ms | 48.9 ms | 2.37× faster; provider alignment reused across six metrics |
| Movement | 4,366.2 ms | 43.0 ms | 101.64× faster; vectorised next-station lookup replaces per-sample dataframe filtering |
| Handover position error | 78.3 ms | 0.6 ms | ~131× faster in a 3,000-observation/5,000-prediction case; sorted nearest-position search replaces pairwise matrix |

The handover calculation also avoids a 15-million-element distance matrix in that benchmark. Track geometry is now reused near both route endpoints and invalidated for route/theme changes. Tree data periodically refreshes as map tiles arrive. Browser frame-rate improvement has not been measured.

## Verification

- Follow-up adds calibrated RSRP parity, provider-specific confidence and import validation tests, plus real Chromium interaction coverage. See the commands below and the GitHub Actions workflow for repeatable checks.
- Final local verification: **48 Python tests**, **10 JavaScript input-validation tests**, **4 Chromium interaction tests**, correctness/import lint and production build passed. The Python suite includes 72 calibrated/uncalibrated browser-model comparison scenarios. MapLibre's worker is explicitly bundled for development and production; browser tests also fail on local HTTP resource errors.
- Offline pipeline, Train Studio and unit tests plus new cache, validation, packaging and outage regressions.
- Python/browser parity runner compares WAN capacity, calibrated/uncalibrated RSRP, Wi-Fi scores and exact service classes over 72 combinations: all six policies, three weather settings, AP capacity zero/300 Mbps, with and without calibration. It runs inside pytest when Node and the installed web dependencies are available; otherwise it explicitly skips. CI installs the dependencies so this coverage runs there.
- Strict TypeScript checking and Vite production build.
- Python correctness/import lint: `python -m ruff check tcs tests --select F,I --no-cache`. Broader pre-existing style rules are not claimed clean.
- Existing generated route bundles are not regenerated by source changes. Rebuild a route to export the new satellite metadata; absent fields remain backward-compatible.

## Remaining review findings

1. **Calibration follow-up completed.** Section corrections override the operator bias in populated sections; other sections retain that operator's global bias. Version 2 fits RSRP on corrected quality to avoid double correction. Legacy mappings retain their original uncorrected-score interpretation. The viewer receives per-sample RSRP coefficients, and calibration confidence/provenance apply only to affected operator rows outside tunnel overrides. Refitting starts from an uncalibrated baseline. Residual standard deviation remains a fit diagnostic, not a claim of field-validated uncertainty; field accuracy still requires representative measurements.
2. **Route endpoints and timetable precision.** Route sampling truncates the trailing fraction of a sample interval; movement assumes uniform spacing. Changing one requires changing the other. Simulated timestamps use a fixed reference date and timetable helpers discard seconds. The distance KPI fix does not change this movement contract.
3. **Import validation follow-up completed.** Both importers validate project/carriage/equipment structure, unique equipment ids, booleans and finite integer counts. Limits are 5 MiB, 100 carriages, 256 equipment types and 1,000 APs/roof units per carriage. Calibration files reject malformed/non-finite parameters. YAML configuration is still trusted local configuration, not a general untrusted-input API.
4. **Browser coverage follow-up completed.** Chromium tests use actual WebGL rendering with a deterministic offline map style and newly generated synthetic Arrow data. Coverage includes playback, keyboard seeking, scrubbing, station selection, scenario/weather changes, Escape dismissal, theme/inspector/help controls, valid/invalid design import, reset, downloaded exports, route selection and timeline inspection. External basemap services and the entire embedded Train Builder are not audited by these tests.
5. **Delivery/dependencies.** GitHub Actions now runs Python, model-parity, input-validation, build and Chromium checks. MapLibre was upgraded to 6.11.2 or later to address the critical sanitizer advisory reported by npm. Six high-severity transitive audit findings remain in the mesh/glTF/texture-compressor/image-size dependency chain; npm's proposed automatic fix downgrades deck.gl across a major version and was not applied. Python dependencies are still broadly ranged without a lockfile. Cold-start profiling was not performed.
6. **Legacy source.** Root-level `part*.html` and `ecml-link-sim.html` appear separate from the maintained Vite application. They remain preserved until their archival status is decided.

No live provider API calls were needed for these checks, and the local `.env` was not inspected.

## Running the follow-up checks

Install Python development dependencies (`python -m pip install -e '.[dev]'`) and run `npm ci` in `web`. Run `python -m pytest -q` from the project root, then `npm run test:inputs`, `npm run build`, `npx playwright install chromium` and `npm run test:browser` from `web`.

Browser fixtures are generated under ignored `web/tests/.generated` using temporary pipeline directories. Set `TCS_TEST_PYTHON` if the test runner should use a different Python executable. Browser traces are retained on failure; generated data, test results, credentials and build artifacts are excluded from Git.
