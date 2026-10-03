# Architecture

```
                    config/route.yaml · networks.yaml · starlink.yaml · simulation.yaml
                                              │
        ┌─────────────────────────────────────┼──────────────────────────────────────┐
        ▼                                     ▼                                      ▼
  sources/osm_route      sources/terrain (Copernicus / OS T50)        sources/ofcom_coverage · opencellid (+ masts.py)
  (Overpass, file)       + sources/lidar (GB, 2 m near the track)     measurements · starlink · timetable
                         + sky_visibility (horizon → solid angle)
        │                                     │                                      │
        └────────────► pipeline/sample_route ─┴─► pipeline/obstruction ─► pipeline/join_coverage ─► pipeline/join_cells
                                                                                     │
                                                              pipeline/movement (speed profile, sim clock)
                                                                                     │
                model/cellular ─► model/satcom ─► model/bonding (link manager) ─► model/wifi ─► model/confidence
                                                                                     │
                                pipeline/export: Parquet (long + wide) · GeoJSON · web bundle (Arrow IPC + meta.json)
                                                                                     │
                                          web/: MapLibre GL + deck.gl viewer, client-side scenario model (sim/model.ts)
```

Design rules

- **Configuration, not geography.** Route, providers, and every numeric assumption live in YAML; the code never
  names an operator or a station.
- **Canonical tables first.** Every source is normalised (`tcs/schema.py`) before modelling; the long-form
  `provider_observation` means adding a provider never changes a schema.
- **Provenance everywhere.** `source_flags`, `confidence` and `model_version` ride with every row into the UI.
- **Fallbacks, never silent.** Each adapter degrades to a flagged synthetic stand-in; the app shows which.
- **One model, two runtimes.** The browser mirrors the link manager / Wi-Fi model (parity-checked at load) so
  scenarios switch instantly; the heavy geometry (routing, horizon, spatial joins) stays in Python/DuckDB.
- **GeoLibre-ready.** Outputs are Parquet/GeoJSON; the deck.gl layers in `web/src/map/layers.ts` are framework
  independent and port into a GeoLibre plugin (MapLibre + deck.gl + DuckDB-WASM) as-is.
