"""Write processed outputs: Parquet tables, GeoJSON, and the web bundle (Arrow IPC + meta.json)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather

from ..config import Settings
from ..model.cellular import das_mask
from ..schema import PROVIDER_OBSERVATION, ROUTE_CONNECTIVITY
from .sample_route import RouteBundle

SAMPLE_COLS = ["sample_id", "distance_m", "latitude", "longitude", "elevation_m", "terrain_m", "bearing_deg", "in_tunnel", "tunnel_name",
               "cutting_depth_m", "embankment_height_m", "on_bridge", "canopy_probability", "urban_density", "sky_visibility", "horizon_deg",
               "speed_kph", "sim_seconds", "next_station", "time_to_next_station_s", "station_nearby"]
PROVIDER_COLS = ["quality_score", "quality_base", "signal_primary", "signal_secondary", "capacity_mbps", "latency_ms", "packet_loss_pct",
                 "available", "confidence", "reason_code", "serving_cell", "serving_distance_m", "handover", "handover_penalty",
                 "radio_technology", "source_flags", "rsrp_slope", "rsrp_intercept"]
SHORT = {"quality_score": "q", "quality_base": "qb", "signal_primary": "sig", "signal_secondary": "sig2", "capacity_mbps": "cap", "latency_ms": "lat",
         "packet_loss_pct": "loss", "available": "avail", "confidence": "conf", "reason_code": "reason", "serving_cell": "cell",
         "serving_distance_m": "celld", "handover": "ho", "handover_penalty": "hp", "radio_technology": "tech", "source_flags": "src",
         "rsrp_slope": "rsrp_slope", "rsrp_intercept": "rsrp_intercept"}


def _cast(df: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    cols = [f.name for f in schema if f.name in df.columns]
    t = pa.Table.from_pandas(df[cols], preserve_index=False)
    return t.cast(pa.schema([schema.field(c) for c in cols]), safe=False)


def export_all(settings: Settings, bundle: RouteBundle, samples: pd.DataFrame, stations: pd.DataFrame, cells: pd.DataFrame,
               obs: pd.DataFrame, rc: pd.DataFrame, prior: pd.DataFrame) -> dict[str, Path]:
    paths = settings.paths()
    out, web = paths["processed"], paths["web"]
    out.mkdir(parents=True, exist_ok=True)
    web.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    samples.to_parquet(out / "route_samples.parquet", index=False)
    written["route_samples"] = out / "route_samples.parquet"
    pa.parquet.write_table(_cast(obs, PROVIDER_OBSERVATION), out / "provider_observation.parquet")
    written["provider_observation"] = out / "provider_observation.parquet"
    pa.parquet.write_table(_cast(rc, ROUTE_CONNECTIVITY), out / "route_connectivity.parquet")
    written["route_connectivity"] = out / "route_connectivity.parquet"
    cells.to_parquet(out / "cells.parquet", index=False)
    prior.to_parquet(out / "coverage_prior.parquet", index=False)
    stations.to_parquet(out / "stations.parquet", index=False)

    # GeoJSON for GIS tools / GeoLibre
    line = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"route_id": settings.route_id, "source": bundle.geometry_source},
                                                       "geometry": {"type": "LineString", "coordinates": [[float(a), float(b)] for a, b in bundle.line_lonlat]}}]}
    (out / "route.geojson").write_text(json.dumps(line), encoding="utf-8")
    st_fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {k: (v.isoformat() if hasattr(v, "isoformat") else (None if pd.isna(v) else v)) for k, v in r.items() if k not in ("latitude", "longitude")},
         "geometry": {"type": "Point", "coordinates": [float(r["longitude"]), float(r["latitude"])]}} for r in stations.to_dict("records")]}
    (out / "stations.geojson").write_text(json.dumps(st_fc, default=str), encoding="utf-8")

    # ---- Web bundle: wide Arrow table (one row per sample) --------------------------------------
    wide = samples[[c for c in SAMPLE_COLS if c in samples.columns]].copy()
    providers = []
    for pid, g in obs.groupby("provider_id", sort=False):
        g = g.set_index("sample_id").reindex(wide["sample_id"].values)
        ptype = g["provider_type"].dropna().iloc[0]
        providers.append({"id": pid, "type": ptype})
        for c in PROVIDER_COLS:
            if c in g.columns:
                wide[f"{pid}_{SHORT[c]}"] = g[c].values
    rcw = rc.set_index("sample_id").reindex(wide["sample_id"].values)
    for c in ["active_links", "bonded_capacity_mbps", "effective_latency_ms", "packet_loss_pct", "per_user_mbps", "active_users", "wifi_service_score", "service_class", "confidence"]:
        wide[f"wan_{c}"] = rcw[c].values
    for c in wide.columns:
        if wide[c].dtype == object:
            wide[c] = wide[c].astype("string")
    table = pa.Table.from_pandas(wide, preserve_index=False)
    for i, f in enumerate(table.schema):            # dictionary-encode strings: ~5x smaller bundle, transparent in Arrow JS
        if pa.types.is_string(f.type) or pa.types.is_large_string(f.type):
            table = table.set_column(i, f.name, table.column(i).dictionary_encode())
    feather.write_feather(table, web / "route.arrow", compression="uncompressed")
    written["web_route"] = web / "route.arrow"
    cells_w = cells[["cell_key", "provider_id", "radio", "latitude", "longitude", "samples", "range_m", "source"]].copy()
    feather.write_feather(pa.Table.from_pandas(cells_w, preserve_index=False), web / "cells.arrow", compression="uncompressed")

    # Tunnels as spans
    tunnels = []
    if samples["in_tunnel"].any():
        t = samples[["sample_id", "distance_m", "in_tunnel", "tunnel_name"]]
        grp = (t["in_tunnel"] != t["in_tunnel"].shift()).cumsum()
        for _, g in t[t["in_tunnel"]].groupby(grp):
            tunnels.append({"name": g["tunnel_name"].dropna().iloc[0] if g["tunnel_name"].notna().any() else "tunnel",
                            "from_m": float(g["distance_m"].min()), "to_m": float(g["distance_m"].max()),
                            "das": bool(das_mask(set(settings.sim["cellular"]["tunnels"].get("das_tunnels", [])), g["tunnel_name"].values, g["distance_m"].values / 1000.0).any())})
    all_sat = settings.starlink["satcom"]["providers"]
    name_by_id = {op["id"]: op["name"] for op in settings.operators} | {p["id"]: p["name"] for p in all_sat}
    for p in providers:
        p["name"] = name_by_id.get(p["id"], p["id"])
        if p["type"] == "cellular":
            op = next(o for o in settings.operators if o["id"] == p["id"])
            p["capacity_prior_mbps"] = op["capacity_prior_mbps"]
        else:
            sp = next(s for s in all_sat if s["id"] == p["id"])
            p["terminal"] = sp["terminal"]
            p["capacity_prior_mbps"] = sp["capacity_prior_mbps"][sp["terminal"]]
            p["capacity_priors"] = sp["capacity_prior_mbps"]          # all terminal classes, so the viewer can switch designs
            dsp = next((d for d in settings.starlink_defaults.get("satcom", {}).get("providers", []) if d["id"] == p["id"]), sp)
            p["terminal_default"] = dsp["terminal"]
            p["enabled"] = sp.get("enabled", True)
            p["enabled_default"] = dsp.get("enabled", True)
            p["service_area"] = sp["service_area"]
            p["min_elevation_deg"] = sp["min_elevation_deg"]
            p["latency_prior_ms"] = sp["latency_prior_ms"]
            p["availability"] = sp["availability"]
    meta = {
        "route": {k: settings.route.get(k) for k in ("id", "name", "country", "operator", "service_id", "direction", "origin_crs", "destination_crs", "sample_spacing_m")},
        "geometry_source": bundle.geometry_source,
        "terrain_source": str(samples["terrain_source"].iloc[0]) if "terrain_source" in samples else "unknown",
        "length_m": float(samples["distance_m"].max()),
        "duration_s": float(samples["sim_seconds"].max()),
        "departure": settings.route.get("timetable", {}).get("departure", "09:00"),
        "n_samples": int(len(samples)),
        "providers": providers,
        "stations": [{"crs": r["crs"], "name": r["name"], "lat": float(r["latitude"]), "lon": float(r["longitude"]), "distance_m": float(r["distance_m"]),
                      "sample_id": int(r["sample_id"]), "stop": bool(r["stop"]), "trainshed": bool(r["trainshed"]),
                      "scheduled": (r["scheduled_time"].strftime("%H:%M") if pd.notna(r.get("scheduled_time")) else None)} for _, r in stations.iterrows()],
        "tunnels": tunnels,
        "line": [[round(float(a), 6), round(float(b), 6)] for a, b in bundle.line_lonlat[::3]],
        "sim": settings.sim,
        "sim_defaults": settings.sim_defaults or settings.sim,
        "provenance": bundle.provenance,
        "coverage_sources": sorted(set(prior["source"].dropna().astype(str))),
        "coverage_share": {str(k): round(float(v), 4) for k, v in prior["source"].fillna("no_coverage_record").value_counts(normalize=True).items()},
        "cell_source": str(cells["source"].iloc[0]) if len(cells) else "none",
        "model_version": settings.sim["model_version"],
        "warnings": bundle.warnings,
    }
    (web / "meta.json").write_text(json.dumps(meta, indent=1, default=_json_default), encoding="utf-8")
    written["web_meta"] = web / "meta.json"
    return written


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)
