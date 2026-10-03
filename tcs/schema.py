"""Canonical tables. Every source is normalised into these before modelling or rendering."""
from __future__ import annotations

import pyarrow as pa

ROUTE_SAMPLES = pa.schema([
    ("sample_id", pa.int64()),
    ("route_id", pa.string()),
    ("distance_m", pa.float64()),
    ("latitude", pa.float64()),
    ("longitude", pa.float64()),
    ("x", pa.float64()),                 # local metric CRS
    ("y", pa.float64()),
    ("elevation_m", pa.float32()),       # railhead (DEM sampled + smoothed)
    ("terrain_m", pa.float32()),         # DEM at the sample
    ("bearing_deg", pa.float32()),
    ("station_nearby", pa.string()),     # CRS code within 600 m or null
    ("in_tunnel", pa.bool_()),
    ("tunnel_name", pa.string()),
    ("cutting_depth_m", pa.float32()),
    ("embankment_height_m", pa.float32()),
    ("on_bridge", pa.bool_()),
    ("canopy_probability", pa.float32()),
    ("urban_density", pa.float32()),     # 0..1 proxy from building density / station proximity
    ("sky_visibility", pa.float32()),    # 0..1 from DEM horizon + tunnel + canopy
    ("speed_kph", pa.float32()),
    ("timestamp_sim", pa.timestamp("s")),
    ("next_station", pa.string()),
    ("time_to_next_station_s", pa.float32()),
    ("geometry_source", pa.string()),
])

# Long-form: one row per (sample, provider). Preferred production shape - adding a provider never changes schema.
PROVIDER_OBSERVATION = pa.schema([
    ("sample_id", pa.int64()),
    ("route_id", pa.string()),
    ("provider_id", pa.string()),
    ("provider_type", pa.string()),      # cellular | satcom
    ("radio_technology", pa.string()),   # 4G | 5G | LEO | null
    ("quality_score", pa.float32()),     # 0..1 model score
    ("signal_primary", pa.float32()),    # RSRP dBm (cellular) | sky visibility (satcom); NaN when not modelled
    ("signal_secondary", pa.float32()),  # SINR dB | obstruction probability
    ("capacity_mbps", pa.float32()),
    ("latency_ms", pa.float32()),
    ("packet_loss_pct", pa.float32()),
    ("available", pa.bool_()),
    ("confidence", pa.float32()),
    ("reason_code", pa.string()),
    ("serving_cell", pa.string()),       # "mcc-mnc-tac-cid" estimated serving cell or null
    ("serving_distance_m", pa.float32()),
    ("handover", pa.bool_()),
    ("source_flags", pa.string()),       # pipe-separated provenance e.g. ofcom_predicted|opencellid|synthetic_rsrp
    ("measured_correction_db", pa.float32()),   # measured correction applied here (tcs/corrections.py), NaN where none
    ("model_version", pa.string()),
])

ROUTE_CONNECTIVITY = pa.schema([
    ("sample_id", pa.int64()),
    ("route_id", pa.string()),
    ("timestamp_sim", pa.timestamp("s")),
    ("distance_m", pa.float64()),
    ("active_links", pa.string()),
    ("wan_policy", pa.string()),
    ("bonded_capacity_mbps", pa.float32()),
    ("effective_latency_ms", pa.float32()),
    ("packet_loss_pct", pa.float32()),
    ("per_user_mbps", pa.float32()),
    ("active_users", pa.float32()),
    ("wifi_service_score", pa.float32()),
    ("service_class", pa.string()),
    ("confidence", pa.float32()),
    ("source_flags", pa.string()),
    ("model_version", pa.string()),
])

CELLS = pa.schema([
    ("cell_key", pa.string()),
    ("provider_id", pa.string()),
    ("radio", pa.string()),
    ("mcc", pa.int32()),
    ("mnc", pa.int32()),
    ("area_or_tac", pa.int64()),
    ("cell_id", pa.int64()),
    ("pci_or_unit", pa.int64()),
    ("latitude", pa.float64()),
    ("longitude", pa.float64()),
    ("x", pa.float64()),
    ("y", pa.float64()),
    ("elevation_m", pa.float32()),
    ("samples", pa.int32()),
    ("range_m", pa.int32()),
    ("source", pa.string()),
])

STATIONS = pa.schema([
    ("crs", pa.string()),
    ("name", pa.string()),
    ("latitude", pa.float64()),
    ("longitude", pa.float64()),
    ("distance_m", pa.float64()),
    ("sample_id", pa.int64()),
    ("stop", pa.bool_()),
    ("trainshed", pa.bool_()),
    ("scheduled_time", pa.timestamp("s")),
])
