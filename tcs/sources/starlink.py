"""Starlink inputs.

Stage 1 needs no feed: service eligibility is a config statement (Land Mobility covers the country) and the
per-sample availability comes from geometry (DEM horizon, tunnels, canopies) in the satcom model.

Stage 2 ingests terminal telemetry. A Starlink dish exposes a local gRPC API (192.168.100.1:9200); the open-source
`starlink-grpc-tools` project dumps `get_history` to CSV (per-second obstruction fraction, downlink throughput,
PoP ping latency and drop rate). Paired with the train's GPS log, that becomes the route-specific empirical
distribution that replaces the generic priors here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_telemetry(path: Path, fields: dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(df[fields["time"]], errors="coerce", utc=True),
        "obstruction": pd.to_numeric(df.get(fields["obstruction"]), errors="coerce"),
        "downlink_mbps": pd.to_numeric(df.get(fields["downlink_bps"]), errors="coerce") / 1e6,
        "latency_ms": pd.to_numeric(df.get(fields["ping_ms"]), errors="coerce"),
        "drop_rate": pd.to_numeric(df.get(fields["drop_rate"]), errors="coerce"),
    })
    return out.dropna(subset=["timestamp"]).reset_index(drop=True)


def join_gps(telemetry: pd.DataFrame, gps: pd.DataFrame) -> pd.DataFrame:
    """Nearest-time join of telemetry to a GPS log (timestamp, latitude, longitude)."""
    gps = gps.sort_values("timestamp")
    return pd.merge_asof(telemetry.sort_values("timestamp"), gps, on="timestamp", direction="nearest", tolerance=pd.Timedelta("2s"))


def empirical_priors(telemetry_on_route: pd.DataFrame, samples: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Per-sample empirical capacity/latency/availability from telemetry attached to route samples."""
    g = telemetry_on_route.groupby("sample_id")
    out = pd.DataFrame({
        "capacity_mbps": g["downlink_mbps"].quantile(0.5),
        "latency_ms": g["latency_ms"].median(),
        "availability": 1.0 - g["drop_rate"].mean().fillna(0),
        "obstruction": g["obstruction"].mean(),
    }).reindex(samples["sample_id"].values)
    return out.rolling(window, min_periods=1, center=True).mean().reset_index().rename(columns={"index": "sample_id"})


def blend_with_priors(sat: pd.DataFrame, emp: pd.DataFrame, weight: float = 0.8) -> pd.DataFrame:
    """Where telemetry exists, blend it into the predictive values (weight = trust in telemetry)."""
    sat = sat.merge(emp, on="sample_id", how="left", suffixes=("", "_emp"))
    has = sat["capacity_mbps_emp"].notna()
    for c in ["capacity_mbps", "latency_ms"]:
        sat.loc[has, c] = (1 - weight) * sat.loc[has, c] + weight * sat.loc[has, f"{c}_emp"]
    sat.loc[has, "confidence"] = np.maximum(sat.loc[has, "confidence"], 0.85)
    sat.loc[has, "source_flags"] = sat.loc[has, "source_flags"] + "|starlink_telemetry"
    return sat.drop(columns=[c for c in sat.columns if c.endswith("_emp")])
