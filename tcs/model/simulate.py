"""Orchestrates the per-sample simulation: cellular + satcom -> link manager -> passenger Wi-Fi -> confidence."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings
from .bonding import link_manager
from .cellular import cellular_observations
from .confidence import cellular_confidence
from .satcom import satcom_observations
from .wifi import passenger_wifi


def simulate(settings: Settings, samples: pd.DataFrame, prior: pd.DataFrame, serving: pd.DataFrame, *, calibration: dict | None = None,
             measured_sample_ids: set[int] | None = None, weather: str = "nominal", policy: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (provider_observation long table, route_connectivity table)."""
    cell = cellular_observations(settings, samples, prior, serving, calibration=calibration)
    cell["confidence"] = cellular_confidence(settings, cell, measured_sample_ids, calibrated=bool(calibration))
    cell = cell.drop(columns=[c for c in cell.columns if c.startswith("_")])
    sat = satcom_observations(settings, samples, weather=weather)
    obs = pd.concat([cell, sat], ignore_index=True)
    obs["serving_distance_m"] = obs["serving_distance_m"].astype(np.float32)

    wan = link_manager(settings, obs, policy=policy)
    wifi = passenger_wifi(settings, samples, wan)
    rc = wan.merge(wifi, on="sample_id").merge(samples[["sample_id", "timestamp_sim", "distance_m"]], on="sample_id")
    rc["route_id"] = settings.route_id
    flags = obs.groupby("sample_id")["source_flags"].agg(lambda s: "|".join(sorted({f for x in s for f in str(x).split("|")})))
    rc["source_flags"] = rc["sample_id"].map(flags)
    rc["model_version"] = settings.sim["model_version"]
    cols = ["sample_id", "route_id", "timestamp_sim", "distance_m", "active_links", "wan_policy", "bonded_capacity_mbps", "effective_latency_ms",
            "packet_loss_pct", "per_user_mbps", "active_users", "wifi_service_score", "service_class", "streaming_probability", "confidence",
            "n_links", "source_flags", "model_version"]
    return obs, rc[cols]
