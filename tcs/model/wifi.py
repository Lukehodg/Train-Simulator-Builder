"""Passenger Wi-Fi demand model: WAN capacity -> per-user experience -> service class."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings

CLASSES = ["EXCELLENT", "GOOD", "USABLE", "POOR", "OUTAGE"]


def passenger_wifi(settings: Settings, samples: pd.DataFrame, wan: pd.DataFrame) -> pd.DataFrame:
    cfg = settings.sim["passenger_wifi"]
    km = samples["distance_m"].values / 1000.0
    lo, hi = cfg["load_factor_range"]
    load = lo + (hi - lo) * (0.5 + 0.5 * np.sin(km / 90.0))          # load varies along the journey (boarding/alighting)
    pax = cfg["passengers"] * load
    active = pax * cfg["active_share"]
    wan_cap = np.minimum(wan["bonded_capacity_mbps"].values, cfg["ap_capacity_mbps"])
    per_user = np.where(active > 0, np.minimum(cfg["per_user_cap_mbps"], wan_cap * 0.85 / np.maximum(active, 1)), 0.0)
    lat = np.nan_to_num(wan["effective_latency_ms"].values, nan=999.0)
    loss = wan["packet_loss_pct"].values
    w = cfg["score_weights"]
    score = 100 * (w["per_user"] * np.clip(per_user / cfg["per_user_target_mbps"], 0, 1) + w["latency"] * np.clip(1 - lat / 250, 0, 1) + w["loss"] * np.clip(1 - loss / 8, 0, 1))
    score = np.where(wan_cap < 1, 0.0, score)
    th = cfg["classes"]
    cls = np.where(score >= th["EXCELLENT"], "EXCELLENT", np.where(score >= th["GOOD"], "GOOD", np.where(score >= th["USABLE"], "USABLE", np.where(score >= th["POOR"], "POOR", "OUTAGE"))))
    streaming = np.clip((per_user - 1.5) / 2.0, 0, 1)
    return pd.DataFrame({"sample_id": wan["sample_id"].values, "per_user_mbps": per_user.astype(np.float32), "active_users": active.astype(np.float32),
                         "wifi_service_score": score.astype(np.float32), "service_class": cls, "streaming_probability": streaming.astype(np.float32)})
