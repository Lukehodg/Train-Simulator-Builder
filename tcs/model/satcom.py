"""Satcom (Starlink) model: obstruction -> availability -> capacity/latency, with reason codes.

Not a tower-distance model. Stage 1 is statistical: P(link) conditioned on sky visibility (DEM horizon, tunnel,
canopy, cutting), terminal type, speed and weather. Stage 2 blends terminal telemetry when available.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings

REASONS = ["OPEN_SKY", "TUNNEL", "DEEP_CUTTING", "STATION_CANOPY", "URBAN_OBSTRUCTION", "TEMPORARY_HANDOVER", "WEATHER_PENALTY", "SERVICE_UNAVAILABLE",
           "NOT_FITTED"]


def satcom_observations(settings: Settings, samples: pd.DataFrame, weather: str = "nominal") -> pd.DataFrame:
    frames = []
    for p in settings.starlink["satcom"]["providers"]:
        term = p["terminal"]
        wcfg = p["availability"]["weather"].get(weather, p["availability"]["weather"]["nominal"])
        fitted = bool(settings.sim.get("satcom_enabled", True) and p.get("enabled", True))   # switched off, or no terminal in the design
        in_service = fitted and settings.route["country"].upper() in {c.upper() for c in p["service_area"]["countries"]}
        sky = samples["sky_visibility"].values.astype(np.float32)
        tun = samples["in_tunnel"].values.astype(bool)
        canopy = samples["canopy_probability"].values
        cutting = samples["cutting_depth_m"].values
        urban = samples["urban_density"].values
        speed = samples["speed_kph"].values if "speed_kph" in samples else np.zeros(len(samples))
        sky_eff = np.clip(sky - wcfg["sky_penalty"] - 0.18 * urban * (1 - canopy), 0, 1)
        sky_eff = np.where(tun, 0.0, sky_eff)
        # Brief beam/satellite handover dips: deterministic pseudo-random by distance so runs are reproducible.
        km = samples["distance_m"].values / 1000.0
        temp = (~tun) & (sky_eff > 0.5) & (np.sin(km * 7.3) * np.sin(km * 2.1 + 1.0) > 0.93)
        thr = p["availability"]["sky_threshold"]
        p_link = np.clip((sky_eff - thr) / (1 - thr), 0, 1) ** 0.5 * (1 - p["availability"]["speed_penalty_per_100kph"] * speed / 100)
        avail = in_service & ~tun & (sky_eff >= thr)
        cap_prior = p["capacity_prior_mbps"][term]
        cap = np.where(avail, cap_prior * np.power(sky_eff, 1.4) * wcfg["capacity_factor"] * np.where(temp, 0.3, 1.0), 0.0)
        lat = p["latency_prior_ms"]["base"] + p["latency_prior_ms"]["obstruction_penalty"] * (1 - sky_eff) + np.where(temp, 40, 0)
        loss = np.where(avail, np.where(sky_eff < 0.6, 1.2, 0.3) + np.where(temp, 3.0, 0.0), 100.0)
        reason = np.full(len(samples), "WEATHER_PENALTY" if weather != "nominal" else "OPEN_SKY", dtype=object)
        reason[temp] = "TEMPORARY_HANDOVER"
        reason[urban > 0.6] = "URBAN_OBSTRUCTION"
        reason[cutting > 10] = "DEEP_CUTTING"
        reason[canopy > 0.5] = "STATION_CANOPY"
        reason[tun] = "TUNNEL"
        if not in_service:
            reason[:] = "SERVICE_UNAVAILABLE" if fitted else "NOT_FITTED"
        conf_cfg = settings.sim["confidence"]
        conf = np.where(tun, conf_cfg["tunnel_deterministic"], conf_cfg["starlink_predictive"]).astype(np.float32)
        frames.append(pd.DataFrame({
            "sample_id": samples["sample_id"].values, "route_id": settings.route_id, "provider_id": p["id"], "provider_type": "satcom",
            "radio_technology": "LEO", "quality_score": sky_eff.astype(np.float32), "signal_primary": sky.astype(np.float32),
            "signal_secondary": (1 - p_link).astype(np.float32), "capacity_mbps": cap.astype(np.float32), "latency_ms": lat.astype(np.float32),
            "packet_loss_pct": loss.astype(np.float32), "available": avail, "confidence": conf, "reason_code": reason,
            "serving_cell": None, "serving_distance_m": np.float32(np.nan), "handover": temp,
            "source_flags": f"{p['mode']}|terrain:{samples['terrain_source'].iloc[0] if 'terrain_source' in samples else 'unknown'}",
            "model_version": settings.sim["model_version"], "quality_base": sky.astype(np.float32), "handover_penalty": temp.astype(np.float32),
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
