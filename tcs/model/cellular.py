"""Cellular link model (layered, per the plan):

Level A  quality = coverage_prior + terrain_adjustment + rail_environment_adjustment + measured_calibration
Level B  cell-aware refinement: serving-cell distance penalty, handover events
Level C  vehicle loss: rooftop antenna vs handset inside the carriage
Then score -> RSRP/SINR (flagged synthetic unless calibrated), throughput, latency, packet loss.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings


def _interp_loss(score: np.ndarray, table: list[list[float]]) -> np.ndarray:
    out = np.full(score.shape, table[-1][1], dtype=np.float32)
    for upper, loss in reversed(table):
        out = np.where(score < upper, loss, out)
    return out


def das_mask(entries: set[str], names: np.ndarray, km: np.ndarray) -> np.ndarray:
    """Tunnels with dedicated in-tunnel coverage: matched by name or by `km:<from>-<to>` ranges (OSM rarely names tunnels)."""
    m = np.zeros(len(names), dtype=bool)
    for e in entries:
        if e.startswith("km:"):
            a, b = (float(x) for x in e[3:].split("-"))
            m |= (km >= a) & (km <= b)
        else:
            m |= np.asarray(names == e)
    return m


def cellular_observations(settings: Settings, samples: pd.DataFrame, prior: pd.DataFrame, serving: pd.DataFrame,
                          calibration: dict | None = None) -> pd.DataFrame:
    cfg = settings.sim["cellular"]
    vcfg = settings.sim["vehicle"]
    vprof = vcfg["profiles"][vcfg["profile"]]
    ops = {op["id"]: op for op in settings.operators}
    das = set(cfg["tunnels"].get("das_tunnels", []))

    df = prior.merge(serving, on=["sample_id", "provider_id"], how="left")
    df = df.merge(samples[["sample_id", "distance_m", "cutting_depth_m", "in_tunnel", "tunnel_name", "urban_density", "speed_kph"]], on="sample_id", how="left")
    s = df["prior_score"].astype(float).values
    has_prior = ~np.isnan(s)
    s = np.where(has_prior, s, 0.45)                       # no record: neutral prior, low confidence (see confidence.py)

    cutting = np.clip(df["cutting_depth_m"].fillna(0).values / cfg["terrain"]["cutting_full_depth_m"], 0, 1) * cfg["terrain"]["cutting_penalty_max"]
    dist_km = df["serving_distance_m"].fillna(np.nan).values / 1000.0
    dist_pen = np.where(np.isnan(dist_km), 0.0, np.clip(dist_km - cfg["cell_distance"]["free_km"], 0, None) * cfg["cell_distance"]["penalty_per_km"])
    vehicle = float(vprof["score_offset"])

    q_base = np.clip(s - cutting - dist_pen, 0, 1)
    raw_quality = np.clip(np.clip(q_base, 0, 1) + vehicle, 0, 1)
    calibrated = np.zeros(len(df), dtype=bool)
    if calibration:
        section_km = float(calibration.get("section_km", 10))
        if not np.isfinite(section_km) or section_km <= 0:
            raise ValueError("calibration section_km must be positive and finite")
        sections = (df["distance_m"].to_numpy() / (section_km * 1000)).astype(int)
        bias = calibration.get("bias", {})
        correction = df["provider_id"].map(bias).fillna(0).to_numpy(copy=True)
        calibrated = df["provider_id"].isin(bias).to_numpy(copy=True)
        for pid, values in calibration.get("sections", {}).items():
            for section, value in values.items():
                mask = (df["provider_id"].to_numpy() == pid) & (sections == int(section))
                correction[mask] = float(value)
                calibrated[mask] = True
        if not np.isfinite(correction).all():
            raise ValueError("calibration biases must be finite")
        q_base = q_base + correction
    q_base = np.clip(q_base, 0, 1).astype(np.float32)
    q = q_base + vehicle
    tun = df["in_tunnel"].fillna(False).values.astype(bool)
    tun_das = tun & das_mask(das, df["tunnel_name"].values, df["distance_m"].values / 1000.0)
    q = np.where(tun_das, cfg["tunnels"]["das_score"], np.where(tun, cfg["tunnels"]["default_score"], q))
    q = np.clip(q, 0, 1).astype(np.float32)

    # Handover degradation
    hp = df["handover_penalty"].fillna(0).values.astype(np.float32)
    hcfg = cfg["handover"]

    rsrp = cfg["rsrp_dbm"]["at_zero"] + (cfg["rsrp_dbm"]["at_one"] - cfg["rsrp_dbm"]["at_zero"]) * q + float(vprof["db_offset"]) * 0
    rsrp_slope = np.full(len(df), cfg["rsrp_dbm"]["at_one"] - cfg["rsrp_dbm"]["at_zero"], dtype=float)
    rsrp_intercept = np.full(len(df), cfg["rsrp_dbm"]["at_zero"], dtype=float)
    mapped = np.zeros(len(df), dtype=bool)
    for pid, mapping in (calibration or {}).get("rsrp_map", {}).items():
        slope, intercept = float(mapping["slope"]), float(mapping["intercept"])
        if not np.isfinite([slope, intercept]).all() or slope <= 0:
            raise ValueError("calibration RSRP mapping must be finite with a positive slope")
        mask = (df["provider_id"].to_numpy() == pid) & ~tun
        rsrp_slope[mask] = slope
        rsrp_intercept[mask] = intercept
        if calibration.get("rsrp_input") != "corrected_quality":
            rsrp_intercept[mask] += slope * (raw_quality[mask] - q[mask])
        mapped[mask] = True
    calibrated = (calibrated | mapped) & ~tun
    rsrp = rsrp_slope * q + rsrp_intercept
    sinr = cfg["sinr_db"]["at_zero"] + (cfg["sinr_db"]["at_one"] - cfg["sinr_db"]["at_zero"]) * q
    tech = df["radio_technology"].fillna("4G").astype(str).values
    cap_prior = np.array([ops[p]["capacity_prior_mbps"].get(t, ops[p]["capacity_prior_mbps"]["4G"]) for p, t in zip(df["provider_id"], tech)], dtype=np.float32)
    floor = cfg["throughput"]["score_floor"]
    usable = np.clip((q - floor) / (1 - floor), 0, 1) ** cfg["throughput"]["curve_exponent"]
    # Mild speed effect (Doppler / scheduler) and multi-cell variance from the serving distance.
    speed_f = 1 - 0.08 * np.clip(df["speed_kph"].fillna(0).values / 200, 0, 1)
    units = float(cfg.get("units_capacity_factor", 1.0) or 1.0)       # several roof units per operator (Train Studio design)
    vcap = float(vprof.get("capacity_factor", 1.0) or 1.0)             # antenna/MIMO class (EDGE Rail 4x4 vs passive 2x2)
    cap = cap_prior * usable * speed_f * (1 - (1 - hcfg["capacity_factor"]) * hp) * units * vcap
    lat = cfg["latency_ms"]["base"] + cfg["latency_ms"]["at_zero_extra"] * (1 - q) + hcfg["latency_spike_ms"] * hp
    loss = _interp_loss(q, cfg["packet_loss_pct"]) + hcfg["packet_loss_pct"] * hp
    avail = q > floor

    flags = np.where(has_prior, df["source"].astype(str).values, "no_coverage_record")
    flags = np.char.add(flags.astype(str), np.where(mapped, "|calibrated_rsrp", "|synthetic_rsrp"))
    flags = np.where(df["serving_cell"].notna().values, np.char.add(flags.astype(str), "|cells"), flags)
    flags = np.char.add(flags.astype(str), np.where(calibrated, "|calibrated", ""))

    out = pd.DataFrame({
        "sample_id": df["sample_id"].values, "route_id": settings.route_id, "provider_id": df["provider_id"].values,
        "provider_type": "cellular", "radio_technology": tech, "quality_score": q,
        "signal_primary": rsrp.astype(np.float32), "signal_secondary": sinr.astype(np.float32),
        "capacity_mbps": cap.astype(np.float32), "latency_ms": lat.astype(np.float32), "packet_loss_pct": loss.astype(np.float32),
        "available": avail, "reason_code": np.where(tun & ~tun_das, "TUNNEL", np.where(avail, "OK", "NO_COVERAGE")),
        "serving_cell": df["serving_cell"].values, "serving_distance_m": df["serving_distance_m"].values.astype(np.float32),
        "handover": df["handover"].fillna(False).values.astype(bool), "source_flags": flags, "model_version": settings.sim["model_version"],
        "quality_base": q_base, "handover_penalty": hp,
        "rsrp_slope": rsrp_slope, "rsrp_intercept": rsrp_intercept,
        "_calibrated": calibrated,
        "_has_prior": has_prior, "_prior_source": df["source"].astype(str).values, "_in_tunnel": tun,
    })
    return out
