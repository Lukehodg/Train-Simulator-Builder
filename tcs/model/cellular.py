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


def portals(in_tunnel: np.ndarray, distance_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For each sample (in route order): the open-air sample just before and just after its tunnel (-1 at a route end)
    and the distance to each. Open-air samples point at themselves at distance 0."""
    n = len(in_tunnel)
    before, after = np.arange(n), np.arange(n)
    for i in range(1, n):
        if in_tunnel[i]:
            before[i] = before[i - 1] if in_tunnel[i - 1] else i - 1
    if n and in_tunnel[0]:
        before[: np.argmin(in_tunnel) if not in_tunnel.all() else n] = -1
    for i in range(n - 2, -1, -1):
        if in_tunnel[i]:
            after[i] = after[i + 1] if in_tunnel[i + 1] else i + 1
    if n and in_tunnel[-1]:
        last_open = np.flatnonzero(~in_tunnel)
        after[(last_open[-1] + 1 if len(last_open) else 0):] = -1
    d_before = np.where(before >= 0, distance_m - distance_m[np.maximum(before, 0)], np.inf)
    d_after = np.where(after >= 0, distance_m[np.maximum(after, 0)] - distance_m, np.inf)
    return before, after, d_before, d_after


def tunnel_quality(q: np.ndarray, sample_pos: np.ndarray, provider: np.ndarray, samples: pd.DataFrame, floor: float, decay_m: float | None) -> np.ndarray:
    """Signal in a tunnel without in-tunnel coverage: the open-air quality at each portal, falling off exponentially
    with distance into the tunnel (e-folding length decay_m) towards the deep-tunnel floor; the better portal wins.
    decay_m None: the floor throughout. q, sample_pos, provider: one row per (sample, network), sample_pos in route order."""
    if not decay_m:
        return np.full(len(q), floor, dtype=float)
    tun = samples["in_tunnel"].fillna(False).to_numpy(dtype=bool)
    before, after, d_before, d_after = portals(tun, samples["distance_m"].to_numpy(dtype=float))
    out = np.full(len(q), np.nan)
    for pid in pd.unique(provider):
        rows = np.flatnonzero(provider == pid)
        by_pos = np.full(len(tun), np.nan)
        by_pos[sample_pos[rows]] = q[rows]
        pos = sample_pos[rows]
        for side, dist in ((before, d_before), (after, d_after)):
            src = side[pos]
            q_out = np.where(src >= 0, by_pos[np.maximum(src, 0)], np.nan)
            low = np.fmin(q_out, floor)                    # a tunnel is never better than the open air outside it
            out[rows] = np.fmax(out[rows], low + (q_out - low) * np.exp(-dist[pos] / float(decay_m)))
    return np.where(np.isnan(out), floor, out)


def cellular_observations(settings: Settings, samples: pd.DataFrame, prior: pd.DataFrame, serving: pd.DataFrame,
                          calibration: dict | None = None, corrections: pd.DataFrame | None = None) -> pd.DataFrame:
    """corrections: measured corrections along the route (tcs/corrections.py): sample_id, provider_id, correction_db,
    weight; added to the score through the calibration's dB scale."""
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
    meas_db, meas_w, meas_q = np.full(len(df), np.nan), np.zeros(len(df)), np.zeros(len(df))
    if corrections is not None and len(corrections):
        m = df[["sample_id", "provider_id"]].merge(corrections[["sample_id", "provider_id", "correction_db", "weight"]], on=["sample_id", "provider_id"], how="left")
        meas_db, meas_w = m["correction_db"].to_numpy(float), m["weight"].fillna(0).to_numpy(float)
        nominal = float(cfg["rsrp_dbm"]["at_one"] - cfg["rsrp_dbm"]["at_zero"])
        slope_db = df["provider_id"].map({k: float(v["slope"]) for k, v in (calibration or {}).get("rsrp_map", {}).items()}).fillna(nominal).to_numpy(float)
        meas_q = np.where(np.isfinite(meas_db), meas_db / slope_db, 0.0)
    tun = df["in_tunnel"].fillna(False).values.astype(bool)
    q_base = q_base + np.where(tun, 0.0, meas_q)            # inside tunnels the correction goes on the tunnel model, below
    q_base = np.clip(q_base, 0, 1).astype(np.float32)
    q = np.clip(q_base + vehicle, 0, 1)
    tun_das = tun & das_mask(das, df["tunnel_name"].values, df["distance_m"].values / 1000.0)
    tcfg = cfg["tunnels"]
    ordered = samples.sort_values("distance_m", kind="stable").reset_index(drop=True)
    pos = df["sample_id"].map(pd.Series(np.arange(len(ordered)), index=ordered["sample_id"].to_numpy())).to_numpy()
    q_tun = tunnel_quality(q, pos, df["provider_id"].to_numpy(), ordered, float(tcfg["default_score"]), tcfg.get("portal_decay_m"))
    q = np.where(tun_das, tcfg["das_score"], np.where(tun, q_tun, q))
    q = np.where(tun, q + meas_q, q)                        # measured inside: corrects the portal or DAS model there
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
        mask = df["provider_id"].to_numpy() == pid              # tunnels too: their signal is the portals' carried inside
        rsrp_slope[mask] = slope
        rsrp_intercept[mask] = intercept
        if calibration.get("rsrp_input") != "corrected_quality":
            legacy = mask & ~tun
            rsrp_intercept[legacy] += slope * (raw_quality[legacy] - q[legacy])
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
    vcap = float(vprof.get("capacity_factor", 1.0) or 1.0)             # antenna/MIMO class (EDGE Rail 4x4 vs passive 2x2)
    cap = cap_prior * usable * speed_f * (1 - (1 - hcfg["capacity_factor"]) * hp) * vcap
    lat = cfg["latency_ms"]["base"] + cfg["latency_ms"]["at_zero_extra"] * (1 - q) + hcfg["latency_spike_ms"] * hp
    loss = _interp_loss(q, cfg["packet_loss_pct"]) + hcfg["packet_loss_pct"] * hp
    avail = q > floor

    flags = np.where(has_prior, df["source"].astype(str).values, "no_coverage_record")
    flags = np.char.add(flags.astype(str), np.where(mapped, "|calibrated_rsrp", "|synthetic_rsrp"))
    flags = np.where(df["serving_cell"].notna().values, np.char.add(flags.astype(str), "|cells"), flags)
    flags = np.char.add(flags.astype(str), np.where(calibrated, "|calibrated", ""))
    flags = np.char.add(flags.astype(str), np.where(np.isfinite(meas_db), "|measured_correction", ""))

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
        "measured_correction_db": meas_db.astype(np.float32),
        "_calibrated": calibrated, "_meas_w": meas_w,
        "_has_prior": has_prior, "_prior_source": df["source"].astype(str).values, "_in_tunnel": tun,
    })
    return out
