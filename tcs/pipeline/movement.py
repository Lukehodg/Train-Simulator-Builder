"""Train movement / time model (Version 1): line-speed caps, accel/decel curves, dwell, timetable stretching."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from ..config import Settings
from ..sources.base import console
from ..sources.timetable import from_yaml, schedule_seconds


def line_speed_kph(settings: Settings, distance_m: np.ndarray, osm_maxspeed: np.ndarray | None) -> np.ndarray:
    cfg = settings.route.get("line_speed_kph", {})
    v = np.full(len(distance_m), float(cfg.get("default", 200)), dtype=np.float64)
    km = distance_m / 1000.0
    for r in cfg.get("restrictions", []):
        v = np.where((km >= r["from_km"]) & (km < r["to_km"]), np.minimum(v, r["kph"]), v)
    if osm_maxspeed is not None:
        ms = np.where(np.isnan(osm_maxspeed), np.inf, osm_maxspeed)
        v = np.minimum(v, ms)
    return v


def movement(settings: Settings, samples: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    rcfg = settings.route
    ds = settings.spacing_m
    n = len(samples)
    a, b = float(rcfg.get("accel_mps2", 0.5)), float(rcfg.get("decel_mps2", 0.7))
    vmax = line_speed_kph(settings, samples["distance_m"].values, samples.get("line_maxspeed_kph", pd.Series(np.nan, index=samples.index)).values) / 3.6
    stops = stations[stations["stop"]].sort_values("distance_m")
    stop_idx = set(int(i) for i in stops["sample_id"].values)
    v = np.zeros(n)
    for i in range(1, n):
        v[i] = 0.0 if i in stop_idx else min(vmax[i], np.sqrt(v[i - 1] ** 2 + 2 * a * ds))
    for i in range(n - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * b * ds))
    dt = np.zeros(n)
    for i in range(1, n):
        vm = max(1.2, (v[i - 1] + v[i]) / 2)
        dt[i] = ds / vm
    dwell = float(rcfg.get("timetable", {}).get("dwell_s", 120))

    # Stretch section run times to match the timetable where the schedule is slower than physics.
    tcfg = rcfg.get("timetable", {})
    calls = from_yaml(tcfg) if tcfg.get("source", "yaml") == "yaml" else pd.DataFrame(columns=["crs", "time", "stop"])
    sched = schedule_seconds(calls, tcfg.get("departure", "00:00"))
    stop_rows = [(str(r["crs"]), int(r["sample_id"])) for _, r in stops.iterrows()]
    for (c0, i0), (c1, i1) in zip(stop_rows[:-1], stop_rows[1:]):
        if c0 in sched and c1 in sched:
            physics = dt[i0 + 1:i1 + 1].sum()
            planned = sched[c1] - sched[c0] - dwell
            if planned > physics > 0:
                dt[i0 + 1:i1 + 1] *= planned / physics
            elif planned < physics:
                console.log(f"[yellow]timetable {c0}->{c1} ({planned:.0f}s) faster than the speed model allows ({physics:.0f}s); keeping physics")
    t = np.cumsum(dt)
    for _, i in stop_rows[1:-1]:
        t[i + 1:] += dwell
    # Effective speed after stretching
    v_eff = np.where(dt > 0, ds / np.maximum(dt, 1e-6), 0.0)
    v_eff[0] = 0.0
    dep = datetime.strptime(tcfg.get("departure", "09:00")[:5], "%H:%M").replace(year=2026, month=1, day=1)
    ts = pd.to_datetime(dep) + pd.to_timedelta(t, unit="s")

    next_station = np.full(n, None, dtype=object)
    ttn = np.zeros(n, dtype=np.float32)
    order = stops.sort_values("distance_m")
    # sample_id is non-decreasing along the sorted stops, so the next stop after sample j is the first with id > j.
    stop_ids = order["sample_id"].to_numpy(dtype=np.int64)
    pos = np.searchsorted(stop_ids, np.arange(n), side="right")
    has_next = pos < len(stop_ids)
    nxt = pos[has_next]
    next_station[has_next] = order["crs"].to_numpy()[nxt]
    ttn[has_next] = t[stop_ids[nxt]] - t[has_next]
    out = samples.copy()
    out["speed_kph"] = (np.minimum(v_eff, vmax) * 3.6).astype(np.float32)
    out["timestamp_sim"] = ts
    out["sim_seconds"] = t.astype(np.float64)
    out["next_station"] = next_station
    out["time_to_next_station_s"] = ttn
    st = stations.copy()
    st["scheduled_time"] = [ts[int(i)] for i in st["sample_id"].values]
    return out, st
