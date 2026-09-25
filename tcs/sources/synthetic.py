"""Offline stand-ins. Everything produced here is flagged `synthetic` in source_flags and capped at low confidence.

They exist so the full pipeline and UI run without any credentials or network, and so unit tests are hermetic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from shapely.geometry import LineString

from ..geo import Projector, local_crs
from .base import Provenance, now_iso

# Approximate coordinates for the reference route so offline mode works out of the box.
FALLBACK_STATIONS = {
    "KGX": (51.5308, -0.1238), "SVG": (51.9017, -0.2069), "PBO": (52.5746, -0.2500), "GRA": (52.9111, -0.6420),
    "NNG": (53.0817, -0.7986), "RET": (53.3153, -0.9475), "DON": (53.5220, -1.1400), "YRK": (53.9581, -1.0931),
    "NTR": (54.3330, -1.4410), "DAR": (54.5205, -1.5474), "DHM": (54.7793, -1.5817), "NCL": (54.9683, -1.6174),
    "MPT": (55.1622, -1.6828), "ALM": (55.3927, -1.6367), "BWK": (55.7742, -2.0110), "DUN": (55.9985, -2.5139),
    "EDB": (55.9520, -3.1883),
}
# km from origin, length km, name, dedicated in-tunnel coverage (assumption)
FALLBACK_TUNNELS_KM = [(0.4, 0.8, "Gasworks Tunnel", True), (34.9, 0.5, "Welwyn North Tunnel", False), (36.0, 0.5, "Welwyn South Tunnel", False),
                       (160.7, 0.9, "Stoke Tunnel", False), (173.9, 0.9, "Peascliffe Tunnel", False), (631.2, 0.6, "Calton Tunnel", True)]


def _hash(n: np.ndarray) -> np.ndarray:
    s = np.sin(n) * 43758.5453
    return s - np.floor(s)


def noise1(x: np.ndarray, seed: float) -> np.ndarray:
    i = np.floor(x)
    f = x - i
    u = f * f * (3 - 2 * f)
    return _hash(i * 1.37 + seed * 911.3) * (1 - u) + _hash((i + 1) * 1.37 + seed * 911.3) * u


def fbm1(x: np.ndarray, seed: float, octaves: int = 4) -> np.ndarray:
    a, s, f, n = 0.5, np.zeros_like(x, dtype=float), 1.0, 0.0
    for o in range(octaves):
        s = s + a * noise1(x * f, seed + o * 17)
        n += a
        a *= 0.5
        f *= 2.1
    return s / n


def _catmull_rom(pts: np.ndarray, per_segment: int = 40) -> np.ndarray:
    p = np.vstack([pts[0], pts, pts[-1]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        t = np.linspace(0, 1, per_segment, endpoint=False)[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3))
    out.append(pts[-1][None, :])
    return np.vstack(out)


def synthetic_route(stations_cfg: list[dict], country: str):
    """Spline through station coordinates with gentle lateral wobble. Returns (line WGS84, stations df, provenance)."""
    rows = []
    for s in stations_cfg:
        if s["crs"] not in FALLBACK_STATIONS:
            raise KeyError(f"offline mode has no coordinates for {s['crs']}; use OSM or supply a route file")
        lat, lon = FALLBACK_STATIONS[s["crs"]]
        rows.append({"crs": s["crs"], "name": s["name"], "lat": lat, "lon": lon})
    st = pd.DataFrame(rows)
    proj = Projector(local_crs(st["lon"].mean(), st["lat"].mean(), country))
    x, y = proj.to_xy(st["lon"].values, st["lat"].values)
    ctrl = []
    rng = np.random.default_rng(7)
    for i in range(len(x)):
        ctrl.append((x[i], y[i]))
        if i < len(x) - 1:
            dx, dy = x[i + 1] - x[i], y[i + 1] - y[i]
            L = np.hypot(dx, dy)
            nx, ny = -dy / L, dx / L
            for k in (1, 2, 3):
                w = (rng.random() - 0.5) * min(4000, L * 0.08)
                ctrl.append((x[i] + dx * k / 4 + nx * w, y[i] + dy * k / 4 + ny * w))
    pts = _catmull_rom(np.array(ctrl))
    lon, lat = proj.to_lonlat(pts[:, 0], pts[:, 1])
    line = LineString(np.column_stack([lon, lat]))
    prov = Provenance("synthetic_route", None, now_iso(), notes="spline through approximate station coordinates", synthetic=True)
    return line, st, prov


def synthetic_prior(samples: pd.DataFrame, operators: list[dict]) -> pd.DataFrame:
    """Per-sample coverage prior in score space, one row per (sample, operator)."""
    km = samples["distance_m"].values / 1000.0
    urban = samples["urban_density"].values if "urban_density" in samples else np.zeros_like(km)
    dips = [(148, 20, 0.30), (335, 18, 0.26), (505, 35, 0.42), (578, 25, 0.36), (262, 10, 0.2), (420, 6, 0.3)]
    frames = []
    for k, op in enumerate(operators):
        seed = 11 + 12 * k
        dip = np.zeros_like(km)
        for c, w, depth in dips:
            z = np.abs(km - c) / w
            dip = np.maximum(dip, np.where(z < 1, depth * (1 - z * z), 0) * (0.85 + 0.15 * k))
        prior = np.clip(0.58 + 0.32 * (fbm1(km / 5, seed) * 2 - 1) + 0.22 * urban - dip, 0.02, 0.95)
        frames.append(pd.DataFrame({
            "sample_id": samples["sample_id"].values, "provider_id": op["id"], "prior_score": prior.astype(np.float32),
            "level_4g": np.nan, "level_5g": np.nan, "radio_technology": np.where(prior > 0.55, "5G", "4G"),
            "source": "synthetic_prior", "postcode": None, "postcode_distance_m": np.nan,
        }))
    return pd.concat(frames, ignore_index=True)


def synthetic_cells(samples: pd.DataFrame, operators: list[dict], proj: Projector) -> pd.DataFrame:
    """Plausible site layout along the corridor: denser in urban stretches, offset 0.5-5 km from the line."""
    rng = np.random.default_rng(3)
    km = samples["distance_m"].values / 1000.0
    urban = samples["urban_density"].values if "urban_density" in samples else np.zeros_like(km)
    b = np.radians(samples["bearing_deg"].values)
    rows = []
    cid = 1000
    for k, op in enumerate(operators):
        d = 1.5 + rng.random() * 2
        while d < km[-1]:
            i = int(np.argmin(np.abs(km - d)))
            rural = 1 - urban[i]
            off = (rng.random() - 0.5) * 2 * (1200 + 3500 * rural)
            x = samples["x"].values[i] + np.cos(b[i]) * off
            y = samples["y"].values[i] - np.sin(b[i]) * off
            lon, lat = proj.to_lonlat(x, y)
            rows.append({"cell_key": f"{op['mcc']}-{op['mnc'][0]}-{cid // 100}-{cid}", "provider_id": op["id"], "radio": "LTE", "mcc": op["mcc"],
                         "mnc": op["mnc"][0], "area_or_tac": cid // 100, "cell_id": cid, "pci_or_unit": -1, "latitude": float(lat), "longitude": float(lon),
                         "x": float(x), "y": float(y), "elevation_m": np.float32(np.nan), "samples": 5, "range_m": 3000, "source": "synthetic_cells"})
            cid += 1
            d += 2.2 + 5.5 * rural + rng.random() * 3.5
    return pd.DataFrame(rows)
