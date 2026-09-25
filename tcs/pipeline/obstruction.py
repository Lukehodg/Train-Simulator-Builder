"""Terrain enrichment: DEM sampling, railhead profile, cutting/embankment depth, sky visibility for satcom."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings
from ..sources.base import console
from ..sources.terrain import CopernicusDEM, OSTerrain50, SyntheticDEM, sky_visibility
from .sample_route import RouteBundle


def make_dem(settings: Settings, bundle: RouteBundle):
    src = settings.terrain.get("source", "copernicus_glo30")
    lon, lat = bundle.samples["longitude"].values, bundle.samples["latitude"].values
    if settings.offline or src == "synthetic":
        return SyntheticDEM(lon0=float(lon[0]), lat0=float(lat[0]))
    if src == "os_terrain50":
        from pathlib import Path

        from ..config import ROOT

        f = settings.terrain.get("os_terrain50_folder")
        folder = (ROOT / f if f and not Path(f).is_absolute() else Path(f)) if f else settings.paths()["raw"] / "os_terrain50"
        try:
            pad = float(settings.terrain.get("horizon_reach_m", 3000)) + 500
            s_ = bundle.samples
            bbox = (float(s_["x"].min()) - pad, float(s_["y"].min()) - pad, float(s_["x"].max()) + pad, float(s_["y"].max()) + pad)
            return OSTerrain50(folder, bbox_bng=bbox if bundle.proj.crs.to_epsg() == 27700 else None)
        except FileNotFoundError as exc:
            console.log(f"[yellow]{exc}; falling back to Copernicus GLO-30")
    dem = CopernicusDEM(settings.paths()["raw"])
    pad = (settings.terrain.get("horizon_reach_m", 3000) + 500) / 111_000 * 1.6
    try:
        # Prepare windows chunk-wise along the route so we never pull whole 1-degree tiles.
        n = len(lon)
        step = max(1, int(20_000 / settings.spacing_m))
        seen = set()
        for i in range(0, n, step):
            j = min(n - 1, i + step)
            box = (round(lon[i:j + 1].min() - pad, 2), round(lat[i:j + 1].min() - pad, 2), round(lon[i:j + 1].max() + pad, 2), round(lat[i:j + 1].max() + pad, 2))
            if box in seen:
                continue
            seen.add(box)
            dem.prepare(*box)
        return dem
    except Exception as exc:  # network / GDAL failures -> synthetic with a loud warning
        console.log(f"[yellow]Copernicus DEM unavailable ({exc}); using synthetic terrain")
        return SyntheticDEM(lon0=float(lon[0]), lat0=float(lat[0]))


def railhead_profile(terrain: np.ndarray, spacing_m: float, max_gradient: float = 0.015, window_m: float = 1500) -> np.ndarray:
    """Railways sit in cuttings and on embankments: smooth the DEM along track and clamp the gradient."""
    w = max(1, int(window_m / spacing_m))
    kernel = np.ones(2 * w + 1) / (2 * w + 1)
    padded = np.pad(terrain, w, mode="edge")
    elev = np.convolve(padded, kernel, mode="valid")
    step = max_gradient * spacing_m
    for _ in range(2):
        for i in range(1, len(elev)):
            elev[i] = np.clip(elev[i], elev[i - 1] - step, elev[i - 1] + step)
        for i in range(len(elev) - 2, -1, -1):
            elev[i] = np.clip(elev[i], elev[i + 1] - step, elev[i + 1] + step)
    return elev.astype(np.float32)


def enrich_terrain(settings: Settings, bundle: RouteBundle) -> pd.DataFrame:
    s = bundle.samples
    dem = make_dem(settings, bundle)
    terrain = dem.sample(s["longitude"].values, s["latitude"].values).astype(np.float32)
    elev = railhead_profile(terrain, settings.spacing_m)
    # Tunnels: the railhead is below terrain by definition; give the profile a floor so cuttings are not absurd.
    diff = terrain - elev
    cutting = np.clip(diff, 0, None)
    embank = np.clip(-diff, 0, None)
    if "osm_cutting" in s:
        cutting = np.where(s["osm_cutting"].values & (cutting < 3), 4.0, cutting)
        embank = np.where(s["osm_embankment"].values & (embank < 2), 3.0, embank)
    cutting = np.where(s["in_tunnel"].values, 0.0, cutting)  # handled by the tunnel flag, not as a cutting

    tcfg = settings.terrain
    sat = settings.starlink["satcom"]["providers"]
    min_el = sat[0]["min_elevation_deg"].get(sat[0]["terminal"], 20) if sat else 20
    console.log(f"sky visibility: {tcfg.get('horizon_azimuths', 16)} rays x {tcfg.get('horizon_reach_m', 3000)} m for {len(s)} samples ({dem.source})")
    sky, horizon = sky_visibility(dem, bundle.proj, s["x"].values, s["y"].values, elev, azimuths=int(tcfg.get("horizon_azimuths", 16)),
                                  reach_m=float(tcfg.get("horizon_reach_m", 3000)), step_m=float(tcfg.get("horizon_step_m", 50)), min_elevation_deg=float(min_el))
    # Cuttings shadow the low sky beyond what a 30 m DEM resolves; add a local penalty, then tunnels/canopies.
    sky = sky - np.clip(cutting / 22.0, 0, 0.55)
    sky = sky * (1 - 0.85 * s["canopy_probability"].values)
    sky = np.where(s["in_tunnel"].values, 0.0, sky)
    out = s.copy()
    out["terrain_m"] = terrain
    out["elevation_m"] = elev
    out["cutting_depth_m"] = cutting.astype(np.float32)
    out["embankment_height_m"] = embank.astype(np.float32)
    out["sky_visibility"] = np.clip(sky, 0, 1).astype(np.float32)
    out["horizon_deg"] = horizon.astype(np.float32)
    out["terrain_source"] = dem.source
    return out
