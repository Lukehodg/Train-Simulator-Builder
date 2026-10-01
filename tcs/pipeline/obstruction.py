"""Terrain enrichment: DEM sampling, railhead profile, cutting/embankment depth, sky visibility for satcom.

In Great Britain, open 2 m LiDAR replaces the 30 m terrain model close to the track wherever the national surveys
cover it: cutting walls and embankments from the bare-earth model, and the near-field skyline (trees, buildings,
cutting walls, bridges over the line) from the surface model. Elsewhere the 30 m model is used as before."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings
from ..sources.base import console
from ..sources.terrain import CopernicusDEM, OSTerrain50, SyntheticDEM, horizon_profile, sky_fraction
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
    lid = lidar_features(settings, bundle) if not getattr(dem, "synthetic", False) else None
    ok = lid["lidar_ok"] if lid is not None else np.zeros(len(s), dtype=bool)
    corridor = float((tcfg.get("lidar") or {}).get("corridor_m", 60))
    observer = np.where(ok, lid["rail_level_m"], elev) if lid is not None else elev
    console.log(f"sky visibility: {tcfg.get('horizon_azimuths', 16)} rays x {tcfg.get('horizon_reach_m', 3000)} m for {len(s)} samples ({dem.source}"
                + (f"; LiDAR within {corridor:.0f} m on {ok.mean():.0%} of them)" if lid is not None else ")"))
    near, far = horizon_profile(dem, bundle.proj, s["x"].values, s["y"].values, observer, azimuths=int(tcfg.get("horizon_azimuths", 16)),
                                reach_m=float(tcfg.get("horizon_reach_m", 3000)), step_m=float(tcfg.get("horizon_step_m", 50)),
                                split_m=corridor if lid is not None else 0.0)
    hz = np.fmax(near, far)
    if lid is not None:                                   # LiDAR sees the near field: trees, buildings, cutting walls
        hz = np.where(ok[:, None], np.fmax(far, np.nan_to_num(lid["horizon_near_deg"], nan=-90.0)), hz)
        import warnings

        with warnings.catch_warnings():                   # one side may have no data: the other side stands for both
            warnings.simplefilter("ignore", RuntimeWarning)
            walls = np.nanmean(np.c_[lid["wall_left_m"], lid["wall_right_m"]], axis=1)
            falls = np.nanmean(np.c_[lid["fall_left_m"], lid["fall_right_m"]], axis=1)
        cutting = np.where(ok & ~s["in_tunnel"].values, walls, cutting)
        embank = np.where(ok & ~s["in_tunnel"].values, falls, embank)
    sky = sky_fraction(hz, float(min_el))
    # Without LiDAR, cuttings shadow the low sky beyond what a 30 m DEM resolves: a local penalty stands in for it.
    sky = np.where(ok, sky, sky - np.clip(cutting / 22.0, 0, 0.55))
    canopy = 0.85 * s["canopy_probability"].values
    roof = np.where(ok, np.fmax(canopy, np.nan_to_num(lid["overhead_fraction"]) if lid is not None else 0.0), canopy)
    sky = sky * (1 - roof)
    sky = np.where(s["in_tunnel"].values, 0.0, sky)
    out = s.copy()
    out["terrain_m"] = terrain
    out["elevation_m"] = elev
    out["cutting_depth_m"] = np.nan_to_num(cutting).astype(np.float32)
    out["embankment_height_m"] = np.nan_to_num(embank).astype(np.float32)
    out["sky_visibility"] = np.clip(sky, 0, 1).astype(np.float32)
    out["horizon_deg"] = hz.mean(axis=1).astype(np.float32)
    out["terrain_source"] = dem.source
    out["lidar"] = ok
    if lid is not None:
        out["lidar_source"] = np.where(ok, lid["lidar_source"], "").astype(str)
        for k in ("rail_level_m", "wall_left_m", "wall_right_m", "overhead_fraction", "obstruction_share"):
            out[k] = np.where(ok, lid[k], np.nan).astype(np.float32)
    return out


LIDAR_VERSION = "lidar-v2"                                # bump when the feature definitions change: the cache key includes it


def lidar_features(settings: Settings, bundle: RouteBundle) -> dict[str, np.ndarray] | None:
    """Per-sample LiDAR features for a GB route (sources.lidar), cached by route geometry; None where it does not apply."""
    import hashlib

    lcfg = settings.terrain.get("lidar") or {}
    if settings.offline or not lcfg.get("enabled", True) or str(settings.route.get("country", "")).upper() not in {"GB", "UK"}:
        return None
    if bundle.proj.crs.to_epsg() != 27700:
        return None
    s = bundle.samples
    corridor, azimuths = float(lcfg.get("corridor_m", 60)), int(settings.terrain.get("horizon_azimuths", 16))
    on_bridge = s["on_bridge"].fillna(False).to_numpy(dtype=bool) if "on_bridge" in s else np.zeros(len(s), dtype=bool)
    roofed = s["canopy_probability"].fillna(0).to_numpy() > 0.3 if "canopy_probability" in s else np.zeros(len(s), dtype=bool)
    key = hashlib.sha1(np.round(np.c_[s["x"].values, s["y"].values, s["bearing_deg"].values], 1).tobytes() + on_bridge.tobytes() + roofed.tobytes()
                       + f"{LIDAR_VERSION}|{corridor}|{azimuths}".encode()).hexdigest()[:16]
    cache = settings.paths()["raw"] / "lidar" / f"features_{key}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        from ..sources.lidar import Lidar, corridor_features

        try:
            lidar = Lidar(settings.paths()["raw"].parent / "shared" / "lidar_index")
        except Exception as exc:  # noqa: BLE001
            console.log(f"[yellow]LiDAR unavailable ({exc}); using the 30 m terrain model")
            return None
        console.log(f"LiDAR: reading a {corridor:.0f} m corridor either side of {len(s):,} samples (England, Wales, Scotland open surveys)")
        f = corridor_features(s["x"].values, s["y"].values, s["bearing_deg"].values, on_bridge, roofed, lidar, corridor_m=corridor,
                              azimuths=azimuths, workers=int(lcfg.get("workers", 8)))
        df = pd.DataFrame({k: v for k, v in f.items() if k not in ("horizon_near_deg", "lidar_failed")})
        for k in range(azimuths):
            df[f"hz_{k:02d}"] = f["horizon_near_deg"][:, k]
        df["lidar_source"] = df["lidar_source"].astype(str)
        if f["lidar_failed"].any():                       # not cached: the next build asks the services again
            console.log(f"[yellow]LiDAR: {int(f['lidar_failed'].sum()):,} samples could not be read (service errors); they use the 30 m "
                        "terrain model in this build")
        else:
            cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache, index=False)
    out = {c: df[c].to_numpy() for c in df.columns if not c.startswith("hz_")}
    out["lidar_ok"] = out["lidar_ok"].astype(bool)
    out["horizon_near_deg"] = df[[f"hz_{k:02d}" for k in range(azimuths)]].to_numpy(dtype=np.float32)
    console.log(f"LiDAR: {out['lidar_ok'].mean():.0%} of samples covered ("
                + ", ".join(f"{k} {v:.0%}" for k, v in pd.Series(out["lidar_source"][out["lidar_ok"]]).value_counts(normalize=True).items()) + ")")
    return out
