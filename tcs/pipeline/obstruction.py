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
    """Per-sample LiDAR features for a GB route (sources.lidar); None where it does not apply.

    Every sample read is kept (data/raw/<route>/lidar), keyed by its position, heading and flags, so a later build
    reads only the samples that are new or that an earlier build could not read (a busy service). Before, one failed
    stretch kept the whole route out of the cache and every publish read every route's LiDAR again."""
    import hashlib

    lcfg = settings.terrain.get("lidar") or {}
    us = lcfg.get("source") == "usgs_3dep"                # US: 3DEP bare earth (no tree / building surface)
    if settings.offline or not lcfg.get("enabled", True):
        return None
    if not us and (str(settings.route.get("country", "")).upper() not in {"GB", "UK"} or bundle.proj.crs.to_epsg() != 27700):
        return None
    s = bundle.samples
    corridor, azimuths = float(lcfg.get("corridor_m", 60)), int(settings.terrain.get("horizon_azimuths", 16))
    on_bridge = s["on_bridge"].fillna(False).to_numpy(dtype=bool) if "on_bridge" in s else np.zeros(len(s), dtype=bool)
    roofed = s["canopy_probability"].fillna(0).to_numpy() > 0.3 if "canopy_probability" in s else np.zeros(len(s), dtype=bool)
    key = hashlib.sha1(f"{LIDAR_VERSION}|{corridor}|{azimuths}|{lcfg.get('source', 'gb')}|{lcfg.get('surface', True)}".encode()).hexdigest()[:16]
    cache = settings.paths()["raw"] / "lidar" / f"samples_{key}.parquet"
    keys = pd.DataFrame({"kx": np.round(s["x"].to_numpy(float) * 10).astype(np.int64), "ky": np.round(s["y"].to_numpy(float) * 10).astype(np.int64),
                         "kb": np.round(s["bearing_deg"].to_numpy(float) * 10).astype(np.int64), "on_bridge": on_bridge, "roofed": roofed})
    kcols = list(keys.columns)
    df = keys
    try:
        if cache.exists():
            df = keys.merge(pd.read_parquet(cache).drop_duplicates(kcols), on=kcols, how="left", validate="many_to_one")
    except Exception as exc:  # noqa: BLE001 - an unreadable cache (a run cut off mid-write) is read again, not a failed build
        console.log(f"[yellow]LiDAR: ignoring an unreadable cache ({exc})")
    todo = df["lidar_ok"].isna().to_numpy() if "lidar_ok" in df else np.ones(len(s), dtype=bool)
    failed = np.zeros(len(s), dtype=bool)
    if todo.any():
        from ..sources.lidar import Lidar, Usgs3dep, corridor_features

        try:
            # point-cloud tiles run to gigabytes along a long route: read in memory, never stored; their project indexes
            # live outside data/raw (which CI caches between runs); only the per-sample features are cached, like GB
            surface = settings.paths()["raw"].parent.parent / "ept_cache" if us and lcfg.get("surface", True) else None
            lidar = Usgs3dep(bundle.proj.crs.to_epsg(), surface=surface) if us else Lidar(settings.paths()["raw"].parent / "shared" / "lidar_index")
        except Exception as exc:  # noqa: BLE001
            console.log(f"[yellow]LiDAR unavailable ({exc}); using the 30 m terrain model")
            if todo.all():
                return None
            lidar = None
        if lidar is not None:
            kept = len(s) - int(todo.sum())
            console.log(f"LiDAR: reading a {corridor:.0f} m corridor either side of {int(todo.sum()):,} samples "
                        + ("(USGS 3DEP bare earth)" if us else "(England, Wales, Scotland open surveys)")
                        + (f"; {kept:,} kept from earlier builds" if kept else ""))
            f = corridor_features(s["x"].values, s["y"].values, s["bearing_deg"].values, on_bridge, roofed, lidar, corridor_m=corridor,
                                  azimuths=azimuths, workers=int(lcfg.get("workers", 8)), todo=todo, outliers=False)
            if us and getattr(lidar, "ept", None) is None:   # bare earth only: nothing is known about trees or buildings
                f["obstruction_share"][:] = np.nan
            elif us:
                got, seen = lidar.surface_cells
                console.log(f"LiDAR: point-cloud surface (trees, buildings) on {got / max(seen, 1):.0%} of the cells read")
            new = pd.DataFrame({k: v for k, v in f.items() if k not in ("horizon_near_deg", "lidar_failed")})
            for k in range(azimuths):
                new[f"hz_{k:02d}"] = f["horizon_near_deg"][:, k]
            new["lidar_source"] = new["lidar_source"].astype(str)
            df = df.copy()
            for c in new.columns:
                df[c] = new[c].to_numpy() if c not in df else np.where(todo, new[c].to_numpy(), df[c].to_numpy())
            failed = f["lidar_failed"]
            if failed.any():                              # not kept: the next build asks the services again
                console.log(f"[yellow]LiDAR: {int(failed.sum()):,} samples could not be read (service errors); they use the 30 m "
                            "terrain model in this build")
            keep = df[~failed & ~df["lidar_ok"].isna().to_numpy()]
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".tmp")
            keep.to_parquet(tmp, index=False)
            tmp.replace(cache)                            # whole or not at all
    df = df.copy()
    df["lidar_ok"] = df["lidar_ok"].fillna(False).astype(bool) if "lidar_ok" in df else False
    df["lidar_source"] = df["lidar_source"].fillna("").astype(str) if "lidar_source" in df else ""
    for c in ("rail_level_m", "wall_left_m", "wall_right_m", "fall_left_m", "fall_right_m", "overhead_fraction", "obstruction_share",
              *(f"hz_{k:02d}" for k in range(azimuths))):
        df[c] = df[c].astype(np.float32) if c in df else np.float32(np.nan)
    df = df.drop(columns=kcols)
    out = {c: df[c].to_numpy() for c in df.columns if not c.startswith("hz_")}
    out["lidar_ok"] = out["lidar_ok"].astype(bool)
    out["horizon_near_deg"] = df[[f"hz_{k:02d}" for k in range(azimuths)]].to_numpy(dtype=np.float32)
    from ..sources.lidar import drop_rail_outliers

    drop_rail_outliers(out, on_bridge)                   # after merging, so kept and new samples are judged together
    console.log(f"LiDAR: {out['lidar_ok'].mean():.0%} of samples covered ("
                + ", ".join(f"{k} {v:.0%}" for k, v in pd.Series(out["lidar_source"][out["lidar_ok"]]).value_counts(normalize=True).items()) + ")")
    return out
