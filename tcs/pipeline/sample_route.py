"""Route geometry -> route_samples (fixed spacing) + stations, with rail-environment flags from OSM tags."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point

from ..config import Settings
from ..geo import Projector, local_crs, sample_line, substring_between
from ..sources import synthetic
from ..sources.base import SourceUnavailable, console
from ..sources.osm_route import RouteGeometry, fetch_route, fetch_stations, load_route_file


@dataclass
class RouteBundle:
    samples: pd.DataFrame
    stations: pd.DataFrame
    proj: Projector
    line_lonlat: list[tuple[float, float]]
    geometry_source: str
    provenance: dict
    warnings: list[str]


def _station_proximity_urban(samples: pd.DataFrame, stations: pd.DataFrame) -> np.ndarray:
    """Fallback urban proxy: proximity to calling points (origin/destination weighted as cities)."""
    d = samples["distance_m"].values
    urban = np.zeros(len(d), dtype=np.float32)
    for k, s in stations.iterrows():
        radius = 20000 if k in (0, len(stations) - 1) else (6000 if s["stop"] else 2500)
        strength = 1.0 if k in (0, len(stations) - 1) else (0.6 if s["stop"] else 0.3)
        urban = np.maximum(urban, strength * np.clip(1 - np.abs(d - s["distance_m"]) / radius, 0, 1))
    return urban


def build_route(settings: Settings) -> RouteBundle:
    rcfg = settings.route
    raw_dir = settings.paths()["raw"]
    stations_cfg = rcfg["stations"]
    crs_codes = [s["crs"] for s in stations_cfg]
    gsrc = rcfg.get("geometry", {}).get("source", "osm")
    warnings: list[str] = []
    geom: RouteGeometry | None = None

    if gsrc in ("osm", "file") and not settings.offline:
        try:
            if gsrc == "osm":
                geom = fetch_route(crs_codes, rcfg["country"], raw_dir, corridor_m=8000, overpass_url=settings.key("OVERPASS_URL"),
                                   timeout=int(rcfg.get("geometry", {}).get("overpass_timeout_s", 180)))
            else:
                st = fetch_stations(crs_codes, rcfg["country"], raw_dir, overpass_url=settings.key("OVERPASS_URL"), offline=False, timeout=120)
                geom = load_route_file(Path(rcfg["geometry"]["file"]), crs_codes, st)
        except SourceUnavailable as exc:
            warnings.append(f"route source '{gsrc}' unavailable ({exc}); using synthetic geometry")
            console.log(f"[yellow]{warnings[-1]}")
    if geom is None:
        line, st, prov = synthetic.synthetic_route(stations_cfg, rcfg["country"])
        geom = RouteGeometry(line=line, segments=pd.DataFrame(), stations=st, source="synthetic", provenance=prov)

    proj = Projector(local_crs(geom.stations["lon"].mean(), geom.stations["lat"].mean(), rcfg["country"]))
    line_xy = proj.line_to_xy(geom.line)
    if geom.source == "file":
        o, d = geom.stations.iloc[0], geom.stations.iloc[-1]
        line_xy = substring_between(line_xy, Point(proj.to_xy(o.lon, o.lat)), Point(proj.to_xy(d.lon, d.lat)))

    s = sample_line(line_xy, settings.spacing_m)
    n = len(s["x"])
    lon, lat = proj.to_lonlat(s["x"], s["y"])
    samples = pd.DataFrame({
        "sample_id": np.arange(n, dtype=np.int64), "route_id": settings.route_id, "distance_m": s["distance_m"],
        "latitude": np.asarray(lat), "longitude": np.asarray(lon), "x": s["x"], "y": s["y"], "bearing_deg": s["bearing_deg"].astype(np.float32),
    })

    # Stations -> along-track distance
    st = geom.stations.copy()
    sx, sy = proj.to_xy(st["lon"].values, st["lat"].values)
    st["distance_m"] = [line_xy.project(Point(a, b)) for a, b in zip(sx, sy)]
    st["sample_id"] = np.clip(np.round(st["distance_m"] / settings.spacing_m).astype(int), 0, n - 1)
    cfg_by_crs = {c["crs"]: c for c in stations_cfg}
    st["stop"] = st["crs"].map(lambda c: bool(cfg_by_crs[c].get("stop", False)))
    st["trainshed"] = st["crs"].map(lambda c: bool(cfg_by_crs[c].get("trainshed", False)))
    st = st.rename(columns={"lat": "latitude", "lon": "longitude"})
    if not st["distance_m"].is_monotonic_increasing:
        warnings.append("station order along the geometry is not monotonic; check route direction / station sequence")

    # Rail environment flags
    in_tunnel = np.zeros(n, dtype=bool)
    tunnel_name = np.full(n, None, dtype=object)
    cutting = np.zeros(n, dtype=bool)
    embank = np.zeros(n, dtype=bool)
    bridge = np.zeros(n, dtype=bool)
    maxspeed = np.full(n, np.nan, dtype=np.float32)
    if len(geom.segments):
        seg = geom.segments
        x0, y0 = proj.to_xy(seg["lon0"].values, seg["lat0"].values)
        x1, y1 = proj.to_xy(seg["lon1"].values, seg["lat1"].values)
        seg_len = np.hypot(x1 - x0, y1 - y0)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        if abs(cum[-1] - line_xy.length) > max(1.0, 1e-4 * line_xy.length):   # the segments must tile the line exactly
            warnings.append(f"OSM segments total {cum[-1]:.0f} m but the line is {line_xy.length:.0f} m; tunnel and cutting positions may be off")
            console.log(f"[yellow]{warnings[-1]}")
        idx = np.clip(np.searchsorted(cum, samples["distance_m"].values, side="right") - 1, 0, len(seg) - 1)
        in_tunnel = seg["tunnel"].values[idx].astype(bool)
        tunnel_name = np.where(in_tunnel, seg["tunnel_name"].values[idx], None)
        cutting = seg["cutting"].values[idx].astype(bool)
        embank = seg["embankment"].values[idx].astype(bool)
        bridge = seg["bridge"].values[idx].astype(bool)
        maxspeed = seg["maxspeed_kph"].values[idx].astype(np.float32)
    elif geom.source == "synthetic":
        km = samples["distance_m"].values / 1000
        for start, length, name, _das in synthetic.FALLBACK_TUNNELS_KM.get(settings.route_id, []):
            m = (km >= start) & (km <= start + length)
            in_tunnel |= m
            tunnel_name[m] = name
    samples["in_tunnel"] = in_tunnel
    samples["tunnel_name"] = tunnel_name
    samples["osm_cutting"] = cutting
    samples["osm_embankment"] = embank
    samples["on_bridge"] = bridge
    samples["line_maxspeed_kph"] = maxspeed

    # Canopy + station proximity
    canopy = np.zeros(n, dtype=np.float32)
    nearby = np.full(n, None, dtype=object)
    d = samples["distance_m"].values
    for _, s_ in st.iterrows():
        dd = np.abs(d - s_["distance_m"])
        canopy = np.maximum(canopy, np.where(dd < (150 if s_["trainshed"] else 60), 1.0 if s_["trainshed"] else 0.35, 0))
        nearby[dd < 600] = s_["crs"]
    samples["canopy_probability"] = canopy
    samples["station_nearby"] = nearby
    samples["urban_density"] = _station_proximity_urban(samples, st)
    samples["geometry_source"] = geom.source

    prov = {"route": geom.provenance.__dict__, "warnings": warnings + geom.warnings}
    return RouteBundle(samples=samples, stations=st, proj=proj, line_lonlat=list(geom.line.coords), geometry_source=geom.source,
                       provenance=prov, warnings=warnings + geom.warnings)


def urban_from_point_density(samples: pd.DataFrame, px: np.ndarray, py: np.ndarray, radius_m: float = 1000, saturate: int = 60) -> np.ndarray:
    """Urban proxy from the density of point features (postcode centroids, cells) within radius of each sample."""
    from shapely.geometry import Point
    from shapely.strtree import STRtree

    tree = STRtree([Point(a, b) for a, b in zip(px, py)])
    out = np.zeros(len(samples), dtype=np.float32)
    step = 10
    for i in range(0, len(samples), step):
        pt = Point(samples["x"].values[i], samples["y"].values[i]).buffer(radius_m)
        out[i:i + step] = min(1.0, len(tree.query(pt)) / saturate)
    return out


def save_bundle(b: RouteBundle, interim: Path) -> None:
    interim.mkdir(parents=True, exist_ok=True)
    b.samples.to_parquet(interim / "samples.parquet", index=False)
    b.stations.to_parquet(interim / "stations.parquet", index=False)
    with open(interim / "route_line.json", "w", encoding="utf-8") as fh:
        json.dump({"coordinates": b.line_lonlat, "crs_epsg": b.proj.crs.to_epsg(), "geometry_source": b.geometry_source, "provenance": b.provenance}, fh)


def load_bundle(interim: Path, country: str) -> RouteBundle:
    samples = pd.read_parquet(interim / "samples.parquet")
    stations = pd.read_parquet(interim / "stations.parquet")
    with open(interim / "route_line.json", "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    from pyproj import CRS

    proj = Projector(CRS.from_epsg(meta["crs_epsg"]))
    return RouteBundle(samples, stations, proj, [tuple(c) for c in meta["coordinates"]], meta["geometry_source"], meta["provenance"], meta["provenance"].get("warnings", []))
