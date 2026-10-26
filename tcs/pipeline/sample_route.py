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

    if gsrc in ("osm", "file") and not settings.offline and settings.country_profile.get("stations") == "ntad_amtrak":
        geom = _us_route(settings, gsrc, crs_codes, raw_dir, warnings)
    elif gsrc in ("osm", "file") and not settings.offline:
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
    if not st["distance_m"].is_monotonic_increasing:
        # a train that runs out and back (Lakeland - Tampa - Lakeland) passes a place twice: look for each station
        # only beyond the one before it
        from shapely.ops import substring

        d, out = 0.0, []
        for a, b in zip(sx, sy):
            d += substring(line_xy, d, line_xy.length).project(Point(a, b)) if d < line_xy.length else 0.0
            out.append(d)
        st["distance_m"] = out
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
    if geom.source != "osm" and rcfg.get("tunnels"):
        # Geometry without tunnel tags (NTAD, a supplied file, offline): the route file's own list, by portal position.
        for t in rcfg["tunnels"]:
            a, b = (line_xy.project(Point(proj.to_xy(p[1], p[0]))) for p in (t["from"], t["to"]))
            m = (samples["distance_m"].values >= min(a, b)) & (samples["distance_m"].values <= max(a, b))
            in_tunnel |= m
            tunnel_name[m] = t["name"]
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

    prov = {"route": geom.provenance.__dict__, "warnings": warnings + geom.warnings, "straight_legs": geom.straight_legs}
    return RouteBundle(samples=samples, stations=st, proj=proj, line_lonlat=list(geom.line.coords), geometry_source=geom.source,
                       provenance=prov, warnings=warnings + geom.warnings)


def _us_route(settings: Settings, gsrc: str, crs_codes: list[str], raw_dir: Path, warnings: list[str]) -> RouteGeometry | None:
    """US route: Amtrak stations from NTAD, the track routed through OpenStreetMap (or a supplied file), and the NTAD line
    of the named Amtrak service (geometry.ntad_route) when Overpass cannot be reached. None = synthetic stand-in."""
    from ..sources import ntad_amtrak

    rcfg = settings.route
    if all("lat" in s and "lon" in s for s in rcfg["stations"]):
        # positions in the route file (written from the operator's GTFS stops by tools/amtrak_routes.py): no lookup, and
        # stations outside the US (Toronto, Montreal, Vancouver) that NTAD does not list work too
        st = pd.DataFrame([{"crs": s["crs"], "name": s["name"], "lat": float(s["lat"]), "lon": float(s["lon"])} for s in rcfg["stations"]])
    else:
        try:
            st = ntad_amtrak.fetch_stations(crs_codes, raw_dir, offline=False)
            st["name"] = [s["name"] for s in rcfg["stations"]]      # NTAD names are towns ("Boston, MA" twice): keep the route file's
        except SourceUnavailable as exc:
            warnings.append(f"Amtrak stations unavailable ({exc}); using synthetic geometry")
            console.log(f"[yellow]{warnings[-1]}")
            return None
    guide = None
    if rcfg.get("geometry", {}).get("guide") == "gtfs_shape":
        from ..sources import gtfs_feed

        g = (rcfg.get("timetable") or {}).get("gtfs") or {}
        try:
            guide = gtfs_feed.trip_shape(gtfs_feed.feed(settings), g["route"], rcfg["origin_crs"], rcfg["destination_crs"], g.get("train"))
        except (SourceUnavailable, KeyError) as exc:
            console.log(f"[yellow]GTFS shape unavailable ({exc}); searching for the track around the station chain")
    try:
        if gsrc == "file":
            return load_route_file(Path(rcfg["geometry"]["file"]), crs_codes, st)
        return fetch_route(crs_codes, rcfg["country"], raw_dir, corridor_m=8000, overpass_url=settings.key("OVERPASS_URL"),
                           timeout=int(rcfg.get("geometry", {}).get("overpass_timeout_s", 180)), stations=st, guide=guide)
    except SourceUnavailable as exc:
        name = rcfg.get("geometry", {}).get("ntad_route")
        console.log(f"[yellow]route source '{gsrc}' unavailable ({exc})" + (f"; trying the NTAD '{name}' line" if name else ""))
        if name:
            try:
                return ntad_amtrak.fetch_route(st, name, Projector(local_crs(st["lon"].mean(), st["lat"].mean(), rcfg["country"])), raw_dir)
            except SourceUnavailable as exc2:
                exc = exc2
        warnings.append(f"route source '{gsrc}' unavailable ({exc}); using synthetic geometry")
        console.log(f"[yellow]{warnings[-1]}")
        return None


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
