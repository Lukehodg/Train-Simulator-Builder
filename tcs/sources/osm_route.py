"""Railway centreline + stations from OpenStreetMap via Overpass.

Strategy: fetch the rail network inside a corridor polygon around the ordered station chain, build a graph,
and route the shortest path station-to-station. This yields a true centreline (not a straight line) and keeps
the way tags we need for the obstruction model: tunnel=yes, cutting=yes, embankment=yes, bridge=yes, maxspeed.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from ..geo import Projector, local_crs
from .base import Provenance, SourceUnavailable, http_get, now_iso

DEFAULT_OVERPASS = "https://overpass-api.de/api/interpreter"
WIDER_CORRIDOR = 2.5          # second try, as a multiple of the corridor, when a leg has no rail path


@dataclass
class RouteGeometry:
    line: LineString                      # WGS84 lon/lat
    segments: pd.DataFrame                # per graph edge along the route: lon0,lat0,lon1,lat1,tags
    stations: pd.DataFrame                # crs,name,lat,lon (resolved)
    source: str
    provenance: Provenance
    warnings: list[str] = field(default_factory=list)
    straight_legs: list[str] = field(default_factory=list)   # "BWK-DUN": legs with no rail path, drawn as a straight line


def _overpass(query: str, raw_dir: Path, name: str, url: str | None, offline: bool, timeout: int) -> dict:
    path = http_get(url or DEFAULT_OVERPASS, raw_dir=raw_dir, name=name, data={"data": query}, ext="json", timeout=timeout + 30, offline=offline)
    # A busy Overpass server answers 200 with an HTML error page, or with JSON whose `remark` says the query timed out
    # and whose elements are cut short. Neither is the railway: drop it from the cache so the next run asks again.
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        path.unlink(missing_ok=True)
        raise SourceUnavailable(f"Overpass sent something other than JSON for {name} ({exc.__class__.__name__})") from exc
    remark = str(doc.get("remark", "")) if isinstance(doc, dict) else ""
    if not isinstance(doc, dict) or re.search(r"runtime error|timed out|out of memory", remark, re.I):
        path.unlink(missing_ok=True)
        raise SourceUnavailable(f"Overpass could not finish the {name} query: {remark[:160] or 'unexpected answer'}")
    return doc


def fetch_stations(crs_codes: list[str], country: str, raw_dir: Path, *, overpass_url: str | None, offline: bool, timeout: int) -> pd.DataFrame:
    codes = "|".join(re.escape(c) for c in crs_codes)
    iso = {"GB": "GB", "UK": "GB"}.get(country.upper(), country.upper())
    q = f"""[out:json][timeout:{timeout}];
area["ISO3166-1"="{iso}"]->.a;
(
  node(area.a)["railway"="station"]["ref:crs"~"^({codes})$"];
  way(area.a)["railway"="station"]["ref:crs"~"^({codes})$"];
);
out center tags;"""
    doc = _overpass(q, raw_dir, "osm_stations", overpass_url, offline, timeout)
    rows = []
    for el in doc.get("elements", []):
        tags = el.get("tags", {})
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        if lat is None:
            continue
        rows.append({"crs": tags.get("ref:crs"), "name": tags.get("name"), "lat": float(lat), "lon": float(lon)})
    df = pd.DataFrame(rows, columns=["crs", "name", "lat", "lon"]).drop_duplicates("crs")
    missing = sorted(set(crs_codes) - set(df["crs"]))
    if missing:
        raise SourceUnavailable(f"OSM has no station with ref:crs for: {missing}")
    return df.set_index("crs").loc[crs_codes].reset_index()


def _corridor_poly(stations: pd.DataFrame, proj: Projector, buffer_m: float) -> str:
    x, y = proj.to_xy(stations["lon"].values, stations["lat"].values)
    chain = LineString(np.column_stack([x, y])).buffer(buffer_m, quad_segs=4).simplify(500)
    ring = chain.exterior.coords if chain.geom_type == "Polygon" else max(chain.geoms, key=lambda g: g.area).exterior.coords
    lon, lat = proj.to_lonlat(np.array([c[0] for c in ring]), np.array([c[1] for c in ring]))
    return " ".join(f"{a:.5f} {b:.5f}" for a, b in zip(lat, lon))


def _parse_maxspeed(v: str | None) -> float | None:
    if not v:
        return None
    m = re.match(r"\s*([\d.]+)\s*(mph)?", v)
    if not m:
        return None
    kph = float(m.group(1)) * (1.609344 if m.group(2) else 1.0)
    return kph


def _segment(p0: tuple[float, float], p1: tuple[float, float], tags: dict, way_id) -> dict:
    """One piece of the routed line with the OSM tags the obstruction and movement models use."""
    return {
        "lon0": p0[0], "lat0": p0[1], "lon1": p1[0], "lat1": p1[1],
        "tunnel": tags.get("tunnel") in ("yes", "building_passage"),
        "tunnel_name": (tags.get("tunnel:name") or tags.get("bridge:name") or ("tunnel" if tags.get("tunnel") else None)) if tags.get("tunnel") else None,
        "cutting": tags.get("cutting") in ("yes", "both", "left", "right"),
        "embankment": tags.get("embankment") in ("yes", "both", "left", "right"),
        "bridge": tags.get("bridge") in ("yes", "viaduct"),
        "maxspeed_kph": _parse_maxspeed(tags.get("maxspeed")),
        "line_name": tags.get("name"),
        "way_id": way_id,
    }


def fetch_route(crs_codes: list[str], country: str, raw_dir: Path, *, corridor_m: float = 8000, overpass_url: str | None = None,
                offline: bool = False, timeout: int = 180, stations: pd.DataFrame | None = None) -> RouteGeometry:
    """stations: already resolved (crs, name, lat, lon), e.g. from NTAD for a US route; None = look them up in OSM by ref:crs."""
    if stations is None:
        stations = fetch_stations(crs_codes, country, raw_dir, overpass_url=overpass_url, offline=offline, timeout=timeout)
    proj = Projector(local_crs(stations["lon"].mean(), stations["lat"].mean(), country))
    coords, seg_rows, straight = _route_in_corridor(stations, proj, corridor_m, raw_dir, overpass_url, offline, timeout)
    warnings: list[str] = []
    if straight:
        # The corridor is a band around the straight station-to-station chain, so a line that swings far off it between
        # two stations (round a hill, an estuary) is clipped away. Ask once more with a wider band before drawing it straight.
        wide = corridor_m * WIDER_CORRIDOR
        try:
            retry = _route_in_corridor(stations, proj, wide, raw_dir, overpass_url, offline, timeout)
        except SourceUnavailable as exc:
            warnings.append(f"no rail path for {', '.join(straight)} within {corridor_m / 1000:g} km, and the wider search failed ({exc})")
        else:
            if len(retry[2]) < len(straight):
                coords, seg_rows, straight = retry
    if straight:
        warnings.append(f"no rail path found for {', '.join(straight)}; drawn as a straight line between the stations, so tunnels, "
                        "cuttings and line speeds are missing there")
    line = LineString(coords)
    prov = Provenance(source="openstreetmap", url=overpass_url or DEFAULT_OVERPASS, fetched_at=now_iso(),
                      notes="ODbL. Rail network routed station-to-station; tags carried per edge.")
    return RouteGeometry(line=line, segments=pd.DataFrame(seg_rows), stations=stations, source="osm", provenance=prov, warnings=warnings,
                         straight_legs=straight)



def _route_in_corridor(stations: pd.DataFrame, proj: Projector, corridor_m: float, raw_dir: Path, overpass_url: str | None, offline: bool,
                       timeout: int) -> tuple[list[tuple[float, float]], list[dict], list[str]]:
    """Route station to station through the rail network inside a band of `corridor_m` around the station chain.
    Returns the line's coordinates, one segment row per piece of it, and the legs that had to be drawn straight."""
    poly = _corridor_poly(stations, proj, corridor_m)
    q = f"""[out:json][timeout:{timeout}];
way["railway"="rail"]["service"!~"yard|siding|spur|crossover"](poly:"{poly}");
out body;
>;
out skel qt;"""
    doc = _overpass(q, raw_dir, "osm_rail", overpass_url, offline, timeout)
    nodes: dict[int, tuple[float, float]] = {}
    ways: list[dict] = []
    for el in doc.get("elements", []):
        if el["type"] == "node":
            nodes[el["id"]] = (el["lon"], el["lat"])
        elif el["type"] == "way":
            ways.append(el)
    if not ways:
        raise SourceUnavailable("Overpass returned no railway ways for the corridor")

    # Graph in metric space
    G = nx.Graph()
    node_xy: dict[int, tuple[float, float]] = {}
    for w in ways:
        tags = w.get("tags", {})
        nds = [n for n in w["nodes"] if n in nodes]
        for a, b in zip(nds[:-1], nds[1:]):
            for n in (a, b):
                if n not in node_xy:
                    node_xy[n] = proj.to_xy(*nodes[n])
            (xa, ya), (xb, yb) = node_xy[a], node_xy[b]
            L = float(np.hypot(xb - xa, yb - ya))
            if L <= 0:
                continue
            # Prefer main running lines when several parallel tracks exist.
            penalty = 1.0
            if tags.get("usage") not in (None, "main", "branch"):
                penalty = 1.3
            if tags.get("service"):
                penalty = 1.5
            G.add_edge(a, b, length=L, weight=L * penalty, way=w["id"], tags=tags)

    # Stations as virtual nodes joined to every track node within `snap_m`: on double/quad track the platforms
    # sit on separate parallel ways, and snapping to a single node can force a detour to the next crossover.
    xy = np.array(list(node_xy.values()))
    ids = np.array(list(node_xy.keys()))
    snap_m = 90.0
    virtual = []
    for k, s in stations.iterrows():
        sx, sy = proj.to_xy(s["lon"], s["lat"])
        d = np.hypot(xy[:, 0] - sx, xy[:, 1] - sy)
        near = np.where(d <= snap_m)[0]
        if len(near) == 0:
            near = np.array([int(np.argmin(d))])
        vid = f"station:{s['crs']}"
        for j in near:
            G.add_edge(vid, int(ids[j]), length=0.0, weight=float(d[j]) * 0.01, way=None, tags={})
        virtual.append(vid)

    coords: list[tuple[float, float]] = []
    seg_rows: list[dict] = []
    straight: list[str] = []
    for i in range(len(virtual) - 1):
        try:
            path = [n for n in nx.shortest_path(G, virtual[i], virtual[i + 1], weight="weight") if not isinstance(n, str)]
        except nx.NetworkXNoPath:
            straight.append(f"{stations.crs[i]}-{stations.crs[i + 1]}")
            a = int(ids[np.argmin(np.hypot(xy[:, 0] - proj.to_xy(stations.lon[i], stations.lat[i])[0], xy[:, 1] - proj.to_xy(stations.lon[i], stations.lat[i])[1]))])
            b = int(ids[np.argmin(np.hypot(xy[:, 0] - proj.to_xy(stations.lon[i + 1], stations.lat[i + 1])[0], xy[:, 1] - proj.to_xy(stations.lon[i + 1], stations.lat[i + 1])[1]))])
            path = [a, b]
        for a, b in zip(path[:-1], path[1:]):
            tags = G.edges[a, b]["tags"] if G.has_edge(a, b) else {}
            (lon0, lat0), (lon1, lat1) = nodes[a], nodes[b]
            if not coords:
                coords.append((lon0, lat0))
            elif coords[-1] != (lon0, lat0):
                # Platform hop between parallel tracks: a short lateral jump. It is part of the line, so it gets an
                # (untagged) segment too; otherwise every flag after it is looked up at the wrong distance.
                seg_rows.append(_segment(coords[-1], (lon0, lat0), {}, None))
                coords.append((lon0, lat0))
            coords.append((lon1, lat1))
            seg_rows.append(_segment((lon0, lat0), (lon1, lat1), tags, G.edges[a, b]["way"] if G.has_edge(a, b) else None))
    return coords, seg_rows, straight


def load_route_file(path: Path, crs_codes: list[str], stations: pd.DataFrame) -> RouteGeometry:
    """GPX / GeoJSON / KML / Shapefile / GeoPackage / GeoParquet supplied by an infrastructure manager or curated."""
    import geopandas as gpd

    if path.suffix.lower() in {".parquet", ".geoparquet"}:
        gdf = gpd.read_parquet(path)
    else:
        gdf = gpd.read_file(path)
    gdf = gdf.to_crs(4326)
    geom = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.geometry.unary_union
    from shapely.ops import linemerge

    merged = linemerge(geom) if geom.geom_type == "MultiLineString" else geom
    if merged.geom_type == "MultiLineString":
        merged = max(merged.geoms, key=lambda g: g.length)
    prov = Provenance(source=f"file:{path.name}", url=None, fetched_at=now_iso())
    return RouteGeometry(line=merged, segments=pd.DataFrame(), stations=stations, source="file", provenance=prov)
