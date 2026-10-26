"""US stations and Amtrak route lines from the US DOT's National Transportation Atlas Database (NTAD).

Two public ArcGIS layers published by the Bureau of Transportation Statistics, no key needed:

- Amtrak Stations: every station with its three-letter Amtrak code (the `crs` field of a US route file).
- Amtrak Routes: one generalised polyline per named service (Acela, Northeast Regional, ...). The track itself is
  routed through OpenStreetMap as in GB (tcs/sources/osm_route.py), which carries tunnels, bridges and line speeds;
  this line stands in when Overpass is unavailable, without those tags.
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from ..geo import Projector
from .base import Provenance, SourceUnavailable, http_get, now_iso

NTAD = "https://services.arcgis.com/xOi1kZaI0eWDREZv/arcgis/rest/services"
STATIONS_URL = f"{NTAD}/NTAD_Amtrak_Stations/FeatureServer/0/query"
ROUTES_URL = f"{NTAD}/NTAD_Amtrak_Routes/FeatureServer/0/query"
JOIN_M = 150.0        # route-line parts whose ends lie this close are joined (the layer is drawn in separate pieces)
SNAP_M = 400.0        # a station joins the line within this distance (the line is generalised, not surveyed)
STATION_MAX_M = 10_000.0   # further than this from every vertex of the line, the station is not on it (New Orleans and
                           # Philadelphia sit 2.3-2.4 km from the nearest vertex of the generalised line)


def _query(url: str, params: dict, raw_dir: Path, name: str, offline: bool) -> dict:
    path = http_get(url, raw_dir=raw_dir, name=name, params={**params, "f": "json"}, ext="json", offline=offline, ttl_days=90)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        path.unlink(missing_ok=True)
        raise SourceUnavailable(f"NTAD sent something other than JSON for {name}") from exc
    if not isinstance(doc, dict) or doc.get("error") or "features" not in doc:
        path.unlink(missing_ok=True)
        raise SourceUnavailable(f"NTAD could not answer the {name} query: {str(doc.get('error') if isinstance(doc, dict) else doc)[:160]}")
    return doc


def fetch_stations(codes: list[str], raw_dir: Path, *, offline: bool = False) -> pd.DataFrame:
    """Amtrak stations by code, in the order given. Columns: crs, name, lat, lon."""
    if any(not c.isalnum() for c in codes):
        raise SourceUnavailable(f"Amtrak station codes are letters and digits only: {codes}")
    where = "Code IN ({})".format(",".join(f"'{c}'" for c in codes))
    doc = _query(STATIONS_URL, {"where": where, "outFields": "Code,StationName,StnType,lat,lon", "returnGeometry": "false"},
                 raw_dir, "ntad_amtrak_stations", offline)
    rows = []
    for f in doc["features"]:
        a = f.get("attributes", {})
        if a.get("lat") is None or a.get("lon") is None:
            continue
        # Bus stops share codes with nothing on the railway; prefer the train station where a code has both.
        rows.append({"crs": a.get("Code"), "name": a.get("StationName"), "lat": float(a["lat"]), "lon": float(a["lon"]),
                     "_train": str(a.get("StnType", "")).upper() == "TRAIN"})
    df = pd.DataFrame(rows, columns=["crs", "name", "lat", "lon", "_train"])
    df = df.sort_values("_train", ascending=False).drop_duplicates("crs").drop(columns="_train")
    missing = sorted(set(codes) - set(df["crs"]))
    if missing:
        raise SourceUnavailable(f"NTAD has no Amtrak station with code: {missing}")
    return df.set_index("crs").loc[codes].reset_index()


def fetch_route(stations: pd.DataFrame, route_name: str, proj: Projector, raw_dir: Path, *, offline: bool = False):
    """The NTAD line of a named Amtrak service, routed station to station. Returns an osm_route.RouteGeometry whose
    segments carry no tunnel, cutting or speed tags (the layer has none)."""
    from .osm_route import RouteGeometry, _segment

    if "'" in route_name:
        raise SourceUnavailable(f"unexpected quote in NTAD route name {route_name!r}")
    doc = _query(ROUTES_URL, {"where": f"name='{route_name}'", "outFields": "name", "outSR": "4326", "returnGeometry": "true"},
                 raw_dir, "ntad_amtrak_route", offline)
    parts = [np.asarray(p, dtype=float)[:, :2] for f in doc["features"] for p in (f.get("geometry") or {}).get("paths", []) if len(p) >= 2]
    if not parts:
        raise SourceUnavailable(f"NTAD has no Amtrak route named {route_name!r}")

    # Graph of the line's vertices in metric space; parts are joined where their ends nearly meet.
    G = nx.Graph()
    lonlat: dict[int, tuple[float, float]] = {}
    xy_parts = []
    nid = 0
    ends = []
    for p in parts:
        x, y = proj.to_xy(p[:, 0], p[:, 1])
        ids = list(range(nid, nid + len(p)))
        nid += len(p)
        for k, i in enumerate(ids):
            lonlat[i] = (float(p[k, 0]), float(p[k, 1]))
        for k in range(len(ids) - 1):
            L = float(np.hypot(x[k + 1] - x[k], y[k + 1] - y[k]))
            G.add_edge(ids[k], ids[k + 1], length=L, weight=L)
        xy_parts.append((np.asarray(x), np.asarray(y), ids))
        ends += [(ids[0], x[0], y[0]), (ids[-1], x[-1], y[-1])]
    allx = np.concatenate([a for a, _, _ in xy_parts])
    ally = np.concatenate([b for _, b, _ in xy_parts])
    allid = np.concatenate([np.asarray(i) for _, _, i in xy_parts])
    for i, ex, ey in ends:
        d = np.hypot(allx - ex, ally - ey)
        for j in np.flatnonzero(d <= JOIN_M):
            if int(allid[j]) != i and not G.has_edge(i, int(allid[j])):
                G.add_edge(i, int(allid[j]), length=float(d[j]), weight=float(d[j]))

    virtual = []
    for _, s in stations.iterrows():
        sx, sy = proj.to_xy(s["lon"], s["lat"])
        d = np.hypot(allx - sx, ally - sy)
        if d.min() > STATION_MAX_M:
            raise SourceUnavailable(f"station {s['crs']} is {d.min() / 1000:.1f} km from the NTAD {route_name!r} line")
        vid = f"station:{s['crs']}"
        for j in np.flatnonzero(d <= max(SNAP_M, float(d.min()) + 1)):
            G.add_edge(vid, int(allid[j]), length=0.0, weight=float(d[j]) * 0.01)
        virtual.append(vid)

    coords: list[tuple[float, float]] = []
    seg_rows: list[dict] = []
    straight: list[str] = []
    for i in range(len(virtual) - 1):
        try:
            path = [n for n in nx.shortest_path(G, virtual[i], virtual[i + 1], weight="weight") if not isinstance(n, str)]
        except nx.NetworkXNoPath:
            straight.append(f"{stations.crs[i]}-{stations.crs[i + 1]}")
            path = [min(G.neighbors(virtual[i]), key=lambda n: G.edges[virtual[i], n]["weight"]),
                    min(G.neighbors(virtual[i + 1]), key=lambda n: G.edges[virtual[i + 1], n]["weight"])]
        for a, b in zip(path[:-1], path[1:]):
            p0, p1 = lonlat[a], lonlat[b]
            if p0 == p1:
                continue
            if not coords:
                coords.append(p0)
            elif coords[-1] != p0:
                seg_rows.append(_segment(coords[-1], p0, {}, None))
                coords.append(p0)
            coords.append(p1)
            seg_rows.append(_segment(p0, p1, {}, None))
    if len(coords) < 2:
        raise SourceUnavailable(f"NTAD {route_name!r} line does not join the stations")
    warnings = [f"track from the NTAD Amtrak Routes line ({route_name}) because OpenStreetMap was unavailable: generalised, and with no "
                "tunnels, bridges or line speeds; list tunnels in the route file's `tunnels` to mark them"]
    if straight:
        warnings.append(f"no NTAD line between {', '.join(straight)}; drawn straight there")
    prov = Provenance(source="ntad_amtrak_routes", url=ROUTES_URL, fetched_at=now_iso(),
                      notes="US DOT BTS National Transportation Atlas Database, Amtrak Routes (public domain); generalised line.")
    return RouteGeometry(line=LineString(coords), segments=pd.DataFrame(seg_rows), stations=stations, source="ntad", provenance=prov,
                         warnings=warnings, straight_legs=straight)


def station_provenance() -> Provenance:
    return Provenance(source="ntad_amtrak_stations", url=STATIONS_URL, fetched_at=now_iso(),
                      notes="US DOT BTS National Transportation Atlas Database, Amtrak Stations (public domain).")
