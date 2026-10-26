"""Write a route file (config/routes/<id>.yaml) for every Amtrak train service in Amtrak's GTFS feed.

    python scripts/amtrak_routes.py                # all services
    python scripts/amtrak_routes.py --only Cardinal --only "Texas Eagle"

Each service is represented by one real weekday train: the one that calls at the most stations (the full run, end to
end), pinned by its train number. The file carries what the build needs and the feed already knows: the stations in
order with their positions (GTFS stops, so stations in Canada work too), the train's calling times as the fallback
timetable, the states the line crosses (for the FCC coverage files, from the Census Bureau's state boundaries along
the feed's shape), and a sample spacing that keeps every route at about 20,000 samples or fewer. The track itself is
found in OpenStreetMap at build time, searched for around the feed's shape (geometry.guide: gtfs_shape).

Run it again after Amtrak changes its timetable: files are rewritten in place. nec_was_bos (the hand-written Acela
route) is left alone.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
GTFS_URL = "https://content.amtrak.com/content/gtfs/GTFS.zip"
STATES_URL = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer/0/query"
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]
SKIP = {
    "acela": "config/routes/nec_was_bos.yaml (hand-written)",
    "commuter rail": "commuter trains, not an Amtrak service",
    "temporary substitute service for train": "bus substitute",
    "lincoln service missouri river runner": "a through train made of two services that have their own files",
}
MAX_SAMPLES = 20_000
SPEED_KPH = 177          # 110 mph: OSM maxspeed tags lower it where they exist, and the train's own calling times set the pace


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower().replace("amtrak ", "")).strip("_")


def km(lon: np.ndarray, lat: np.ndarray) -> float:
    la, lo = np.radians(lat), np.radians(lon)
    return float((6371.0 * np.hypot(np.diff(la), np.diff(lo) * np.cos(la[1:]))).sum())


def hhmm(t: str) -> str:
    h, m = int(t.split(":")[0]), int(t.split(":")[1])
    return f"{h % 24:02d}:{m:02d}"


def states_along(lon: np.ndarray, lat: np.ndarray, http: requests.Session) -> list[str]:
    """US states (postal codes) the line passes through, from the Census Bureau's TIGERweb, every ~5 km."""
    step = max(1, len(lon) // 400)
    pts, out = np.c_[lon, lat][::step], []
    dense = []
    for (a, b), (c, d) in zip(pts[:-1], pts[1:]):           # shapes have straight runs of up to ~30 km: fill them in
        n = max(1, int(math.hypot((c - a) * math.cos(math.radians(b)), d - b) * 111 / 5))
        dense += [(a + (c - a) * k / n, b + (d - b) * k / n) for k in range(n)]
    dense.append(tuple(pts[-1]))
    for i in range(0, len(dense), 500):
        geom = {"points": [[round(x, 5), round(y, 5)] for x, y in dense[i:i + 500]], "spatialReference": {"wkid": 4326}}
        r = http.post(STATES_URL, data={"geometry": json.dumps(geom), "geometryType": "esriGeometryMultipoint", "inSR": 4326,
                                        "spatialRel": "esriSpatialRelIntersects", "outFields": "STUSAB", "returnGeometry": "false",
                                        "f": "json"}, timeout=120)
        r.raise_for_status()
        out += [f["attributes"]["STUSAB"] for f in r.json().get("features", [])]
    return sorted(set(out), key=out.index)                  # in the order first met along the line


def representative(feed: dict[str, pd.DataFrame], name: str) -> dict:
    routes, trips, st, cal, stops, shapes = (feed[k] for k in ("routes", "trips", "stop_times", "calendar", "stops", "shapes"))
    rid = routes.loc[routes["route_long_name"].str.casefold() == name.casefold(), "route_id"]
    weekday = cal.loc[cal[WEEKDAYS].astype(int).sum(axis=1) > 0, "service_id"]
    t = trips[trips["route_id"].isin(rid) & trips["service_id"].isin(weekday)].copy()
    n = st[st["trip_id"].isin(t["trip_id"])].groupby("trip_id").size()
    first = st[st["trip_id"].isin(t["trip_id"])].sort_values(["trip_id", "seq"]).groupby("trip_id")["departure_time"].first()
    t["stops"] = t["trip_id"].map(n)
    t["dep"] = t["trip_id"].map(first).map(lambda s: int(s.split(":")[0]) * 60 + int(s.split(":")[1]))
    t["shape_pts"] = t["shape_id"].map(shapes.groupby("shape_id").size()).fillna(0)
    best = t.sort_values(["stops", "shape_pts", "dep"], ascending=[False, False, True]).iloc[0]
    rows = st[st["trip_id"] == best["trip_id"]].sort_values("seq")
    code = stops.set_index("stop_id")
    cs = [str(code.loc[s, "stop_code"]) if isinstance(code.loc[s, "stop_code"], str) else s for s in rows["stop_id"]]
    times = list(rows["departure_time"].iloc[:-1]) + [rows["arrival_time"].iloc[-1]]
    sh = None
    if isinstance(best["shape_id"], str) and best["shape_pts"] > 1:
        s = shapes[shapes["shape_id"] == best["shape_id"]].sort_values("shape_pt_sequence")
        sh = s[["shape_pt_lon", "shape_pt_lat"]].to_numpy(dtype=float)
    else:                                                    # a train with no shape: borrow one of the service's
        other = t[t["shape_pts"] > 1].sort_values("shape_pts", ascending=False)
        if len(other):
            s = shapes[shapes["shape_id"] == other["shape_id"].iloc[0]].sort_values("shape_pt_sequence")
            sh = s[["shape_pt_lon", "shape_pt_lat"]].to_numpy(dtype=float)
    return {"train": str(best["trip_short_name"]), "codes": cs, "times": [hhmm(x) for x in times], "shape": sh,
            "names": [str(code.loc[s, "stop_name"]) for s in rows["stop_id"]],
            "lat": [float(code.loc[s, "stop_lat"]) for s in rows["stop_id"]], "lon": [float(code.loc[s, "stop_lon"]) for s in rows["stop_id"]]}


def direction(lon0: float, lat0: float, lon1: float, lat1: float) -> str:
    dx, dy = (lon1 - lon0) * math.cos(math.radians((lat0 + lat1) / 2)), lat1 - lat0
    return ("north" if dy > 0 else "south") + "bound" if abs(dy) >= abs(dx) else ("east" if dx > 0 else "west") + "bound"


def q(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def write(name: str, r: dict, states: list[str], length_km: float, feed_version: str | None) -> Path:
    o, d = r["codes"][0], r["codes"][-1]
    rid = f"{slug(name)}_{o}_{d}".lower()
    spacing = 50 if length_km * 1000 / 50 <= MAX_SAMPLES else int(math.ceil(length_km * 1000 / MAX_SAMPLES / 25.0) * 25)
    lines = [f"# {name}, {r['names'][0]} -> {r['names'][-1]} (Amtrak train {r['train']}). Generated by scripts/amtrak_routes.py from",
             f"# Amtrak's GTFS feed{f' (version {feed_version})' if feed_version else ''}: edit that script, not this file, and run it again.",
             "route:",
             f"  id: {rid}",
             f"  name: {q(name)}",
             "  country: US",
             "  operator: Amtrak",
             f"  direction: {direction(r['lon'][0], r['lat'][0], r['lon'][-1], r['lat'][-1])}",
             f"  origin_crs: {o}",
             f"  destination_crs: {d}"]
    if spacing != 50:
        lines.append(f"  sample_spacing_m: {spacing}          # {length_km:,.0f} km: kept to ~{MAX_SAMPLES:,} samples")
    lines += ["  geometry:",
              "    source: osm                 # track routed through OpenStreetMap ...",
              "    guide: gtfs_shape           # ... searched for around the feed's shape of this train",
              f"    ntad_route: {q(name)}     # NTAD Amtrak Routes line used when Overpass is unavailable",
              f"  fcc_states: [{', '.join(states)}]",
              "  stations:"]
    for c, nm, la, lo in zip(r["codes"], r["names"], r["lat"], r["lon"]):
        lines.append(f"  - {{crs: {c}, name: {q(nm)}, lat: {la:.6f}, lon: {lo:.6f}, stop: true, trainshed: false}}")
    lines += ["  timetable:",
              f"    # Train {r['train']} from Amtrak's GTFS feed (keyless, refreshed weekly); the calls below are used if the feed can't be",
              "    # read or no longer has this train.",
              "    source: gtfs",
              "    gtfs:",
              f"      url: {GTFS_URL}",
              f"      route: {q(name)}",
              f"      train: '{r['train']}'",
              f"    departure: '{r['times'][0]}'",
              "    calls:"]
    seen = set()
    for c, t in zip(r["codes"], r["times"]):
        if c not in seen:
            lines.append(f"      {c}: '{t}'")
            seen.add(c)
    lines += ["    dwell_s: 90",
              "  line_speed_kph:",
              f"    default: {SPEED_KPH}                # OSM maxspeed tags lower it where present; the calling times set the pace between stops",
              "    restrictions: []",
              ""]
    path = ROOT / "config" / "routes" / f"{rid}.yaml"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", action="append", help="GTFS route_long_name to write (repeatable); default every service")
    ap.add_argument("--feed", type=Path, help="a downloaded GTFS.zip (default: fetch Amtrak's)")
    args = ap.parse_args()
    http = requests.Session()
    data = args.feed.read_bytes() if args.feed else http.get(GTFS_URL, timeout=300).content
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        feed = {n[:-4]: pd.read_csv(z.open(n), dtype=str) for n in z.namelist() if n.endswith(".txt")}
    feed["stop_times"]["seq"] = feed["stop_times"]["stop_sequence"].astype(int)
    feed["shapes"]["shape_pt_sequence"] = feed["shapes"]["shape_pt_sequence"].astype(int)
    version = feed.get("feed_info", pd.DataFrame()).get("feed_version", pd.Series([None])).iloc[0]
    rail = feed["routes"][feed["routes"]["route_type"] == "2"]["route_long_name"].drop_duplicates()
    names = [n for n in rail if n.casefold() not in SKIP and (not args.only or n in args.only)]
    for n in sorted(rail):
        if n.casefold() in SKIP:
            print(f"skip  {n}: {SKIP[n.casefold()]}")
    for name in sorted(names):
        r = representative(feed, name)
        line = r["shape"] if r["shape"] is not None else np.c_[r["lon"], r["lat"]]
        length = km(line[:, 0], line[:, 1])
        states = states_along(line[:, 0], line[:, 1], http)
        path = write(name, r, states, length, version)
        print(f"wrote {path.relative_to(ROOT)}: train {r['train']}, {len(r['codes'])} stations, ~{length:,.0f} km, states {' '.join(states)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
