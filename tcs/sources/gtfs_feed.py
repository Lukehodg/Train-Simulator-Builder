"""Timetable from an operator's GTFS feed (Amtrak publishes one, keyless, refreshed weekly).

`timetable.source: gtfs` in a route file picks one real train from the feed: a trip on `timetable.gtfs.route` (the
GTFS route_long_name, e.g. "Acela") that starts at the route's origin and ends at its destination, runs on a weekday,
and is either train number `train` or the first to leave at or after `depart_after`. Its calling times replace the
route file's `calls`, and stations it does not call at become passing points. When the feed can't be read, the route
file's own `calls` are used, so a build never stops for the timetable.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from .base import SourceUnavailable, console, http_get

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]


def _hhmm(t: str) -> str:
    """GTFS times can run past 24:00 (a trip that crosses midnight): fold back onto the clock."""
    h, m = int(t[:t.index(":")]), int(t.split(":")[1])
    return f"{h % 24:02d}:{m:02d}"


def _mins(t: str) -> int:
    h, m = t.split(":")[:2]
    return int(h) * 60 + int(m)


def pick_trip(feed: Path, route_name: str, origin: str, destination: str, train: str | None = None, depart_after: str | None = None) -> dict:
    """{'train', 'departure', 'calls': {code: 'HH:MM'}, 'feed_version', 'stops': [codes in order]} for the chosen trip."""
    with zipfile.ZipFile(feed) as z:
        read = lambda n: pd.read_csv(z.open(n), dtype=str)
        routes, trips, st, cal = read("routes.txt"), read("trips.txt"), read("stop_times.txt"), read("calendar.txt")
        stops = read("stops.txt")
        info = read("feed_info.txt") if "feed_info.txt" in z.namelist() else pd.DataFrame()
    rid = routes.loc[routes["route_long_name"].str.casefold() == route_name.casefold(), "route_id"]
    if rid.empty:
        raise SourceUnavailable(f"GTFS feed has no route named {route_name!r}")
    weekday = cal.loc[cal[WEEKDAYS].astype(int).sum(axis=1) > 0, "service_id"]
    t = trips[trips["route_id"].isin(rid) & trips["service_id"].isin(weekday)]
    if train:
        t = t[t["trip_short_name"] == str(train)]
    code = stops.set_index("stop_id")["stop_code"].fillna(stops.set_index("stop_id").index.to_series())
    st = st[st["trip_id"].isin(t["trip_id"])].copy()
    st["seq"] = st["stop_sequence"].astype(int)
    st["code"] = st["stop_id"].map(code).fillna(st["stop_id"])
    st = st.sort_values(["trip_id", "seq"])
    ends = st.groupby("trip_id").agg(first=("code", "first"), last=("code", "last"), dep=("departure_time", "first"))
    ends = ends[(ends["first"] == origin) & (ends["last"] == destination)]
    if depart_after and not train:
        later = ends[ends["dep"].map(_mins) >= _mins(depart_after)]
        ends = later if len(later) else ends
    if ends.empty:
        raise SourceUnavailable(f"GTFS feed has no weekday {route_name} trip {origin} -> {destination}" + (f" numbered {train}" if train else ""))
    trip_id = ends["dep"].map(_mins).idxmin()
    rows = st[st["trip_id"] == trip_id]
    times = list(rows["departure_time"].iloc[:-1]) + [rows["arrival_time"].iloc[-1]]     # departures, and the arrival at the end
    calls = {c: _hhmm(x) for c, x in zip(rows["code"], times)}
    return {"train": str(t.loc[t["trip_id"] == trip_id, "trip_short_name"].iloc[0]), "departure": calls[origin], "calls": calls,
            "stops": list(rows["code"]), "feed_version": str(info["feed_version"].iloc[0]) if "feed_version" in info else None}


def apply(settings) -> dict | None:
    """Resolve a GTFS timetable into settings.route['timetable'] (calls, departure, `resolved`). None when not used."""
    tcfg = settings.route.setdefault("timetable", {})
    if tcfg.get("source") != "gtfs":
        return None
    g = tcfg.get("gtfs") or {}
    try:
        feed = http_get(g["url"], raw_dir=settings.paths()["raw"], name="gtfs", ext="zip", timeout=300, ttl_days=7, offline=settings.offline)
        if not zipfile.is_zipfile(feed):
            feed.unlink(missing_ok=True)
            raise SourceUnavailable("GTFS download is not a zip")
        r = pick_trip(feed, g["route"], settings.route["origin_crs"], settings.route["destination_crs"], g.get("train"), g.get("depart_after"))
    except (SourceUnavailable, KeyError) as exc:
        console.log(f"[yellow]GTFS timetable unavailable ({exc}); using the route file's calls")
        return None
    tcfg["calls"], tcfg["departure"] = r["calls"], r["departure"]
    tcfg["resolved"] = {"source": "gtfs", "route": g["route"], "train": r["train"], "feed_version": r["feed_version"], "stops": r["stops"]}
    console.log(f"timetable: {g['route']} train {r['train']} from the GTFS feed, {r['departure']} {settings.route['origin_crs']} -> "
                f"{r['calls'][settings.route['destination_crs']]} {settings.route['destination_crs']}, {len(r['calls'])} calls")
    return r
