"""Timetable from Network Rail's SCHEDULE feed (Network Rail Open Data, free account; GB routes).

`timetable.source: network_rail` in a route file picks one real passenger train from the daily full extract
(CIF_ALL_FULL_DAILY, JSON): one that calls at the route's origin and later at its destination on a typical weekday (the
coming Wednesday), operated by `timetable.network_rail.toc` (ATOC code, or a list) when given, and either headcode or
train UID `train` or the first to leave at or after `depart_after` (default: the route file's `departure`). Its public
calling times replace the route file's `calls`, and route stations it does not call at become passing points, as for a
GTFS train (gtfs_feed.py). Short-term plans are honoured: a cancellation (STP C) drops the train for that day, an
overlay (STP O) replaces the permanent schedule.

TIPLOCs map to station codes (CRS) from the extract's own TIPLOC records, topped up from CORPUS (the location reference
file from the same service). Some big stations have platforms under a TIPLOC that carries no station code (Victoria's
Brighton side is VICTRIC, Waterloo's main line WATRLMN, St Pancras' HS1 platforms STPANCI): a route file lists those under
`timetable.network_rail.tiplocs` ({CRS: [TIPLOC, ...]}). Where the feed codes a call differently from the route file
(HS1's domestic platforms at Stratford International and Ebbsfleet are SDI and EBF), `aliases` ({feed CRS: route CRS})
renames it. Needs NETWORK_RAIL_USERNAME and NETWORK_RAIL_PASSWORD (repository secrets in CI). When the
feed can't be read, the route file's own `calls` are used, so a build never stops for the timetable.

The extract is ~135 MB gzipped (~3.5 GB of JSON lines, ~720k schedules), downloaded once a day and shared by every GB
route; each route scans it line by line and parses only the schedules that name both its end stations.
"""
from __future__ import annotations

import base64
import gzip
import json
import os
import re
from datetime import date, timedelta
from pathlib import Path

from .base import SourceUnavailable, console, http_get

BASE = "https://publicdatafeeds.networkrail.co.uk/ntrod"
SCHEDULE_URL = f"{BASE}/CifFileAuthenticate?type=CIF_ALL_FULL_DAILY&day=toc-full"
CORPUS_URL = f"{BASE}/SupportingFileAuthenticate?type=CORPUS"
PASSENGER_STATUS = {"P", "1"}               # permanent / short-term passenger trains (B, 5 are buses; the rest freight, ECS)
BUS_CATEGORY = {"BR", "BS"}
_HEAD = re.compile(r'"CIF_stp_indicator":"(\w)","CIF_train_uid":"([^"]+)"')
_FIELD = {k: re.compile(rf'"{k}":"([^"]*)"') for k in ("schedule_days_runs", "schedule_start_date", "schedule_end_date")}


def _auth() -> dict[str, str]:
    user, pw = os.environ.get("NETWORK_RAIL_USERNAME"), os.environ.get("NETWORK_RAIL_PASSWORD")
    if not user or not pw:
        raise SourceUnavailable("NETWORK_RAIL_USERNAME / NETWORK_RAIL_PASSWORD not set")
    # requests drops the header when the service redirects to its file store (another host), as it should
    return {"Authorization": "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()}


def shared_dir(settings) -> Path:
    return settings.paths()["raw"].parent / "_network_rail"


def download(settings) -> tuple[Path, Path | None]:
    """(SCHEDULE extract, CORPUS or None): refreshed daily, shared by every route, an older copy used if the service is down."""
    raw = shared_dir(settings)
    headers = None if settings.offline else _auth()
    sched = http_get(SCHEDULE_URL, raw_dir=raw, name="schedule", headers=headers, ext="json.gz", timeout=600, ttl_days=1,
                     offline=settings.offline, stale_ok=True)
    try:
        corpus = http_get(CORPUS_URL, raw_dir=raw, name="corpus", headers=headers, ext="json.gz", timeout=300, ttl_days=30,
                          offline=settings.offline, stale_ok=True)
    except SourceUnavailable as exc:
        console.log(f"[yellow]CORPUS unavailable ({exc}); station codes from the SCHEDULE extract only")
        corpus = None
    return sched, corpus


def _open(path: Path):
    with open(path, "rb") as fh:
        gz = fh.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rt", encoding="utf-8") if gz else open(path, "r", encoding="utf-8")


def crs_tiplocs(sched: Path, corpus: Path | None) -> dict[str, set[str]]:
    """CRS -> the TIPLOCs at that station (a big station has several: KGX is KNGX, STP is STPX and STPXBOX ...)."""
    out: dict[str, set[str]] = {}

    def add(crs, tiploc):
        crs, tiploc = (crs or "").strip(), (tiploc or "").strip()
        if crs and tiploc:
            out.setdefault(crs, set()).add(tiploc)

    with _open(sched) as fh:
        for line in fh:
            if line.startswith('{"TiplocV1"'):
                t = json.loads(line)["TiplocV1"]
                add(t.get("crs_code"), t.get("tiploc_code"))
            elif line.startswith('{"JsonSchedule') or line.startswith('{"JsonAssociation'):
                break                                   # TIPLOC records come first
    if corpus is not None:
        with _open(corpus) as fh:
            for r in json.load(fh).get("TIPLOCDATA", []):
                add(r.get("3ALPHA"), r.get("TIPLOC"))
    return out


def _runs_on(s: dict, day: date) -> bool:
    days = s.get("schedule_days_runs") or ""
    return (s.get("schedule_start_date", "") <= day.isoformat() <= s.get("schedule_end_date", "")
            and len(days) == 7 and days[day.weekday()] == "1")


def _hhmm(t: str | None) -> str | None:
    """Public times are HHMM; '0000' is the placeholder for no public time (a real midnight call is rare enough)."""
    if not t or len(t) < 4 or not t[:4].isdigit() or t[:4] == "0000":
        return None
    return f"{t[:2]}:{t[2:4]}"


def _mins(t: str) -> int:
    return int(t[:2]) * 60 + int(t[3:5])


def _journey(s: dict, origin: set[str], dest: set[str], code: dict[str, str]) -> dict | None:
    """The part of a schedule from a public departure at the origin to a later public arrival at the destination."""
    seg = s.get("schedule_segment") or {}
    locs = seg.get("schedule_location") or []
    i0 = next((i for i, l in enumerate(locs) if l.get("tiploc_code") in origin and _hhmm(l.get("public_departure"))), None)
    if i0 is None:
        return None
    i1 = next((i for i in range(i0 + 1, len(locs)) if locs[i].get("tiploc_code") in dest and _hhmm(locs[i].get("public_arrival"))), None)
    if i1 is None:
        return None
    calls: dict[str, str] = {}
    for i in range(i0, i1 + 1):
        l = locs[i]
        t = _hhmm(l.get("public_arrival")) if i == i1 else (_hhmm(l.get("public_departure")) or _hhmm(l.get("public_arrival")))
        crs = code.get(l.get("tiploc_code"))
        if t and crs and crs not in calls:
            calls[crs] = t
    return {"uid": s.get("CIF_train_uid"), "train": seg.get("signalling_id") or s.get("CIF_train_uid"), "toc": s.get("atoc_code"),
            "stp": s.get("CIF_stp_indicator"), "calls": calls, "stops": list(calls)}


def typical_weekday(today: date | None = None) -> date:
    """The coming Wednesday (today when it is one): clear of weekend engineering and Monday/Friday patterns."""
    today = today or date.today()
    return today + timedelta(days=(2 - today.weekday()) % 7)


def pick_train(sched: Path, tiplocs: dict[str, set[str]], origin_crs: str, dest_crs: str, *, day: date, toc: str | list | None = None,
               train: str | None = None, depart_after: str | None = None) -> dict:
    """{'train', 'uid', 'toc', 'departure', 'calls': {crs: 'HH:MM'}, 'stops', 'date'} for the chosen train on `day`."""
    origin, dest = tiplocs.get(origin_crs, set()), tiplocs.get(dest_crs, set())
    if not origin or not dest:
        raise SourceUnavailable(f"no TIPLOC for {origin_crs if not origin else dest_crs}")
    code = {t: c for c, ts in tiplocs.items() for t in ts}
    tocs = {toc} if isinstance(toc, str) else set(toc or [])
    o_keys, d_keys = [f'"{t}"' for t in origin], [f'"{t}"' for t in dest]
    best: dict[str, tuple[str, dict]] = {}          # uid -> (stp, journey) of its schedule valid on the day
    overridden: set[str] = set()                    # uids with a cancellation or an overlay on the day
    with _open(sched) as fh:
        for line in fh:
            if not line.startswith('{"JsonScheduleV1"'):
                continue
            m = _HEAD.search(line, 0, 400)
            stp = m.group(1) if m else None
            has_route = any(k in line for k in o_keys) and any(k in line for k in d_keys)
            if not has_route:
                if stp in ("C", "O"):                   # elsewhere on the network, but it may stop this UID running here
                    f = {k: (r.group(1) if (r := rx.search(line)) else "") for k, rx in _FIELD.items()}
                    if _runs_on(f, day):
                        overridden.add(m.group(2))
                continue
            s = json.loads(line)["JsonScheduleV1"]
            if not _runs_on(s, day):
                continue
            uid = s.get("CIF_train_uid")
            if s.get("CIF_stp_indicator") in ("C", "O"):
                overridden.add(uid)
            if s.get("CIF_stp_indicator") == "C":
                continue
            if s.get("train_status") not in PASSENGER_STATUS or (s.get("schedule_segment") or {}).get("CIF_train_category") in BUS_CATEGORY:
                continue
            if tocs and s.get("atoc_code") not in tocs:
                continue
            j = _journey(s, origin, dest, code)
            if j is not None and (uid not in best or s.get("CIF_stp_indicator") == "O"):
                best[uid] = (s.get("CIF_stp_indicator"), j)
    # A permanent schedule whose UID has a cancellation or an overlay that day does not run as planned (an overlay that
    # still serves the route was kept above in its place).
    trains = [j for uid, (stp, j) in best.items() if stp == "O" or uid not in overridden]
    if train:
        trains = [j for j in trains if train in (j["train"], j["uid"])]
    if not trains:
        raise SourceUnavailable(f"SCHEDULE has no passenger train {origin_crs} -> {dest_crs} on {day}" + (f" ({train})" if train else "")
                                + (f" for {','.join(sorted(tocs))}" if tocs else ""))
    dep = lambda j: _mins(j["calls"][origin_crs]) if origin_crs in j["calls"] else _mins(next(iter(j["calls"].values())))
    if depart_after and not train:
        later = [j for j in trains if dep(j) >= _mins(depart_after)]
        trains = later or trains
    j = min(trains, key=lambda j: (dep(j), -len(j["calls"])))
    return j | {"departure": j["calls"].get(origin_crs) or next(iter(j["calls"].values())), "date": day.isoformat()}


def apply(settings) -> dict | None:
    """Resolve a Network Rail timetable into settings.route['timetable'] (calls, departure, `resolved`). None when not used."""
    tcfg = settings.route.setdefault("timetable", {})
    if tcfg.get("source") != "network_rail":
        return None
    n = tcfg.get("network_rail") or {}
    origin, dest = settings.route["origin_crs"], settings.route["destination_crs"]
    try:
        sched, corpus = download(settings)
        tiplocs = crs_tiplocs(sched, corpus)
        for crs, extra in (n.get("tiplocs") or {}).items():
            tiplocs.setdefault(crs, set()).update(extra)
        r = pick_train(sched, tiplocs, origin, dest, day=typical_weekday(), toc=n.get("toc"),
                       train=n.get("train"), depart_after=n.get("depart_after") or tcfg.get("departure"))
    except (SourceUnavailable, OSError, ValueError) as exc:
        console.log(f"[yellow]Network Rail timetable unavailable ({exc}); using the route file's calls")
        return None
    alias = n.get("aliases") or {}
    r["calls"] = {alias.get(c, c): t for c, t in r["calls"].items()}
    r["stops"] = [alias.get(c, c) for c in r["stops"]]
    tcfg["calls"], tcfg["departure"] = r["calls"], r["departure"]
    tcfg["resolved"] = {"source": "network_rail", "train": r["train"], "uid": r["uid"], "toc": r["toc"], "date": r["date"], "stops": r["stops"]}
    console.log(f"timetable: {r['toc']} {r['train']} ({r['uid']}) from Network Rail SCHEDULE for {r['date']}, {r['departure']} {origin} -> "
                f"{r['calls'][dest]} {dest}, {len(r['calls'])} calls")
    return r
