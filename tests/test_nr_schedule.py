"""Network Rail SCHEDULE timetables (tcs/sources/nr_schedule.py) on a small hand-made extract."""
from __future__ import annotations

import gzip
import json
from datetime import date

import pytest

from tcs.config import load_settings
from tcs.sources import nr_schedule
from tcs.sources.base import SourceUnavailable

DAY = date(2026, 10, 21)                     # a Wednesday


def _loc(kind, tiploc, arr=None, dep=None, pas=None):
    return {"location_type": kind, "record_identity": kind, "tiploc_code": tiploc, "public_arrival": arr, "public_departure": dep,
            "arrival": arr and arr + "H", "departure": dep, "pass": pas}


def _sched(uid, stp, locs, *, toc="GR", status="P", days="1111100", start="2026-05-17", end="2026-12-12", headcode="1S07", category="XX"):
    # Keys in the feed's own order: the STP indicator and UID sit next to each other near the start of the line
    return {"JsonScheduleV1": {"CIF_bank_holiday_running": None, "CIF_stp_indicator": stp, "CIF_train_uid": uid, "atoc_code": toc,
                               "schedule_days_runs": days, "schedule_end_date": end,
                               "schedule_segment": {"signalling_id": headcode, "CIF_train_category": category,
                                                    "schedule_location": locs} if locs is not None else {},
                               "schedule_start_date": start, "train_status": status}}


def _kgx_edb(t0="0900", via_dep="0947", arr="1320"):
    return [_loc("LO", "KNGX", dep=t0), _loc("LI", "STEVNGE", pas="0918"), _loc("LI", "PBRO", arr="0945", dep=via_dep),
            _loc("LI", "YORK", arr="1050", dep="1052"), _loc("LT", "EDINBUR", arr=arr)]


def _extract(tmp_path, schedules):
    lines = [{"JsonTimetableV1": {"Metadata": {"type": "full"}}}]
    lines += [{"TiplocV1": {"tiploc_code": t, "crs_code": c}} for t, c in
              [("KNGX", "KGX"), ("STEVNGE", "SVG"), ("PBRO", "PBO"), ("YORK", "YRK"), ("EDINBUR", "EDB"), ("GLGQHL", None)]]
    lines += schedules + [{"EOF": True}]
    p = tmp_path / "schedule.json.gz"
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        fh.writelines(json.dumps(x, separators=(",", ":")) + "\n" for x in lines)
    corpus = tmp_path / "corpus.json.gz"
    with gzip.open(corpus, "wt", encoding="utf-8") as fh:
        json.dump({"TIPLOCDATA": [{"TIPLOC": "GLGQHL", "3ALPHA": "GLQ"}, {"TIPLOC": "KNGXMCR", "3ALPHA": "KGX"}, {"TIPLOC": " ", "3ALPHA": " "}]}, fh)
    return p, corpus


def test_crs_tiplocs_merge_the_extract_and_corpus(tmp_path):
    sched, corpus = _extract(tmp_path, [])
    t = nr_schedule.crs_tiplocs(sched, corpus)
    assert t["KGX"] == {"KNGX", "KNGXMCR"} and t["GLQ"] == {"GLGQHL"} and " " not in t


def test_picks_the_first_passenger_train_after_the_departure(tmp_path):
    sched, corpus = _extract(tmp_path, [
        _sched("A00001", "P", _kgx_edb("0830", arr="1250"), headcode="1S05"),
        _sched("A00002", "P", _kgx_edb("0900"), headcode="1S07"),
        _sched("A00003", "P", _kgx_edb("0930", arr="1350"), headcode="1S09"),
        _sched("A00004", "P", _kgx_edb("0905"), status="B", category="BR"),          # a bus
        _sched("A00005", "P", _kgx_edb("0901"), toc="XC"),                           # another operator
        _sched("A00006", "P", _kgx_edb("0902"), days="0000011"),                     # weekends only
        _sched("A00007", "P", _kgx_edb("0903"), end="2026-10-01"),                   # expired
    ])
    t = nr_schedule.crs_tiplocs(sched, corpus)
    r = nr_schedule.pick_train(sched, t, "KGX", "EDB", day=DAY, toc="GR", depart_after="09:00")
    assert (r["train"], r["uid"], r["departure"]) == ("1S07", "A00002", "09:00")
    assert r["calls"] == {"KGX": "09:00", "PBO": "09:47", "YRK": "10:52", "EDB": "13:20"}     # Stevenage passed, not called
    assert r["stops"] == ["KGX", "PBO", "YRK", "EDB"] and r["date"] == "2026-10-21"
    assert nr_schedule.pick_train(sched, t, "KGX", "EDB", day=DAY, toc=["XC"])["uid"] == "A00005"
    assert nr_schedule.pick_train(sched, t, "KGX", "EDB", day=DAY, train="1S09")["uid"] == "A00003"
    assert nr_schedule.pick_train(sched, t, "KGX", "EDB", day=DAY, toc="GR", depart_after="23:00")["uid"] == "A00001"  # none later: earliest
    with pytest.raises(SourceUnavailable):
        nr_schedule.pick_train(sched, t, "EDB", "KGX", day=DAY)                                # wrong direction


def test_cancellations_and_overlays_change_the_day(tmp_path):
    sched, corpus = _extract(tmp_path, [
        _sched("B00001", "P", _kgx_edb("0900")),
        _sched("B00001", "C", None, start="2026-10-19", end="2026-10-23"),          # cancelled this week
        _sched("B00002", "P", _kgx_edb("0910")),
        _sched("B00002", "O", _kgx_edb("0912", via_dep="0950"), start="2026-10-21", end="2026-10-21"),   # retimed on the day
        _sched("B00003", "P", _kgx_edb("0905")),
        _sched("B00003", "O", [_loc("LO", "KNGX", dep="0905"), _loc("LT", "YORK", arr="1100")], start="2026-10-20", end="2026-10-22"),  # cut short
    ])
    r = nr_schedule.pick_train(sched, nr_schedule.crs_tiplocs(sched, corpus), "KGX", "EDB", day=DAY, depart_after="09:00")
    assert r["uid"] == "B00002" and r["calls"]["KGX"] == "09:12" and r["calls"]["PBO"] == "09:50"


def test_typical_weekday_is_the_coming_wednesday():
    assert nr_schedule.typical_weekday(date(2026, 10, 17)) == DAY          # a Saturday
    assert nr_schedule.typical_weekday(DAY) == DAY


def test_apply_sets_the_train_and_falls_back_without_credentials(tmp_path, monkeypatch):
    s = load_settings(offline=False)
    before = dict(s.route["timetable"]["calls"])
    assert s.route["timetable"]["source"] == "network_rail"
    monkeypatch.delenv("NETWORK_RAIL_USERNAME", raising=False)
    monkeypatch.delenv("NETWORK_RAIL_PASSWORD", raising=False)
    assert nr_schedule.apply(s) is None                                    # no account: the route file's calls stand
    assert s.route["timetable"]["calls"] == before and "resolved" not in s.route["timetable"]

    sched, corpus = _extract(tmp_path, [_sched("A00002", "P", _kgx_edb("0900"))])
    monkeypatch.setattr(nr_schedule, "download", lambda settings: (sched, corpus))
    monkeypatch.setattr(nr_schedule, "typical_weekday", lambda: DAY)
    r = nr_schedule.apply(s)
    tt = s.route["timetable"]
    assert r["uid"] == "A00002" and tt["departure"] == "09:00" and tt["calls"]["EDB"] == "13:20"
    assert tt["resolved"] == {"source": "network_rail", "train": "1S07", "uid": "A00002", "toc": "GR", "date": "2026-10-21",
                              "stops": ["KGX", "PBO", "YRK", "EDB"]}


def test_every_gb_route_asks_for_a_network_rail_train():
    from tcs.config import list_routes

    gb = [r for r in list_routes() if r["country"] != "US"]
    assert len(gb) >= 18
    for r in gb:
        s = load_settings(offline=True, route_id=r["id"])
        tt = s.route["timetable"]
        assert tt["source"] == "network_rail" and tt["network_rail"]["toc"] and tt["calls"], r["id"]


def test_route_file_adds_tiplocs_with_no_station_code(tmp_path, monkeypatch):
    locs = _kgx_edb()
    locs[0] = _loc("LO", "KNGXHL", dep="0900")                             # a platform TIPLOC the feed gives no CRS
    sched, corpus = _extract(tmp_path, [_sched("A00009", "P", locs)])
    s = load_settings(offline=False)
    monkeypatch.setattr(nr_schedule, "download", lambda settings: (sched, corpus))
    monkeypatch.setattr(nr_schedule, "typical_weekday", lambda: DAY)
    assert nr_schedule.apply(s) is None                                    # not found by station code alone
    s.route["timetable"]["network_rail"]["tiplocs"] = {"KGX": ["KNGXHL"]}
    r = nr_schedule.apply(s)
    assert r["uid"] == "A00009" and r["calls"]["KGX"] == "09:00"
    s.route["timetable"]["network_rail"]["aliases"] = {"PBO": "PBR"}         # the feed's code for a call -> the route file's
    r = nr_schedule.apply(s)
    assert "PBR" in r["calls"] and "PBO" not in r["calls"] and r["stops"][1] == "PBR"
