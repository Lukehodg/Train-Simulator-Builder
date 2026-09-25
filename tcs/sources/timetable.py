"""Timetables: YAML calling pattern (default), GTFS stop_times, or Network Rail CIF schedule extracts.

CIF (from Network Rail Open Data SCHEDULE feed or the RDG timetable) is fixed-width; we read BS/LO/LI/LT records
for one train UID and map TIPLOCs to CRS via a CORPUS reference JSON (also from Network Rail Open Data).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def _hhmm(s: str) -> str | None:
    s = s.strip()
    if len(s) < 4 or not s[:4].isdigit():
        return None
    return f"{s[:2]}:{s[2:4]}" + (":30" if s[4:5] == "H" else "")


def from_yaml(cfg: dict) -> pd.DataFrame:
    calls = cfg.get("calls", {})
    return pd.DataFrame([{"crs": k, "time": v, "stop": True} for k, v in calls.items()])


def from_cif(path: Path, train_uid: str, corpus_json: Path) -> pd.DataFrame:
    with open(corpus_json, "r", encoding="utf-8") as fh:
        corpus = json.load(fh)
    tiploc_to_crs = {r["TIPLOC"]: r["3ALPHA"] for r in corpus.get("TIPLOCDATA", []) if r.get("3ALPHA")}
    rows: list[dict] = []
    capture = False
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            rec = line[:2]
            if rec == "BS":
                capture = line[3:9] == train_uid and line[2] != "D"
                continue
            if not capture:
                continue
            tiploc = line[2:9].strip()
            if rec == "LO":
                rows.append({"tiploc": tiploc, "arr": None, "dep": _hhmm(line[10:15]), "stop": True})
            elif rec == "LI":
                arr, dep, pas = _hhmm(line[10:15]), _hhmm(line[15:20]), _hhmm(line[20:25])
                rows.append({"tiploc": tiploc, "arr": arr, "dep": dep or pas, "stop": bool(arr or dep)})
            elif rec == "LT":
                rows.append({"tiploc": tiploc, "arr": _hhmm(line[10:15]), "dep": None, "stop": True})
                capture = False
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"train UID {train_uid} not found in {path.name}")
    df["crs"] = df["tiploc"].map(tiploc_to_crs)
    df["time"] = df["dep"].fillna(df["arr"])
    return df[["crs", "time", "stop", "tiploc"]]


def from_gtfs(folder: Path, trip_id: str) -> pd.DataFrame:
    st = pd.read_csv(folder / "stop_times.txt", dtype=str)
    stops = pd.read_csv(folder / "stops.txt", dtype=str)
    st = st[st["trip_id"] == trip_id].sort_values("stop_sequence", key=lambda s: s.astype(int))
    st = st.merge(stops[["stop_id", "stop_code", "stop_name"]], on="stop_id", how="left")
    return pd.DataFrame({"crs": st["stop_code"], "time": st["departure_time"].str[:5], "stop": True, "name": st["stop_name"]})


def schedule_seconds(calls: pd.DataFrame, departure: str) -> dict[str, float]:
    """Seconds after departure for each CRS in a calling pattern (handles midnight wrap)."""
    base = datetime.strptime(departure[:5], "%H:%M")
    out = {}
    for _, r in calls.iterrows():
        if not isinstance(r.get("time"), str):
            continue
        t = datetime.strptime(r["time"][:5], "%H:%M")
        if t < base:
            t += timedelta(days=1)
        out[r["crs"]] = (t - base).total_seconds()
    return out
