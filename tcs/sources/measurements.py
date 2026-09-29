"""Measured data -> canonical measurement table used for calibration and validation.

Supported inputs (all CSV; column names mapped through `mapping` so new formats need no code):
- Ofcom mobile signal-strength measurement datasets (drive tests)
- Ofcom Connectivity on Trains Measurement Study segment annexes (segment pass-rate/classification)
- Network Survey (Android) exports: GPS + RSRP/RSRQ/SINR + MCC/MNC/TAC/CI
- Onboard modem / multi-WAN controller logs (Peplink, Icomera, Cradlepoint, ...) with GPS
- Throughput/latency test logs

Canonical columns: timestamp, latitude, longitude, provider_id, radio, rsrp_dbm, rsrq_db, sinr_db,
throughput_mbps, latency_ms, mcc, mnc, tac, cell_id, source, kind (point | segment)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

CANON = ["timestamp", "latitude", "longitude", "provider_id", "radio", "rsrp_dbm", "rsrq_db", "sinr_db", "throughput_mbps",
         "latency_ms", "mcc", "mnc", "tac", "cell_id", "source", "kind"]

# Default column guesses per format (case-insensitive substrings).
PRESETS: dict[str, dict[str, list[str]]] = {
    "network_survey": {
        "timestamp": ["time", "timestamp"], "latitude": ["latitude", "lat"], "longitude": ["longitude", "lon", "lng"],
        "rsrp_dbm": ["rsrp"], "rsrq_db": ["rsrq"], "sinr_db": ["snr", "sinr"], "mcc": ["mcc"], "mnc": ["mnc"],
        "tac": ["tac", "lac"], "cell_id": ["ci", "cellid", "cid"], "radio": ["radio", "technology", "rat"],
    },
    "ofcom_drive": {
        "timestamp": ["date", "time"], "latitude": ["latitude", "lat"], "longitude": ["longitude", "lon", "lng"], "provider": ["operator", "mno", "network"],
        "rsrp_dbm": ["rsrp"], "rsrq_db": ["rsrq"], "sinr_db": ["sinr", "snr"], "radio": ["technology", "rat", "generation"],
    },
    "ofcom_train_segments": {
        "segment_id": ["segment"], "latitude": ["latitude", "lat"], "longitude": ["longitude", "lon", "lng"], "provider": ["operator", "mno"],
        "pass_rate": ["pass", "rate"], "classification": ["class", "category"],
    },
    "modem_log": {
        "timestamp": ["time"], "latitude": ["latitude", "lat"], "longitude": ["longitude", "lon", "lng"], "provider": ["carrier", "operator", "sim"],
        "rsrp_dbm": ["rsrp"], "rsrq_db": ["rsrq"], "sinr_db": ["sinr"], "throughput_mbps": ["throughput", "mbps", "rx_rate"],
        "latency_ms": ["latency", "rtt", "ping"], "radio": ["rat", "tech"],
    },
}


def _words(text: str) -> list[str]:
    """'gpsLat', 'GPS_Lat (deg)' -> ['gps', 'lat', 'deg']."""
    return re.findall(r"[a-z0-9]+", re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text)).lower())


def _find(cols: list[str], needles: list[str], taken: set[str] = frozenset()) -> str | None:
    """The column a canonical field most likely lives in. Whole words win over parts of words, and a needle shorter than
    four letters must be a whole word: 'lat' is not the start of 'latency', nor 'ci' the middle of 'precision'."""
    free = [c for c in cols if c not in taken]
    words = {c: _words(c) for c in free}
    for n in needles:                                     # a column that is exactly the needle, or has it as a word
        for c in free:
            if "".join(words[c]) == n or n in words[c]:
                return c
    for n in needles:                                     # longer needles may also be part of a word ('rsrp' in 'rsrpdbm')
        if len(n) >= 4:
            for c in free:
                if n in c.lower():
                    return c
    return None


def _provider_from(value: str, operators: list[dict]) -> str | None:
    """The operator a free-text network name refers to, matched on whole words (so 'Three' is not EE, and '3' is Three).
    A name that matches no operator, or more than one, is left unassigned."""
    v = f" {' '.join(_words(value))} "
    hits = []
    for op in operators:
        names = [op["id"], op.get("name", ""), op.get("ofcom_key", ""), *op.get("aliases", [])]
        if any(n and f" {' '.join(_words(n))} " in v for n in names):
            hits.append(op["id"])
    return hits[0] if len(hits) == 1 else None


def load_measurements(path: Path, preset: str, operators: list[dict], mapping: dict[str, str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    cols = list(df.columns)
    guess: dict[str, str | None] = {}
    for k, needles in PRESETS[preset].items():            # a column feeds one field only
        guess[k] = _find(cols, needles, {c for c in guess.values() if c})
    if mapping:
        guess.update(mapping)
    out = pd.DataFrame(index=df.index)
    for c in CANON:
        src = guess.get(c)
        out[c] = df[src] if src in df.columns else np.nan
    if "provider" in guess and guess["provider"] in df.columns:
        out["provider_id"] = df[guess["provider"]].map(lambda v: _provider_from(v, operators))
    elif out["mcc"].notna().any():
        mm = {(int(op["mcc"]), int(m)): op["id"] for op in operators for m in op.get("mnc", [])}
        out["provider_id"] = [mm.get((int(a), int(b))) if pd.notna(a) and pd.notna(b) else None for a, b in zip(out["mcc"], out["mnc"])]
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    for c in ["latitude", "longitude", "rsrp_dbm", "rsrq_db", "sinr_db", "throughput_mbps", "latency_ms"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["source"] = f"{preset}:{path.name}"
    out["kind"] = "segment" if preset == "ofcom_train_segments" else "point"
    if preset == "ofcom_train_segments":
        out["pass_rate"] = pd.to_numeric(df[guess["pass_rate"]], errors="coerce") if guess.get("pass_rate") else np.nan
        out["classification"] = df[guess["classification"]] if guess.get("classification") else None
    out = out[out["latitude"].notna() & out["longitude"].notna()]
    return out.reset_index(drop=True)


def attach_to_route(meas: pd.DataFrame, samples: pd.DataFrame, proj, max_distance_m: float = 250) -> pd.DataFrame:
    """Join each measurement to its nearest route sample (drops measurements farther than max_distance_m)."""
    from ..geo import nearest_sample_index

    x, y = proj.to_xy(meas["longitude"].values, meas["latitude"].values)
    idx, d = nearest_sample_index(samples["x"].values, samples["y"].values, np.asarray(x), np.asarray(y))
    meas = meas.copy()
    meas["sample_id"] = samples["sample_id"].values[idx]
    meas["route_distance_m"] = d
    return meas[meas["route_distance_m"] <= max_distance_m].reset_index(drop=True)
