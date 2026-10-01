"""Measured data -> canonical measurement table used for calibration and validation.

Supported inputs (all CSV; column names mapped through `mapping` so new formats need no code):
- Ofcom mobile signal-strength measurement datasets (drive tests)
- Ofcom Connectivity on Trains Measurement Study segment annexes (segment pass-rate/classification)
- Network Survey (Android) exports: GPS + RSRP/RSRQ/SINR + MCC/MNC/TAC/CI
- Onboard modem / multi-WAN controller logs (Peplink, Icomera, Cradlepoint, ...) with GPS
- Throughput/latency test logs
- Network Rail Yellow Train LTE scanner logs (Rail Data Marketplace): every carrier of every network each second,
  reduced to the strongest per network (the one a modem would use); the condensed parquet form is read too
- Network Rail Global View 4G and 5G scanner logs (Rail Data Marketplace, 2026 on): the same reduction; date and
  time in separate columns (day first), and each data row carries one field more than the header

Large files are read in chunks, keeping only the rows inside `bbox` (the route's surroundings).

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
    "yellow_train": {
        "timestamp": ["datetime"], "latitude": ["latitude"], "longitude": ["longitude"], "provider": ["operator"],
        "rsrp_dbm": ["cal_rsrp"],                      # RSRP corrected for the measurement antenna, not the raw scanner value
        "rsrq_db": ["rsrq"], "sinr_db": ["sinr"], "mnc": ["mnc"], "cell_id": ["pci"], "device": ["train"],
    },
    "global_view_4g": {
        "timestamp": ["date"], "time_of_day": ["time"], "latitude": ["latitude"], "longitude": ["longitude"], "provider": ["operator"],
        "rsrp_dbm": ["rsrp"], "rsrq_db": ["rsrq"], "sinr_db": ["sinr"], "mnc": ["mnc"], "mcc": ["mcc"], "cell_id": ["cellid", "pci"],
        "device": ["train"], "channel": ["earfcn"],
    },
    "global_view_5g": {                                # NR: SS-RSRP per beam; the strongest per second and network is kept
        "timestamp": ["date"], "time_of_day": ["time"], "latitude": ["latitude"], "longitude": ["longitude"], "provider": ["operator"],
        "rsrp_dbm": ["rsrp"], "rsrq_db": ["rsrq"], "sinr_db": ["sinr"], "mnc": ["mnc"], "mcc": ["mcc"], "cell_id": ["pci"],
        "device": ["train"], "channel": ["nrarfcn"],
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


# Scanners log every carrier they hear; a modem uses the strongest, so keep that one per second, device and network.
BEST_SERVER = {"yellow_train", "global_view_4g", "global_view_5g"}
FIXED = {"yellow_train": {"radio": "4G"}, "global_view_4g": {"radio": "4G"}, "global_view_5g": {"radio": "5G"}}
DATE_FORMAT = {"global_view_4g": "%d/%m/%Y %H:%M:%S", "global_view_5g": "%d/%m/%Y %H:%M:%S"}   # with time_of_day appended
RSRP_VALID = (-160.0, -20.0)                               # scanners log 0 or -200 for "no reading"


def load_measurements(path: Path, preset: str, operators: list[dict], mapping: dict[str, str] | None = None,
                      bbox: tuple[float, float, float, float] | None = None, chunksize: int = 1_000_000, near=None) -> pd.DataFrame:
    """bbox: (lon_min, lat_min, lon_max, lat_max); rows outside it are dropped as each chunk is read.
    near: optional (lon, lat) -> bool mask, e.g. near_route(): a tighter filter applied after the bbox, as each chunk is read."""
    path = Path(path)
    guess: dict[str, str | None] | None = None
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        f = pq.ParquetFile(path)
        guess = _guess(f.schema_arrow.names, preset, mapping)
        chunks = (b.to_pandas() for b in f.iter_batches(batch_size=chunksize))
    else:
        # index_col=False: a row with one field more than the header (Global View files) must not shift every column
        chunks = pd.read_csv(path, low_memory=False, chunksize=chunksize, encoding="utf-8-sig", index_col=False)
    parts = []
    import warnings

    warnings.filterwarnings("ignore", "Length of header", pd.errors.ParserWarning)   # the extra field is expected (above)
    for df in chunks:
        if guess is None:
            guess = _guess(list(df.columns), preset, mapping)
        if (bbox is not None or near is not None) and guess.get("latitude") in df.columns and guess.get("longitude") in df.columns:
            lon = pd.to_numeric(df[guess["longitude"]], errors="coerce").to_numpy(dtype=float)
            lat = pd.to_numeric(df[guess["latitude"]], errors="coerce").to_numpy(dtype=float)
            keep = np.isfinite(lon) & np.isfinite(lat)
            if bbox is not None:
                keep &= (lon >= bbox[0]) & (lon <= bbox[2]) & (lat >= bbox[1]) & (lat <= bbox[3])
            if near is not None and keep.any():
                keep[keep] = near(lon[keep], lat[keep])
            df = df[keep]                                 # before any per-row work: a national file stays small
        part = _canonical(df, guess, preset, operators, path)
        parts.append(_best_server(part) if preset in BEST_SERVER else part)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=[*CANON, "device"])
    return (_best_server(out) if preset in BEST_SERVER else out).reset_index(drop=True)   # a second can straddle two chunks


def near_route(samples: pd.DataFrame, proj, max_distance_m: float):
    """(lon, lat) -> mask of the points that may lie within max_distance_m of a route sample: those in a grid cell
    (max_distance_m square) next to one holding a sample. Cheap, so millions of rows around a city can be cut first."""
    cell = max(float(max_distance_m), 1.0)
    sx = np.floor(samples["x"].to_numpy(dtype=float) / cell).astype(np.int64)
    sy = np.floor(samples["y"].to_numpy(dtype=float) / cell).astype(np.int64)
    cells = np.unique(np.concatenate([(sx + dx) * 4_000_003 + (sy + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]))

    def mask(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        x, y = proj.to_xy(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        ok[ok] = np.isin(np.floor(x[ok] / cell).astype(np.int64) * 4_000_003 + np.floor(y[ok] / cell).astype(np.int64), cells)
        return ok
    return mask


def route_bbox(samples: pd.DataFrame, margin_deg: float = 0.03) -> tuple[float, float, float, float]:
    """The route's surroundings (about 2-3 km around it): measurement files covering the whole network are cut to this."""
    return (float(samples["longitude"].min()) - margin_deg, float(samples["latitude"].min()) - margin_deg,
            float(samples["longitude"].max()) + margin_deg, float(samples["latitude"].max()) + margin_deg)


def _guess(cols: list[str], preset: str, mapping: dict[str, str] | None) -> dict[str, str | None]:
    guess: dict[str, str | None] = {}
    for k, needles in PRESETS[preset].items():            # a column feeds one field only
        guess[k] = _find(cols, needles, {c for c in guess.values() if c})
    if mapping:
        guess.update(mapping)
    return guess


def _best_server(m: pd.DataFrame) -> pd.DataFrame:
    m = m[m["provider_id"].notna() & m["rsrp_dbm"].notna()]
    return m.sort_values("rsrp_dbm", ascending=False).drop_duplicates(["timestamp", "device", "provider_id"]).sort_index()


def _canonical(df: pd.DataFrame, guess: dict[str, str | None], preset: str, operators: list[dict], path: Path) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in CANON:
        src = guess.get(c)
        out[c] = df[src] if src in df.columns else np.nan
    if "provider" in guess and guess["provider"] in df.columns:
        names = df[guess["provider"]]
        out["provider_id"] = names.map({v: _provider_from(v, operators) for v in names.dropna().unique()})   # each name once
    elif out["mcc"].notna().any():
        mm = {(int(op["mcc"]), int(m)): op["id"] for op in operators for m in op.get("mnc", [])}
        out["provider_id"] = [mm.get((int(a), int(b))) if pd.notna(a) and pd.notna(b) else None for a, b in zip(out["mcc"], out["mnc"])]
    if guess.get("time_of_day") in df.columns:              # date and time in separate columns
        out["timestamp"] = out["timestamp"].astype(str) + " " + df[guess["time_of_day"]].astype(str)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True, format=DATE_FORMAT.get(preset))
    for c in ["latitude", "longitude", "rsrp_dbm", "rsrq_db", "sinr_db", "throughput_mbps", "latency_ms"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["rsrp_dbm"] = out["rsrp_dbm"].where(out["rsrp_dbm"].between(*RSRP_VALID))
    if guess.get("channel") in df.columns:                   # EARFCN / NR-ARFCN: which band a reading is on
        out["channel"] = pd.to_numeric(df[guess["channel"]], errors="coerce")
    out["source"] = f"{preset}:{path.name}"
    out["kind"] = "segment" if preset == "ofcom_train_segments" else "point"
    out["device"] = df[guess["device"]].astype(str) if guess.get("device") in df.columns else ""
    for c, v in FIXED.get(preset, {}).items():
        out[c] = out[c].fillna(v) if c in out else v
    if preset == "ofcom_train_segments":
        out["pass_rate"] = pd.to_numeric(df[guess["pass_rate"]], errors="coerce") if guess.get("pass_rate") else np.nan
        out["classification"] = df[guess["classification"]] if guess.get("classification") else None
    return out[out["latitude"].notna() & out["longitude"].notna()]


def attach_to_route(meas: pd.DataFrame, samples: pd.DataFrame, proj, max_distance_m: float = 250) -> pd.DataFrame:
    """Join each measurement to its nearest route sample (drops measurements farther than max_distance_m)."""
    from ..geo import nearest_sample_index

    meas = meas[near_route(samples, proj, max_distance_m)(meas["longitude"].to_numpy(dtype=float), meas["latitude"].to_numpy(dtype=float))]
    if meas.empty:                                        # the exact nearest-point search below is the slow part
        return meas.assign(sample_id=pd.Series(dtype="int64"), route_distance_m=pd.Series(dtype=float)).reset_index(drop=True)
    x, y = proj.to_xy(meas["longitude"].values, meas["latitude"].values)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    idx, d = nearest_sample_index(samples["x"].values, samples["y"].values, x, y)
    meas = meas.copy()
    meas["sample_id"] = samples["sample_id"].values[idx]
    meas["route_distance_m"] = d
    return meas[meas["route_distance_m"] <= max_distance_m].reset_index(drop=True)
