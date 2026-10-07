"""OpenCellID community cell database -> corridor cell table.

Bulk per-MCC download (CSV.gz, token required) filtered with DuckDB to the route corridor. `getInArea` is also
wrapped for small refreshes but is capped/rate-limited by the service. Records are logical cells: several may
share one physical site, and a missing cell never implies no service (see docs/data-sources.md).
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from ..geo import Projector
from .base import Provenance, SourceUnavailable, console, http_get, now_iso

COLUMNS = ["radio", "mcc", "net", "area", "cell", "unit", "lon", "lat", "range", "samples", "changeable", "created", "updated", "averageSignal"]


def plmns(op: dict) -> list[tuple[int, int]]:
    """The (mcc, mnc) pairs a network's cells carry: its `plmn` list ("mcc-mnc") where a network spans several MCCs
    (the US), else its `mcc` with each of its `mnc` codes."""
    if op.get("plmn"):
        return [(int(a), int(b)) for a, b in (str(p).split("-") for p in op["plmn"])]
    return [(int(op["mcc"]), int(mnc)) for mnc in op.get("mnc", [])]


def _mnc_map(operators: list[dict]) -> dict[tuple[int, int], str]:
    out = {}
    for op in operators:
        for key in plmns(op):
            out[key] = op["id"]
    return out


def fetch_cells_bulk(settings, proj: Projector, samples: pd.DataFrame, raw_dir: Path, *, corridor_m: float) -> pd.DataFrame:
    token = settings.key("OPENCELLID_TOKEN")
    if not token:
        raise SourceUnavailable("OPENCELLID_TOKEN not set")
    ocfg = settings.networks["opencellid"]
    mccs = sorted({mcc for op in settings.operators for mcc, _ in plmns(op)})
    frames = []
    shared = raw_dir.parent / "shared"          # one national download serves every route (OpenCellID allows 2 per file per day)
    for mcc in mccs:
        url = ocfg["bulk_url"].format(token=token, mcc=mcc)
        gz = http_get(url, raw_dir=shared, name="opencellid_bulk", ext="csv.gz", offline=settings.offline, ttl_days=30)
        with open(gz, "rb") as fh:
            magic = fh.read(2)
        if magic != bytes([0x1F, 0x8B]):
            msg = gz.read_text(encoding="utf-8", errors="ignore")[:160]
            gz.unlink(missing_ok=True)
            raise SourceUnavailable(f"OpenCellID bulk download for MCC {mcc} is not a gzip file ({msg.strip()})")
        frames.append(_filter_corridor(gz, proj, samples, corridor_m, ocfg))
    cells = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    return _normalise(cells, settings, proj)


def _filter_corridor(csv_path: Path, proj: Projector, samples: pd.DataFrame, corridor_m: float, ocfg: dict) -> pd.DataFrame:
    # Coarse bbox filter in DuckDB (fast, streaming over the gz), fine filter by distance below.
    lon, lat = proj.to_lonlat(samples["x"].values, samples["y"].values)
    pad = corridor_m / 111_000 * 1.6
    con = duckdb.connect()
    df = con.execute(
        f"""
        SELECT radio, mcc, net, area, cell, unit, lon, lat, range, samples
        FROM read_csv('{csv_path.as_posix()}', header=false, columns={{'radio':'VARCHAR','mcc':'INT','net':'INT','area':'BIGINT','cell':'BIGINT','unit':'BIGINT','lon':'DOUBLE','lat':'DOUBLE','range':'INT','samples':'INT','changeable':'INT','created':'BIGINT','updated':'BIGINT','averageSignal':'INT'}})
        WHERE lon BETWEEN {lon.min() - pad} AND {lon.max() + pad}
          AND lat BETWEEN {lat.min() - pad} AND {lat.max() + pad}
          AND samples >= {int(ocfg.get('min_samples', 1))}
          AND radio IN ({','.join(repr(r) for r in ocfg.get('radios', ['LTE', 'NR', 'UMTS', 'GSM']))})
        """
    ).df()
    if df.empty:
        return df
    x, y = proj.to_xy(df["lon"].values, df["lat"].values)
    df["x"], df["y"] = x, y
    # distance to the route polyline via a coarse sample grid (every 10th sample, 500 m) is enough for a corridor cut
    sx, sy = samples["x"].values[::10], samples["y"].values[::10]
    from shapely.geometry import Point
    from shapely.strtree import STRtree

    tree = STRtree([Point(a, b) for a, b in zip(sx, sy)])
    idx = tree.nearest([Point(a, b) for a, b in zip(x, y)])
    d = np.hypot(sx[idx] - x, sy[idx] - y)
    return df[d <= corridor_m + 500].reset_index(drop=True)


def fetch_cells_area(settings, proj: Projector, bbox_lonlat: tuple[float, float, float, float], raw_dir: Path) -> pd.DataFrame:
    token = settings.key("OPENCELLID_TOKEN")
    if not token:
        raise SourceUnavailable("OPENCELLID_TOKEN not set")
    w, s, e, n = bbox_lonlat
    url = settings.networks["opencellid"]["area_url"].format(token=token, bbox=f"{s},{w},{n},{e}")
    p = http_get(url, raw_dir=raw_dir, name="opencellid_area", ext="json", offline=settings.offline)
    import json

    with open(p, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    df = pd.DataFrame(doc.get("cells", []))
    if df.empty:
        return df
    df = df.rename(columns={"mnc": "net", "lac": "area", "cellid": "cell"})
    df["x"], df["y"] = proj.to_xy(df["lon"].values, df["lat"].values)
    return _normalise(df, settings, proj)


def _normalise(df: pd.DataFrame, settings, proj: Projector) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    mmap = _mnc_map(settings.operators)
    df["provider_id"] = [mmap.get((int(a), int(b))) for a, b in zip(df["mcc"], df["net"])]
    df = df[df["provider_id"].notna()].copy()
    df["cell_key"] = df["mcc"].astype(str) + "-" + df["net"].astype(str) + "-" + df["area"].astype(str) + "-" + df["cell"].astype(str)
    out = pd.DataFrame({
        "cell_key": df["cell_key"], "provider_id": df["provider_id"], "radio": df["radio"],
        "mcc": df["mcc"].astype(int), "mnc": df["net"].astype(int), "area_or_tac": df["area"].astype("int64"),
        "cell_id": df["cell"].astype("int64"), "pci_or_unit": df.get("unit", pd.Series(-1, index=df.index)).fillna(-1).astype("int64"),
        "latitude": df["lat"].astype(float), "longitude": df["lon"].astype(float), "x": df["x"], "y": df["y"],
        "elevation_m": np.float32(np.nan), "samples": df.get("samples", pd.Series(0, index=df.index)).fillna(0).astype(int),
        "range_m": df.get("range", pd.Series(0, index=df.index)).fillna(0).astype(int), "source": "opencellid",
    }).reset_index(drop=True)
    console.log(f"OpenCellID: {len(out)} corridor cells across {out['provider_id'].nunique()} operators")
    return out


def provenance() -> Provenance:
    return Provenance("opencellid", "https://opencellid.org", now_iso(), notes="CC BY-SA 4.0; community contributed logical cells")
