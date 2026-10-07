"""Ookla Open Data: crowd-sourced mobile speed tests per ~600 m map tile (zoom-16 quadkeys), quarterly, keyless
(registry.opendata.aws/speedtest-global-performance, CC BY-NC-SA 4.0).

All networks are pooled and the tests are mostly handsets indoors or on foot, so this is not a per-network prior and
not what a train sees. It is used as an independent check on an uncalibrated model: do the stretches where the model
predicts weak mobile service line up with the tiles where people's phones measure slow? The comparison goes into
meta.json (`ookla_check`); the model itself is not changed.

The quarterly file (~185 MB, worldwide) is read once, cut to the route's bounding box, and only that cut is kept
(data/raw/<route>/ookla/).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .base import USER_AGENT, SourceUnavailable, console, http_get

BUCKET = "https://ookla-open-data.s3.amazonaws.com"
PREFIX = "parquet/performance/type=mobile/"
ZOOM = 16
LICENCE = "Speedtest by Ookla Global Fixed and Mobile Network Performance Maps (CC BY-NC-SA 4.0)"


def latest_key(timeout: int = 60) -> str:
    r = requests.get(f"{BUCKET}/", params={"list-type": "2", "prefix": PREFIX}, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    r.raise_for_status()
    keys = sorted(k for k in re.findall(r"<Key>(.*?)</Key>", r.text) if k.endswith("_performance_mobile_tiles.parquet"))
    if not keys:
        raise SourceUnavailable("Ookla open data lists no mobile tile files")
    return keys[-1]


def quarter_of(key: str) -> str:
    m = re.search(r"year=(\d{4})/quarter=(\d)", key)
    return f"{m.group(1)}-Q{m.group(2)}" if m else key


def quadkey(lon: np.ndarray, lat: np.ndarray, zoom: int = ZOOM) -> np.ndarray:
    """Bing-maps quadkeys (the tiles Ookla aggregates to) for points, as strings."""
    lat = np.clip(lat, -85.05112878, 85.05112878)
    n = 1 << zoom
    x = np.clip(((lon + 180.0) / 360.0 * n).astype(np.int64), 0, n - 1)
    s = np.sin(np.radians(lat))
    y = np.clip(((0.5 - np.log((1 + s) / (1 - s)) / (4 * np.pi)) * n).astype(np.int64), 0, n - 1)
    return tile_quadkeys(x, y, zoom)


def tile_quadkeys(x: np.ndarray, y: np.ndarray, zoom: int = ZOOM) -> np.ndarray:
    digits = np.zeros((len(x), zoom), dtype=np.int64)
    for i in range(zoom):
        mask = 1 << (zoom - 1 - i)
        digits[:, i] = ((x & mask) > 0).astype(np.int64) + 2 * ((y & mask) > 0).astype(np.int64)
    return np.array(["".join(map(str, d)) for d in digits])


def corridor_tiles(settings, bbox: tuple[float, float, float, float], raw_dir: Path) -> tuple[pd.DataFrame, str]:
    """Ookla mobile tiles inside bbox (lon0, lat0, lon1, lat1) from the newest quarter (or `ookla.key` in the networks file)."""
    import pyarrow.parquet as pq

    cfg = settings.networks.get("ookla") or {}
    key = cfg.get("key") or latest_key()
    q = quarter_of(key)
    clip = raw_dir / "ookla" / f"{q}.parquet"
    if clip.exists():
        return pd.read_parquet(clip), q
    if settings.offline:
        raise SourceUnavailable("offline and no Ookla clip cached")
    big = http_get(f"{BUCKET}/{key}", raw_dir=raw_dir.parent / "shared", name="ookla", ext="parquet", timeout=900, ttl_days=120)
    lon0, lat0, lon1, lat1 = bbox
    cols = ["quadkey", "tile_x", "tile_y", "avg_d_kbps", "avg_u_kbps", "avg_lat_ms", "tests", "devices"]
    t = pq.read_table(big, columns=cols, filters=[("tile_x", ">=", lon0), ("tile_x", "<=", lon1), ("tile_y", ">=", lat0), ("tile_y", "<=", lat1)])
    df = t.to_pandas()
    clip.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(clip, index=False)
    big.unlink(missing_ok=True)                 # keep only the corridor cut
    return df, q


def along_route(samples: pd.DataFrame, tiles: pd.DataFrame) -> pd.DataFrame:
    """Per sample: test-weighted Ookla speeds over the sample's tile and its 8 neighbours (tiles are sparse off main
    roads), and the number of tests behind them. NaN where no test was taken nearby."""
    lon, lat = samples["longitude"].to_numpy(float), samples["latitude"].to_numpy(float)
    n = 1 << ZOOM
    x = np.clip(((lon + 180.0) / 360.0 * n).astype(np.int64), 0, n - 1)
    s = np.sin(np.radians(lat))
    y = np.clip(((0.5 - np.log((1 + s) / (1 - s)) / (4 * np.pi)) * n).astype(np.int64), 0, n - 1)
    t = tiles.set_index("quadkey")
    w = np.zeros(len(samples))
    d = np.zeros(len(samples))
    u = np.zeros(len(samples))
    lt = np.zeros(len(samples))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            qk = tile_quadkeys(x + dx, y + dy)
            hit = t.reindex(qk)
            k = hit["tests"].fillna(0).to_numpy(float)
            w += k
            d += k * hit["avg_d_kbps"].fillna(0).to_numpy(float)
            u += k * hit["avg_u_kbps"].fillna(0).to_numpy(float)
            lt += k * hit["avg_lat_ms"].fillna(0).to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.DataFrame({"sample_id": samples["sample_id"].to_numpy(), "ookla_down_mbps": np.where(w > 0, d / w / 1000.0, np.nan),
                             "ookla_up_mbps": np.where(w > 0, u / w / 1000.0, np.nan), "ookla_latency_ms": np.where(w > 0, lt / w, np.nan),
                             "ookla_tests": w.astype(np.int64)})


def check(samples: pd.DataFrame, obs: pd.DataFrame, ookla: pd.DataFrame, quarter: str, section_m: float = 5000.0) -> dict:
    """Model vs Ookla along the route, by section: how often the model's best mobile network and the phones' measured
    speed agree on which sections are the slow ones (rank correlation), and the medians side by side."""
    cell = obs[obs["provider_type"] == "cellular"] if "provider_type" in obs else obs
    best = cell.assign(c=np.where(cell["available"], cell["capacity_mbps"], 0.0)).groupby("sample_id")["c"].max()
    df = samples[["sample_id", "distance_m"]].merge(ookla, on="sample_id", how="left")
    df["model_best_mbps"] = df["sample_id"].map(best)
    df["section"] = (df["distance_m"] // section_m).astype(int)
    sec = df.dropna(subset=["ookla_down_mbps"]).groupby("section").agg(ookla=("ookla_down_mbps", "median"), model=("model_best_mbps", "median"),
                                                                      tests=("ookla_tests", "max"))
    out = {"quarter": quarter, "source": LICENCE, "samples_with_tests": round(float(df["ookla_down_mbps"].notna().mean()), 3),
           "median_ookla_down_mbps": _r(df["ookla_down_mbps"].median()), "median_model_best_mbps": _r(df["model_best_mbps"].median()),
           "sections": int(len(sec)), "section_km": section_m / 1000.0}
    if len(sec) >= 5:
        out["section_rank_correlation"] = _r(sec["ookla"].rank().corr(sec["model"].rank()), 3)
        slow_o = sec["ookla"] <= sec["ookla"].quantile(0.2)
        slow_m = sec["model"] <= sec["model"].quantile(0.2)
        out["slowest_fifth_overlap"] = _r((slow_o & slow_m).sum() / max(int(slow_o.sum()), 1), 3)
    return out


def _r(v, nd: int = 1):
    return None if v is None or pd.isna(v) else round(float(v), nd)


def for_build(settings, samples: pd.DataFrame, raw_dir: Path) -> tuple[pd.DataFrame, str] | None:
    """Per-sample Ookla columns for a route, or None (logged) when the data can't be read."""
    pad = 0.02
    bbox = (samples["longitude"].min() - pad, samples["latitude"].min() - pad, samples["longitude"].max() + pad, samples["latitude"].max() + pad)
    try:
        tiles, q = corridor_tiles(settings, bbox, raw_dir)
    except (SourceUnavailable, requests.RequestException, OSError) as exc:
        console.log(f"[yellow]Ookla open data unavailable ({exc}); no crowd-sourced speed check")
        return None
    o = along_route(samples, tiles)
    console.log(f"Ookla {q}: {len(tiles):,} mobile tiles in the corridor; phones tested within ~600 m of {o['ookla_down_mbps'].notna().mean():.0%} of the route")
    return o, q
