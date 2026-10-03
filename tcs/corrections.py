"""Measured corrections: where scanner trains have measured a route, the model's error at those measurements, smoothed
along the track, corrects its prediction there.

Signal at a given spot repeats from trip to trip far more closely than any coverage map predicts it (a mast's own
power and aim, a cutting, a bend behind a hill), so earlier passes are the best guide to later ones. Per route,
network and 50 m route point:

    error_j      = measured_j - predicted_j - level        (one level per measurement set and network across every
                                                             route it covers, from open-air route points, so a scanner
                                                             that reads low everywhere does not shift a route but a
                                                             route the model over- or under-rates is corrected)
    correction_i = sum_j w_j g(d_ij) error_j / (sum_j w_j g(d_ij) + SHRINK)
                                                            (in tunnels from in-tunnel points only, outside from
                                                             open-air ones only: each corrects its own model)

with g a Gaussian along the track (SCALE_M) and w_j the measurement set's weight: 1 for the current Global View logs,
0.1 for the 2018-19 Yellow Train logs (older networks, but cuttings and hills have not moved). SHRINK pulls a
correction towards zero where few measurements are near, so a stretch measured once moves less than one measured
often. Corrections are in dB of RSRP, added to the score through the calibration's dB scale, so everything downstream
(capacity, latency, the viewer) follows; inside a measured tunnel the correction goes on the portal or DAS model.

Tested on later trips (Global View, corrections from 16 Mar - 6 Apr 2026 plus Yellow Train, tested from 7 Apr through
the model): the error fell from 9.7 to 8.3 dB on average, on every route (docs/validation.md). `tcs check-national`
repeats that test on each run.
The corrections (config/route_corrections.parquet) are derived results: position, network and dB, no measurements.
They are fitted against the calibrated model of the time: re-run `tcs correct-routes` after re-calibrating.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SCALE_M = 100.0               # Gaussian scale along the track
SHRINK = 0.5                  # weight units pulling a correction towards zero where measurements are sparse
SET_WEIGHT = {"global_view_4g": 1.0, "yellow_train": 0.1}
MIN_DB = 0.3                  # smaller corrections are not stored
MATCH_M = 60.0                # a stored correction applies to a route sample within this distance
COLUMNS = ["route_id", "provider_id", "latitude", "longitude", "correction_db", "weight"]
INTERIM_FILE = "measured_corrections.parquet"     # this build's corrections, matched to its samples


def levels(j: pd.DataFrame) -> pd.Series:
    """A measurement set's own level per network (provider_id -> dB): the median of measured minus predicted over its
    open-air route points on every route it covers. j: as for errors(), pooled over the set's routes."""
    j = j[~j["_in_tunnel"].astype(bool) & np.isfinite(j["rsrp_dbm"]) & np.isfinite(j["signal_primary"])]
    return (j["rsrp_dbm"] - j["signal_primary"]).groupby(j["provider_id"]).median()


def errors(j: pd.DataFrame, weight: float, level: pd.Series | None = None) -> pd.DataFrame:
    """Model error at measured route points: sample_id, provider_id, distance_m, in_tunnel, error_db, w.
    j: measured route points with the model's prediction there (national.predict): sample_id, provider_id, rsrp_dbm,
    signal_primary, _in_tunnel, distance_m. level: the set's levels() over all its routes; None takes it from j alone,
    which also removes the route's own offset."""
    j = j[np.isfinite(j["rsrp_dbm"]) & np.isfinite(j["signal_primary"])]
    if j.empty:
        return pd.DataFrame(columns=["sample_id", "provider_id", "distance_m", "in_tunnel", "error_db", "w"])
    level = levels(j) if level is None else level
    diff = j["rsrp_dbm"] - j["signal_primary"]
    return pd.DataFrame({"sample_id": j["sample_id"].to_numpy(), "provider_id": j["provider_id"].to_numpy(), "distance_m": j["distance_m"].to_numpy(float),
                         "in_tunnel": j["_in_tunnel"].astype(bool).to_numpy(),
                         "error_db": (diff - j["provider_id"].map(level).fillna(0.0)).to_numpy(float), "w": float(weight)})


def smooth(err: pd.DataFrame, samples: pd.DataFrame, scale_m: float = SCALE_M, shrink: float = SHRINK) -> pd.DataFrame:
    """Per network, the shrunk Gaussian-weighted mean error along the track at every route sample with measurements
    within 3 scales: sample_id, provider_id, correction_db, weight. Tunnel samples take in-tunnel errors only and the
    rest open-air ones only (when both frames say which is which)."""
    out = []
    split = "in_tunnel" in err and "in_tunnel" in samples
    s_tun = samples["in_tunnel"].fillna(False).to_numpy(bool) if split else np.zeros(len(samples), bool)
    e_tun = err["in_tunnel"].astype(bool).to_numpy() if split else np.zeros(len(err), bool)
    for inside in (False, True):
        sel = samples[s_tun == inside]
        dist, sid = sel["distance_m"].to_numpy(float), sel["sample_id"].to_numpy()
        for pid, g in err[e_tun == inside].groupby("provider_id"):
            order = np.argsort(g["distance_m"].to_numpy())
            d, e, w = (g[c].to_numpy(float)[order] for c in ("distance_m", "error_db", "w"))
            lo, hi = np.searchsorted(d, dist - 3 * scale_m), np.searchsorted(d, dist + 3 * scale_m)
            corr, wsum = np.zeros(len(dist)), np.zeros(len(dist))
            for i in np.flatnonzero(hi > lo):
                k = w[lo[i]:hi[i]] * np.exp(-0.5 * ((d[lo[i]:hi[i]] - dist[i]) / scale_m) ** 2)
                wsum[i] = k.sum()
                corr[i] = (k * e[lo[i]:hi[i]]).sum() / (wsum[i] + shrink)
            keep = wsum > 0
            out.append(pd.DataFrame({"sample_id": sid[keep], "provider_id": pid, "correction_db": corr[keep], "weight": wsum[keep]}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["sample_id", "provider_id", "correction_db", "weight"])


def to_table(route_id: str, corr: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """Stored form: position instead of sample id, so a route re-sampled along a slightly different line still matches."""
    c = corr[corr["correction_db"].abs() >= MIN_DB].merge(samples[["sample_id", "latitude", "longitude"]], on="sample_id")
    return pd.DataFrame({"route_id": route_id, "provider_id": c["provider_id"], "latitude": c["latitude"].round(5), "longitude": c["longitude"].round(5),
                         "correction_db": c["correction_db"].round(1).astype("float32"), "weight": c["weight"].round(2).astype("float32")})


def save(table: pd.DataFrame, path: Path, meta: dict) -> None:
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    t = pa.Table.from_pandas(table[COLUMNS], preserve_index=False)
    t = t.replace_schema_metadata({**(t.schema.metadata or {}), b"tcs": json.dumps(meta).encode()})
    pq.write_table(t, path, compression="zstd")


def meta(path: Path) -> dict:
    import json

    import pyarrow.parquet as pq

    if not Path(path).exists():
        return {}
    md = pq.read_schema(path).metadata or {}
    return json.loads(md.get(b"tcs", b"{}"))


def for_route(path: Path | None, route_id: str, samples: pd.DataFrame, proj) -> pd.DataFrame | None:
    """The stored corrections for this route, matched to its samples (nearest within MATCH_M): sample_id, provider_id,
    correction_db, weight. None if there are none."""
    if not path or not Path(path).exists():
        return None
    import pyarrow.parquet as pq

    t = pq.read_table(path, filters=[("route_id", "==", route_id)]).to_pandas()
    if t.empty:
        return None
    x, y = proj.to_xy(t["longitude"].to_numpy(float), t["latitude"].to_numpy(float))
    from shapely import points
    from shapely.strtree import STRtree

    sx, sy = samples["x"].to_numpy(float), samples["y"].to_numpy(float)
    idx = STRtree(points(sx, sy)).nearest(points(np.asarray(x), np.asarray(y)))
    near = np.hypot(sx[idx] - np.asarray(x), sy[idx] - np.asarray(y)) <= MATCH_M
    out = pd.DataFrame({"sample_id": samples["sample_id"].to_numpy()[idx[near]], "provider_id": t["provider_id"].to_numpy()[near],
                        "correction_db": t["correction_db"].to_numpy(float)[near], "weight": t["weight"].to_numpy(float)[near]})
    return out.drop_duplicates(["sample_id", "provider_id"]) if len(out) else None


def path_of(settings) -> Path | None:
    from .config import ROOT

    f = settings.sim["cellular"].get("measured_corrections")
    return ROOT / f if f else None


def for_build(settings, bundle, interim: Path) -> pd.DataFrame | None:
    """This route's corrections, matched to the samples just built and kept beside them (reports re-simulate from them).
    Only on real track geometry: a stand-in line does not pass where the measurements were taken."""
    corr = for_route(path_of(settings), settings.route_id, bundle.samples, bundle.proj) if bundle.geometry_source in ("osm", "file") else None
    f = Path(interim) / INTERIM_FILE
    if corr is None:
        f.unlink(missing_ok=True)
    else:
        corr.to_parquet(f, index=False)
    return corr


def load_interim(interim: Path) -> pd.DataFrame | None:
    f = Path(interim) / INTERIM_FILE
    return pd.read_parquet(f) if f.exists() else None
