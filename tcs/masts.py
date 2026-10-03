"""Mast positions fitted to scanner measurements: where each network's 4G masts (eNodeBs) stand, worked out from how
their signal rises and falls along the track.

Scanner logs such as Network Rail's Global View name the cell behind every reading (its E-UTRAN cell identity; the
eNodeB, i.e. the mast, is that number without its last 8 bits). Each mast is seen on many trips, so for each one:

    RSRP_j = P_c(j) - 10 n log10( sqrt(dx_j^2 + dy_j^2 + H^2) )

  readings j    one per 50 m square and cell (the median of the readings there), so a train standing at a platform
                does not outweigh a mile of line
  P_c           one level per cell (sector and carrier: power, antenna gain and direction differ), solved for exactly
  n             the path-loss exponent, between 2 and 5
  X, Y (dx, dy) the mast; fitted from several starts either side of the track

with a robust (soft L1, 6 dB) loss, since single readings fade by +-10 dB. Readings along one line cannot tell left of
the track from right; `side_clear` says whether the best fit on one side beat the best on the other by 20 % or more.
A mast seen from two lines, or where the line curves, is placed on both axes.

Tested on later trips (masts placed from 16 Mar - 5 Apr 2026, readings from 8 Apr - 2 May): each reading's signal from
distance to its mast is within 8.8 dB on average, against 10.5 dB with OpenCellID's positions for the same masts,
which are typically 1 km out (docs/validation.md). Only the measurements go in: no other mast list is used, so the
result is the scanner data's alone. The output (config/masts.csv) holds positions and fit figures, no readings.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

BIN_M = 50.0                  # readings are reduced to one median per square of this size and cell
H_M = 25.0                    # mast height over the track: only shapes the curve within ~100 m of the mast
MIN_LOCATIONS = 20            # squares a mast needs before it is placed
MIN_TRIPS = 2                 # and separate trips (train and day)
SIDE_MARGIN = 0.2             # the best fit on one side of the track beats the other side's by this much: side known
STARTS_ACROSS_M = (-1500.0, -500.0, -150.0, 150.0, 500.0, 1500.0)
SOURCE = "fitted_masts"        # how the cells table marks a mast placed here
COLUMNS = ["network", "mnc", "enb", "latitude", "longitude", "locations", "trips", "fit_rms_db", "side_clear"]


def readings(path: Path, operators: list[dict], preset: str = "global_view_4g", before: str | None = None) -> pd.DataFrame:
    """Scanner readings with the cell behind each, in British National Grid: x, y, rsrp_dbm, network, mnc, enb, cell,
    trip. before: keep only readings before this date (to test placed masts on later trips)."""
    from pyproj import Transformer

    from .sources.measurements import load_measurements

    m = load_measurements(Path(path), preset, operators)
    m = m[m["provider_id"].notna() & m["rsrp_dbm"].notna() & m["timestamp"].notna()]
    m = m[pd.to_numeric(m["cell_id"], errors="coerce").gt(255) & pd.to_numeric(m["mnc"], errors="coerce").notna()]
    if before:
        m = m[m["timestamp"] < pd.Timestamp(before, tz="UTC")]
    cell = pd.to_numeric(m["cell_id"]).astype("int64").to_numpy()
    x, y = Transformer.from_crs(4326, 27700, always_xy=True).transform(m["longitude"].to_numpy(float), m["latitude"].to_numpy(float))
    return pd.DataFrame({"x": x, "y": y, "rsrp_dbm": m["rsrp_dbm"].to_numpy(float), "network": m["provider_id"].to_numpy(),
                         "mnc": pd.to_numeric(m["mnc"]).astype("int64").to_numpy(), "enb": cell // 256, "cell": cell,
                         "trip": m["device"].astype(str).to_numpy() + "|" + m["timestamp"].dt.strftime("%Y-%m-%d").to_numpy(),
                         "timestamp": m["timestamp"].to_numpy()})


def bin_readings(r: pd.DataFrame) -> pd.DataFrame:
    """One row per mast, cell and 50 m square: median signal, how many readings, where (mean position)."""
    r = r.assign(bx=np.floor(r["x"] / BIN_M).astype("int64"), by=np.floor(r["y"] / BIN_M).astype("int64"))
    return (r.groupby(["network", "mnc", "enb", "cell", "bx", "by"], sort=False)
             .agg(rsrp_dbm=("rsrp_dbm", "median"), n=("rsrp_dbm", "size"), x=("x", "mean"), y=("y", "mean")).reset_index())


def fit_mast(x: np.ndarray, y: np.ndarray, rsrp: np.ndarray, n: np.ndarray, cell: np.ndarray) -> dict | None:
    """Position (and exponent) of one mast from its binned readings; None if no start converged."""
    from scipy.optimize import least_squares

    w = np.sqrt(np.minimum(n, 10.0))                   # repeat readings in a square count, up to a point
    _, ci = np.unique(cell, return_inverse=True)
    wsum = np.bincount(ci, weights=w)

    def residual(p):
        loss = 10 * p[2] * np.log10(np.sqrt((x - p[0]) ** 2 + (y - p[1]) ** 2 + H_M * H_M))
        level = np.bincount(ci, weights=w * (rsrp + loss)) / wsum      # each cell's own level, solved exactly
        return np.sqrt(w) * (rsrp - (level[ci] - loss))

    # start where the signal is strongest relative to its cell's typical level, then step across the local track line
    rel = rsrp - np.array([np.median(rsrp[ci == k]) for k in range(len(wsum))])[ci]
    k = int(np.argmax(rel))
    px, py = float(x[k]), float(y[k])
    near = np.hypot(x - px, y - py) < 2000.0
    if near.sum() >= 3:
        _, vecs = np.linalg.eigh(np.cov(np.vstack([x[near], y[near]])))
        along = vecs[:, 1]
    else:
        along = np.array([1.0, 0.0])
    across = np.array([-along[1], along[0]])
    fits = []
    for s in STARTS_ACROSS_M:
        try:
            o = least_squares(residual, x0=[px + across[0] * s, py + across[1] * s, 3.0],
                              bounds=([px - 20000, py - 20000, 2.0], [px + 20000, py + 20000, 5.0]),
                              loss="soft_l1", f_scale=6.0, x_scale=[500.0, 500.0, 1.0], max_nfev=200)
        except (ValueError, np.linalg.LinAlgError):
            continue
        side = np.sign((o.x[0] - px) * across[0] + (o.x[1] - py) * across[1])
        fits.append((float(o.cost), o.x, side))
    if not fits:
        return None
    cost, p, side = min(fits, key=lambda f: f[0])
    other = [f[0] for f in fits if f[2] != side]
    clear = bool(other) and (min(other) - cost) > SIDE_MARGIN * max(cost, 1e-9)
    res = residual(p) / np.sqrt(w)
    return {"x": float(p[0]), "y": float(p[1]), "exponent": float(p[2]), "fit_rms_db": float(np.sqrt(np.mean(res ** 2))), "side_clear": clear}


def _fit_group(args):
    key, g = args
    out = fit_mast(g["x"].to_numpy(), g["y"].to_numpy(), g["rsrp_dbm"].to_numpy(), g["n"].to_numpy(float), g["cell"].to_numpy())
    return (key, out)


def locate(r: pd.DataFrame, workers: int | None = None, log=print) -> pd.DataFrame:
    """Every mast with enough readings, placed: network, mnc, enb, latitude, longitude, locations, trips, fit_rms_db,
    side_clear."""
    from concurrent.futures import ProcessPoolExecutor

    from pyproj import Transformer

    b = bin_readings(r)
    keys = ["network", "mnc", "enb"]
    trips = r.groupby(keys)["trip"].nunique()
    locs = b.groupby(keys).size()
    ok = locs[(locs >= MIN_LOCATIONS) & (trips.reindex(locs.index).fillna(0) >= MIN_TRIPS)].index
    b = b.set_index(keys).loc[ok].reset_index()
    groups = [(k, g) for k, g in b.groupby(keys, sort=True)]
    log(f"masts: {r.groupby(keys).ngroups:,} seen, {len(groups):,} with {MIN_LOCATIONS}+ locations on {MIN_TRIPS}+ trips; placing them")
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            done = list(ex.map(_fit_group, groups, chunksize=32))
    else:
        done = [_fit_group(g) for g in groups]
    rows = [{"network": k[0], "mnc": int(k[1]), "enb": int(k[2]), **f, "locations": int(locs[k]), "trips": int(trips[k])}
            for k, f in done if f is not None]
    out = pd.DataFrame(rows, columns=[*keys, "x", "y", "exponent", "fit_rms_db", "side_clear", "locations", "trips"])
    lon, lat = Transformer.from_crs(27700, 4326, always_xy=True).transform(out["x"].to_numpy(), out["y"].to_numpy())
    out["latitude"], out["longitude"] = np.round(lat, 5), np.round(lon, 5)
    out["fit_rms_db"] = out["fit_rms_db"].round(1)
    log(f"masts: {len(out):,} placed; side of the track clear for {out['side_clear'].mean():.0%}" if len(out) else "masts: none placed")
    return out[COLUMNS].sort_values(["network", "enb"]).reset_index(drop=True)


def save(masts: pd.DataFrame, path: Path, source: str, period: tuple[str, str]) -> None:
    head = (f"# Mast positions fitted by `tcs locate-masts` to {source}, {period[0]} to {period[1]} (tcs/masts.py).\n"
            "# Positions and fit figures only; the measurements are not kept here. Re-run the command rather than editing.\n")
    Path(path).write_text(head + masts[COLUMNS].to_csv(index=False), encoding="utf-8")


def load(path: Path) -> pd.DataFrame:
    """config/masts.csv (empty frame if the file is missing)."""
    if not Path(path).exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, comment="#", dtype={"network": str})
