"""Learned adjustment: what the calibrated model gets wrong along the track, learned from where trains have measured
and applied everywhere, including stretches no train has measured.

Measured corrections (tcs/corrections.py) fix the model where scanner trains have run, and only there: on the 2026
check, test points with earlier Global View trips nearby are predicted to about 7 dB, close to the trip-to-trip noise,
while those without are about 10 dB out. The model's error is not random along the track, though. It follows what is
known at every route point (the Ofcom level and its spread, how far the nearest mast is and how many there are, the
cutting, the skyline, the land around), so gradient-boosted trees fitted to the errors where trains measured predict
part of them where none did:

    adjustment_db = trees(features)        features: FEATURES below, one row per route point and network

The target is each measured route point's error against the calibrated model (no measured corrections), less the
measurement set's own level per network (corrections.levels), so the trees learn the pattern, not a scanner's
offset. `tcs learn-adjustment` fits them (scikit-learn, `pip install -e ".[fit]"`) and stores the trees as a table
(config/learned_adjustment.parquet: split features, thresholds and leaf values, no measurements); building a route
only walks the stored trees with numpy. The adjustment enters the score through the calibration's dB scale, like a
measured correction, and measured corrections are then fitted on top of it (`tcs correct-routes` after this).

Tested on later trips (`tcs check-national`: trees and corrections from Global View before 7 April 2026 plus the
Yellow Train logs, tested from 7 April): docs/validation.md.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

NETWORKS = ["ee", "o2", "three", "vodafone"]
SAMPLE_FEATURES = ["cutting_depth_m", "embankment_height_m", "sky_visibility", "horizon_deg", "urban_density", "line_maxspeed_kph",
                   "canopy_probability", "terrain_m", "elevation_m", "rail_level_m", "wall_left_m", "wall_right_m", "overhead_fraction",
                   "obstruction_share", "on_bridge", "osm_cutting"]
PRIOR_FEATURES = ["level_4g", "level_spread", "n_uprn", "postcode_distance_m", "prior_score"]
FEATURES = [*SAMPLE_FEATURES, *PRIOR_FEATURES, "serving_km", "handover", "in_tunnel", *(f"net_{n}" for n in NETWORKS),
            "cells_2km", "cells_5km", "nearest_cell_km", "relief_1km"]
PARAMS = {"loss": "absolute_error", "max_iter": 400, "learning_rate": 0.05, "max_leaf_nodes": 63, "min_samples_leaf": 100,
          "l2_regularization": 1.0}
RELIEF_SAMPLES = 41           # route points either side (about 1 km at 50 m) the land is averaged over
INTERIM_FILE = "learned_adjustment.parquet"     # this build's adjustment, per route point and network
TREE_COLUMNS = ["tree", "node", "feature", "threshold", "missing_left", "left", "right", "leaf", "value"]


def features(samples: pd.DataFrame, prior: pd.DataFrame, serving: pd.DataFrame, cells: pd.DataFrame | None,
             rows: pd.DataFrame) -> pd.DataFrame:
    """FEATURES for each row of `rows` (sample_id, provider_id), in that order. Missing inputs stay NaN: the trees
    send them down the branch they learned for missing values."""
    from shapely import points
    from shapely.strtree import STRtree

    s = samples.set_index("sample_id")
    sid, pid = rows["sample_id"].to_numpy(), rows["provider_id"].to_numpy()
    key = pd.MultiIndex.from_arrays([sid, pid])
    out = pd.DataFrame(index=np.arange(len(rows)))
    for c in SAMPLE_FEATURES:
        out[c] = pd.to_numeric(s[c], errors="coerce").reindex(sid).to_numpy(float) if c in s else np.nan
    p = prior.set_index(["sample_id", "provider_id"])
    for c in PRIOR_FEATURES:
        out[c] = pd.to_numeric(p[c], errors="coerce").reindex(key).to_numpy(float) if c in p else np.nan
    sv = serving.set_index(["sample_id", "provider_id"])
    out["serving_km"] = sv["serving_distance_m"].reindex(key).to_numpy(float) / 1000.0
    out["handover"] = sv["handover"].reindex(key).astype(float).to_numpy()
    out["in_tunnel"] = s["in_tunnel"].fillna(False).astype(float).reindex(sid).to_numpy()
    for n in NETWORKS:
        out[f"net_{n}"] = (pid == n).astype(float)
    out[["cells_2km", "cells_5km", "nearest_cell_km"]] = np.nan
    if cells is not None and len(cells):
        xy = s[["x", "y"]].reindex(sid).to_numpy(float)
        for n in pd.unique(pid):
            m, cc = pid == n, cells[cells["provider_id"] == n]
            if cc.empty:
                continue
            cxy = cc[["x", "y"]].to_numpy(float)
            t, pts = STRtree(points(cxy)), points(xy[m])
            for col, r in (("cells_2km", 2000.0), ("cells_5km", 5000.0)):
                hit = t.query(pts, predicate="dwithin", distance=r)[0]
                out.loc[m, col] = np.bincount(hit, minlength=int(m.sum()))
            near = t.nearest(pts)
            out.loc[m, "nearest_cell_km"] = np.hypot(*(xy[m] - cxy[near]).T) / 1000.0
    land = s.sort_values("distance_m")["terrain_m"].astype(float)
    relief = land - land.rolling(RELIEF_SAMPLES, center=True, min_periods=5).mean()
    out["relief_1km"] = relief.reindex(sid).to_numpy(float)
    return out[FEATURES]


def fit(x: pd.DataFrame, error_db: np.ndarray, weight: np.ndarray | None = None) -> pd.DataFrame:
    """Gradient-boosted trees on the errors (least absolute error), returned as a tree table (TREE_COLUMNS)."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    x = x[FEATURES].copy()
    empty = [c for c in FEATURES if x[c].isna().all()]
    x[empty] = 0.0                                          # nothing measured carries it (scikit-learn cannot bin an empty column)
    m = HistGradientBoostingRegressor(**PARAMS).fit(x, error_db, sample_weight=weight)
    return to_table(m)


def to_table(model) -> pd.DataFrame:
    """A fitted HistGradientBoostingRegressor's trees as plain rows; the baseline goes in as tree -1, one leaf."""
    parts = [pd.DataFrame({"tree": [-1], "node": [0], "feature": [-1], "threshold": [np.nan], "missing_left": [False],
                           "left": [0], "right": [0], "leaf": [True], "value": [float(np.ravel(model._baseline_prediction)[0])]})]
    for t, (pred,) in enumerate(model._predictors):
        nd = pred.nodes
        if (nd["is_categorical"] != 0).any():
            raise ValueError("categorical splits are not supported")
        parts.append(pd.DataFrame({"tree": t, "node": np.arange(len(nd)), "feature": nd["feature_idx"].astype(int),
                                   "threshold": nd["num_threshold"].astype(float), "missing_left": nd["missing_go_to_left"].astype(bool),
                                   "left": nd["left"].astype(int), "right": nd["right"].astype(int), "leaf": nd["is_leaf"].astype(bool),
                                   "value": nd["value"].astype(float)}))
    return pd.concat(parts, ignore_index=True)[TREE_COLUMNS]


def predict(table: pd.DataFrame, x: pd.DataFrame) -> np.ndarray:
    """Walk the stored trees: the sum of the leaf each row reaches in every tree, plus the baseline."""
    xv = x[FEATURES].to_numpy(float)
    out = np.full(len(xv), float(table.loc[table["tree"] == -1, "value"].iloc[0]))
    rows = np.arange(len(xv))
    for _, t in table[table["tree"] >= 0].groupby("tree", sort=False):
        t = t.sort_values("node")
        feat, thr, ml = t["feature"].to_numpy(), t["threshold"].to_numpy(), t["missing_left"].to_numpy()
        left, right, leaf, val = t["left"].to_numpy(), t["right"].to_numpy(), t["leaf"].to_numpy(), t["value"].to_numpy()
        node = np.zeros(len(xv), dtype=int)
        live = ~leaf[node]
        while live.any():
            n = node[live]
            v = xv[rows[live], feat[n]]
            go_left = np.where(np.isnan(v), ml[n], v <= thr[n])
            node[live] = np.where(go_left, left[n], right[n])
            live = ~leaf[node]
        out += val[node]
    return out


def save(table: pd.DataFrame, path: Path, meta: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    t = pa.Table.from_pandas(table[TREE_COLUMNS], preserve_index=False)
    t = t.replace_schema_metadata({**(t.schema.metadata or {}), b"tcs": json.dumps({"features": FEATURES, **meta}).encode()})
    pq.write_table(t, path, compression="zstd")


def load(path: Path | None) -> pd.DataFrame | None:
    """The stored trees, or None (no file, or trees on a different feature list than this code computes)."""
    if not path or not Path(path).exists():
        return None
    import pyarrow.parquet as pq

    t = pq.read_table(path)
    if json.loads((t.schema.metadata or {}).get(b"tcs", b"{}")).get("features") != FEATURES:
        raise ValueError(f"{path} was fitted on other features; re-run `tcs learn-adjustment`")
    return t.to_pandas()


def meta(path: Path) -> dict:
    import pyarrow.parquet as pq

    if not Path(path).exists():
        return {}
    return json.loads((pq.read_schema(path).metadata or {}).get(b"tcs", b"{}"))


def path_of(settings) -> Path | None:
    from ..config import ROOT

    f = settings.sim["cellular"].get("learned_adjustment")
    return ROOT / f if f else None


def for_route(table: pd.DataFrame | None, samples: pd.DataFrame, prior: pd.DataFrame, serving: pd.DataFrame,
              cells: pd.DataFrame | None) -> pd.DataFrame | None:
    """The adjustment at every route point and network: sample_id, provider_id, adjustment_db. None without trees."""
    if table is None:
        return None
    rows = prior[["sample_id", "provider_id"]].reset_index(drop=True)
    return rows.assign(adjustment_db=predict(table, features(samples, prior, serving, cells, rows)))


def for_build(settings, samples: pd.DataFrame, prior: pd.DataFrame, serving: pd.DataFrame, cells: pd.DataFrame | None,
              interim: Path) -> pd.DataFrame | None:
    """This route's adjustment, kept beside its samples (reports re-simulate from them)."""
    adj = for_route(load(path_of(settings)), samples, prior, serving, cells)
    f = Path(interim) / INTERIM_FILE
    if adj is None:
        f.unlink(missing_ok=True)
    else:
        adj.to_parquet(f, index=False)
    return adj


def load_interim(interim: Path) -> pd.DataFrame | None:
    f = Path(interim) / INTERIM_FILE
    return pd.read_parquet(f) if f.exists() else None
