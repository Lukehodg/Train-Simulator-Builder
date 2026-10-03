"""Cells along the corridor and candidate serving cells per sample (top-N by distance, per operator)."""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from .. import masts
from ..config import ROOT, Settings
from ..sources import synthetic
from ..sources.base import SourceUnavailable, console
from ..sources.opencellid import fetch_cells_bulk
from .sample_route import RouteBundle

FITTED_SOURCE = masts.SOURCE


def corridor_cells(settings: Settings, bundle: RouteBundle, samples: pd.DataFrame) -> pd.DataFrame:
    if not settings.offline:
        corridor_m = float(settings.route.get("corridor_m", 2500))
        try:
            cells = fetch_cells_bulk(settings, bundle.proj, samples, settings.paths()["raw"], corridor_m=corridor_m)
            if len(cells):
                return with_fitted_masts(settings, bundle.proj, samples, cells, corridor_m)
        except SourceUnavailable as exc:
            console.log(f"[yellow]OpenCellID: {exc}")
    console.log("[yellow]cell sites: synthetic stand-in (flagged)")
    return synthetic.synthetic_cells(samples, settings.operators, bundle.proj)


def with_fitted_masts(settings: Settings, proj, samples: pd.DataFrame, cells: pd.DataFrame, corridor_m: float) -> pd.DataFrame:
    """Masts placed from scanner logs (config/masts.csv, `tcs locate-masts`) stand in for OpenCellID's cells of the same
    mast (network and eNodeB): one row per mast at the fitted position. OpenCellID positions are averages of where phones
    heard a cell, typically ~1 km from the mast; the fitted ones predict later trips' signal from distance markedly
    better (docs/validation.md). Masts the logs never placed keep OpenCellID's cells."""
    path = (settings.networks.get("fitted_masts") or {}).get("file")
    if not path or cells.empty:
        return cells
    m = masts.load(ROOT / path)
    ops = {op["id"]: op for op in settings.operators}
    m = m[m["network"].isin(ops)].reset_index(drop=True)
    if m.empty:
        return cells
    x, y = proj.to_xy(m["longitude"].to_numpy(float), m["latitude"].to_numpy(float))
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    sx, sy = samples["x"].to_numpy()[::10], samples["y"].to_numpy()[::10]       # every 500 m is enough for a corridor cut
    from shapely import points
    from shapely.strtree import STRtree

    idx = STRtree(points(sx, sy)).nearest(points(x, y))
    near = np.hypot(sx[idx] - x, sy[idx] - y) <= corridor_m + 500
    m, x, y = m[near].reset_index(drop=True), x[near], y[near]
    if m.empty:
        return cells
    mcc = m["network"].map({k: int(op["mcc"]) for k, op in ops.items()})
    fitted = pd.DataFrame({
        "cell_key": mcc.astype(str) + "-" + m["mnc"].astype(str) + "-enb-" + m["enb"].astype(str), "provider_id": m["network"], "radio": "LTE",
        "mcc": mcc.astype(int), "mnc": m["mnc"].astype(int), "area_or_tac": np.int64(-1), "cell_id": m["enb"].astype("int64"), "pci_or_unit": np.int64(-1),
        "latitude": m["latitude"].astype(float), "longitude": m["longitude"].astype(float), "x": x, "y": y, "elevation_m": np.float32(np.nan),
        "samples": m["locations"].astype(int), "range_m": 0, "source": FITTED_SOURCE,
    })
    placed = set(zip(fitted["mnc"].tolist(), fitted["cell_id"].tolist()))
    lte = (cells["radio"] == "LTE").to_numpy()
    same = np.array([(int(a), int(c) // 256) in placed for a, c in zip(cells["mnc"], cells["cell_id"])], dtype=bool) & lte
    console.log(f"masts placed from scanner logs: {len(fitted)} in the corridor, standing in for {int(same.sum())} OpenCellID cells")
    return pd.concat([cells[~same], fitted], ignore_index=True)


def candidate_cells(settings: Settings, samples: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Top-N nearest cells per (sample, operator) within the candidate radius. Pure DuckDB, no spatial extension needed."""
    ocfg = settings.networks["opencellid"]
    radius = float(ocfg.get("candidate_radius_m", 12000))
    topn = int(ocfg.get("candidates_per_sample", 5))
    con = duckdb.connect()
    con.register("s", samples[["sample_id", "x", "y"]])
    con.register("c", cells[["cell_key", "provider_id", "radio", "x", "y", "samples"]])
    df = con.execute(f"""
        SELECT sample_id, provider_id, cell_key, radio, samples AS observations, dist_m
        FROM (
            SELECT s.sample_id, c.provider_id, c.cell_key, c.radio, c.samples,
                   sqrt((c.x - s.x)*(c.x - s.x) + (c.y - s.y)*(c.y - s.y)) AS dist_m,
                   row_number() OVER (PARTITION BY s.sample_id, c.provider_id ORDER BY sqrt((c.x - s.x)*(c.x - s.x) + (c.y - s.y)*(c.y - s.y))) AS rn
            FROM s JOIN c ON abs(c.x - s.x) < {radius} AND abs(c.y - s.y) < {radius}
        ) WHERE rn <= {topn} AND dist_m <= {radius}
        ORDER BY sample_id, provider_id, rn
    """).df()
    return df


def serving_cells(settings: Settings, samples: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Sequential serving-cell choice with hysteresis per operator; emits handover flags.

    Output per (sample, operator): serving_cell, serving_distance_m, handover (bool), handover_penalty (0..1 decaying).
    """
    hcfg = settings.sim["cellular"]["handover"]
    ratio = float(hcfg["hysteresis_ratio"])
    max_d = float(hcfg["max_serving_distance_km"]) * 1000
    dur = int(hcfg["duration_samples"])
    n = len(samples)
    out = []
    for pid, grp in candidates.groupby("provider_id"):
        best = grp.sort_values(["sample_id", "dist_m"]).drop_duplicates("sample_id").set_index("sample_id")
        # Plain lists/dicts: per-sample pandas groupby and .iloc inside the loop below dominated the pipeline runtime.
        nearest_key = best["cell_key"].reindex(range(n)).tolist()
        nearest_d = best["dist_m"].reindex(range(n)).tolist()
        by_sample: dict[int, dict] = {}
        for sid, key, d in zip(grp["sample_id"].tolist(), grp["cell_key"].tolist(), grp["dist_m"].tolist()):
            by_sample.setdefault(sid, {})[key] = d
        serving = np.full(n, None, dtype=object)
        sdist = np.full(n, np.nan)
        ho = np.zeros(n, dtype=bool)
        pen = np.zeros(n, dtype=np.float32)
        cur = None
        left = 0
        for i in range(n):
            cands = by_sample.get(i, {})
            if cur is not None and cur in cands:
                cd = cands[cur]
            else:
                cd = np.inf
            nk, nd = nearest_key[i], nearest_d[i]
            if cur is None or (isinstance(nk, str) and nk != cur and (nd < cd * ratio or cd > max_d)):
                if cur is not None and isinstance(nk, str):
                    ho[i] = True
                    left = dur
                cur = nk if isinstance(nk, str) else cur
                cd = cands.get(cur, np.nan) if cur else np.nan
            serving[i] = cur
            sdist[i] = cd if np.isfinite(cd) else np.nan
            if left > 0:
                pen[i] = left / dur
                left -= 1
        out.append(pd.DataFrame({"sample_id": np.arange(n), "provider_id": pid, "serving_cell": serving, "serving_distance_m": sdist.astype(np.float32),
                                 "handover": ho, "handover_penalty": pen}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["sample_id", "provider_id", "serving_cell", "serving_distance_m", "handover", "handover_penalty"])
