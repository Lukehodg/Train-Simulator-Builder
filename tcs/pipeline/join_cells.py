"""Cells along the corridor and candidate serving cells per sample (top-N by distance, per operator)."""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from ..config import Settings
from ..sources import synthetic
from ..sources.base import SourceUnavailable, console
from ..sources.opencellid import fetch_cells_bulk
from .sample_route import RouteBundle


def corridor_cells(settings: Settings, bundle: RouteBundle, samples: pd.DataFrame) -> pd.DataFrame:
    if not settings.offline:
        try:
            cells = fetch_cells_bulk(settings, bundle.proj, samples, settings.paths()["raw"], corridor_m=float(settings.route.get("corridor_m", 2500)))
            if len(cells):
                return cells
        except SourceUnavailable as exc:
            console.log(f"[yellow]OpenCellID: {exc}")
    console.log("[yellow]cell sites: synthetic stand-in (flagged)")
    return synthetic.synthetic_cells(samples, settings.operators, bundle.proj)


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
        nearest_key = best["cell_key"].reindex(range(n))
        nearest_d = best["dist_m"].reindex(range(n)).values
        by_sample = {sid: dict(zip(g["cell_key"], g["dist_m"])) for sid, g in grp.groupby("sample_id")}
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
            nk, nd = nearest_key.iloc[i], nearest_d[i]
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
