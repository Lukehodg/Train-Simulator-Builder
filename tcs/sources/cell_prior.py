"""Coverage prior from cell-site distance, for countries without a regulator's coverage predictions (the US).

Each network's prior at a route sample is read from the distance to its nearest site in the corridor cell table
(OpenCellID), with a small lift where a 5G (NR) site is close and where several sites surround the line. The knots
are assumptions in the country's networks file (`site_prior`) until field measurements calibrate them; the rows are
tagged `opencellid_sites` so the viewer and the reports say the prior is a site-distance estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from shapely import points
from shapely.strtree import STRtree

SOURCE = "opencellid_sites"
DEFAULT = {"knots": [[0.0, 0.84], [1.0, 0.78], [3.0, 0.64], [6.0, 0.46], [10.0, 0.30], [15.0, 0.15], [25.0, 0.05]],
           "nr_lift": 0.04, "nr_radius_km": 3.0, "density_radius_km": 5.0, "density_lift_per_site": 0.01, "density_lift_max": 0.06}


def _nearest_m(pts, xy: np.ndarray) -> np.ndarray:
    """Distance (m) from each point to the nearest of xy."""
    tree = STRtree(points(xy[:, 0], xy[:, 1]))
    (src, _), dist = tree.query_nearest(pts, return_distance=True, all_matches=False)
    out = np.full(len(pts), np.inf)
    out[src] = dist
    return out


def site_prior(samples: pd.DataFrame, cells: pd.DataFrame, operators: list[dict], cfg: dict | None = None) -> pd.DataFrame:
    """One row per (sample, network) with the columns join_coverage expects. Networks with no site in the corridor get
    no rows, so the model sees them as having no coverage record."""
    c = {**DEFAULT, **(cfg or {})}
    knots = np.asarray(c["knots"], dtype=float)
    pts = points(samples["x"].to_numpy(float), samples["y"].to_numpy(float))
    frames = []
    for op in operators:
        mine = cells[cells["provider_id"] == op["id"]] if len(cells) else cells
        if mine is None or not len(mine):
            continue
        # Several logical cells share a mast: collapse to sites ~50 m apart so the density lift counts masts, not sectors.
        site_xy = np.unique(np.round(mine[["x", "y"]].to_numpy(float) / 50.0) * 50.0, axis=0)
        d = _nearest_m(pts, site_xy)
        score = np.interp(d / 1000.0, knots[:, 0], knots[:, 1])
        within = STRtree(points(site_xy[:, 0], site_xy[:, 1])).query(pts, predicate="dwithin", distance=c["density_radius_km"] * 1000.0)
        n_near = np.bincount(within[0], minlength=len(pts))
        score += np.minimum(np.maximum(n_near - 1, 0) * c["density_lift_per_site"], c["density_lift_max"])
        nr = mine[mine["radio"].astype(str).str.upper() == "NR"]
        has_nr = np.zeros(len(pts), dtype=bool)
        if len(nr):
            has_nr = _nearest_m(pts, nr[["x", "y"]].to_numpy(float)) <= c["nr_radius_km"] * 1000.0
            score = score + np.where(has_nr, c["nr_lift"], 0.0)
        score = np.clip(score, 0.02, 0.95)
        frames.append(pd.DataFrame({
            "sample_id": samples["sample_id"].to_numpy(), "provider_id": op["id"], "prior_score": score.astype(np.float32),
            "level_4g": np.nan, "level_5g": np.nan, "radio_technology": np.where(has_nr, "5G", "4G"),
            "source": SOURCE, "postcode": None, "postcode_distance_m": np.nan,
        }))
    if not frames:
        return pd.DataFrame(columns=["sample_id", "provider_id", "prior_score", "level_4g", "level_5g", "radio_technology", "source", "postcode",
                                     "postcode_distance_m"])
    return pd.concat(frames, ignore_index=True)
