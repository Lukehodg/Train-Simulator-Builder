"""Calibrate the predicted score against measurements (Ofcom drive tests, train study, modem logs).

Produces, per operator: an additive score bias, a score->RSRP linear map (so RSRP stops being synthetic),
residual spread (drives confidence bands) and per-section bias for sections with enough observations.
Result is saved as data/interim/<route>/calibration.json and applied by the cellular model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def fit(prior_obs: pd.DataFrame, measurements: pd.DataFrame, section_km: float = 10.0, min_points: int = 30,
        nominal_rsrp: tuple[float, float] = (-120.0, -74.0)) -> dict:
    """prior_obs: provider observations before calibration (quality_score, sample_id, provider_id, distance_m).
    measurements: canonical measurement rows with sample_id, provider_id, rsrp_dbm (point kind).
    nominal_rsrp: the model's (score=0, score=1) RSRP anchors; the bias is measured against this nominal map."""
    m = measurements[(measurements["kind"] == "point") & measurements["rsrp_dbm"].notna() & measurements["provider_id"].notna()]
    joined = m.merge(prior_obs[["sample_id", "provider_id", "quality_score", "distance_m"]], on=["sample_id", "provider_id"], how="inner")
    out: dict = {"bias": {}, "rsrp_map": {}, "residual_std": {}, "sections": {}, "n_points": {}}
    for pid, g in joined.groupby("provider_id"):
        if len(g) < min_points:
            continue
        q = g["quality_score"].values.astype(float)
        r = g["rsrp_dbm"].values.astype(float)
        # score -> RSRP linear fit
        a, b = np.polyfit(q, r, 1) if np.std(q) > 1e-3 else (46.0, -120.0)
        pred = a * q + b
        resid = r - pred
        # implied score from observed RSRP under the model's nominal map -> additive bias in score space
        a0, b0 = nominal_rsrp[1] - nominal_rsrp[0], nominal_rsrp[0]
        q_obs = np.clip((r - b0) / a0, 0, 1)
        out["bias"][pid] = float(np.mean(q_obs - q))
        out["rsrp_map"][pid] = {"slope": float(a), "intercept": float(b)}
        out["residual_std"][pid] = float(np.std(resid))
        out["n_points"][pid] = int(len(g))
        sec = (g["distance_m"].values / (section_km * 1000)).astype(int)
        sdf = pd.DataFrame({"sec": sec, "d": q_obs - q}).groupby("sec")["d"].agg(["mean", "count"])
        out["sections"][pid] = {int(k): float(v["mean"]) for k, v in sdf.iterrows() if v["count"] >= max(10, min_points // 3)}
    return out


def save(cal: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cal, fh, indent=2)


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
