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
    if not np.isfinite(section_km) or section_km <= 0 or min_points < 2:
        raise ValueError("calibration requires positive section_km and at least two points")
    if not np.isfinite(nominal_rsrp).all() or nominal_rsrp[1] <= nominal_rsrp[0]:
        raise ValueError("nominal RSRP anchors must be finite and increasing")
    m = measurements[(measurements["kind"] == "point") & np.isfinite(measurements["rsrp_dbm"]) & measurements["provider_id"].notna()]
    joined = m.merge(prior_obs[["sample_id", "provider_id", "quality_score", "distance_m"]], on=["sample_id", "provider_id"], how="inner")
    joined = joined[np.isfinite(joined["quality_score"]) & np.isfinite(joined["distance_m"])]
    out: dict = {"version": 2, "section_km": section_km, "rsrp_input": "corrected_quality",
                 "bias": {}, "rsrp_map": {}, "residual_std": {}, "sections": {}, "n_points": {}}
    for pid, g in joined.groupby("provider_id"):
        if len(g) < min_points:
            continue
        q = g["quality_score"].values.astype(float)
        r = g["rsrp_dbm"].values.astype(float)
        # implied score from observed RSRP under the model's nominal map -> additive bias in score space
        a0, b0 = nominal_rsrp[1] - nominal_rsrp[0], nominal_rsrp[0]
        q_obs = np.clip((r - b0) / a0, 0, 1)
        out["bias"][pid] = float(np.mean(q_obs - q))
        out["n_points"][pid] = int(len(g))
        sec = (g["distance_m"].values / (section_km * 1000)).astype(int)
        sdf = pd.DataFrame({"sec": sec, "d": q_obs - q}).groupby("sec")["d"].agg(["mean", "count"])
        out["sections"][pid] = {int(k): float(v["mean"]) for k, v in sdf.iterrows() if v["count"] >= max(10, min_points // 3)}
        corrections = np.array([out["sections"][pid].get(int(k), out["bias"][pid]) for k in sec])
        corrected = np.clip(q + corrections, 0, 1)
        # Fit on the same corrected score used at inference, avoiding a second bias application.
        a, b = np.polyfit(corrected, r, 1) if np.std(corrected) > 1e-3 else (a0, float(np.mean(r - a0 * corrected)))
        if a <= 0:
            a, b = a0, float(np.mean(r - a0 * corrected))
        out["rsrp_map"][pid] = {"slope": float(a), "intercept": float(b)}
        out["residual_std"][pid] = float(np.std(r - (a * corrected + b)))
    return out


def save(cal: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cal, fh, indent=2)


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        cal = json.load(fh)
    if not isinstance(cal, dict):
        raise ValueError("calibration must be an object")
    section_km = cal.get("section_km", 10)
    if not isinstance(section_km, (int, float)) or not np.isfinite(section_km) or section_km <= 0:
        raise ValueError("calibration section_km must be positive and finite")
    for key in ("bias", "rsrp_map", "sections"):
        if not isinstance(cal.get(key, {}), dict):
            raise ValueError(f"calibration {key} must be an object")
    for value in cal.get("bias", {}).values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError("calibration biases must be finite numbers")
    for mapping in cal.get("rsrp_map", {}).values():
        if not isinstance(mapping, dict) or any(not isinstance(mapping.get(k), (int, float)) or not np.isfinite(mapping[k]) for k in ("slope", "intercept")) or mapping["slope"] <= 0:
            raise ValueError("calibration RSRP mapping must have finite coefficients and positive slope")
    for sections in cal.get("sections", {}).values():
        if not isinstance(sections, dict) or any(not str(k).isdigit() or not isinstance(v, (int, float)) or not np.isfinite(v) for k, v in sections.items()):
            raise ValueError("calibration sections require non-negative indices and finite biases")
    return cal
