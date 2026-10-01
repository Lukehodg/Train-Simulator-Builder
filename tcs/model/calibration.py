"""Calibrate the predicted score against measurements (Ofcom drive tests, train study, modem logs).

Produces, per operator: an additive score bias, a score->RSRP linear map (so RSRP stops being synthetic),
residual spread (drives confidence bands) and per-section bias for sections with enough observations.
A route's own fit (`tcs calibrate`) is saved as data/interim/<route>/calibration.json; without one, the national
calibration in config/calibration.yaml (`tcs calibrate-national`, fitted over every route) applies.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..config import ROOT, Settings


def fit(prior_obs: pd.DataFrame, measurements: pd.DataFrame, section_km: float = 10.0, min_points: int = 30,
        nominal_rsrp: tuple[float, float] = (-120.0, -74.0), sections: bool = True, fit_scale: bool = True) -> dict:
    """prior_obs: provider observations before calibration (quality_score, sample_id, provider_id, distance_m).
    measurements: canonical measurement rows with sample_id, provider_id, rsrp_dbm (point kind).
    nominal_rsrp: the model's (score=0, score=1) RSRP anchors; the bias is measured against this nominal map.
    sections: also fit a bias per section_km stretch of the route (off for a fit pooled over several routes).
    fit_scale: also fit the score->RSRP slope and intercept; off keeps the nominal map, so only the bias moves. A slope
    fitted by least squares shrinks when the predicted score is noisy, squeezing predicted signal into a narrow band:
    the right average, but the wrong share of the route in each signal band and a misplaced usable-signal threshold."""
    if not np.isfinite(section_km) or section_km <= 0 or min_points < 2:
        raise ValueError("calibration requires positive section_km and at least two points")
    if not np.isfinite(nominal_rsrp).all() or nominal_rsrp[1] <= nominal_rsrp[0]:
        raise ValueError("nominal RSRP anchors must be finite and increasing")
    m = measurements[(measurements["kind"] == "point") & np.isfinite(measurements["rsrp_dbm"]) & measurements["provider_id"].notna()]
    if "_in_tunnel" in prior_obs:                          # tunnel signal comes from the portals, not this correction
        prior_obs = prior_obs[~prior_obs["_in_tunnel"].astype(bool)]
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
        out["sections"][pid] = {int(k): float(v["mean"]) for k, v in sdf.iterrows() if v["count"] >= max(10, min_points // 3)} if sections else {}
        corrections = np.array([out["sections"][pid].get(int(k), out["bias"][pid]) for k in sec])
        corrected = np.clip(q + corrections, 0, 1)
        # Fit on the same corrected score used at inference, avoiding a second bias application.
        if not fit_scale:
            a, b = a0, b0
        else:
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
        return check(json.load(fh))


def national_file(settings: Settings) -> Path | None:
    rel = settings.sim["cellular"].get("national_calibration")
    return ROOT / rel if rel else None


def national_doc(settings: Settings) -> dict | None:
    """config/calibration.yaml as written by `tcs calibrate-national`: parameters, provenance and accuracy."""
    path = national_file(settings)
    if path is None or not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or None


def for_route(settings: Settings, interim: Path) -> dict | None:
    """The calibration the model uses on a route: the route's own fit if it has one, else the national one, else none."""
    own = load(interim / "calibration.json")
    if own is not None:
        return own
    doc = national_doc(settings)
    return check(dict(doc["calibration"], scope="national")) if doc and doc.get("calibration") else None


def check(cal) -> dict:
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


def describe(settings: Settings, interim: Path) -> dict | None:
    """Which calibration a route's model uses and how well it held up against measurements it was not fitted to,
    for reports and the viewer. None: the model is uncalibrated."""
    if (interim / "calibration.json").exists():
        return {"scope": "route", "source": "field measurements attached to this route"}
    doc = national_doc(settings)
    if not doc or not doc.get("calibration"):
        return None
    v = doc.get("validation") or {}
    cal = doc["calibration"]
    return {"scope": "national", "source": doc.get("source"), "fit_period": doc.get("fit_period"), "test_period": doc.get("test_period"),
            "fitted_on": doc.get("fitted_on"), "routes": len(doc.get("routes") or []), "points": doc.get("points"),
            "max_distance_m": doc.get("max_distance_m"), "environment": doc.get("environment"),
            "bias": cal.get("bias"), "residual_std": cal.get("residual_std"),
            "overall": (v.get("calibrated") or {}).get("overall"), "overall_before": (v.get("uncalibrated") or {}).get("overall"),
            "networks": (v.get("calibrated") or {}).get("networks"), "settings": (v.get("calibrated") or {}).get("settings"),
            "held_out_routes": v.get("held_out_routes"),
            "route": (v.get("routes") or {}).get(settings.route_id),      # None: no measurements along this route
            "current_check": _check_for(doc.get("current_check"), settings.route_id),
            "five_g": _five_g_for(doc.get("five_g"), settings.route_id)}


def _check_for(c: dict | None, rid: str) -> dict | None:
    """The model against later measurements (tcs check-national), nationally and on this route."""
    if not c:
        return None
    return {k: c.get(k) for k in ("source", "fit_period", "test_period", "level_offset_db", "overall", "networks")} | {"route": (c.get("routes") or {}).get(rid)}


def _five_g_for(f: dict | None, rid: str) -> dict | None:
    if not f:
        return None
    return {k: f.get(k) for k in ("source", "period", "usable_dbm", "bands", "unmeasured_bands", "networks")} | {"route": (f.get("routes") or {}).get(rid)}
