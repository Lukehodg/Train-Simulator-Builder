"""Validation metrics by route section: MAE/RMSE, outage precision/recall, classification accuracy, availability error."""
from __future__ import annotations

import numpy as np
import pandas as pd

RSRP_CLASS_EDGES = [-140, -110, -100, -90, 0]      # dBm -> OUTAGE/POOR/USABLE/STRONG
SCORE_CLASS_EDGES = [-1, 0.2, 0.4, 0.6, 2]


def _classify(x: np.ndarray, edges: list[float]) -> np.ndarray:
    return np.clip(np.digitize(x, edges[1:-1]), 0, 3)


def report(obs: pd.DataFrame, meas: pd.DataFrame, section_km: float = 10.0) -> pd.DataFrame:
    m = meas[(meas["kind"] == "point") & meas["provider_id"].notna()]
    j = m.merge(obs, on=["sample_id", "provider_id"], how="inner", suffixes=("_obs", ""))
    if j.empty:
        return pd.DataFrame(columns=["provider_id", "section", "n", "rsrp_mae", "rsrp_rmse", "class_acc", "outage_precision", "outage_recall", "avail_err_pp"])
    j["section"] = (j["distance_m"] / (section_km * 1000)).astype(int)
    rows = []
    for (pid, sec), g in j.groupby(["provider_id", "section"]):
        has_rsrp = g["rsrp_dbm"].notna()
        pred_rsrp = g["signal_primary"].values
        d = (pred_rsrp - g["rsrp_dbm"].values)[has_rsrp.values]
        pc = _classify(g["quality_score"].values, SCORE_CLASS_EDGES)
        oc = _classify(g["rsrp_dbm"].fillna(-140).values, RSRP_CLASS_EDGES)
        pred_out, obs_out = pc == 0, oc == 0
        tp = np.sum(pred_out & obs_out)
        prec = tp / max(1, pred_out.sum())
        rec = tp / max(1, obs_out.sum())
        pred_av = g["available"].mean()
        obs_av = (g["rsrp_dbm"] > -115).mean()
        rows.append({"provider_id": pid, "section": int(sec), "section_from_km": sec * section_km, "n": int(len(g)),
                     "rsrp_mae": float(np.mean(np.abs(d))) if len(d) else np.nan, "rsrp_rmse": float(np.sqrt(np.mean(d ** 2))) if len(d) else np.nan,
                     "class_acc": float(np.mean(pc == oc)), "outage_precision": float(prec), "outage_recall": float(rec),
                     "avail_err_pp": float((pred_av - obs_av) * 100)})
    return pd.DataFrame(rows).sort_values(["provider_id", "section"]).reset_index(drop=True)


def handover_position_error(pred_handovers_m: np.ndarray, observed_changes_m: np.ndarray) -> dict:
    """Metres between each observed serving-cell change and the nearest predicted handover."""
    if len(pred_handovers_m) == 0 or len(observed_changes_m) == 0:
        return {"n": 0}
    d = np.abs(observed_changes_m[:, None] - pred_handovers_m[None, :]).min(axis=1)
    return {"n": int(len(d)), "median_m": float(np.median(d)), "p90_m": float(np.percentile(d, 90))}
