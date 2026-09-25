"""Confidence / provenance: every estimate carries a confidence in the plan's bands.

0.90-1.00 measured/validated · 0.70-0.89 strong source + calibrated · 0.40-0.69 prediction with partial
infrastructure support · 0.10-0.39 sparse/synthetic · 0.00 unknown
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings


def cellular_confidence(settings: Settings, obs: pd.DataFrame, measured_sample_ids: set[int] | None = None, calibrated: bool = False) -> np.ndarray:
    c = settings.sim["confidence"]
    has_cells = obs["serving_cell"].notna().values
    conf = np.full(len(obs), c["synthetic_prior"], dtype=np.float32)
    ofcom = obs["_prior_source"].astype(str).str.startswith("ofcom").values
    conf = np.where(ofcom & ~has_cells, c["ofcom_prior"], conf)
    conf = np.where(ofcom & has_cells, c["ofcom_prior_with_cells"], conf)
    conf = np.where(~obs["_has_prior"].values, 0.15, conf)
    if calibrated:
        conf = np.where(ofcom, conf + c["calibrated_bonus"], conf)
    if measured_sample_ids:
        m = obs["sample_id"].isin(measured_sample_ids).values
        conf = np.where(m, c["measured"], conf)
    conf = np.where(obs["_in_tunnel"].values, np.maximum(conf, c["tunnel_deterministic"]), conf)
    return np.clip(conf, 0, 1).astype(np.float32)
