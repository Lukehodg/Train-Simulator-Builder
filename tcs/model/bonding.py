"""Multi-WAN link manager: link scoring + policy -> combined onboard WAN per sample (vectorised).

score = w_cap * norm(capacity) + w_lat * norm(latency) + w_loss * norm(packet_loss) + w_stab * stability + w_conf * confidence
Bonding never sums advertised speeds: bonded = efficiency * sum(usable) * congestion.
The same logic is mirrored in web/src/sim/bonding.ts so the UI can switch policies without re-running Python.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Settings

POLICIES = ["FAILOVER", "WEIGHTED_LOAD_BALANCING", "PACKET_BONDING", "CELLULAR_PRIMARY_STARLINK_BACKUP", "STARLINK_PRIMARY_CELLULAR_BACKUP", "POLICY_BASED"]


def link_scores(settings: Settings, obs: pd.DataFrame) -> np.ndarray:
    w = settings.sim["wan"]["score_weights"]
    nrm = settings.sim["wan"]["normalisation"]
    nc = np.clip(obs["capacity_mbps"].values / nrm["capacity_mbps"], 0, 1)
    nl = np.clip(1 - obs["latency_ms"].values / nrm["latency_ms"], 0, 1)
    npl = np.clip(1 - obs["packet_loss_pct"].values / nrm["packet_loss_pct"], 0, 1)
    hp = obs["handover_penalty"].values if "handover_penalty" in obs else obs["handover"].values.astype(float)
    stab = np.where(hp > 0, 0.3, 1.0)           # degraded window, mirrored in web/src/sim/model.ts
    sc = w["capacity"] * nc + w["latency"] * nl + w["packet_loss"] * npl + w["stability"] * stab + w["confidence"] * obs["confidence"].values
    return np.where(obs["available"].values, sc, 0.0).astype(np.float32)


def link_manager(settings: Settings, obs: pd.DataFrame, policy: str | None = None) -> pd.DataFrame:
    wcfg = settings.sim["wan"]
    policy = policy or wcfg["policy"]
    obs = obs.copy()
    obs["score"] = link_scores(settings, obs)
    n = int(obs["sample_id"].max()) + 1
    providers = list(obs["provider_id"].unique())
    ptype = obs.drop_duplicates("provider_id").set_index("provider_id")["provider_type"].to_dict()
    # Wide matrices (samples x providers)
    def mat(col, fill=0.0):
        m = np.full((n, len(providers)), fill, dtype=np.float64)
        for j, p in enumerate(providers):
            g = obs[obs["provider_id"] == p].set_index("sample_id")[col].reindex(range(n))
            m[:, j] = g.fillna(fill).values
        return m
    score = mat("score")
    cap = mat("capacity_mbps")
    lat = mat("latency_ms", 999)
    loss = mat("packet_loss_pct", 100)
    conf = mat("confidence")
    avail = mat("available") > 0.5
    is_cell = np.array([ptype[p] == "cellular" for p in providers])
    usable = avail & (score >= wcfg["minimum_link_score"])

    chosen = np.zeros_like(usable)
    best = np.argmax(score, axis=1)
    best_cell = np.argmax(np.where(is_cell[None, :], score, -1), axis=1)
    best_sat = np.argmax(np.where(~is_cell[None, :], score, -1), axis=1)
    rows = np.arange(n)
    if policy == "FAILOVER":
        chosen[rows, best] = usable[rows, best]
    elif policy == "CELLULAR_PRIMARY_STARLINK_BACKUP":
        primary = usable[rows, best_cell] & is_cell[best_cell]
        chosen[rows[primary], best_cell[primary]] = True
        backup = ~primary & usable[rows, best_sat] & ~is_cell[best_sat]
        chosen[rows[backup], best_sat[backup]] = True
    elif policy == "STARLINK_PRIMARY_CELLULAR_BACKUP":
        primary = usable[rows, best_sat] & ~is_cell[best_sat]
        chosen[rows[primary], best_sat[primary]] = True
        backup = ~primary & usable[rows, best_cell] & is_cell[best_cell]
        chosen[rows[backup], best_cell[backup]] = True
    else:  # bonding / load balancing / policy-based: every usable link participates
        chosen = usable.copy()
    # Degraded fallback: hold whatever is available even below the minimum score.
    none = ~chosen.any(axis=1)
    hold = none & avail[rows, best]
    chosen[rows[hold], best[hold]] = True

    csum = (cap * chosen).sum(axis=1)
    min_lat = np.where(chosen, lat, np.inf).min(axis=1)
    min_loss = np.where(chosen, loss, np.inf).min(axis=1)
    wl = (lat * cap * chosen).sum(axis=1) / np.maximum(csum, 1e-6)
    any_ = chosen.any(axis=1)
    if policy == "PACKET_BONDING":
        bonded = wcfg["bonding_efficiency"] * csum * wcfg["congestion_factor"]
        eff_lat = min_lat + 5
        eff_loss = min_loss * 0.7
    elif policy in ("WEIGHTED_LOAD_BALANCING", "POLICY_BASED"):
        bonded = csum * wcfg["load_balancing_efficiency"]
        eff_lat = np.where(csum > 0, wl, min_lat)
        eff_loss = min_loss
    else:
        bonded, eff_lat, eff_loss = csum, min_lat, min_loss
    bonded = np.where(any_, bonded, 0.0)
    eff_lat = np.where(any_, eff_lat, np.nan)
    eff_loss = np.where(any_, eff_loss, 100.0)
    active = ["+".join(p for p, c in zip(providers, row) if c) for row in chosen]
    wconf = np.where(any_, (conf * chosen).sum(axis=1) / np.maximum(chosen.sum(axis=1), 1), 0.0)
    return pd.DataFrame({"sample_id": rows, "active_links": active, "wan_policy": policy, "bonded_capacity_mbps": bonded.astype(np.float32),
                         "effective_latency_ms": eff_lat.astype(np.float32), "packet_loss_pct": eff_loss.astype(np.float32), "confidence": wconf.astype(np.float32),
                         "n_links": chosen.sum(axis=1)})
