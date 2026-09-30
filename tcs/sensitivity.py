"""How far the headline results move when the main uncertain assumptions are wrong.

Each assumption is set to a pessimistic and an optimistic value, one at a time and then all together, and the
simulation is re-run from the same cached route data. The ranges are not statistical confidence intervals: they show
which assumptions matter and how much, which is what a reader weighing a prediction needs to know.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .config import Settings

COVERAGE_ERROR_DB = 5.0       # a plausible error of operators' outdoor coverage predictions at the track
MEASURES = ("streaming_share_pct", "usable_share_pct", "bonded_median_mbps", "bonded_p10_mbps", "outage_km")


@dataclass
class Case:
    key: str
    label: str
    phrase: str                                # the assumption in running text ("depends most on ...")
    low: str                                   # the pessimistic value, in words
    high: str                                  # the optimistic value, in words
    apply_low: Callable[[Settings], None]
    apply_high: Callable[[Settings], None] | None   # None: the central assumption is already the optimistic end


def _db_to_quality(s: Settings, db: float) -> float:
    r = s.sim["cellular"]["rsrp_dbm"]
    return db / float(r["at_one"] - r["at_zero"])


def _vehicle(s: Settings) -> dict:
    return s.sim["vehicle"]["profiles"][s.sim["vehicle"]["profile"]]


def _scale_cellular(f: float):
    def go(s: Settings) -> None:
        for op in s.operators:
            op["capacity_prior_mbps"] = {k: v * f for k, v in op["capacity_prior_mbps"].items()}
    return go


def _shift_signal(db: float):
    def go(s: Settings) -> None:
        _vehicle(s)["score_offset"] = float(_vehicle(s)["score_offset"]) + _db_to_quality(s, db)
    return go


def _scale_satellite(f: float):
    def go(s: Settings) -> None:
        for p in s.starlink["satcom"]["providers"]:
            p["capacity_prior_mbps"] = {k: v * f for k, v in p["capacity_prior_mbps"].items()}
    return go


def _online_share(share: float):
    def go(s: Settings) -> None:
        s.sim["passenger_wifi"]["active_share"] = share
    return go


def cases(s: Settings) -> list[Case]:
    """The assumptions varied for this scenario; the antenna case applies only where EDGE Rail antennas are fitted."""
    online = float(s.sim["passenger_wifi"]["active_share"])
    out = [
        Case("cellular", "Mobile network capacity (cell load, spectrum)", "mobile network capacity", "×0.6 (busy cells)", "×1.3",
             _scale_cellular(0.6), _scale_cellular(1.3)),
        Case("coverage", "Accuracy of operators' coverage predictions", "the accuracy of the operators' coverage predictions", f"signal {COVERAGE_ERROR_DB:.0f} dB weaker",
             f"signal {COVERAGE_ERROR_DB:.0f} dB stronger", _shift_signal(-COVERAGE_ERROR_DB), _shift_signal(COVERAGE_ERROR_DB)),
        Case("satellite", "Satellite capacity (beam load)", "satellite capacity", "×0.5", "×1.25", _scale_satellite(0.5), _scale_satellite(1.25)),
        Case("demand", "Passengers online at once", "how many passengers are online at once", f"{max(online, 0.5) * 100:.0f} % of passengers",
             f"{min(online, 0.25) * 100:.0f} % of passengers", _online_share(max(online, 0.5)), _online_share(min(online, 0.25))),
    ]
    if s.sim["vehicle"]["profile"] == "EDGE_RAIL_ACTIVE_ANTENNA":
        v = _vehicle(s)
        gain, factor = float(v.get("db_offset", 6)), float(v.get("capacity_factor", 1.35))

        half = float(v["score_offset"]) / 2

        def weaker_antenna(t: Settings) -> None:
            tv = _vehicle(t)
            tv["score_offset"] = float(tv["score_offset"]) - half   # remove half the modelled link-budget gain; composes with
            tv["capacity_factor"] = 1 + (factor - 1) / 2            # the coverage shift in the all-together case
        out.append(Case("antenna", "EDGE Rail antenna benefit", "the EDGE Rail antenna's assumed benefit", f"half the assumed gain (+{gain / 2:g} dB, ×{1 + (factor - 1) / 2:.2f})",
                        "as assumed", weaker_antenna, None))
    return out


def run(settings: Settings, samples: pd.DataFrame, prior: pd.DataFrame, serving: pd.DataFrame, *, calibration: dict | None = None,
        weather: str = "nominal") -> dict:
    """{"central": kpis, "cases": [{key, label, low, high, low_kpis, high_kpis}], "combined": {"low": kpis, "high": kpis}}."""
    from .model.simulate import simulate
    from .report import kpis

    spacing = float(settings.spacing_m)

    def outcome(mutators: list[Callable[[Settings], None]]) -> dict:
        s = copy.deepcopy(settings)
        for m in mutators:
            m(s)
        _, rc = simulate(s, samples, prior, serving, calibration=calibration, weather=weather)
        k = kpis(rc, samples, spacing)
        return {m: k[m] for m in MEASURES}

    central = outcome([])
    rows = []
    for c in cases(settings):
        rows.append({"key": c.key, "label": c.label, "phrase": c.phrase, "low": c.low, "high": c.high,
                     "low_kpis": outcome([c.apply_low]), "high_kpis": outcome([c.apply_high]) if c.apply_high else central})
    every = cases(settings)
    combined = {"low": outcome([c.apply_low for c in every]),
                "high": outcome([c.apply_high for c in every if c.apply_high]) if any(c.apply_high for c in every) else central}
    return {"central": central, "cases": rows, "combined": combined}


def span(sens: dict, measure: str) -> tuple[float, float]:
    """Lowest and highest value of a measure across every case, the combined ones included."""
    vals = [sens["central"][measure], sens["combined"]["low"][measure], sens["combined"]["high"][measure]]
    for r in sens["cases"]:
        vals += [r["low_kpis"][measure], r["high_kpis"][measure]]
    return min(vals), max(vals)
