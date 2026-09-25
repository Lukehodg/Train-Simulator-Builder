"""Hermetic end-to-end run on synthetic sources with coarse spacing (fast)."""
import numpy as np
import pandas as pd
import pytest

from tcs.config import load_settings
from tcs.model.bonding import POLICIES, link_manager
from tcs.model.simulate import simulate
from tcs.pipeline.join_cells import candidate_cells, corridor_cells, serving_cells
from tcs.pipeline.join_coverage import coverage_prior
from tcs.pipeline.movement import movement
from tcs.pipeline.obstruction import enrich_terrain
from tcs.pipeline.sample_route import build_route


@pytest.fixture(scope="module")
def pipeline():
    s = load_settings(offline=True)
    s.route["sample_spacing_m"] = 500
    s.terrain["horizon_azimuths"] = 8
    s.terrain["horizon_reach_m"] = 1500
    b = build_route(s)
    b.samples = enrich_terrain(s, b)
    prior, b.samples = coverage_prior(s, b)
    cells = corridor_cells(s, b, b.samples)
    serving = serving_cells(s, b.samples, candidate_cells(s, b.samples, cells))
    b.samples, stations = movement(s, b.samples, b.stations)
    obs, rc = simulate(s, b.samples, prior, serving)
    return s, b, stations, prior, serving, obs, rc


def test_route_samples_monotonic(pipeline):
    s, b, stations, *_ = pipeline
    d = b.samples["distance_m"].values
    assert np.all(np.diff(d) > 0)
    assert stations["distance_m"].is_monotonic_increasing
    assert b.samples["geometry_source"].iloc[0] == "synthetic"


def test_time_model(pipeline):
    s, b, stations, *_ = pipeline
    t = b.samples["sim_seconds"].values
    assert np.all(np.diff(t) > 0)
    assert 3 * 3600 < t[-1] < 6 * 3600
    assert b.samples["speed_kph"].max() <= 200.5
    stops = stations[stations["stop"]]
    assert stops["scheduled_time"].is_monotonic_increasing


def test_tunnels_zero_starlink(pipeline):
    s, b, stations, prior, serving, obs, rc = pipeline
    sl = obs[obs["provider_id"] == "starlink"].set_index("sample_id")
    tun = b.samples[b.samples["in_tunnel"]]["sample_id"]
    if len(tun):
        assert (~sl.loc[tun, "available"]).all()
        assert (sl.loc[tun, "reason_code"] == "TUNNEL").all()
        assert (sl.loc[tun, "confidence"] >= 0.9).all()


def test_provider_observation_shape(pipeline):
    s, b, stations, prior, serving, obs, rc = pipeline
    n = len(b.samples)
    assert len(obs) == n * (len(s.operators) + len(s.satcom_providers))
    assert set(obs["provider_type"]) == {"cellular", "satcom"}
    assert obs["confidence"].between(0, 1).all()
    assert obs["quality_score"].between(0, 1).all()
    assert obs.loc[obs["provider_type"] == "cellular", "source_flags"].str.contains("synthetic").all()


def test_bonding_never_exceeds_sum(pipeline):
    s, b, stations, prior, serving, obs, rc = pipeline
    for pol in POLICIES:
        wan = link_manager(s, obs, policy=pol)
        total = obs.groupby("sample_id")["capacity_mbps"].sum().reindex(wan["sample_id"]).values
        assert np.all(wan["bonded_capacity_mbps"].values <= total + 1e-3), pol
        assert wan["bonded_capacity_mbps"].min() >= 0


def test_wifi_classes(pipeline):
    *_, rc = pipeline
    assert set(rc["service_class"]).issubset({"EXCELLENT", "GOOD", "USABLE", "POOR", "OUTAGE"})
    assert rc["wifi_service_score"].between(0, 100).all()
    assert (rc.loc[rc["bonded_capacity_mbps"] < 1, "service_class"] == "OUTAGE").all()


def test_handset_profile_is_worse(pipeline):
    s, b, stations, prior, serving, obs, rc = pipeline
    s2 = load_settings(offline=True)
    s2.route["sample_spacing_m"] = 500
    s2.sim["vehicle"]["profile"] = "PASSENGER_HANDSET_INSIDE_CARRIAGE"
    obs2, _ = simulate(s2, b.samples, prior, serving)
    a = obs[obs["provider_type"] == "cellular"]["capacity_mbps"].mean()
    b_ = obs2[obs2["provider_type"] == "cellular"]["capacity_mbps"].mean()
    assert b_ < a
