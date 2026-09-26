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


def test_next_station_is_first_stop_ahead(pipeline):
    _, b, stations, *_ = pipeline
    stops = stations[stations["stop"]].sort_values("distance_m")
    t = b.samples["sim_seconds"].values
    for j in range(len(b.samples)):
        ahead = stops[stops["sample_id"] > j]
        if len(ahead):
            k = int(ahead.iloc[0]["sample_id"])
            assert b.samples["next_station"].iat[j] == ahead.iloc[0]["crs"]
            assert b.samples["time_to_next_station_s"].iat[j] == pytest.approx(t[k] - t[j], rel=1e-6)
        else:
            assert pd.isna(b.samples["next_station"].iat[j])


def test_calibrate_twice_keeps_calibration(pipeline, tmp_path, monkeypatch):
    """`tcs calibrate` after a calibrated `tcs run` must reproduce the same fit, not measure (and save) the residual."""
    from typer.testing import CliRunner

    from tcs import config
    from tcs.cli import app
    from tcs.model.calibration import load
    from tcs.pipeline.sample_route import save_bundle

    s, b, _, prior, serving, obs, _ = pipeline
    for name in ("RAW", "INTERIM", "PROCESSED"):
        monkeypatch.setattr(config, name, tmp_path / name.lower())
    interim, processed = tmp_path / "interim" / s.route_id, tmp_path / "processed" / s.route_id
    save_bundle(b, interim)
    prior.to_parquet(interim / "coverage_prior.parquet", index=False)
    serving.to_parquet(interim / "serving.parquet", index=False)
    processed.mkdir(parents=True)
    obs.to_parquet(processed / "provider_observation.parquet", index=False)      # an uncalibrated `tcs run`

    # Survey says EE is 0.1 worse in score space than predicted.
    ee = obs[(obs["provider_id"] == "ee") & (obs["reason_code"] != "TUNNEL")].merge(b.samples[["sample_id", "latitude", "longitude"]], on="sample_id")
    r = s.sim["cellular"]["rsrp_dbm"]
    rsrp = r["at_zero"] + (r["at_one"] - r["at_zero"]) * np.clip(ee["quality_score"] - 0.1, 0, 1)
    csv = tmp_path / "survey.csv"
    pd.DataFrame({"latitude": ee["latitude"], "longitude": ee["longitude"], "rsrp": rsrp, "mcc": 234, "mnc": 30}).to_csv(csv, index=False)

    args = ["calibrate", str(csv), "--route", s.route_id]
    res = CliRunner().invoke(app, args)
    assert res.exit_code == 0, res.output
    bias1 = load(interim / "calibration.json")["bias"]["ee"]
    assert abs(bias1 + 0.1) < 0.02
    obs_cal, _ = simulate(s, b.samples, prior, serving, calibration=load(interim / "calibration.json"))
    obs_cal.to_parquet(processed / "provider_observation.parquet", index=False)  # `tcs run` again, now calibrated
    res = CliRunner().invoke(app, args)
    assert res.exit_code == 0, res.output
    assert load(interim / "calibration.json")["bias"]["ee"] == pytest.approx(bias1)


@pytest.mark.parametrize("calibrated", [False, True])
def test_browser_model_parity(pipeline, calibrated):
    import copy
    import json
    import shutil
    import subprocess
    from itertools import product

    from tcs.config import ROOT

    node = shutil.which("node")
    if not node or not (ROOT / "web/node_modules/typescript").exists():
        pytest.skip("Install Node.js and web dependencies to run browser parity checks")
    s, b, _, prior, serving, obs, _ = pipeline
    cal = {"version": 2, "rsrp_input": "corrected_quality", "section_km": 10,
           "bias": {"ee": 0.08}, "sections": {"ee": {"1": -0.05}},
           "rsrp_map": {"ee": {"slope": 42, "intercept": -119}}} if calibrated else None
    obs, _ = simulate(s, b.samples, prior, serving, calibration=cal)
    providers = [{"id": p["id"], "type": "cellular", "capacity_prior_mbps": p["capacity_prior_mbps"]} for p in s.operators]
    for p in s.starlink["satcom"]["providers"]:
        providers.append({**p, "type": "satcom", "capacity_prior_mbps": p["capacity_prior_mbps"][p["terminal"]]})
    data = {"n": len(b.samples), "meta": {"providers": providers, "sim": s.sim, "route": s.route}, "base": {}}
    for field, col in {"inTunnel": "in_tunnel", "tunnelName": "tunnel_name", "distance": "distance_m",
                       "speed": "speed_kph", "urban": "urban_density", "canopy": "canopy_probability",
                       "cutting": "cutting_depth_m"}.items():
        data[field] = b.samples[col].astype(object).where(b.samples[col].notna(), None).tolist()
    data["inTunnel"] = [int(v) for v in data["inTunnel"]]
    for pid, g in obs.groupby("provider_id", sort=False):
        g = g.sort_values("sample_id")
        data["base"][pid] = {field: g[col].tolist() for field, col in
                              {"qb": "quality_base", "hp": "handover_penalty", "conf": "confidence", "tech": "radio_technology"}.items()}
        if g["provider_type"].iloc[0] == "cellular":
            data["base"][pid]["rsrpSlope"] = g["rsrp_slope"].tolist()
            data["base"][pid]["rsrpIntercept"] = g["rsrp_intercept"].tolist()
    cases = []
    for policy in POLICIES:
        for ap_capacity, weather in product([0, 300], ["nominal", "rain", "storm"]):
            settings = copy.deepcopy(s)
            settings.sim["passenger_wifi"]["ap_capacity_mbps"] = ap_capacity
            scenario_obs, result = simulate(settings, b.samples, prior, serving, policy=policy, weather=weather, calibration=cal)
            case_data = {**data, "meta": {**data["meta"], "sim": settings.sim}}
            cases.append({"data": case_data, "scenario": {"policy": policy, "vehicle": settings.sim["vehicle"]["profile"], "weather": weather},
                          "expected": {"rsrp": {pid: g.sort_values("sample_id")["signal_primary"].tolist() for pid, g in scenario_obs[scenario_obs["provider_type"] == "cellular"].groupby("provider_id")},
                                       "bonded": result["bonded_capacity_mbps"].tolist(), "wifi": result["wifi_service_score"].tolist(),
                                       "cls": result["service_class"].tolist()}})
    completed = subprocess.run([node, str(ROOT / "web/tests/model-parity.mjs")], input=json.dumps(cases), text=True, capture_output=True)
    assert completed.returncode == 0, (completed.stdout + completed.stderr)[-5000:]


def test_calibration_applies_only_to_calibrated_provider_and_sections(pipeline):
    from tcs.model.cellular import cellular_observations
    from tcs.model.confidence import cellular_confidence

    s, b, _, prior, serving, *_ = pipeline
    prior = prior.copy()
    prior["source"] = "ofcom_test"
    cal = {"version": 2, "rsrp_input": "corrected_quality", "section_km": 10, "bias": {"ee": 0.05},
           "sections": {"ee": {"1": -0.1}}, "rsrp_map": {"ee": {"slope": 40, "intercept": -115}}}
    raw = cellular_observations(s, b.samples, prior, serving)
    adjusted = cellular_observations(s, b.samples, prior, serving, cal)
    ee = (adjusted["provider_id"] == "ee") & ~adjusted["_in_tunnel"]
    other = adjusted["provider_id"] != "ee"
    assert np.allclose(adjusted.loc[ee, "signal_primary"], 40 * adjusted.loc[ee, "quality_score"] - 115)
    assert np.allclose(raw.loc[other, "quality_score"], adjusted.loc[other, "quality_score"])
    assert not adjusted.loc[other, "source_flags"].str.contains("calibrated").any()
    distances = adjusted["sample_id"].map(b.samples.set_index("sample_id")["distance_m"])
    correction = np.where((distances >= 10000) & (distances < 20000), -0.1, 0.05)
    assert np.allclose(adjusted.loc[ee, "quality_base"], np.clip(raw.loc[ee, "quality_base"] + correction[ee], 0, 1))
    before = cellular_confidence(s, raw)
    after = cellular_confidence(s, adjusted)
    assert np.allclose(before[other], after[other])
    assert np.all(after[ee] > before[ee])
