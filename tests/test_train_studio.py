import json
from pathlib import Path

import pytest

from tcs.config import load_settings
from tcs.sources.train_studio import apply_to_settings, derive, load_project

PROJECT = {
    "app": "motion-applied-train-studio", "version": 1,
    "project": {
        "title": "Azuma 5-car", "slideNo": "11", "threeD": True, "showNumbers": False, "showLegend": False, "cards": [],
        "types": [
            {"id": "t1", "slug": "edge-mini", "name": "EDGE Mini", "color": "#1b7a8c", "placement": "roof", "link": True, "hardwareKind": "satcom", "presetId": "edge-mini"},
            {"id": "t2", "slug": "edge-core", "name": "EDGE Core", "color": "#333333", "placement": "rack", "link": True, "hardwareKind": "router", "presetId": "edge-core"},
        ],
        "cars": [
            {"id": "a", "type": "cabL", "aps": 1, "sw": True, "edge": 1, "fleet": True, "custom": {"t1": True, "t2": True}, "mounts": {}},
            {"id": "b", "type": "mid", "aps": 2, "sw": True, "edge": 0, "fleet": False, "custom": {}, "mounts": {}},
            {"id": "c", "type": "mid", "aps": 1, "sw": False, "edge": 1, "fleet": False, "custom": {}, "mounts": {}},   # no switch -> APs not counted
            {"id": "d", "type": "mid", "aps": 1, "sw": True, "edge": 0, "fleet": False, "custom": {}, "mounts": {}},
            {"id": "e", "type": "cabR", "aps": 1, "sw": True, "edge": 1, "fleet": False, "custom": {}, "mounts": {}},
        ],
    },
}


def test_derive_design(tmp_path):
    f = tmp_path / "azuma.train.json"
    f.write_text(json.dumps(PROJECT), encoding="utf-8")
    s = load_settings(offline=True)
    d = derive(load_project(f), s.sim["train"])
    assert len(d.carriages) == 5 and d.cellular_units == 3 and d.satcom_units == 1 and d.satcom_terminal == "mini"
    assert d.aps_total == 6 and d.aps_connected == 5          # carriage 3 has no switch
    assert d.passengers == 56 + 76 * 3 + 56
    assert d.fleet_connect and d.policy == "PACKET_BONDING"
    assert d.vehicle_profile == "EDGE_RAIL_ACTIVE_ANTENNA"
    assert d.fitted_networks == ["ee", "vodafone", "three"]     # three EDGE Rail units, one network each
    assert d.ap_capacity_mbps == 120 * 5
    assert any("no switch" in w for w in d.warnings)


def test_apply_to_settings(tmp_path):
    f = tmp_path / "p.train.json"
    f.write_text(json.dumps(PROJECT), encoding="utf-8")
    s = load_settings(offline=True)
    d = derive(load_project(f), s.sim["train"])
    apply_to_settings(s, d)
    assert s.sim["vehicle"]["profile"] == "EDGE_RAIL_ACTIVE_ANTENNA"
    assert s.sim["passenger_wifi"]["passengers"] == d.passengers
    assert s.sim["passenger_wifi"]["ap_capacity_mbps"] == 600
    assert s.starlink["satcom"]["providers"][0]["terminal"] == "mini"
    assert s.sim["satcom_enabled"] is True
    assert s.sim["train"]["design"]["n_carriages"] == 5
    assert s.sim["cellular"]["fitted_networks"] == ["ee", "vodafone", "three"]


def test_lner_designs_fit_four_networks_and_count_starlink_tiles():
    """The LNER sets: four active antennas (one per UK network) under Fleet Connect; one Starlink tile on 5 cars, two on 9."""
    import numpy as np
    import pandas as pd

    from tcs.model.satcom import satcom_observations
    from tcs.sources.train_studio import load_design

    root = Path(__file__).resolve().parents[1] / "examples"
    caps = {}
    for n, tiles in ((5, 1), (9, 2)):
        s = load_settings(offline=True)
        d = load_design(root / f"lner-{n}car.train.json", s)
        assert len(d.carriages) == n and d.aps_connected == n and d.satcom_units == tiles and not d.warnings
        assert d.fitted_networks == ["ee", "vodafone", "three", "o2"] and d.policy == "PACKET_BONDING"
        apply_to_settings(s, d)
        assert s.starlink["satcom"]["providers"][0]["terminals"] == tiles
        m = 4
        samples = pd.DataFrame({"sample_id": np.arange(m), "sky_visibility": np.ones(m), "in_tunnel": np.zeros(m, bool), "canopy_probability": np.zeros(m),
                                "cutting_depth_m": np.zeros(m), "urban_density": np.zeros(m), "distance_m": np.arange(m) * 50.0})
        caps[n] = satcom_observations(s, samples)["capacity_mbps"].to_numpy()
    assert np.allclose(caps[9], 2 * caps[5])                    # two tiles bond to twice one tile's capacity under the same sky


def test_no_satcom_no_units(tmp_path):
    p = json.loads(json.dumps(PROJECT))
    for c in p["project"]["cars"]:
        c["edge"] = 0; c["custom"] = {}; c["fleet"] = False
    f = tmp_path / "bare.train.json"
    f.write_text(json.dumps(p), encoding="utf-8")
    s = load_settings(offline=True)
    d = derive(load_project(f), s.sim["train"])
    assert d.cellular_units == 0 and d.vehicle_profile == "PASSENGER_HANDSET_INSIDE_CARRIAGE" and d.policy == "FAILOVER"
    apply_to_settings(s, d)
    assert s.sim["satcom_enabled"] is False and s.sim["cellular"]["fitted_networks"] is None   # passengers' own phones


def test_rejects_foreign_file(tmp_path):
    f = tmp_path / "x.json"
    f.write_text(json.dumps({"app": "other", "version": 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_project(f)


@pytest.mark.parametrize("value", [-1, 0.5, None, "3", True, 1001, float("inf")])
def test_rejects_invalid_equipment_count(value):
    with pytest.raises(ValueError, match="integer"):
        derive({"cars": [{"aps": value}]}, {})


@pytest.mark.parametrize("project", [[], {}, {"cars": {}}, {"cars": []}, {"cars": [{"custom": []}]},
                                     {"cars": [{}], "types": [{"id": "x", "name": "X"}, {"id": "x", "name": "X"}]}])
def test_rejects_invalid_structure(project):
    with pytest.raises(ValueError):
        derive(project, {})


def test_calibration_mapping_does_not_apply_bias_twice():
    import numpy as np
    import pandas as pd

    from tcs.model.calibration import fit

    q = np.linspace(0.2, 0.7, 100)
    obs = pd.DataFrame({"sample_id": range(100), "provider_id": "ee", "quality_score": q, "distance_m": np.arange(100) * 50})
    meas = pd.DataFrame({"sample_id": range(100), "provider_id": "ee", "rsrp_dbm": -120 + 46 * (q + 0.1), "kind": "point"})
    cal = fit(obs, meas)
    mapping = cal["rsrp_map"]["ee"]
    assert mapping["slope"] == pytest.approx(46)
    assert mapping["intercept"] == pytest.approx(-120)
    assert cal["section_km"] == 10


@pytest.mark.parametrize("cal", [[], {"bias": []}, {"bias": {"ee": float("inf")}},
                                 {"section_km": 0}, {"rsrp_map": {"ee": {"slope": -1, "intercept": -120}}}])
def test_rejects_malformed_calibration(tmp_path, cal):
    from tcs.model.calibration import load

    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(cal), encoding="utf-8")
    with pytest.raises(ValueError):
        load(path)
