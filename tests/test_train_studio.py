import json

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
    assert abs(d.units_capacity_factor - (1 + 0.85 + 0.85 ** 2)) < 1e-6
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
    assert s.sim["satcom_enabled"] is False and s.sim["cellular"]["units_capacity_factor"] == 1.0


def test_rejects_foreign_file(tmp_path):
    f = tmp_path / "x.json"
    f.write_text(json.dumps({"app": "other", "version": 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_project(f)
