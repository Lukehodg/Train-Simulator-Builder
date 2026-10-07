"""United States: the country profile (config/countries/US), multi-MCC networks, NTAD stations and route lines, the
OpenCellID site-distance prior, and the country-aware publish and report helpers. Hermetic: no network."""
import json

import numpy as np
import pandas as pd
import pytest

from tcs.config import load_settings
from tcs.geo import Projector, local_crs
from tcs.report import LIVE_COVERAGE_MIN, is_gb, live_coverage_share
from tcs.sources import cell_prior, ntad_amtrak
from tcs.sources.base import SourceUnavailable
from tcs.sources.opencellid import _mnc_map, plmns

US_ROUTE = "nec_was_bos"


def test_us_profile_replaces_gb_inputs():
    s = load_settings(offline=True, route_id=US_ROUTE)
    assert s.country == "US" and not s.is_gb
    assert [op["id"] for op in s.operators] == ["att", "verizon", "tmobile"]
    assert s.terrain["source"] == "copernicus_glo30" and s.terrain["lidar"]["enabled"] is False
    assert s.sim["cellular"]["national_calibration"] is None and s.sim["cellular"]["measured_corrections"] is None
    assert s.sim["presets"]["edge_rail_fleet_connect"]["fitted_networks"] == ["att", "verizon", "tmobile"]
    assert s.sim["presets"]["edge_rail_fleet_connect"]["policy"] == "PACKET_BONDING"      # the rest of the preset is kept
    assert s.networks["fitted_masts"]["file"] is None


def test_gb_routes_keep_the_gb_defaults():
    s = load_settings(offline=True)
    assert s.is_gb and s.country_profile == {}
    assert [op["id"] for op in s.operators] == ["ee", "o2", "vodafone", "three"]
    assert s.sim["cellular"]["national_calibration"] == "config/calibration.yaml"


def test_plmns_span_several_mccs():
    s = load_settings(offline=True, route_id=US_ROUTE)
    m = _mnc_map(s.operators)
    assert m[(310, 410)] == "att" and m[(313, 100)] == "att"          # FirstNet
    assert m[(311, 480)] == "verizon" and m[(310, 4)] == "verizon"
    assert m[(310, 260)] == "tmobile" and m[(312, 250)] == "tmobile"  # ex-Sprint
    assert len(m) == sum(len(op["plmn"]) for op in s.operators)        # no PLMN listed under two networks
    assert plmns({"mcc": 234, "mnc": [30, 31]}) == [(234, 30), (234, 31)]   # GB style unchanged


def _line_samples(n=200, spacing=50.0):
    x = np.arange(n) * spacing
    return pd.DataFrame({"sample_id": np.arange(n), "x": x, "y": np.zeros(n), "distance_m": x})


def test_site_prior_falls_with_distance_and_lifts_for_5g():
    samples = _line_samples()
    ops = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    cells = pd.DataFrame({"provider_id": ["a", "b"], "radio": ["LTE", "NR"], "x": [0.0, 0.0], "y": [500.0, 500.0]})
    p = cell_prior.site_prior(samples, cells, ops)
    a = p[p["provider_id"] == "a"].set_index("sample_id")["prior_score"]
    b = p[p["provider_id"] == "b"].set_index("sample_id")["prior_score"]
    assert a.is_monotonic_decreasing and a.iloc[0] > 0.7 and a.iloc[-1] < a.iloc[0] - 0.2
    assert (b.iloc[:20] > a.iloc[:20]).all()                           # the 5G site lifts its network nearby
    assert "c" not in set(p["provider_id"])                            # no site: no record, not a guess
    assert set(p["source"]) == {cell_prior.SOURCE}


def test_ntad_stations_keep_order_and_prefer_trains(monkeypatch, tmp_path):
    doc = {"features": [
        {"attributes": {"Code": "BOS", "StationName": "Boston, MA", "StnType": "BUS", "lat": 1.0, "lon": 1.0}},
        {"attributes": {"Code": "BOS", "StationName": "Boston, MA", "StnType": "TRAIN", "lat": 42.35, "lon": -71.06}},
        {"attributes": {"Code": "WAS", "StationName": "Washington, DC", "StnType": "TRAIN", "lat": 38.90, "lon": -77.01}},
    ]}
    monkeypatch.setattr(ntad_amtrak, "_query", lambda *a, **k: doc)
    st = ntad_amtrak.fetch_stations(["WAS", "BOS"], tmp_path)
    assert list(st["crs"]) == ["WAS", "BOS"] and st.loc[1, "lat"] == pytest.approx(42.35)
    with pytest.raises(SourceUnavailable):
        ntad_amtrak.fetch_stations(["WAS", "NYP"], tmp_path)
    with pytest.raises(SourceUnavailable):
        ntad_amtrak.fetch_stations(["WAS') OR (1=1"], tmp_path)


def test_ntad_route_joins_parts_between_stations(monkeypatch, tmp_path):
    # Two pieces of line with a 60 m gap between them, and a spur that must not be taken.
    a = [[-77.00 + 0.01 * i, 39.0] for i in range(11)]                  # -77.00 .. -76.90
    b = [[-76.8993 + 0.01 * i, 39.0] for i in range(11)]                # starts ~60 m east of a's end
    spur = [[-76.95, 39.0], [-76.95, 39.2]]
    monkeypatch.setattr(ntad_amtrak, "_query", lambda *a_, **k: {"features": [{"geometry": {"paths": [a, b, spur]}}]})
    st = pd.DataFrame({"crs": ["AAA", "BBB"], "name": ["A", "B"], "lat": [39.0005, 38.9995], "lon": [-76.999, -76.80]})
    proj = Projector(local_crs(-76.9, 39.0, "US"))
    g = ntad_amtrak.fetch_route(st, "Test", proj, tmp_path)
    assert g.source == "ntad" and not g.straight_legs
    length = proj.line_to_xy(g.line).length
    assert 16_000 < length < 17_500                                     # ~0.2 degrees of longitude at 39 N, no spur
    assert len(g.segments) and not g.segments["tunnel"].any()


def test_us_route_builds_offline_with_its_tunnels():
    from tcs.pipeline.sample_route import build_route

    s = load_settings(offline=True, route_id=US_ROUTE)
    s.route["sample_spacing_m"] = 200
    b = build_route(s)
    assert b.geometry_source == "synthetic" and not b.warnings
    names = set(b.samples.loc[b.samples["in_tunnel"], "tunnel_name"])
    assert {"North River Tunnels", "East River Tunnels", "New York Penn Station", "B&P Tunnel"} <= names
    nyp = b.stations.set_index("crs").loc["NYP", "distance_m"]
    tun = b.samples[b.samples["tunnel_name"] == "New York Penn Station"]["distance_m"]
    assert abs(tun.mean() - nyp) < 1500                                 # the station box sits at the station


def test_live_coverage_and_gate_by_country():
    gb = {"route": {"country": "GB"}, "coverage_share": {"ofcom_api": 0.95, "no_coverage_record": 0.05}}
    us = {"route": {"country": "US"}, "coverage_share": {"opencellid_sites": 0.97, "no_coverage_record": 0.03}}
    us_stand_in = {"route": {"country": "US"}, "coverage_share": {"synthetic_prior": 1.0}}
    assert live_coverage_share(gb) >= LIVE_COVERAGE_MIN and is_gb(gb)
    assert live_coverage_share(us) >= LIVE_COVERAGE_MIN and not is_gb(us)
    assert live_coverage_share(us_stand_in) == 0.0
    assert is_gb({"route": {}})                                         # bundles without a country are GB


def test_glossary_is_country_specific():
    from tcs.report_docx import GB_ONLY_TERMS, GLOSSARY, SITES_TERM

    terms = {g[0] for g in GLOSSARY}
    assert GB_ONLY_TERMS <= terms and SITES_TERM in terms


def test_index_and_catalogue_carry_the_country(tmp_path, monkeypatch):
    from tcs.config import list_routes

    countries = {r["id"]: r["country"] for r in list_routes()}
    assert countries[US_ROUTE] == "US" and countries["ecml_kgx_edb"] == "GB"
    import tcs.cli as cli

    data = tmp_path / "web" / "public" / "data" / US_ROUTE
    data.mkdir(parents=True)
    (data / "meta.json").write_text(json.dumps({"route": {"id": US_ROUTE, "name": "NEC", "country": "US"}, "stations": [{"name": "A"}, {"name": "B"}]}))
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    idx = json.loads(cli.write_index().read_text())
    assert idx["routes"][0]["country"] == "US"


def test_fcc_files_are_classified_by_technology_and_environment():
    from tcs.sources.fcc_bdc import classify

    assert classify({"technology_code": "400", "file_name": "bdc_36_130077_4GLTE_mobile_broadband_invehicle.zip"}) == ("4G", "invehicle")
    assert classify({"technology_code": "500", "file_name": "bdc_36_130077_5GNR_7_1_outdoor_stationary.zip"}) == ("5G", "stationary")
    assert classify({"technology_code": "500", "file_name": "bdc_36_130077_5GNR_35/3_in-vehicle.zip"})[0] is None   # 7/1 contains 35/3


def test_fcc_prior_scores_in_vehicle_highest(monkeypatch):
    from tcs.sources import fcc_bdc

    s = load_settings(offline=True, route_id=US_ROUTE)
    samples = pd.DataFrame({"sample_id": np.arange(4)})
    t, f = np.array([True, False, False, False]), np.array([False, True, False, False])
    flags = {"att": {"invehicle_5g": t, "invehicle_4g": t | f, "stationary_5g": t, "stationary_4g": t | f | np.array([0, 0, 1, 0], bool)}}
    monkeypatch.setattr(fcc_bdc, "coverage_flags", lambda *a, **k: flags)
    p = fcc_bdc.fcc_prior(s, samples, None)["prior_score"].to_numpy()
    assert p[0] > p[1] > p[2] > p[3] and set(fcc_bdc.fcc_prior(s, samples, None)["source"]) == {"fcc_bdc"}
    assert live_coverage_share({"coverage_share": {"fcc_bdc": 0.95}}) >= LIVE_COVERAGE_MIN
