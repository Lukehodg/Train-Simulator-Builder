"""Regression tests for cache integrity, validation and passenger service availability."""
import os
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import requests

from tcs.config import load_settings
from tcs.model.wifi import passenger_wifi
from tcs.sources import base
from tcs.validate.metrics import handover_position_error, report


def test_no_access_points_means_wifi_outage():
    settings = load_settings(offline=True)
    settings.sim["passenger_wifi"]["ap_capacity_mbps"] = 0
    samples = pd.DataFrame({"distance_m": [0.0]})
    wan = pd.DataFrame({"sample_id": [0], "bonded_capacity_mbps": [500.0],
                        "effective_latency_ms": [20.0], "packet_loss_pct": [0.0]})
    result = passenger_wifi(settings, samples, wan)
    assert result.loc[0, "service_class"] == "OUTAGE"
    assert result.loc[0, "wifi_service_score"] == 0


def test_missing_rsrp_is_not_a_measured_outage():
    obs = pd.DataFrame({"sample_id": [0, 1], "provider_id": ["ee", "ee"],
                        "distance_m": [0.0, 50.0], "signal_primary": [-80.0, -80.0],
                        "quality_score": [0.9, 0.9], "available": [True, True]})
    meas = pd.DataFrame({"sample_id": [0, 1], "provider_id": ["ee", "ee"],
                         "kind": ["point", "point"], "rsrp_dbm": [-80.0, np.nan]})
    result = report(obs, meas)
    assert result.loc[0, "n"] == 1
    assert result.loc[0, "class_acc"] == 1
    assert result.loc[0, "avail_err_pp"] == 0
    assert report(obs, meas.iloc[1:]).empty


def test_nearest_handover_matches_brute_force():
    rng = np.random.default_rng(8)
    predicted = rng.uniform(0, 10000, 200)
    observed = np.concatenate(([-100, 20000], rng.uniform(0, 10000, 100)))
    expected = np.abs(observed[:, None] - predicted).min(axis=1)
    result = handover_position_error(predicted, observed)
    assert result["median_m"] == pytest.approx(np.median(expected))
    assert result["p90_m"] == pytest.approx(np.percentile(expected, 90))


def test_interrupted_download_preserves_cache(tmp_path, monkeypatch):
    good = Mock(status_code=200)
    good.iter_content.return_value = iter([b"complete"])
    broken = Mock(status_code=200)

    def chunks():
        yield b"partial"
        raise requests.ConnectionError("interrupted")

    broken.iter_content.side_effect = lambda _: chunks()
    monkeypatch.setattr(base.requests, "get", Mock(side_effect=[good, broken]))
    monkeypatch.setattr(base.time, "sleep", lambda _: None)
    args = dict(raw_dir=tmp_path, name="test", retries=1)
    path = base.http_get("https://example.test/data", **args)
    os.utime(path, (0, 0))
    with pytest.raises(base.SourceUnavailable):
        base.http_get("https://example.test/data", ttl_days=0, **args)
    assert path.read_bytes() == b"complete"
    assert not list(tmp_path.rglob("*.part"))
    good.close.assert_called_once()
    broken.close.assert_called_once()
    os.utime(path, (0, 0))
    assert base.http_get("https://example.test/data", offline=True, **args) == path


def test_unknown_policy_rejected():
    from tcs.model.bonding import link_manager

    with pytest.raises(ValueError, match="unknown WAN policy"):
        link_manager(load_settings(offline=True), pd.DataFrame(), policy="TYPO")


def test_outage_distance_cannot_exceed_route_length():
    from tcs.report import kpis

    samples = pd.DataFrame({"distance_m": [0.0, 500.0, 750.0], "sim_seconds": [0, 10, 15]})
    rc = pd.DataFrame({"service_class": ["OUTAGE"] * 3, "bonded_capacity_mbps": [100.0] * 3,
                       "per_user_mbps": [0.0] * 3, "effective_latency_ms": [20.0] * 3,
                       "wifi_service_score": [0.0] * 3, "confidence": [0.5] * 3})
    result = kpis(rc, samples, 500)
    assert result["outage_km"] == 0.8  # Rounded 750 m, including an AP outage with healthy WAN.
    assert result["outage_km"] == result["route_length_km"]


def test_heat_map_bands_follow_the_legend_edges():
    """The route heat maps band every sample exactly as their legends state (0 = Excellent ... 4 = Very poor)."""
    from tcs.report import HEAT_METRICS, band_shares, heat_bands, typical_bands

    rsrp = [-80.0, -80.1, -90.0, -100.0, -110.0, -110.1, np.nan]
    cap = [200.0, 199.9, 100.0, 50.0, 10.0, 9.9, 0.0]
    lat = [39.9, 40.0, 60.0, 100.0, 149.9, 150.0, np.nan]         # no latency = no link
    n = len(rsrp)
    samples = pd.DataFrame({"sample_id": range(n)})
    obs = pd.DataFrame({"sample_id": [*range(n), *range(n)], "provider_type": ["cellular"] * (2 * n),
                        "signal_primary": [*rsrp, *([-125.0] * n)]})   # a weaker second network never lowers the strongest
    rc = pd.DataFrame({"sample_id": range(n)[::-1], "bonded_capacity_mbps": cap[::-1], "effective_latency_ms": lat[::-1]})   # aligned by id, not row order
    bands = heat_bands(samples, rc, obs)
    assert bands["signal"].tolist() == [0, 1, 1, 2, 3, 4, 4]
    assert bands["throughput"].tolist() == [0, 1, 1, 2, 3, 4, 4]
    assert bands["latency"].tolist() == [0, 1, 2, 3, 3, 4, 4]
    assert all(len(m["ranges"]) == 5 for m in HEAT_METRICS.values())
    assert sum(band_shares(bands["signal"])) == pytest.approx(100)

    # page-scale runs take the band at least half of them reach: one bad sample does not colour a run, half of them do
    edges, typical = typical_bands(np.array([0, 0, 0, 4, 0, 0, 4, 4]), 2)
    assert edges.tolist() == [0, 4, 8] and typical.tolist() == [0, 4]


def test_outage_stretches_are_found_longest_first_with_their_place():
    from tcs.report import outage_stretches

    cls = ["GOOD", "OUTAGE", "OUTAGE", "GOOD", "OUTAGE", "OUTAGE", "OUTAGE", "GOOD"]
    samples = pd.DataFrame({"sample_id": range(8), "distance_m": [i * 50.0 for i in range(8)],
                            "in_tunnel": [False, False, False, False, True, True, True, False],
                            "tunnel_name": [None, None, None, None, "Stoke Tunnel", "Stoke Tunnel", "Stoke Tunnel", None]})
    rc = pd.DataFrame({"sample_id": range(8), "service_class": cls})
    stations = pd.DataFrame({"name": ["Alpha", "Bravo", "Charlie"], "distance_m": [0.0, 180.0, 350.0]})
    out = outage_stretches(samples, rc, stations)
    assert out["length_km"].tolist() == [0.15, 0.1]           # three points then two, 50 m each
    first = out.iloc[0]
    assert (first["from"], first["to"], first["tunnel"]) == ("Bravo", "Charlie", "Stoke Tunnel")
    assert out.iloc[1]["tunnel"] == ""
    assert outage_stretches(samples, rc.assign(service_class="GOOD"), stations).empty


def test_packaging_missing_build_preserves_existing_output(tmp_path, monkeypatch):
    from tcs import package

    monkeypatch.setattr(package, "ROOT", tmp_path)
    marker = tmp_path / "dist" / "keep.txt"
    marker.parent.mkdir()
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no index.html"):
        package.package(skip_build=True)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_osm_segments_tile_the_routed_line_across_platform_hops(tmp_path, monkeypatch):
    """A leg ending on one track and the next starting on a parallel one adds a lateral hop to the line. The segment
    table must include it, or every tunnel/cutting flag after the station is looked up at the wrong distance."""
    from shapely.geometry import Point

    from tcs.geo import Projector, local_crs
    from tcs.sources import osm_route

    lat0, off = 51.5, 0.0007                                    # second track about 78 m north of the first
    nodes = {1: (-0.10, lat0), 2: (-0.09, lat0), 3: (-0.08, lat0),                         # track 1: A -> B
             4: (-0.08, lat0 + off), 5: (-0.07, lat0 + off), 6: (-0.06, lat0 + off), 7: (-0.05, lat0 + off)}   # track 2: B -> C
    ways = [{"type": "way", "id": 10, "nodes": [1, 2, 3], "tags": {"railway": "rail"}},
            {"type": "way", "id": 11, "nodes": [4, 5], "tags": {"railway": "rail"}},
            {"type": "way", "id": 12, "nodes": [5, 6], "tags": {"railway": "rail", "tunnel": "yes", "tunnel:name": "Test Tunnel"}},
            {"type": "way", "id": 13, "nodes": [6, 7], "tags": {"railway": "rail"}}]
    stations = [("AAA", -0.10, lat0), ("BBB", -0.08, lat0 + off / 2), ("CCC", -0.05, lat0 + off)]   # B sits between both tracks

    def fake_overpass(query, raw_dir, name, url, offline, timeout):
        if name == "osm_stations":
            return {"elements": [{"type": "node", "lat": la, "lon": lo, "tags": {"ref:crs": c, "name": c}} for c, lo, la in stations]}
        return {"elements": [{"type": "node", "id": k, "lon": lo, "lat": la} for k, (lo, la) in nodes.items()] + ways}

    monkeypatch.setattr(osm_route, "_overpass", fake_overpass)
    geom = osm_route.fetch_route(["AAA", "BBB", "CCC"], "GB", tmp_path)
    proj = Projector(local_crs(-0.075, lat0, "GB"))
    line = proj.line_to_xy(geom.line)
    seg = geom.segments
    x0, y0 = proj.to_xy(seg["lon0"].values, seg["lat0"].values)
    x1, y1 = proj.to_xy(seg["lon1"].values, seg["lat1"].values)
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(x1 - x0, y1 - y0))])
    assert cum[-1] == pytest.approx(line.length, abs=0.5)      # the segments tile the whole line, hop included
    k = int(np.flatnonzero(seg["tunnel"].values)[0])
    portal = line.project(Point(*proj.to_xy(*nodes[5])))
    assert cum[k] == pytest.approx(portal, abs=0.5)            # the tunnel starts where it really is, not ~78 m early


def test_ofcom_quota_refusal_stops_the_source_instead_of_filling_stand_ins(tmp_path, monkeypatch):
    """A 403/429 mid-route is a quota or key problem, not 'no data for this postcode': the source must stop so the
    route falls back to a labelled source, rather than carrying on and labelling neutral stand-ins as Ofcom."""
    from tcs.sources import ofcom_coverage

    settings = load_settings(offline=True)
    settings.offline = False
    monkeypatch.setenv("OFCOM_API_KEY", "test-key")
    calls = []

    def refused(url, **kw):
        calls.append(url)
        raise base.SourceUnavailable("ofcom_api: HTTP 403 for https://example.invalid")
    monkeypatch.setattr(ofcom_coverage, "http_get", refused)
    postcodes = pd.DataFrame({"postcode": ["AB1 2CD", "AB1 2CE", "AB1 2CF"]})
    with pytest.raises(base.SourceUnavailable, match="refused"):
        ofcom_coverage.fetch_ofcom_api(postcodes, settings, tmp_path)
    assert len(calls) == 1                                      # no point spending the rest of the route on refusals

    def not_found(url, **kw):
        raise base.SourceUnavailable("ofcom_api: HTTP 404 for https://example.invalid")
    monkeypatch.setattr(ofcom_coverage, "http_get", not_found)
    with pytest.raises(base.SourceUnavailable, match="no parsable"):   # a genuine gap is still skipped, postcode by postcode
        ofcom_coverage.fetch_ofcom_api(postcodes, settings, tmp_path)


def test_coverage_counts_as_live_only_when_most_of_it_is_ofcom():
    from tcs.report import _coverage_row, live_coverage_share

    full = {"coverage_sources": ["no_coverage_record", "ofcom_predicted"], "coverage_share": {"ofcom_predicted": 0.97, "no_coverage_record": 0.03}}
    partial = {"coverage_sources": ["no_coverage_record", "ofcom_predicted"], "coverage_share": {"ofcom_predicted": 0.116, "no_coverage_record": 0.884}}
    assert _coverage_row(full)[2] == "live"
    assert _coverage_row(partial)[2] == "partly stand-in" and "12 %" in _coverage_row(partial)[1]
    assert _coverage_row({"coverage_sources": ["synthetic_prior"], "coverage_share": {"synthetic_prior": 1.0}})[2] == "synthetic stand-in"
    assert live_coverage_share({"coverage_sources": ["ofcom_predicted"]}) == 1.0   # bundles built before the share was recorded


def test_in_tunnel_coverage_is_listed_per_route_not_shared():
    """The ECML's King's Cross and Edinburgh km ranges used to apply to every route, giving other routes' first
    2.5 km of tunnels phantom in-tunnel coverage."""
    from tcs.model.cellular import das_mask

    ecml = load_settings(offline=True)
    other = load_settings(offline=True, route_id="tfw_cdf_man")
    assert ecml.sim["cellular"]["tunnels"]["das_tunnels"]
    assert other.sim["cellular"]["tunnels"]["das_tunnels"] == []
    assert other.sim_defaults["cellular"]["tunnels"]["das_tunnels"] == []     # what the viewer re-simulates from
    names, km = np.array(["Tunnel"], dtype=object), np.array([1.0])
    assert das_mask(set(ecml.sim["cellular"]["tunnels"]["das_tunnels"]), names, km).all()
    assert not das_mask(set(other.sim["cellular"]["tunnels"]["das_tunnels"]), names, km).any()
