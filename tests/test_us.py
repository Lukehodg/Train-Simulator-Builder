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
    assert s.terrain["source"] == "copernicus_glo30" and s.terrain["lidar"]["source"] == "usgs_3dep"
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


def _gtfs(tmp_path):
    import zipfile

    files = {
        "routes.txt": "route_id,route_long_name\n1,Acela\n2,Northeast Regional\n",
        "calendar.txt": "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
                        "wk,1,1,1,1,1,0,0,20261001,20261031\nsat,0,0,0,0,0,1,0,20261001,20261031\n",
        "trips.txt": "route_id,service_id,trip_id,trip_short_name\n1,wk,a,2100\n1,wk,b,2150\n1,sat,c,2000\n1,wk,d,2101\n2,wk,e,170\n",
        "stops.txt": "stop_id,stop_code,stop_name\nWAS,WAS,Washington\nNYP,NYP,New York\nBOS,BOS,Boston\n",
        "stop_times.txt": "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
                          "a,05:00:00,05:00:00,WAS,1\na,07:50:00,07:55:00,NYP,2\na,10:40:00,10:40:00,BOS,3\n"
                          "b,23:00:00,23:00:00,WAS,1\nb,25:50:00,25:52:00,NYP,2\nb,28:30:00,28:30:00,BOS,3\n"
                          "c,06:00:00,06:00:00,WAS,1\nc,10:00:00,10:00:00,BOS,2\n"
                          "d,06:30:00,06:30:00,BOS,1\nd,11:00:00,11:00:00,WAS,2\n"
                          "e,06:10:00,06:10:00,WAS,1\ne,11:00:00,11:00:00,BOS,2\n",
        "feed_info.txt": "feed_publisher_name,feed_version\nAmtrak,20261007\n",
    }
    p = tmp_path / "gtfs.zip"
    with zipfile.ZipFile(p, "w") as z:
        for n, body in files.items():
            z.writestr(n, body)
    return p


def test_gtfs_picks_a_weekday_train_end_to_end(tmp_path):
    from tcs.sources.gtfs_feed import pick_trip

    feed = _gtfs(tmp_path)
    r = pick_trip(feed, "acela", "WAS", "BOS", depart_after="04:00")
    assert r["train"] == "2100" and r["departure"] == "05:00" and r["calls"] == {"WAS": "05:00", "NYP": "07:55", "BOS": "10:40"}
    assert r["feed_version"] == "20261007"
    late = pick_trip(feed, "Acela", "WAS", "BOS", depart_after="12:00")    # Saturday-only and northbound-only trips skipped
    assert late["train"] == "2150" and late["calls"]["BOS"] == "04:30"     # GTFS 28:30 is 04:30 the next morning
    with pytest.raises(SourceUnavailable):
        pick_trip(feed, "Acela", "WAS", "BOS", train="9999")


def test_gtfs_train_sets_calls_and_passing_points(tmp_path, monkeypatch):
    from tcs.pipeline.movement import movement
    from tcs.pipeline.sample_route import build_route
    from tcs.sources import gtfs_feed

    s = load_settings(offline=True, route_id=US_ROUTE)
    s.route["sample_spacing_m"] = 400
    b = build_route(s)
    r = {"train": "2252", "departure": "06:40", "feed_version": "x", "stops": ["WAS", "BAL", "NYP", "BOS"],
         "calls": {"WAS": "06:40", "BAL": "07:12", "NYP": "09:51", "BOS": "13:46"}}
    monkeypatch.setattr(gtfs_feed, "http_get", lambda *a, **k: _gtfs(tmp_path))
    monkeypatch.setattr(gtfs_feed, "pick_trip", lambda *a, **k: r)
    gtfs_feed.apply(s)
    assert s.route["timetable"]["departure"] == "06:40" and s.route["timetable"]["resolved"]["train"] == "2252"
    samples, st = movement(s, b.samples, b.stations)
    assert set(st.loc[st["stop"], "crs"]) == {"WAS", "BAL", "NYP", "BOS"}      # BWI, PHL, ... become passing points
    assert samples["sim_seconds"].max() >= (13 * 60 + 46 - (6 * 60 + 40)) * 60 - 1


def test_gtfs_unavailable_keeps_the_route_file_calls(monkeypatch):
    from tcs.sources import gtfs_feed

    s = load_settings(offline=True, route_id=US_ROUTE)
    before = dict(s.route["timetable"]["calls"])
    assert gtfs_feed.apply(s) is None                                        # offline with no cached feed
    assert s.route["timetable"]["calls"] == before and "resolved" not in s.route["timetable"]


def test_ookla_tiles_join_the_route_and_compare_with_the_model():
    from tcs.sources import ookla

    assert list(ookla.quadkey(np.array([-160.0406]), np.array([70.6336]))) == ["0022133222330201"]   # a tile from the real file
    lon = np.linspace(-74.0, -73.8, 60)                                                 # ~290 m apart: two samples a tile
    samples = pd.DataFrame({"sample_id": np.arange(60), "longitude": lon, "latitude": np.full(60, 40.75), "distance_m": np.arange(60) * 1000.0})
    qk = ookla.quadkey(lon, np.full(60, 40.75))
    speed = np.where(np.arange(60) < 30, 20_000, 200_000)                               # slow first half, fast second
    tiles = pd.DataFrame({"quadkey": qk[::2], "avg_d_kbps": speed[::2], "avg_u_kbps": 10_000, "avg_lat_ms": 30, "tests": 4})
    o = ookla.along_route(samples, tiles)
    assert o["ookla_down_mbps"].notna().all() and o.loc[0, "ookla_down_mbps"] == pytest.approx(20.0)
    obs = pd.DataFrame({"sample_id": np.repeat(np.arange(60), 2), "provider_type": "cellular", "available": True,
                        "capacity_mbps": np.repeat(np.where(np.arange(60) < 30, 15.0, 90.0), 2)})
    c = ookla.check(samples, obs, o, "2026-Q2")
    assert c["section_rank_correlation"] > 0.8 and c["slowest_fifth_overlap"] == 1.0 and c["samples_with_tests"] == 1.0


def test_fcc_hexagon_csv_files_flag_the_samples_inside(tmp_path):
    import zipfile

    import h3

    from tcs.sources.fcc_bdc import _clip, _file_type_rank, _hits

    lat, lon = np.array([40.75, 40.75, 41.50]), np.array([-73.99, -73.80, -72.00])
    covered = [h3.latlng_to_cell(40.75, -73.99, 8), h3.latlng_to_cell(35.0, -100.0, 8)]   # one on the route, one far off
    z = tmp_path / "hex.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("bdc_36_130077_4GLTE_mobile_broadband_h3.csv", "h3_res8_id,env\n" + "\n".join(f"{c},1" for c in covered))
    clip = _clip(z, (-74.5, 40.0, -71.0, 42.0))
    assert list(clip["h3"]) == [covered[0]]                                      # the far cell is cut away
    assert list(_hits(clip, lon, lat, None)) == [0]
    row = {"subcategory": "Hexagon Coverage", "file_type": "csv", "file_name": "x.csv"}
    assert _file_type_rank(row) < 9 and _file_type_rank({"subcategory": "Raw Coverage", "file_type": "gpkg"}) < _file_type_rank(row)


def test_fcc_files_split_by_their_environment_column(tmp_path):
    import zipfile

    import geopandas as gpd
    from shapely.geometry import box

    from tcs.sources.fcc_bdc import _clip, _file_type_rank, _hits, classify

    g = gpd.GeoDataFrame({"environment": [0, 1]}, geometry=[box(-74.1, 40.7, -73.9, 40.8), box(-74.0, 40.74, -73.98, 40.76)], crs=4326)
    g.to_file(tmp_path / "cov.gpkg", driver="GPKG")
    z = tmp_path / "raw.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.write(tmp_path / "cov.gpkg", "bdc_36_130077_4GLTE_mobile_broadband_D25.gpkg")
    clip = _clip(z, (-75.0, 40.0, -73.0, 41.0))
    lon, lat = np.array([-73.99, -73.95]), np.array([40.75, 40.75])
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs=4326)
    assert list(_hits(clip[clip["env"] == "invehicle"], lon, lat, pts)) == [0]
    assert sorted(_hits(clip[clip["env"] == "stationary"], lon, lat, pts)) == [0, 1]
    raw = {"subcategory": "Raw Coverage", "file_type": "gis", "file_name": "bdc_11_130077_4GLTE_mobile_broadband_D25_29sep2026"}
    hexa = {"subcategory": "Hexagon Coverage", "file_type": "gis", "file_name": "bdc_11_130077_4GLTE_mobile_broadband_h3_D25_29sep2026"}
    assert _file_type_rank(raw) < _file_type_rank(hexa) < 9
    assert classify({"technology_code": "500", "file_name": "bdc_11_130077_5GNR_35_3_mobile_broadband_D25"})[0] is None
    assert classify({"technology_code": "500", "file_name": "bdc_11_130077_5GNR_7_1_mobile_broadband_D25"})[0] == "5G"


def test_ookla_capacity_scale_follows_local_phone_speeds():
    from tcs.sources import ookla

    n = 400
    samples = pd.DataFrame({"sample_id": np.arange(n), "distance_m": np.arange(n) * 50.0, "in_tunnel": np.arange(n) >= 380})
    down = np.select([np.arange(n) < 130, np.arange(n) < 260], [400.0, 200.0], 100.0)   # city, suburbs, rural
    o = pd.DataFrame({"sample_id": np.arange(n), "ookla_down_mbps": down, "ookla_tests": np.where(np.arange(n) % 50 < 45, 30, 2)})
    sc = ookla.capacity_scale(samples, o)
    assert sc[50] > 1.2 and sc[300] < 0.85 and sc.min() >= 0.6 and sc.max() <= 1.6
    assert np.isfinite(sc).all()
    assert np.allclose(ookla.capacity_scale(samples, o.assign(ookla_tests=0)), 1.0)   # too few tests anywhere: no change


def test_capacity_scale_multiplies_cellular_capacity():
    from tcs.model.simulate import simulate
    from tcs.pipeline.join_cells import candidate_cells, corridor_cells, serving_cells
    from tcs.pipeline.join_coverage import coverage_prior
    from tcs.pipeline.movement import movement
    from tcs.pipeline.obstruction import enrich_terrain
    from tcs.pipeline.sample_route import build_route

    s = load_settings(offline=True, route_id=US_ROUTE)
    s.route["sample_spacing_m"] = 1000
    b = build_route(s)
    b.samples = enrich_terrain(s, b)
    cells = corridor_cells(s, b, b.samples)
    prior, b.samples = coverage_prior(s, b, cells=cells)
    serving = serving_cells(s, b.samples, candidate_cells(s, b.samples, cells))
    b.samples, _ = movement(s, b.samples, b.stations)
    base, _ = simulate(s, b.samples, prior, serving)
    b.samples["capacity_scale"] = np.float32(1.5)
    scaled, _ = simulate(s, b.samples, prior, serving)
    c0 = base[base["provider_type"] == "cellular"]["capacity_mbps"].to_numpy()
    c1 = scaled[scaled["provider_type"] == "cellular"]["capacity_mbps"].to_numpy()
    assert c0.max() > 0 and np.allclose(c1, 1.5 * c0, rtol=1e-4)


def test_point_cloud_surface_sits_on_the_3dep_ground():
    """Heights above the cloud's own ground are hung on the 3DEP terrain, whatever the cloud's vertical datum."""
    from pyproj import Transformer
    from shapely.geometry import box
    from shapely.strtree import STRtree

    from tcs.sources import usgs_ept
    from tcs.sources.lidar import Grid

    epsg = 32618
    g = Grid.blank((500000.0, 4400000.0, 500040.0, 4400040.0))            # 20 x 20 cells of 2 m
    dtm = np.full(g.arr.shape, 10.0, dtype=np.float32)
    to_wm = Transformer.from_crs(epsg, 3857, always_xy=True)
    xs, ys = np.meshgrid(np.arange(500001.0, 500040.0, 2.0), np.arange(4400001.0, 4400040.0, 2.0))
    wx, wy = to_wm.transform(xs.ravel(), ys.ravel())
    ground = np.c_[wx, wy, np.full(wx.size, 40.0), np.full(wx.size, 2)]         # the cloud's ground is 30 m off (datum)
    tx, ty = to_wm.transform([500011.0], [4400031.0])
    tree = np.array([[tx[0], ty[0], 55.0, 5], [tx[0], ty[0], 140.0, 18]])           # a 15 m tree, and high noise

    import io

    import laspy

    pts = np.vstack([ground, tree])
    hdr = laspy.LasHeader(point_format=6, version="1.4")
    hdr.offsets, hdr.scales = pts[:, :3].min(axis=0), np.array([0.01, 0.01, 0.01])
    las = laspy.LasData(hdr)
    las.x, las.y, las.z, las.classification = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3].astype(np.uint8)
    buf = io.BytesIO()
    las.write(buf, do_compress=True)                                                 # a real LAZ tile, served from memory

    class Http:
        def get(self, url, timeout=None):
            assert url.endswith("/ept-data/0-0-0-0.laz")
            return type("R", (), {"content": buf.getvalue(), "raise_for_status": lambda self: None})()

    pr = object.__new__(usgs_ept._Project)
    pr.name, pr.base, pr.http, pr.width, pr.span = "fake_2020", "https://x/fake_2020", Http(), 1e6, 256
    pr.x0, pr.y0 = float(wx.min()) - 5e5, float(wy.min()) - 5e5                     # far corner: float32 offsets must hold
    pr.nodes = lambda bbox, depth: ["0-0-0-0"]

    e = object.__new__(usgs_ept.Ept)
    lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(500020.0, 4400020.0)
    e.meta, e.tree = [{"name": "fake_2020", "url": "u"}], STRtree([box(lon - 1, lat - 1, lon + 1, lat + 1)])
    e.to_ll, e.to_wm = Transformer.from_crs(epsg, 4326, always_xy=True), to_wm
    e.from_wm = Transformer.from_crs(3857, epsg, always_xy=True)
    e.projects, e.memo, e.memo_bytes = {"u": pr}, usgs_ept.OrderedDict(), 0
    import threading
    e.lock = threading.Lock()
    top, name = e.heights(g, dtm)
    assert name == "fake_2020"
    r, c = int((g.y1 - 4400031.0) // 2), int((500011.0 - g.x0) // 2)
    assert top[r, c] == pytest.approx(25.0, abs=0.01)                             # 10 m ground + 15 m tree; noise dropped
    assert np.nanmedian(top) == pytest.approx(10.0, abs=0.01)
    assert len(e.memo) == 1 and e.memo_bytes == sum(x.nbytes for x in e.memo[("fake_2020", "0-0-0-0")])


def test_point_cloud_gaps_take_their_neighbours():
    from tcs.sources.usgs_ept import _fill_gaps

    a = np.full((5, 5), np.nan, dtype=np.float32)
    a[1:4, 1:4] = 20.0
    a[2, 2] = np.nan                                  # a hole in a crown
    out = _fill_gaps(a)
    assert out[2, 2] == pytest.approx(20.0) and out[0, 0] == pytest.approx(20.0)
    assert np.isnan(_fill_gaps(np.full((3, 3), np.nan, dtype=np.float32))).all()


def test_gtfs_shape_guides_the_track_search(tmp_path):
    import zipfile

    from tcs.sources.gtfs_feed import trip_shape
    from tcs.sources.osm_route import _guide_polys

    feed = _gtfs(tmp_path)
    with zipfile.ZipFile(feed, "a") as z:              # trip a has a 3-point shape, trip b a 2-point one; train 2150 is b
        z.writestr("shapes.txt", "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence\n"
                                 "s1,38.90,-77.00,1\ns1,40.75,-74.00,2\ns1,42.35,-71.06,3\ns2,38.90,-77.00,1\ns2,42.35,-71.06,2\n")
    with zipfile.ZipFile(feed) as z:
        files = {n: z.read(n).decode() for n in z.namelist()}
    files["trips.txt"] = "route_id,service_id,trip_id,trip_short_name,shape_id\n1,wk,a,2100,s1\n1,wk,b,2150,s2\n1,sat,c,2000,\n1,wk,d,2101,\n2,wk,e,170,\n"
    with zipfile.ZipFile(feed, "w") as z:
        for n, body in files.items():
            z.writestr(n, body)
    assert trip_shape(feed, "Acela", "WAS", "BOS").shape == (3, 2)            # most detailed shape by default
    assert trip_shape(feed, "Acela", "WAS", "BOS", train="2150").shape == (2, 2)
    assert trip_shape(feed, "Acela", "BOS", "WAS").shape == (3, 2)            # trip d has no shape: borrow the service's best
    with pytest.raises(SourceUnavailable):
        trip_shape(feed, "Northeast Regional", "WAS", "BOS")                   # no trip of the service has one
    guide = trip_shape(feed, "Acela", "WAS", "BOS")
    proj = Projector(local_crs(-74.0, 40.6, "US"))
    polys = _guide_polys(guide, proj, 8000)
    assert len(polys) == 3                                                     # ~630 km: 200 + 200 + 230 (a short tail joins)
    from shapely.geometry import Point, Polygon

    last = [float(v) for v in polys[-1].split()]
    assert Polygon(list(zip(last[1::2], last[::2]))).contains(Point(-71.06, 42.35))   # the far end is searched too
    assert all(len(p.split()) % 2 == 0 for p in polys)                         # "lat lon lat lon ..." for Overpass


def test_multi_day_timetable_counts_each_midnight():
    from tcs.sources.timetable import schedule_seconds

    calls = pd.DataFrame({"crs": ["LAX", "ELP", "SND", "DRT", "LRK", "CHI"], "time": ["01:00", "17:45", "23:46", "02:12", "00:44", "14:49"]})
    s = schedule_seconds(calls, "01:00")
    assert s["DRT"] == pytest.approx((25 * 60 + 12) * 60) and s["LRK"] == pytest.approx((47 * 60 + 44) * 60)
    assert s["CHI"] == pytest.approx((61 * 60 + 49) * 60)
    assert list(s.values()) == sorted(s.values())


def test_generated_amtrak_routes_are_complete():
    """Every generated route file: positions for every station, a pinned GTFS train, its calls, and FCC states."""
    import yaml

    from tcs.config import list_routes

    gen = [r for r in list_routes() if r["country"] == "US" and r["id"] != "nec_was_bos"]
    assert len(gen) >= 40
    for r in gen:
        route = yaml.safe_load(open(r["file"], encoding="utf-8"))["route"]
        codes = [s["crs"] for s in route["stations"]]
        assert all("lat" in s and "lon" in s for s in route["stations"]), r["id"]
        assert route["timetable"]["gtfs"]["train"] and route["geometry"]["guide"] == "gtfs_shape", r["id"]
        assert set(route["timetable"]["calls"]) == set(codes) and route["fcc_states"], r["id"]
        assert route["origin_crs"] == codes[0] and route["destination_crs"] == codes[-1], r["id"]
        assert 50 <= route.get("sample_spacing_m", 50) <= 300, r["id"]


def test_guide_is_stitched_through_every_station():
    from shapely.geometry import LineString, Point

    from tcs.sources.osm_route import _stitch_guide

    proj = Projector(local_crs(-97.0, 34.0, "US"))
    # a shape for Oklahoma City -> Fort Worth only, with a bend at Norman; the train now runs on to Dallas
    shape = np.array([[-97.51, 35.47], [-97.40, 35.22], [-97.33, 32.75]])
    st = pd.DataFrame({"crs": ["OKC", "NOR", "FTW", "DAL"], "lon": [-97.51, -97.44, -97.33, -96.81], "lat": [35.47, 35.22, 32.75, 32.78]})
    g = _stitch_guide(shape, st, proj)
    line = LineString(np.column_stack(proj.to_xy(g[:, 0], g[:, 1])))
    for _, s in st.iterrows():
        assert line.distance(Point(*proj.to_xy(s.lon, s.lat))) < 1.0           # passes through every station, Dallas too
    assert any(np.allclose(p, [-97.40, 35.22]) for p in g)                     # and follows the shape between them
    back = _stitch_guide(shape[::-1].copy(), st, proj)                         # a shape of the return train is turned round
    assert np.allclose(back, g, atol=1e-6)
    # Lakeland -> Tampa -> Lakeland: places are found in order, so the second Lakeland call follows the way back
    there_and_back = np.array([[-81.95, 28.05], [-82.45, 27.95], [-81.95, 28.06], [-81.70, 28.20]])
    st2 = pd.DataFrame({"crs": ["LKL", "TPA", "LAK", "WPK"], "lon": [-81.95, -82.45, -81.95, -81.70], "lat": [28.05, 27.95, 28.06, 28.20]})
    g2 = _stitch_guide(there_and_back, st2, proj)
    assert len(g2) >= 7 and np.allclose(g2[-1], [-81.70, 28.20])


def test_us_bundles_are_reused_until_their_inputs_change(tmp_path):
    import datetime as dt

    from tcs import bundle_store

    cfg = tmp_path / "config"
    (cfg / "countries" / "US").mkdir(parents=True)
    (cfg / "countries" / "US" / "profile.yaml").write_text("a: 1\n")
    (cfg / "simulation.yaml").write_text("model_version: '1'\n")
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("route: {id: a}\n")
    b.write_text("route: {id: b}\n")
    files = {"a": a, "b": b}
    now = dt.datetime(2026, 10, 8, tzinfo=dt.timezone.utc)
    name_a = bundle_store.asset_name("a", bundle_store.fingerprint(a, cfg))
    kept = [{"name": name_a, "createdAt": "2026-09-01T00:00:00Z"}, {"name": "b--000000000000.tar.gz", "createdAt": "2026-10-01T00:00:00Z"}]
    assert bundle_store.plan(["a", "b"], files, kept, now, config_dir=cfg) == (["b"], [name_a])     # b's file changed since
    assert bundle_store.plan(["a", "b"], files, kept, now, force=True, config_dir=cfg) == (["a", "b"], [])
    assert bundle_store.plan(["a"], files, kept, now + dt.timedelta(days=200), config_dir=cfg) == (["a"], [])   # too old
    (cfg / "countries" / "US" / "profile.yaml").write_text("a: 2\n")                                       # US settings changed
    assert bundle_store.plan(["a"], files, kept, now, config_dir=cfg) == (["a"], [])
