"""LiDAR along the track: grid references, mosaicking, and what each route sample takes from the terrain and surface
models (cutting walls, the skyline, bridges over the line, viaducts, gaps in the surveys), on synthetic ground."""
import numpy as np
import pytest

from tcs.sources.lidar import RES, Grid, corridor_features, grid_ref, tile_bounds

TRACK_X = 1000.0             # a straight line running north along x = 1000, rail at 100 m


def dtm(x, y):
    z = np.full(np.shape(x), 100.0)
    lat = np.abs(x - TRACK_X)
    cut = (y > 0) & (y < 1000)                                    # a 10 m cutting: walls rise from 8 m to 18 m off the line
    z = np.where(cut, 100.0 + np.clip(lat - 8.0, 0, 10), z)
    via = (y > 2000) & (y < 2600)                                 # a valley under a viaduct the map did not tag
    return np.where(via, 80.0, z)


def dsm(x, y):
    z = dtm(x, y)
    trees = (y > 0) & (y < 1000) & (x - TRACK_X > 12) & (x - TRACK_X < 20)   # a 20 m tree line on the east side
    z = np.where(trees, z + 20.0, z)
    bridge = (y >= 1500) & (y < 1514) & (np.abs(x - TRACK_X) < 30)          # a road bridge over the line, deck 8 m up
    z = np.where(bridge, 108.0, z)
    via = (y > 2000) & (y < 2600) & (np.abs(x - TRACK_X) < 4)               # the viaduct deck carrying the line
    return np.where(via, 100.0, z)


class FakeLidar:
    """Serves the analytic ground above, with no survey north of y = 3000."""

    def read(self, bbox, layer):
        g = Grid.blank(bbox)
        h, w = g.arr.shape
        xs = g.x0 + (np.arange(w) + 0.5) * RES
        ys = g.y1 - (np.arange(h) + 0.5) * RES
        X, Y = np.meshgrid(xs, ys)
        a = (dsm if layer == "dsm" else dtm)(X, Y).astype(np.float32)
        a[Y > 3000] = np.nan
        g.arr[:] = a
        return g, "lidar_test"


def _features(ys, on_bridge=None):
    ys = np.asarray(ys, dtype=float)
    n = len(ys)
    return corridor_features(np.full(n, TRACK_X), ys, np.zeros(n), np.zeros(n, bool) if on_bridge is None else on_bridge,
                             np.zeros(n, bool), FakeLidar(), corridor_m=60, segment=3, workers=1, retry_pause_s=0, log=lambda *a: None)


def test_grid_references_and_tile_bounds():
    assert grid_ref(325800, 673900) == "NT27" and grid_ref(325800, 673900, 2) == "NT2573" and grid_ref(530000, 185000) == "TQ38"
    assert tile_bounds("NT27SE_50CM_DTM_PHASE3.tif") == (325000, 670000, 330000, 675000)        # a quarter of the 10 km square
    assert tile_bounds("NO12_1M_DTM_PHASE1.tif") == (310000, 720000, 320000, 730000)
    assert tile_bounds("NG2104_50cm_DSM_ScotlandNationalLiDAR.tif") == (121000, 804000, 122000, 805000)
    assert tile_bounds("not_a_tile.tif") is None


def test_mosaic_fills_only_what_is_missing():
    g = Grid.blank((0, 0, 8, 8))
    g.place(np.full((2, 4), 5.0, dtype=np.float32), left=0, top=8)            # top half from the first source
    g.place(np.full((4, 4), 7.0, dtype=np.float32), left=0, top=8)            # the second only fills the rest
    assert g.arr[0, 0] == 5 and g.arr[3, 0] == 7 and g.missing() == 0
    assert g.sample(np.array([1.0, 1.0, 99.0]), np.array([7.0, 1.0, 1.0])).tolist()[:2] == [5.0, 7.0]
    assert np.isnan(g.sample(np.array([99.0]), np.array([1.0])))[0]


def test_cutting_walls_trees_and_open_track():
    f = _features([500.0, 1200.0])
    assert f["lidar_ok"].all()
    assert f["rail_level_m"] == pytest.approx([100, 100], abs=0.5)
    assert f["wall_left_m"][0] == pytest.approx(10, abs=0.5) and f["wall_right_m"][0] == pytest.approx(10, abs=0.5)
    assert f["wall_left_m"][1] < 0.5 and f["wall_right_m"][1] < 0.5          # open track beyond the cutting
    hz = f["horizon_near_deg"]
    east, west = hz[0, 4], hz[0, 12]                                         # 16 azimuths: 4 = 90 deg (east), 12 = 270 deg (west)
    assert east > 45                                                         # the tree line, 16 m above the antenna 12 m away
    assert 10 < west < 45                                                    # only the 10 m cutting wall on the west
    assert hz[1].max() < 5                                                   # nothing above the antenna on open track
    assert f["obstruction_share"][0] > f["obstruction_share"][1]


def test_bridges_over_the_line_and_viaducts_carrying_it():
    f = _features([1500.0, 2300.0])
    assert f["overhead_fraction"][0] == pytest.approx(7 / 25, abs=0.05)      # 14 m of deck over 48 m of track
    assert f["rail_level_m"][1] == pytest.approx(100, abs=0.5)               # the deck, not the valley 20 m below
    assert f["overhead_fraction"][1] == 0
    assert f["fall_left_m"][1] > 15                                          # and it reads as standing high above the ground


def test_track_the_map_tags_as_a_bridge_takes_the_deck():
    f = _features([2300.0], on_bridge=np.array([True]))
    assert f["rail_level_m"][0] == pytest.approx(100, abs=0.5)


def test_no_survey_leaves_the_sample_to_the_terrain_model():
    f = _features([3500.0, 3600.0])
    assert not f["lidar_ok"].any() and np.isnan(f["rail_level_m"]).all()


def test_england_service_zero_means_no_data():
    """Outside England the Environment Agency service answers with zeros rather than its no-data value."""
    import io

    import rasterio
    from rasterio.transform import from_origin

    from tcs.sources.lidar import EnglandWCS

    buf = io.BytesIO()
    with rasterio.MemoryFile() as mf:
        with mf.open(driver="GTiff", height=4, width=4, count=1, dtype="float32", crs="EPSG:27700", transform=from_origin(0, 8, 2, 2)) as ds:
            ds.write(np.zeros((1, 4, 4), dtype=np.float32))
        buf.write(mf.read())

    class Resp:
        status_code, content = 200, buf.getvalue()

    class Http:
        class session:
            @staticmethod
            def get(*a, **k):
                return Resp()

    g = Grid.blank((0, 0, 8, 8))
    EnglandWCS(Http()).fill(g, "dtm")
    assert g.missing() == 1.0


def test_a_failed_read_is_retried_and_reported_not_taken_as_no_survey():
    calls = {"n": 0}

    class Flaky(FakeLidar):
        def read(self, bbox, layer):
            calls["n"] += 1
            g, src = super().read(bbox, layer)
            if calls["n"] == 1:                                              # the first try fails; the retry pass gets it
                g.arr[:] = np.nan
                g.failed = True
                return g, None
            return g, src

    class Down(FakeLidar):
        def read(self, bbox, layer):
            g = Grid.blank(bbox)
            g.failed = True
            return g, None

    f = corridor_features(np.full(2, TRACK_X), np.array([500.0, 1200.0]), np.zeros(2), np.zeros(2, bool), np.zeros(2, bool), Flaky(),
                          corridor_m=60, segment=3, workers=1, retry_pause_s=0, log=lambda *a: None)
    assert f["lidar_ok"].all() and not f["lidar_failed"].any()
    f = corridor_features(np.full(2, TRACK_X), np.array([500.0, 1200.0]), np.zeros(2), np.zeros(2, bool), np.zeros(2, bool), Down(),
                          corridor_m=60, segment=3, workers=1, retry_pause_s=0, log=lambda *a: None)
    assert not f["lidar_ok"].any() and f["lidar_failed"].all()


def test_england_requests_stay_inside_the_survey_extent():
    from tcs.sources.lidar import EnglandWCS

    asked = []

    class Http:
        class session:
            @staticmethod
            def get(q, **k):
                asked.append(q)
                raise __import__("requests").ConnectionError("down")

    g = Grid.blank((300000, 900000, 300400, 900400))                         # the far north of Scotland: not asked at all
    EnglandWCS(Http()).fill(g, "dsm")
    assert not asked and not g.failed
    g = Grid.blank((400000, 657400, 400400, 657800))                         # straddles the top of the surface model's extent
    EnglandWCS(Http()).fill(g, "dsm")
    assert "subset=N(657400,657600)" in asked[0] and g.failed                # asked only for what it has; the error is reported


def test_reports_say_how_much_of_the_route_the_lidar_covers():
    import pandas as pd

    from tcs.pipeline.export import _lidar_meta
    from tcs.report import _terrain_row, lidar_text

    s = pd.DataFrame({"lidar": [True, True, True, False], "lidar_source": ["lidar_ea", "lidar_ea", "lidar_wales", ""]})
    m = {"terrain_source": "copernicus_glo30", **_lidar_meta(s)}
    assert m["lidar_share"] == 0.75 and m["lidar_sources"] == {"lidar_ea": 0.6667, "lidar_wales": 0.3333}
    assert lidar_text(m) == "open 2 m LiDAR (Environment Agency 67 %, Welsh Government 33 %; OGL) on 75 % of the route"
    assert _terrain_row(m)[1].startswith("Copernicus DEM GLO-30 (30 m); near the track, open 2 m LiDAR") and _terrain_row(m)[2] == "live"
    assert lidar_text({"lidar_share": 0.5, "lidar_sources": {"lidar_scotland": 1.0}}) == "open 2 m LiDAR (Scottish public sector; OGL) on 50 % of the route"
    assert lidar_text({"terrain_source": "copernicus_glo30"}) == "" and _lidar_meta(pd.DataFrame({"x": [1]})) == {}


def test_only_the_named_samples_are_read_and_a_second_slower_pass_retries_what_still_fails():
    reads = []

    class Counting(FakeLidar):
        def read(self, bbox, layer):
            reads.append(bbox)
            return super().read(bbox, layer)

    ys = np.arange(0.0, 900.0, 50.0)
    only = np.zeros(len(ys), bool)
    only[7] = True
    f = corridor_features(np.full(len(ys), TRACK_X), ys, np.zeros(len(ys)), np.zeros(len(ys), bool), np.zeros(len(ys), bool), Counting(),
                          corridor_m=60, segment=3, workers=1, retry_pause_s=0, only=only, log=lambda *a: None)
    assert f["lidar_ok"][6:9].all() and not f["lidar_ok"][:6].any() and not f["lidar_ok"][9:].any()   # just the stretch holding sample 7
    assert len(reads) == 2                                                    # one stretch: terrain and surface
    calls = {"n": 0}

    class TwiceDown(FakeLidar):
        def read(self, bbox, layer):
            calls["n"] += 1
            g, src = super().read(bbox, layer)
            if calls["n"] <= 2:                                              # the first try and the first retry fail
                g.arr[:] = np.nan
                g.failed = True
                return g, None
            return g, src

    f = corridor_features(np.full(2, TRACK_X), np.array([500.0, 1200.0]), np.zeros(2), np.zeros(2, bool), np.zeros(2, bool), TwiceDown(),
                          corridor_m=60, segment=3, workers=1, retry_pause_s=0, log=lambda *a: None)
    assert f["lidar_ok"].all() and not f["lidar_failed"].any()


def test_a_build_keeps_what_it_read_and_the_next_reads_only_what_failed(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import pandas as pd
    from pyproj import CRS

    import tcs.config as config
    import tcs.sources.lidar as lidar_mod
    from tcs.config import load_settings
    from tcs.geo import Projector
    from tcs.pipeline.obstruction import lidar_features

    monkeypatch.setattr(config, "RAW", tmp_path / "raw")
    state = {"build": 1, "reads": 0}

    class Service(FakeLidar):
        def __init__(self, *a, **k):
            pass

        def read(self, bbox, layer):
            state["reads"] += 1
            g, src = super().read(bbox, layer)
            if state["build"] == 1 and bbox[1] > 1000:                       # first build: the service fails north of y = 1000
                g.arr[:] = np.nan
                g.failed = True
                return g, None
            return g, src

    monkeypatch.setattr(lidar_mod, "Lidar", Service)
    s = load_settings(route_id="ecml_kgx_edb")
    s.offline = False
    s.terrain["lidar"] = {"enabled": True, "workers": 1}
    ys = np.arange(0.0, 2000.0, 50.0)
    samples = pd.DataFrame({"x": np.full(len(ys), TRACK_X), "y": ys, "bearing_deg": 0.0, "on_bridge": False, "canopy_probability": 0.0})
    bundle = SimpleNamespace(samples=samples, proj=Projector(CRS.from_epsg(27700)))
    monkeypatch.setattr(lidar_mod.time, "sleep", lambda s: None)
    first = lidar_features(s, bundle)
    assert first["lidar_ok"][ys < 900].all() and not first["lidar_ok"][ys >= 1200].any()       # stretches reaching below y = 1000 read
    state.update(build=2, reads=0)
    second = lidar_features(s, bundle)
    assert second["lidar_ok"].all()
    assert 0 < state["reads"] <= 2 * (int((ys > 900).sum()) // 6 + 2)          # only the failed stretches were read again
    state.update(build=3, reads=0)
    assert lidar_features(s, bundle)["lidar_ok"].all() and state["reads"] == 0   # complete: served from the cache
