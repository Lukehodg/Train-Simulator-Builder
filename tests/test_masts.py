"""tcs/masts.py: placing masts from how their signal rises and falls along the track, on synthetic scanner readings."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")

from tcs import masts  # noqa: E402

RNG = np.random.default_rng(7)


def _pass(mast, xs, ys, trip, levels=(-60.0, -66.0), exponent=3.5, noise_db=4.0, network="ee", mnc=30, enb=4242):
    """One trip's readings every 10 m along a line, from two cells (sectors) of one mast."""
    out = []
    for k, level in enumerate(levels):
        d = np.sqrt((xs - mast[0]) ** 2 + (ys - mast[1]) ** 2 + masts.H_M ** 2)
        rsrp = level - 10 * exponent * np.log10(d) + RNG.normal(0, noise_db, len(xs))
        out.append(pd.DataFrame({"x": xs, "y": ys, "rsrp_dbm": rsrp, "network": network, "mnc": mnc, "enb": enb,
                                 "cell": enb * 256 + k, "trip": trip, "timestamp": pd.Timestamp("2026-04-01", tz="UTC")}))
    return pd.concat(out, ignore_index=True)


def test_a_mast_beside_a_straight_line_is_placed_along_it_and_at_its_distance_from_it():
    mast = (400_300.0, 300_000.0)                                     # 300 m east of a north-south line
    ys = np.arange(296_000.0, 304_000.0, 10.0)
    xs = np.full_like(ys, 400_000.0)
    r = pd.concat([_pass(mast, xs, ys, "t1"), _pass(mast, xs, ys, "t2")], ignore_index=True)
    m = masts.locate(r, workers=1, log=lambda *_: None)
    assert len(m) == 1
    from pyproj import Transformer

    x, y = Transformer.from_crs(4326, 27700, always_xy=True).transform(m.longitude[0], m.latitude[0])
    assert abs(y - mast[1]) < 80                                      # along the line
    assert abs(abs(x - 400_000.0) - 300) < 80                         # distance from it; which side cannot be told
    assert not m.side_clear[0]


def test_a_mast_seen_from_two_lines_is_fixed_on_both_axes():
    mast = (500_400.0, 200_300.0)
    t = np.arange(-4000.0, 4000.0, 10.0)
    r = pd.concat([_pass(mast, 500_000.0 + t * 0, 200_000.0 + t, "a"),          # north-south, 400 m west of the mast
                   _pass(mast, 500_000.0 + t, 199_500.0 + t * 0, "b")],         # east-west, 800 m south of it
                  ignore_index=True)
    m = masts.locate(r, workers=1, log=lambda *_: None)
    from pyproj import Transformer

    x, y = Transformer.from_crs(4326, 27700, always_xy=True).transform(m.longitude[0], m.latitude[0])
    assert np.hypot(x - mast[0], y - mast[1]) < 100
    assert m.side_clear[0]


def test_a_mast_needs_two_trips_and_the_file_round_trips(tmp_path):
    ys = np.arange(296_000.0, 304_000.0, 10.0)
    one = _pass((400_300.0, 300_000.0), np.full_like(ys, 400_000.0), ys, "only")
    assert masts.locate(one, workers=1, log=lambda *_: None).empty           # one trip: not placed
    two = pd.concat([one, one.assign(trip="again")], ignore_index=True)
    m = masts.locate(two, workers=1, log=lambda *_: None)
    masts.save(m, tmp_path / "masts.csv", source="synthetic", period=("2026-04-01", "2026-04-01"))
    back = masts.load(tmp_path / "masts.csv")
    assert list(back.columns) == masts.COLUMNS and len(back) == 1 and back.network[0] == "ee"
    assert masts.load(tmp_path / "missing.csv").empty


def test_a_placed_mast_stands_in_for_opencellids_cells_of_that_mast(tmp_path):
    from pyproj import CRS

    from tcs.config import load_settings
    from tcs.geo import Projector
    from tcs.pipeline.join_cells import with_fitted_masts

    s = load_settings(offline=True)
    proj = Projector(CRS.from_epsg(27700))
    ys = np.arange(300_000.0, 310_000.0, 50.0)
    samples = pd.DataFrame({"x": np.full_like(ys, 400_000.0), "y": ys})
    lon, lat = proj.to_lonlat(np.array([400_200.0, 400_150.0, 450_000.0]), np.array([305_000.0, 302_000.0, 305_000.0]))

    def ocid(cell_id, x, y):
        lo, la = proj.to_lonlat(x, y)
        return {"cell_key": f"234-30-1-{cell_id}", "provider_id": "ee", "radio": "LTE", "mcc": 234, "mnc": 30, "area_or_tac": 1,
                "cell_id": cell_id, "pci_or_unit": -1, "latitude": la, "longitude": lo, "x": x, "y": y, "elevation_m": np.nan,
                "samples": 5, "range_m": 1000, "source": "opencellid"}

    cells = pd.DataFrame([ocid(100 * 256 + 1, 401_200.0, 305_400.0), ocid(100 * 256 + 2, 400_900.0, 304_100.0),   # mast 100, ~1 km out
                          ocid(101 * 256 + 1, 399_500.0, 302_000.0)])                                            # mast 101: not placed
    pd.DataFrame({"network": ["ee", "ee", "ee"], "mnc": [30, 30, 30], "enb": [100, 102, 103], "latitude": lat, "longitude": lon,
                  "locations": [40, 30, 25], "trips": [3, 2, 2], "fit_rms_db": [5.0, 6.0, 6.0], "side_clear": [False, False, False]}
                 ).to_csv(tmp_path / "masts.csv", index=False)
    s.networks["fitted_masts"] = {"file": str(tmp_path / "masts.csv")}
    out = with_fitted_masts(s, proj, samples, cells, corridor_m=2500)
    assert set(out["cell_key"]) == {"234-30-1-25857", "234-30-enb-100", "234-30-enb-102"}   # 103 is 50 km away
    placed = out.set_index("cell_key").loc["234-30-enb-100"]
    assert placed["source"] == "fitted_masts" and abs(placed["x"] - 400_200) < 1 and abs(placed["y"] - 305_000) < 1
    s.networks["fitted_masts"] = {"file": None}
    assert with_fitted_masts(s, proj, samples, cells, corridor_m=2500).equals(cells)


def test_route_meta_names_opencellid_and_counts_the_placed_masts():
    from tcs.pipeline.export import _cell_source, _fitted_masts_meta

    cells = pd.DataFrame({"cell_key": ["a", "b", "234-30-enb-1"], "source": ["opencellid", "opencellid", "fitted_masts"]})
    obs = pd.DataFrame({"provider_type": ["cellular"] * 4 + ["satcom"], "serving_cell": ["a", "234-30-enb-1", "234-30-enb-1", None, None]})
    assert _cell_source(cells) == "opencellid"                                  # the database behind the route, as before
    assert _fitted_masts_meta(cells, obs) == {"fitted_masts": {"masts": 1, "serving_share": round(2 / 3, 4)}}
    plain = cells[cells.source == "opencellid"]
    assert _cell_source(plain) == "opencellid" and _fitted_masts_meta(plain, obs) == {}
    assert _cell_source(cells.iloc[0:0]) == "none"
