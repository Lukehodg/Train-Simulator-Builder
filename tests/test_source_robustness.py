"""Downloads that fail part-way, busy services and odd inputs: a build keeps what it has instead of falling back to
stand-ins or asking again for what it already holds."""
import os
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import requests

from tcs.sources import base


def test_overpass_answers_are_kept_whichever_mirror_gave_them(tmp_path, monkeypatch):
    """The cache was keyed by mirror: an answer from the second mirror was never found offline, and online the busy
    first mirror was asked again before it."""
    from tcs.sources import osm_route

    asked = []

    def post(url, **kw):
        asked.append(url)
        if url == osm_route.OVERPASS_ENDPOINTS[0]:
            return Mock(status_code=504, close=Mock(), raise_for_status=Mock(side_effect=requests.HTTPError("504 Server Error: Gateway Timeout")))
        r = Mock(status_code=200, close=Mock())
        r.iter_content.return_value = iter([b'{"elements": [1]}'])
        return r

    monkeypatch.setattr(base.requests, "post", post)
    monkeypatch.setattr(base.time, "sleep", lambda _: None)
    monkeypatch.setattr(osm_route.time, "sleep", lambda _: None)
    assert osm_route._overpass("q", tmp_path, "osm_rail", None, False, 10) == {"elements": [1]}
    asked.clear()
    assert osm_route._overpass("q", tmp_path, "osm_rail", None, False, 10) == {"elements": [1]}
    assert osm_route._overpass("q", tmp_path, "osm_rail", None, True, 10) == {"elements": [1]}
    assert not asked


def test_an_expired_copy_is_used_when_the_service_is_down_if_the_caller_allows(tmp_path, monkeypatch):
    good = Mock(status_code=200)
    good.iter_content.return_value = iter([b"railway"])
    monkeypatch.setattr(base.requests, "get", Mock(side_effect=[good, requests.ConnectionError("down"), requests.ConnectionError("down")]))
    monkeypatch.setattr(base.time, "sleep", lambda _: None)
    args = dict(raw_dir=tmp_path, name="t", retries=1)
    path = base.http_get("https://example.test/x", **args)
    os.utime(path, (0, 0))                                                 # expired
    assert base.http_get("https://example.test/x", stale_ok=True, **args) == path
    with pytest.raises(base.SourceUnavailable):                            # the default still says the service is down
        base.http_get("https://example.test/x", **args)


def test_a_cut_off_dem_window_is_fetched_again_not_a_synthetic_route(tmp_path, monkeypatch):
    """A run killed while saving a DEM window left a broken .npz, and every later build fell back to synthetic
    terrain (and so switched LiDAR off) until someone deleted it."""
    from tcs.sources import terrain

    dem = terrain.CopernicusDEM(tmp_path)
    name = terrain._tile_name(51, 0)
    (dem.raw_dir / f"{name}_0.1000_51.1000_0.2000_51.2000.npz").write_bytes(b"PK\x03\x04 cut off")

    class Ds:
        transform = __import__("rasterio").transform.from_origin(0, 52, 0.01, 0.01)
        nodata, res = -9999.0, (0.01, 0.01)

        def read(self, band, window):
            return np.full((int(window.height), int(window.width)), 42.0)

        def window_bounds(self, win):
            return __import__("rasterio").windows.bounds(win, self.transform)

    monkeypatch.setattr(dem, "_open", lambda n: Ds())
    dem.prepare(0.1, 51.1, 0.2, 51.2)
    assert dem.windows and float(dem.windows[0].data.mean()) == 42.0
    again = terrain.CopernicusDEM(tmp_path)
    monkeypatch.setattr(again, "_open", Mock(side_effect=AssertionError("asked again")))
    again.prepare(0.1, 51.1, 0.2, 51.2)                                    # the rewritten window is read from the cache
    assert not list(dem.raw_dir.glob("*.tmp*"))


def test_readings_whose_time_could_not_be_read_are_not_collapsed_into_one():
    from tcs.sources.measurements import _best_server

    m = pd.DataFrame({"timestamp": pd.to_datetime([None, None, None], utc=True), "device": "d", "provider_id": "ee",
                      "rsrp_dbm": [-90.0, -95.0, -100.0]})
    assert len(_best_server(m)) == 3
    t = pd.Timestamp("2026-01-01", tz="UTC")
    m = pd.DataFrame({"timestamp": [t, t], "device": "d", "provider_id": "ee", "rsrp_dbm": [-90.0, -80.0]})
    assert _best_server(m)["rsrp_dbm"].tolist() == [-80.0]                 # same moment: the strongest server still wins
