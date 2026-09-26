import io

import numpy as np
import pandas as pd
import pytest
import requests
from rich.console import Console
from shapely.geometry import LineString

from tcs.geo import Projector, local_crs, offset_xy, sample_line
from tcs.model.calibration import fit
from tcs.sources import base
from tcs.sources.base import SourceUnavailable, http_get, redact
from tcs.sources.measurements import PRESETS
from tcs.sources.ofcom_coverage import _aggregate, _parse_payload
from tcs.sources.terrain import SyntheticDEM, sky_visibility
from tcs.sources.timetable import _hhmm, schedule_seconds


def test_local_crs():
    assert local_crs(-1.5, 53.0, "GB").to_epsg() == 27700
    assert local_crs(2.35, 48.85, "FR").to_epsg() == 32631


def test_sample_line_spacing():
    line = LineString([(0, 0), (1000, 0), (1000, 1000)])
    s = sample_line(line, 50)
    assert len(s["x"]) == 41
    assert np.isclose(s["distance_m"][-1], 2000)
    assert np.isclose(s["bearing_deg"][1], 90)      # eastbound
    assert np.isclose(s["bearing_deg"][-2], 0)      # northbound


def test_offset_right_hand_side():
    x, y = offset_xy(np.array([0.0]), np.array([0.0]), np.array([0.0]), 10)   # heading north -> right = east
    assert np.isclose(x[0], 10) and np.isclose(y[0], 0)


def test_ofcom_payload_parsing():
    """Real contract: list of {PostCode, Availability:[per-UPRN rows]} with Mc_<MNO> 0-4 levels; sample keys carry stray spaces."""
    ops = [{"id": "ee", "name": "EE", "ofcom_key": "EE"}, {"id": "vodafone", "name": "Vodafone", "ofcom_key": "VO"}, {"id": "three", "name": "Three", "ofcom_key": "TH"}]
    payload = [{"PostCode": "N1C4TB", "Availability": [
        {" uprn ": "1", " Mc_EE ": 4, " Mc_TH ": 3, " Mc_O2 ": 4, " Mc_VO ": 2},
        {"uprn": "2", "Mc_EE": 2, "Mc_TH": 3, "Mc_O2": 4, "Mc_VO": 1},
        {"uprn": "3", "Mc_EE": 3, "Mc_TH": 0, "Mc_O2": 4, "Mc_VO": 2},
    ]}]
    got = _parse_payload(payload, ops)
    assert got["ee"]["levels"] == [4, 2, 3] and got["ee"]["n_uprn"] == 3
    assert got["vodafone"]["levels"] == [2, 1, 2]
    assert got["three"]["levels"] == [3, 3, 0]
    assert _aggregate(got["ee"]["levels"], "median") == 3.0 and _aggregate(got["ee"]["levels"], "max") == 4.0
    assert _aggregate([], "median") is None


def test_timetable_helpers():
    assert _hhmm("0947 ") == "09:47" and _hhmm("0947H") == "09:47:30" and _hhmm("     ") is None
    secs = schedule_seconds(pd.DataFrame({"crs": ["KGX", "EDB"], "time": ["09:00", "13:20"]}), "09:00")
    assert secs["EDB"] == 4 * 3600 + 20 * 60


def test_sky_visibility_open_and_blocked():
    dem = SyntheticDEM(-0.12, 51.53)
    proj = Projector(local_crs(-0.12, 51.53, "GB"))
    x, y = proj.to_xy(np.array([-0.12, -1.0]), np.array([51.53, 54.4]))
    elev = dem.sample(*proj.to_lonlat(x, y))
    sky, hz = sky_visibility(dem, proj, np.asarray(x), np.asarray(y), elev, azimuths=8, reach_m=1000, step_m=100)
    assert sky.shape == (2,) and np.all((sky >= 0) & (sky <= 1))
    # a sample placed 200 m below the terrain must be (nearly) fully blocked
    sky2, _ = sky_visibility(dem, proj, np.asarray(x[:1]), np.asarray(y[:1]), elev[:1] - 200, azimuths=8, reach_m=1000, step_m=100)
    assert sky2[0] < 0.3


def test_calibration_fit_recovers_bias():
    n = 400
    q = np.random.default_rng(1).uniform(0.2, 0.9, n)
    obs = pd.DataFrame({"sample_id": np.arange(n), "provider_id": "ee", "quality_score": q, "distance_m": np.arange(n) * 50.0})
    rsrp = -120 + 46 * (q + 0.1)             # measurements say the prior is 0.1 too pessimistic
    meas = pd.DataFrame({"sample_id": np.arange(n), "provider_id": "ee", "rsrp_dbm": rsrp, "kind": "point"})
    cal = fit(obs, meas)
    assert abs(cal["bias"]["ee"] - 0.1) < 0.02
    assert abs(cal["rsrp_map"]["ee"]["slope"] - 46) < 1


def test_measurement_presets_have_position():
    for k, v in PRESETS.items():
        assert "latitude" in v and "longitude" in v, k


TOKEN_URL = "https://opencellid.org/ocid/downloads?token=pk.secret123&type=mcc&file=234.csv.gz"


def test_redact_masks_credentials():
    assert redact(TOKEN_URL) == "https://opencellid.org/ocid/downloads?token=***&type=mcc&file=234.csv.gz"
    msg = "Max retries exceeded with url: /cell/getInArea?key=abc123&BBOX=1,2,3,4 (Caused by NewConnectionError)"
    assert "abc123" not in redact(msg) and "BBOX=1,2,3,4" in redact(msg)


@pytest.mark.parametrize("failure", ["http_403", "connection_error"])
def test_http_get_never_exposes_token(tmp_path, monkeypatch, failure):
    def fake_get(url, **kw):
        if failure == "connection_error":
            raise requests.ConnectionError(f"Max retries exceeded with url: {url[url.index('/ocid'):]}")
        return type("Resp", (), {"status_code": 403, "close": lambda self: None})()

    log = io.StringIO()
    monkeypatch.setattr(base, "console", Console(file=log, width=400))
    monkeypatch.setattr(base.requests, "get", fake_get)
    monkeypatch.setattr(base.time, "sleep", lambda s: None)
    with pytest.raises(SourceUnavailable) as exc:
        http_get(TOKEN_URL, raw_dir=tmp_path, name="opencellid_bulk")
    assert "pk.secret123" not in str(exc.value)
    assert "fetching" in log.getvalue() and "pk.secret123" not in log.getvalue()


def test_fallback_coordinates_cover_route_catalogue():
    """Offline mode needs a coordinate for every station of every route in config/ (a new route file must add its own)."""
    import yaml

    from tcs.config import list_routes
    from tcs.sources.fallback_stations import STATIONS

    for r in list_routes():
        with open(r["file"], encoding="utf-8") as fh:
            stations = yaml.safe_load(fh)["route"]["stations"]
        missing = [s["crs"] for s in stations if s["crs"] not in STATIONS]
        assert not missing, f"{r['id']}: no offline coordinates for {missing}"
    for crs, (lat, lon) in STATIONS.items():
        assert 49.8 < lat < 60.9 and -8.7 < lon < 1.8, crs          # Great Britain bounding box
