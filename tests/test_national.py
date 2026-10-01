"""National calibration: signal carried into tunnels from the portals, which calibration a route uses, and the pooled
fit-and-test (`tcs calibrate-national`)."""
import numpy as np
import pandas as pd
import pytest
import yaml

from tcs.config import load_settings
from tcs.model import calibration
from tcs.model.cellular import portals, tunnel_quality


def test_portals_find_the_open_air_either_side():
    tun = np.array([1, 1, 0, 0, 1, 1, 1, 0, 1], dtype=bool)      # a tunnel at each end of the route and one between
    before, after, d_before, d_after = portals(tun, np.arange(9) * 50.0)
    assert before.tolist() == [-1, -1, 2, 3, 3, 3, 3, 7, 7]
    assert after.tolist() == [2, 2, 2, 3, 7, 7, 7, 7, -1]
    assert d_before[5] == 100 and d_after[5] == 100 and np.isinf(d_before[0]) and np.isinf(d_after[8])


def test_tunnel_signal_fades_from_the_portals_to_the_floor():
    n = 42
    samples = pd.DataFrame({"in_tunnel": [False] + [True] * (n - 2) + [False], "distance_m": np.arange(n) * 50.0})
    q, pos, ee = np.r_[0.8, np.zeros(n - 2), 0.4], np.arange(n), np.array(["ee"] * n)
    inside = tunnel_quality(q, pos, ee, samples, floor=0.04, decay_m=150)[1:-1]
    assert inside[0] == pytest.approx(0.04 + 0.76 * np.exp(-50 / 150))      # 50 m in from the stronger portal
    assert inside[-1] == pytest.approx(0.04 + 0.36 * np.exp(-50 / 150))
    assert inside[20] == pytest.approx(0.04, abs=1e-3)                        # 1 km in: the deep-tunnel floor
    assert (np.diff(inside[:20]) < 0).all()
    assert (tunnel_quality(q, pos, ee, samples, 0.04, None)[1:-1] == 0.04).all()   # no decay set: the floor throughout
    dark = np.r_[0.01, np.zeros(n - 2), 0.01]
    assert tunnel_quality(dark, pos, ee, samples, 0.04, 150)[1:-1].max() <= 0.01 + 1e-9   # never better than outside


def test_routes_without_their_own_fit_use_the_national_calibration(tmp_path):
    s = load_settings(offline=True)
    doc = {"source": "Test survey (somewhere)", "fit_period": {"from": "2018-06-01", "to": "2018-12-31"},
           "calibration": {"version": 2, "section_km": 10, "rsrp_input": "corrected_quality", "bias": {"ee": 0.1},
                           "rsrp_map": {"ee": {"slope": 40.0, "intercept": -110.0}}, "residual_std": {"ee": 9.0}, "sections": {}, "n_points": {"ee": 100}},
           "validation": {"routes": {s.route_id: {"calibrated": {"points": 50, "mae_db": 8.0}}}}}
    f = tmp_path / "calibration.yaml"
    f.write_text(yaml.safe_dump(doc))
    s.sim["cellular"]["national_calibration"] = str(f)
    interim = tmp_path / "interim"
    interim.mkdir()
    cal = calibration.for_route(s, interim)
    assert cal["scope"] == "national" and cal["bias"] == {"ee": 0.1}
    about = calibration.describe(s, interim)
    assert about["source"] == "Test survey (somewhere)" and about["route"]["calibrated"]["mae_db"] == 8.0
    calibration.save({"version": 2, "bias": {"ee": -0.2}, "rsrp_map": {}, "sections": {}}, interim / "calibration.json")
    assert calibration.for_route(s, interim)["bias"] == {"ee": -0.2}            # a route's own fit wins
    assert calibration.describe(s, interim)["scope"] == "route"
    (interim / "calibration.json").unlink()
    s.sim["cellular"]["national_calibration"] = None
    assert calibration.for_route(s, interim) is None and calibration.describe(s, interim) is None


def test_checked_in_calibration_matches_the_model_settings():
    """config/calibration.yaml was fitted with particular cutting and tunnel terms; simulation.yaml must carry them."""
    from tcs.national import environment_of

    s = load_settings(offline=True)
    doc = calibration.national_doc(s)
    if doc is None:
        pytest.skip("no national calibration checked in")
    assert doc["environment"] == environment_of(s)
    calibration.check(doc["calibration"])
    assert set(doc["calibration"]["bias"]) <= {op["id"] for op in s.operators}


def test_national_fit_recovers_a_known_offset_and_tests_out_of_sample(tmp_path, monkeypatch):
    """Measurements made from the model itself, 0.1 better in score: the pooled fit must find that, keep the model's dB
    scale, and report accuracy for the later period that beats the uncalibrated model's."""
    from tcs import config, national
    from tcs.model.cellular import cellular_observations
    from tcs.pipeline.join_cells import candidate_cells, corridor_cells, serving_cells
    from tcs.pipeline.join_coverage import coverage_prior
    from tcs.pipeline.movement import movement
    from tcs.pipeline.obstruction import enrich_terrain
    from tcs.pipeline.sample_route import build_route, save_bundle

    for name in ("RAW", "INTERIM", "PROCESSED"):
        monkeypatch.setattr(config, name, tmp_path / name.lower())
    s = load_settings(offline=True)
    s.route["sample_spacing_m"] = 500
    s.terrain["horizon_azimuths"] = 8
    s.terrain["horizon_reach_m"] = 1500
    b = build_route(s)
    b.samples = enrich_terrain(s, b)
    prior, b.samples = coverage_prior(s, b)
    serving = serving_cells(s, b.samples, candidate_cells(s, b.samples, corridor_cells(s, b, b.samples)))
    b.samples, _ = movement(s, b.samples, b.stations)
    b.geometry_source = "osm"                                     # as if drawn from OSM: only those routes are matched
    interim = s.paths()["interim"]
    save_bundle(b, interim)
    prior.to_parquet(interim / "coverage_prior.parquet", index=False)
    serving.to_parquet(interim / "serving.parquet", index=False)
    monkeypatch.setattr(national, "load_settings", lambda route_id=None, **k: load_settings(offline=True, route_id=route_id))

    obs = cellular_observations(s, b.samples, prior, serving)
    obs = obs[~obs["_in_tunnel"]].merge(b.samples[["sample_id", "latitude", "longitude"]], on="sample_id")
    names = {op["id"]: op["name"] for op in s.operators}
    rows = []
    for year in ("2018-07-01", "2019-03-01"):
        for k, (_, r) in enumerate(obs.iterrows()):
            rows.append({"latitude": r["latitude"], "longitude": r["longitude"], "datetime": pd.Timestamp(year) + pd.Timedelta(seconds=k),
                         "train": "Train1", "operator": names[r["provider_id"]], "mnc": 0, "pci": 1, "rsrq": -10, "sinr": 10,
                         "cal_rsrp": -120 + 46 * min(1.0, r["quality_score"] + 0.1)})
    csv = tmp_path / "yt.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    doc = national.run([s.route_id], csv, "yellow_train", "2019-01-01", log=lambda *a: None)
    assert doc["routes"] == [s.route_id] and doc["fit_period"]["from"].startswith("2018") and doc["test_period"]["from"].startswith("2019")
    v = doc["validation"]
    after, before = v["calibrated"]["overall"], v["uncalibrated"]["overall"]
    assert after["points"] == before["points"] > 0
    assert after["mae_db"] < 1.0 < before["mae_db"]
    assert v["routes"][s.route_id]["held_out"] is None                       # one route: nothing to hold out
    for pid, mp in doc["calibration"]["rsrp_map"].items():
        assert (mp["slope"], mp["intercept"]) == (46, -120)                   # the model's scale, kept
        assert doc["calibration"]["bias"][pid] == pytest.approx(0.1, abs=0.03)
    assert "rsrp_dbm" not in yaml.safe_dump(doc)                              # derived figures only, never the measurements
