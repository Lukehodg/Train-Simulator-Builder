"""tcs/corrections.py: earlier measurements' errors, smoothed along the track, correcting the prediction there."""
import numpy as np
import pandas as pd
import pytest

from tcs import corrections
from tcs.config import load_settings
from tcs.model.cellular import cellular_observations
from tcs.pipeline.join_cells import candidate_cells, corridor_cells, serving_cells
from tcs.pipeline.join_coverage import coverage_prior
from tcs.pipeline.movement import movement
from tcs.pipeline.obstruction import enrich_terrain
from tcs.pipeline.sample_route import build_route


def _points(dist, err, provider="ee", tunnel=False, level=0.0):
    return pd.DataFrame({"sample_id": np.arange(len(dist)), "provider_id": provider, "distance_m": dist, "_in_tunnel": tunnel,
                         "signal_primary": -90.0, "rsrp_dbm": -90.0 + level + np.asarray(err, float)})


def test_errors_drop_the_sets_own_level_and_keep_tunnels_apart():
    err = np.zeros(20)
    err[5:9] = 6.0                                                          # a stretch 6 dB better than modelled
    j = _points(np.arange(0, 1000, 50.0), err, level=-7.0)                  # read by a scanner 7 dB low everywhere
    j.loc[3, "_in_tunnel"] = True
    j.loc[3, "rsrp_dbm"] = -60.0                                            # far better inside than the portal model says
    e = corrections.errors(j, weight=0.1)
    assert len(e) == 20 and e["w"].eq(0.1).all() and e["in_tunnel"].sum() == 1
    open_air = e[~e["in_tunnel"]]
    assert open_air.loc[~open_air["distance_m"].between(250, 400), "error_db"].abs().max() == pytest.approx(0.0)   # its own level is gone
    assert open_air.loc[open_air["distance_m"].between(250, 400), "error_db"].tolist() == pytest.approx([6.0] * 4)
    assert e.loc[e["in_tunnel"], "error_db"].iloc[0] == pytest.approx(37.0)  # the level comes from open air only


def test_a_set_level_over_all_its_routes_keeps_a_route_the_model_overrates():
    dist = np.arange(0, 1000, 50.0)
    a = _points(dist, np.zeros(20), level=-7.0)                             # scanner 7 dB low; the model right here
    b = _points(dist, np.full(20, -4.0), level=-7.0)                        # ...and 4 dB too hopeful on this route
    big = pd.concat([a] * 3 + [b])                                          # most of the set's points are on the first route
    level = corrections.levels(big)
    assert level["ee"] == pytest.approx(-7.0)
    assert corrections.errors(b, 1.0, level)["error_db"].tolist() == pytest.approx([-4.0] * 20)   # kept
    assert corrections.errors(b, 1.0)["error_db"].abs().max() == pytest.approx(0.0)              # the route's own level: lost


def test_tunnel_and_open_air_errors_each_correct_their_own_samples():
    samples = pd.DataFrame({"sample_id": np.arange(40), "distance_m": np.arange(40) * 50.0, "in_tunnel": np.arange(40) >= 20})
    err = pd.DataFrame({"provider_id": "ee", "distance_m": [950.0] * 10 + [1050.0] * 10, "in_tunnel": [False] * 10 + [True] * 10,
                        "error_db": [-3.0] * 10 + [8.0] * 10, "w": 1.0})
    c = corrections.smooth(err, samples).set_index("sample_id")["correction_db"]
    assert c[19] == pytest.approx(-3.0 * 10 / 10.5, abs=0.05) and c[20] == pytest.approx(8.0 * 10 / 10.5, abs=0.05)
    assert (c[c.index < 20] < 0).all() and (c[c.index >= 20] > 0).all()     # nothing leaks across the portal


def test_a_correction_is_shrunk_where_measurements_are_few_and_fades_away_from_them():
    samples = pd.DataFrame({"sample_id": np.arange(100), "distance_m": np.arange(100) * 50.0})
    many = pd.DataFrame({"provider_id": "ee", "distance_m": np.full(20, 1000.0), "error_db": 6.0, "w": 1.0})
    one = pd.DataFrame({"provider_id": "ee", "distance_m": [4000.0], "error_db": [6.0], "w": [1.0]})
    c = corrections.smooth(pd.concat([many, one]), samples).set_index("sample_id")["correction_db"]
    assert c[20] == pytest.approx(6.0 * 20 / 20.5, abs=0.05)                 # measured often: almost the full error
    assert c[80] == pytest.approx(6.0 * 1 / 1.5, abs=0.05)                   # measured once: pulled towards zero
    assert c[20] > c[21] > c[22] > c[24] and 27 not in c.index              # fades over ~100 m, none beyond 300 m
    assert 50 not in c.index                                                 # nothing measured near: no correction


def test_corrections_move_the_prediction_by_their_size_in_open_air_and_in_tunnels(tmp_path):
    s = load_settings(offline=True)
    s.route["sample_spacing_m"] = 500
    s.terrain["horizon_azimuths"] = 8
    s.terrain["horizon_reach_m"] = 1500
    b = build_route(s)
    b.samples = enrich_terrain(s, b)
    prior, b.samples = coverage_prior(s, b)
    serving = serving_cells(s, b.samples, candidate_cells(s, b.samples, corridor_cells(s, b, b.samples)))
    b.samples, _ = movement(s, b.samples, b.stations)
    base = cellular_observations(s, b.samples, prior, serving)
    open_air = base[~base["_in_tunnel"] & base["quality_score"].between(0.2, 0.7) & (base["provider_id"] == "ee")]
    sid = int(open_air["sample_id"].iloc[0])
    corr = pd.DataFrame({"sample_id": [sid], "provider_id": ["ee"], "correction_db": [-4.6], "weight": [3.0]})
    got = cellular_observations(s, b.samples, prior, serving, corrections=corr)
    row = lambda d: d[(d["sample_id"] == sid) & (d["provider_id"] == "ee")].iloc[0]
    assert row(got)["signal_primary"] - row(base)["signal_primary"] == pytest.approx(-4.6, abs=0.05)
    assert "measured_correction" in row(got)["source_flags"] and row(got)["measured_correction_db"] == pytest.approx(-4.6)
    # inside a tunnel the correction goes on the tunnel model
    inside = base[base["_in_tunnel"] & base["quality_score"].between(0.1, 0.8) & (base["provider_id"] == "ee")]
    tid = int(inside["sample_id"].iloc[0])
    got_t = cellular_observations(s, b.samples, prior, serving, corrections=pd.DataFrame(
        {"sample_id": [tid], "provider_id": ["ee"], "correction_db": [5.0], "weight": [3.0]}))
    rt = lambda d: d[(d["sample_id"] == tid) & (d["provider_id"] == "ee")].iloc[0]
    assert rt(got_t)["signal_primary"] - rt(base)["signal_primary"] == pytest.approx(5.0, abs=0.05)
    assert "measured_correction" in rt(got_t)["source_flags"] and rt(got_t)["quality_base"] == pytest.approx(rt(base)["quality_base"])
    others = got[(got["sample_id"] != sid) | (got["provider_id"] != "ee")]
    assert others["measured_correction_db"].isna().all() and not others["source_flags"].str.contains("measured_correction").any()
    # stored by position and matched back to the samples
    table = corrections.to_table("r1", corr, b.samples)
    corrections.save(table, tmp_path / "c.parquet", meta={"sources": []})
    back = corrections.for_route(tmp_path / "c.parquet", "r1", b.samples, b.proj)
    assert back[["sample_id", "provider_id"]].values.tolist() == [[sid, "ee"]] and back["correction_db"].iloc[0] == pytest.approx(-4.6)
    assert corrections.for_route(tmp_path / "c.parquet", "other_route", b.samples, b.proj) is None
    assert corrections.meta(tmp_path / "c.parquet") == {"sources": []}
