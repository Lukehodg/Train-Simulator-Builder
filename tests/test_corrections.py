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


def test_errors_drop_the_sets_own_level_and_tunnels():
    err = np.zeros(20)
    err[5:9] = 6.0                                                          # a stretch 6 dB better than modelled
    j = _points(np.arange(0, 1000, 50.0), err, level=-7.0)                  # read by a scanner 7 dB low everywhere
    j.loc[3, "_in_tunnel"] = True
    e = corrections.errors(j, weight=0.1)
    assert len(e) == 19 and e["w"].eq(0.1).all()
    assert e.loc[~e["distance_m"].between(250, 400), "error_db"].abs().max() == pytest.approx(0.0)   # its own level is gone
    assert e.loc[e["distance_m"].between(250, 400), "error_db"].tolist() == pytest.approx([6.0] * 4)


def test_a_correction_is_shrunk_where_measurements_are_few_and_fades_away_from_them():
    samples = pd.DataFrame({"sample_id": np.arange(100), "distance_m": np.arange(100) * 50.0})
    many = pd.DataFrame({"provider_id": "ee", "distance_m": np.full(20, 1000.0), "error_db": 6.0, "w": 1.0})
    one = pd.DataFrame({"provider_id": "ee", "distance_m": [4000.0], "error_db": [6.0], "w": [1.0]})
    c = corrections.smooth(pd.concat([many, one]), samples).set_index("sample_id")["correction_db"]
    assert c[20] == pytest.approx(6.0 * 20 / 20.5, abs=0.05)                 # measured often: almost the full error
    assert c[80] == pytest.approx(6.0 * 1 / 1.5, abs=0.05)                   # measured once: pulled towards zero
    assert c[20] > c[21] > c[22] > c[24] and 27 not in c.index              # fades over ~100 m, none beyond 300 m
    assert 50 not in c.index                                                 # nothing measured near: no correction


def test_corrections_move_the_prediction_by_their_size_and_leave_tunnels_to_the_portal_model(tmp_path):
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
    others = got[(got["sample_id"] != sid) | (got["provider_id"] != "ee")]
    assert others["measured_correction_db"].isna().all() and not others["source_flags"].str.contains("measured_correction").any()
    # stored by position and matched back to the samples
    table = corrections.to_table("r1", corr, b.samples)
    corrections.save(table, tmp_path / "c.parquet", meta={"sources": []})
    back = corrections.for_route(tmp_path / "c.parquet", "r1", b.samples, b.proj)
    assert back[["sample_id", "provider_id"]].values.tolist() == [[sid, "ee"]] and back["correction_db"].iloc[0] == pytest.approx(-4.6)
    assert corrections.for_route(tmp_path / "c.parquet", "other_route", b.samples, b.proj) is None
    assert corrections.meta(tmp_path / "c.parquet") == {"sources": []}
