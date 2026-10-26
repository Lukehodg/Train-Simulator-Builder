"""tcs/model/adjustment.py: trees learned from the model's errors where trains measured, stored as a table and walked
with numpy when a route is built."""
import numpy as np
import pandas as pd
import pytest

from tcs.config import ROOT, load_settings
from tcs.model import adjustment
from tcs.model.cellular import cellular_observations
from tcs.pipeline.join_cells import candidate_cells, corridor_cells, serving_cells
from tcs.pipeline.join_coverage import coverage_prior
from tcs.pipeline.movement import movement
from tcs.pipeline.obstruction import enrich_terrain
from tcs.pipeline.sample_route import build_route


def _offline_route():
    s = load_settings(offline=True)
    s.route["sample_spacing_m"] = 500
    s.terrain["horizon_azimuths"] = 8
    s.terrain["horizon_reach_m"] = 1500
    b = build_route(s)
    b.samples = enrich_terrain(s, b)
    prior, b.samples = coverage_prior(s, b)
    cells = corridor_cells(s, b, b.samples)
    serving = serving_cells(s, b.samples, candidate_cells(s, b.samples, cells))
    b.samples, _ = movement(s, b.samples, b.stations)
    return s, b, prior, serving, cells


def test_stored_trees_predict_what_scikit_learn_predicts_missing_values_included():
    pytest.importorskip("sklearn")
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(3000, len(adjustment.FEATURES))), columns=adjustment.FEATURES)
    x.iloc[::7, 3] = np.nan                                             # missing values take their learned branch
    y = 4 * np.tanh(x.iloc[:, 0]) - 3 * (x.iloc[:, 3].fillna(2) > 0) + rng.normal(scale=0.5, size=len(x))
    m = HistGradientBoostingRegressor(**{**adjustment.PARAMS, "max_iter": 40}).fit(x, y)
    table = adjustment.to_table(m)
    new = pd.DataFrame(rng.normal(size=(500, len(adjustment.FEATURES))), columns=adjustment.FEATURES)
    new.iloc[::5, 3] = np.nan
    assert adjustment.predict(table, new) == pytest.approx(m.predict(new), abs=1e-9)


def test_the_adjustment_moves_the_prediction_by_its_size_and_is_saved_with_its_feature_list(tmp_path):
    s, b, prior, serving, cells = _offline_route()
    rows = prior[["sample_id", "provider_id"]]
    x = adjustment.features(b.samples, prior, serving, cells, rows)
    assert list(x.columns) == adjustment.FEATURES and len(x) == len(rows)
    assert x["net_ee"].sum() == (rows["provider_id"] == "ee").sum() and (x["cells_5km"] >= x["cells_2km"]).all()
    # a table that is one constant leaf: the same dB everywhere
    table = pd.DataFrame({"tree": [-1, 0], "node": [0, 0], "feature": [-1, -1], "threshold": [np.nan, np.nan], "missing_left": [False, False],
                          "left": [0, 0], "right": [0, 0], "leaf": [True, True], "value": [-3.0, 0.5]})
    adj = adjustment.for_route(table, b.samples, prior, serving, cells)
    assert adj["adjustment_db"].to_numpy() == pytest.approx(-2.5)
    base = cellular_observations(s, b.samples, prior, serving)
    got = cellular_observations(s, b.samples, prior, serving, adjustment=adj)
    keep = ~base["_in_tunnel"] & base["quality_score"].between(0.2, 0.7)              # away from the score's ends
    assert (got["signal_primary"] - base["signal_primary"])[keep].to_numpy() == pytest.approx(-2.5, abs=0.05)
    assert got["source_flags"][keep].str.contains("learned_adjustment").all()
    assert not base["source_flags"].str.contains("learned_adjustment").any()
    adjustment.save(table, tmp_path / "a.parquet", meta={"sources": []})
    assert adjustment.load(tmp_path / "a.parquet")["value"].tolist() == [-3.0, 0.5]
    assert adjustment.meta(tmp_path / "a.parquet")["features"] == adjustment.FEATURES
    assert adjustment.load(tmp_path / "missing.parquet") is None


def test_the_stored_trees_match_this_codes_features_and_the_us_leaves_them_out():
    s = load_settings()
    path = adjustment.path_of(s)
    assert path == ROOT / "config" / "learned_adjustment.parquet"
    assert adjustment.load(path) is not None                            # fitted on the features computed here
    from tcs.config import list_routes

    us = next(r["id"] for r in list_routes() if r.get("country") == "US")
    assert adjustment.path_of(load_settings(route_id=us)) is None       # learned on GB scanner trains only
