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


def test_packaging_missing_build_preserves_existing_output(tmp_path, monkeypatch):
    from tcs import package

    monkeypatch.setattr(package, "ROOT", tmp_path)
    marker = tmp_path / "dist" / "keep.txt"
    marker.parent.mkdir()
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no index.html"):
        package.package(skip_build=True)
    assert marker.read_text(encoding="utf-8") == "keep"
