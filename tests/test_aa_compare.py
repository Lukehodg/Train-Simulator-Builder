"""aa/compare_route.py: the link maths behind the active antenna vs passive installs comparison (aa/METHOD.md)."""
import numpy as np
import pytest

pytest.importorskip("scipy")

from aa.compare_route import ASSUME, cable_loss_db, carrier_mbps, link_terms, noise_dbm  # noqa: E402


def test_coax_loss_follows_the_cable_data_sheet():
    assert cable_loss_db(900, 10, 0) == pytest.approx(1.28, abs=0.01)       # 0.128 dB/m at 900 MHz
    assert cable_loss_db(2500, 10, 1) == pytest.approx(3.23, abs=0.01)      # 0.223 dB/m at 2.5 GHz, plus fittings
    assert cable_loss_db(3500, 10, 0) > cable_loss_db(800, 10, 0)


def test_the_cable_only_costs_anything_where_the_link_is_noise_limited():
    n = noise_dbm(ASSUME["noise_figure_db"])
    for sinr_db, rsrp_dbm, cable_matters in ((-3.0, n - 3.0 - 7.1, True),   # signal at the noise floor, no interference
                                             (0.0, -80.0, False)):           # strong signal, as strong interference
        s, i, nn, inr = link_terms([rsrp_dbm], [sinr_db], 7.1, 0.0, ASSUME)
        _, _, nc, _ = link_terms([rsrp_dbm], [sinr_db], 7.1, 4.0, ASSUME)
        a = carrier_mbps(s, i, nn, 10, 4, 2, ASSUME)[0]
        p = carrier_mbps(s, i, nc, 10, 4, 2, ASSUME)[0]
        assert (a / p > 1.1) == cable_matters, (sinr_db, a, p, inr)


def test_more_receive_branches_help_most_when_interference_behaves_like_noise():
    s, i, n = np.array([1.0]), np.array([0.5]), np.array([1e-4])           # interference-limited, SINR about 3 dB
    four = {c: carrier_mbps(s, i, n, 10, 4, 2, ASSUME, c)[0] for c in ("like noise", "correlated")}
    two = {c: carrier_mbps(s, i, n, 10, 2, 2, ASSUME, c)[0] for c in ("like noise", "correlated")}
    assert four["like noise"] / two["like noise"] > 1.3                     # combining lifts the signal over interference
    assert 0.9 < four["correlated"] / two["correlated"] < 1.15              # only diversity left: little either way
    assert carrier_mbps(np.array([1e6]), np.array([1.0]), np.array([1.0]), 10, 4, 2, ASSUME)[0] == pytest.approx(
        10 * ASSUME["cell_share"] * 2 * ASSUME["eta_max"], rel=0.01)       # two layers at the modulation ceiling


def test_bonding_and_networks_are_set_per_install():
    import pandas as pd

    from aa.compare_route import ALL_NETWORKS, PROFILES, SCENARIOS, with_scenario

    df = pd.DataFrame({f"{k}_{n}": [10.0] for k in PROFILES for n in ALL_NETWORKS})
    central = with_scenario(df, SCENARIOS["central (both 0.85)"])
    assert central.T_A.item() == central.T_P2.item() == pytest.approx(0.85 * 30)          # three networks each
    better = with_scenario(df, SCENARIOS["Fleet Connect 0.95, router 0.75"])
    assert better.T_A.item() / better.T_P4.item() == pytest.approx(0.95 / 0.75)
    o2 = with_scenario(df, SCENARIOS["router adds O2 (4 networks vs 3)"])
    assert o2.T_P4.item() == pytest.approx(0.85 * 40) and o2.T_A.item() == pytest.approx(0.85 * 30)   # EDGE Rail stays on 3 units
