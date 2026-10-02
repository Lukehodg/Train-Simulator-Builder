"""aa/antenna_patterns.py: reading antenna reports, on synthetic reports drawn with matplotlib in the vendor's layout
(a gain chart, a pair of polar pattern cuts, a port-to-port S-parameter chart), and the gain sums."""
import numpy as np
import pytest

pytest.importorskip("pdfplumber")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42                     # real text in the PDF, as in the vendor's reports
matplotlib.rcParams["path.simplify"] = False                 # every point drawn, as in the vendor's reports
import matplotlib.pyplot as plt  # noqa: E402

from aa.antenna_patterns import (  # noqa: E402
    Cut,
    band_summary,
    horizon_gain,
    lin_mean_db,
    read_cuts,
    read_xy,
    summarise,
)

RED, BLUE, GREEN, GREY = (1, 0, 0), (0, 0, 1), (0, 0.5, 0), (0.5, 0.5, 0.5)


def _cartesian(path, curves, xlim, ylim, xticks, yticks, legend):
    """A chart page: grey grid, tick labels, one curve per (x, y, colour); a legend line beside its label."""
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0.15, 0.35, 0.7, 0.4])
    ax.set_xlim(*xlim), ax.set_ylim(*ylim), ax.set_xticks(xticks), ax.set_yticks(yticks)
    ax.grid(True, color=GREY)
    for x, y, c in curves:
        ax.plot(x, y, color=c, linewidth=1)
    fig.lines.append(plt.Line2D([0.1, 0.13], [0.3, 0.3], color=curves[0][2], transform=fig.transFigure))
    fig.text(0.14, 0.3, legend, va="center")
    fig.savefig(path)
    plt.close(fig)


def _polar_pair(path, vertical, horizontal):
    """Two polar charts drawn the vendor's way: rings every 10 dB from -40 at the centre to 0 at the outside, dB labels
    along the right-hand radius (just inside their ring), angle labels outside; horizontal cut on top (0 deg at the
    right, +90 at the bottom of the page), vertical cut below (0 deg at the top, +90 at the right)."""
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 595), ax.set_ylim(842, 0), ax.axis("off")              # page points, y down
    R = 100.0

    def chart(cx, cy, deg, db, phi_of, colour, name, legend_y):
        for k in range(1, 5):
            t = np.linspace(0, 2 * np.pi, 200)
            ax.plot(cx + R * k / 4 * np.cos(t), cy + R * k / 4 * np.sin(t), color=GREY, linewidth=0.5)
        for k, v in enumerate((-40, -30, -20, -10, 0)):
            ax.text(cx + R * k / 4 - 5, cy + 6, str(v), ha="center", va="center", fontsize=6)
        for a in range(-150, 181, 30):
            p = np.radians(phi_of(a))
            ax.text(cx + 1.15 * R * np.cos(p), cy + 1.15 * R * np.sin(p), str(a), ha="center", va="center", fontsize=6)
        r = (np.asarray(db) + 40) / 40 * R
        p = np.radians(phi_of(np.asarray(deg)))
        ax.plot(cx + r * np.cos(p), cy + r * np.sin(p), color=colour, linewidth=1)
        ax.plot([cx - 120, cx - 100], [legend_y, legend_y], color=colour)
        ax.text(cx - 98, legend_y, f"{name}-Plane: Copolar", va="center", fontsize=6)

    chart(300, 230, *horizontal, lambda a: a, RED, "Horizontal", 360)              # screen angle = azimuth
    chart(300, 560, *vertical, lambda a: np.asarray(a) - 90, BLUE, "Vertical", 690)  # 0 at the top
    fig.savefig(path)
    plt.close(fig)


def test_reads_a_gain_chart_in_its_own_units(tmp_path):
    f = tmp_path / "x_694 to 6425 MHz_Cellular1_gain.pdf"
    mhz = np.array([700, 800, 960, 1800, 2600, 3500])
    gain = np.array([3.0, 6.0, 7.0, 9.0, 7.5, 7.0])
    _cartesian(f, [(mhz, gain, RED)], (500, 6500), (0, 12), range(500, 7000, 500), range(0, 13), "Gain")
    g = read_xy(f, RED)
    assert np.interp(mhz, g.x, g.y) == pytest.approx(gain, abs=0.05)


def test_reads_s21_even_where_it_runs_off_the_chart(tmp_path):
    f = tmp_path / "x.s_Cell 1 and Cell 2.pdf"
    ghz = np.linspace(0.5, 6.5, 300)
    s21 = -15 - 8 * ghz                                       # runs below the -60 dB floor of the chart past 5.6 GHz
    _cartesian(f, [(ghz, s21, GREEN)], (0, 7), (-60, 5), range(0, 8), range(-60, 6, 5), "S-Parameter: S21")
    s = read_xy(f, GREEN)
    assert np.interp([1.0, 3.0, 5.0], s.x, s.y) == pytest.approx([-23, -39, -55], abs=0.3)


def test_reads_both_pattern_cuts_with_their_own_angle_conventions(tmp_path):
    f = tmp_path / "x_790_cellular1.pdf"
    deg = np.arange(-180, 180, 1.0)
    v = -10 * np.abs(np.sin(np.radians(deg + 30)))            # strongest at -30 and 150 deg, weakest at 60 and -120
    h = -6 * (1 + np.cos(np.radians(deg - 120))) / 2          # weakest at azimuth 120
    _polar_pair(f, (deg, v), (deg, h))
    vc, hc = read_cuts(f)
    for cut, want in ((vc, v), (hc, h)):
        got = np.array([cut.at(a) for a in deg[::15]])
        assert got == pytest.approx(want[::15], abs=0.4)


def test_horizon_gain_anchors_the_horizontal_cut_to_the_vertical_one():
    deg = np.arange(-180, 180, 1.0)
    vertical = Cut(deg, np.where(np.abs(deg) == 0, 0.0, -3.0))     # peak at the zenith, -3 dB everywhere else
    horizontal = Cut(deg, np.where(np.abs(deg) == 90, -10.0, 0.0))  # its own maximum ahead and behind
    g = horizon_gain(7.0, vertical, horizontal)
    az = np.arange(-180.0, 180.0, 1.0)
    assert g[az == 0][0] == pytest.approx(4.0) and g[az == 90][0] == pytest.approx(-6.0)   # peak - 3 dB ahead, a null at the side
    s = band_summary({"1": (7.0, vertical, horizontal), "2": (7.0, vertical, Cut(deg, np.where(np.abs(deg) == 0, -10.0, 0.0)))})
    assert s["horizon_best_port_worst_dbi"] > s["horizon_worst_per_port_dbi"]                # two ports cover each other's nulls
    assert lin_mean_db([0, 0]) == 0 and lin_mean_db([0, -100]) == pytest.approx(-3.01, abs=0.01)


def test_a_whole_report_is_summarised_per_uk_band(tmp_path):
    rep = tmp_path / "report"
    (rep / "Gain").mkdir(parents=True)
    (rep / "2D_Pattern" / "Cellular1").mkdir(parents=True)
    _cartesian(rep / "Gain" / "x_694 to 6425 MHz_Cellular1_gain.pdf", [(np.array([700, 1000, 1500, 4000]), np.array([6, 6, 8, 8]), RED)],
               (500, 6500), (0, 12), range(500, 7000, 500), range(0, 13), "Gain")
    deg = np.arange(-180, 180, 1.0)
    for mhz in (790, 1850):
        _polar_pair(rep / "2D_Pattern" / "Cellular1" / f"x_{mhz}_cellular1.pdf", (deg, -3 * np.abs(np.cos(np.radians(deg)))), (deg, 0 * deg))
    r = summarise(rep, tmp_path / "out")
    h = r["horizon"].set_index("band")
    assert set(h.index) == {"800", "1800"}
    assert h.loc["800", "horizon_mean_per_port_dbi"] == pytest.approx(6.0, abs=0.3)      # the horizon is the peak here
    assert h.loc["1800", "peak_gain_dbi"] == pytest.approx(8.0, abs=0.1)
    assert (tmp_path / "out" / "horizon_by_band.csv").exists() and (tmp_path / "out" / "gain_cellular1.csv").exists()
