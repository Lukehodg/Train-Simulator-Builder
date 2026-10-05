"""Field throughput from an active antenna's own logs: periodic download tests along the line.

Reads a log with one row per time bin (thingName, bin_time [ms], lat, lon, download_rate_*) where a download test
runs every few bins and the bins between tests read 0. Finds the test cadence, counts test slots that returned
nothing (no service at that moment), works out train speed from the positions, and summarises the rate per test:
overall, moving and standing, and by speed band, with a chart along each trip.

    python aa/field_throughput.py data/aa/vendor/<log>.csv --out data/aa/field/<name>

Data stays out of git (aa/README.md); this writes to the --out folder only. Without signal (RSRP/SINR) columns the
log cannot fit SINR -> throughput; it gives the throughput distribution an active antenna reached in service.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MOVING_KPH = 15.0
BANDS = ((15, 80), (80, 140), (140, 250))
GAP_S = 60.0                                   # a longer pause between rows starts a new logging block


def load(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, index_col=0)
    rate = next(c for c in d.columns if c.startswith("download_rate"))
    d = d.rename(columns={rate: "rate"})
    d["t"] = pd.to_datetime(d["bin_time"], unit="ms")
    return d.sort_values("t").reset_index(drop=True)


def speed_kph(d: pd.DataFrame, span: int = 10) -> np.ndarray:
    """Speed from positions `span` bins apart (a few tens of seconds: GPS jitter averages out)."""
    from pyproj import Geod

    lon, lat, ts = d["lon"].to_numpy(), d["lat"].to_numpy(), d["bin_time"].to_numpy() / 1000
    _, _, dist = Geod(ellps="WGS84").inv(lon[:-span], lat[:-span], lon[span:], lat[span:])
    out = np.full(len(d), np.nan)
    out[span // 2: span // 2 + len(dist)] = dist / np.maximum(ts[span:] - ts[:-span], 1) * 3.6
    return pd.Series(out).bfill().ffill().to_numpy()


def tests(d: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """One row per test slot: the tests that returned a rate, plus the slots in the cadence that returned none."""
    d = d.assign(block=(d["t"].diff().dt.total_seconds() > GAP_S).cumsum(), speed_kph=speed_kph(d))
    got = d[d["rate"] > 0]
    gaps = got.groupby("block")["t"].diff().dt.total_seconds().dropna()
    cadence = float(gaps.mode().iloc[0]) if len(gaps) else np.nan
    missed = []
    for _, g in got.groupby("block"):
        gp = g["t"].diff().dt.total_seconds().to_numpy()
        for i in np.flatnonzero(gp > 1.5 * cadence):
            for k in range(1, int(round(gp[i] / cadence))):          # the slots between two results
                row = g.iloc[i - 1].copy()
                row["t"] = row["t"] + pd.Timedelta(seconds=k * cadence)
                row["rate"] = 0.0
                missed.append(row)
    out = pd.concat([got, pd.DataFrame(missed)], ignore_index=True) if missed else got.copy()
    return out.sort_values("t").reset_index(drop=True), cadence


def quantiles(s: pd.Series) -> dict:
    return {"tests": int(len(s)), "p10": round(float(s.quantile(0.1)), 1), "p50": round(float(s.quantile(0.5)), 1),
            "p90": round(float(s.quantile(0.9)), 1), "mean": round(float(s.mean()), 1)}


def summarise(t: pd.DataFrame, cadence: float) -> dict:
    mv = t["speed_kph"] > MOVING_KPH
    r = t["rate"]
    return {
        "cadence_s": cadence, "test_slots": int(len(t)), "slots_without_a_result": int((r <= 0).sum()),
        "all": quantiles(r), "moving": quantiles(r[mv]), "standing": quantiles(r[~mv]),
        "by_speed_kph": {f"{lo}-{hi}": quantiles(r[(t["speed_kph"] > lo) & (t["speed_kph"] <= hi)]) for lo, hi in BANDS},
        "moving_share_below": {str(x): round(float((r[mv] < x).mean()), 3) for x in (1, 5, 10, 25)},
        "successive_moving_tests_correlation": round(float(r[mv].reset_index(drop=True).autocorr()), 2),
    }


def chart(t: pd.DataFrame, s: dict, unit: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pyproj import Geod

    INK, MUTED, GRID, BLUE, PALE = "#1f2328", "#59636e", "#e6e8eb", "#2a78d6", "#9cc3ef"
    trips = [g for _, g in t.groupby("block") if (g["speed_kph"] > MOVING_KPH).mean() > 0.3]
    fig = plt.figure(figsize=(11, 6.2), dpi=150)
    gs = fig.add_gridspec(len(trips), 2, width_ratios=[2.2, 1], wspace=0.28, hspace=0.45)
    geod = Geod(ellps="WGS84")
    ymax = float(np.ceil(t["rate"].quantile(0.995) / 10) * 10)
    for k, g in enumerate(trips):
        ax = fig.add_subplot(gs[k, 0])
        _, _, step = geod.inv(g["lon"].values[:-1], g["lat"].values[:-1], g["lon"].values[1:], g["lat"].values[1:])
        km = np.r_[0, np.cumsum(step)] / 1000
        ax.scatter(km, g["rate"], s=10, color=PALE, edgecolor="none", zorder=2)
        ax.plot(km, g["rate"].rolling(9, center=True, min_periods=3).median(), color=BLUE, lw=2, zorder=3)
        ax.set_ylim(0, ymax)
        ax.set_title(f"Trip {k + 1}: {g['t'].min():%H:%M}–{g['t'].max():%H:%M}, {len(g)} tests", loc="left", fontsize=10, color=INK)
        ax.set_ylabel(f"download rate ({unit})", fontsize=8.5, color=MUTED)
        if k == len(trips) - 1:
            ax.set_xlabel("distance along the trip (km)", fontsize=8.5, color=MUTED)
        if k == 0:
            ax.text(0.99, 0.95, "dots: each test · line: running median of 9 tests (~4.5 min)", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8, color=MUTED)
    ax = fig.add_subplot(gs[:, 1])
    rows = [("standing", s["standing"])] + [(f"{k} km/h", v) for k, v in s["by_speed_kph"].items()]
    for i, (name, q) in enumerate(rows):
        ax.plot([q["p10"], q["p90"]], [i, i], color=PALE, lw=6, solid_capstyle="round", zorder=2)
        ax.plot(q["p50"], i, "o", color=BLUE, ms=8, mec="white", mew=2, zorder=3)
        ax.text(q["p90"] + ymax * 0.02, i, f"{q['p50']:.0f}  ({q['tests']} tests)", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9)
    ax.set_xlim(0, ymax * 1.25)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel(f"download rate ({unit}): p10 – median – p90", fontsize=8.5, color=MUTED)
    ax.set_title("By train speed", loc="left", fontsize=10, color=INK)
    for a in fig.axes:
        a.grid(True, color=GRID, lw=0.8, zorder=0)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(GRID)
        a.tick_params(colors=MUTED, labelsize=8)
    fig.suptitle(f"Active antenna, download tests every {s['cadence_s']:.0f} s: median {s['moving']['p50']:.0f} {unit} moving, "
                 f"{s['slots_without_a_result']} of {s['test_slots']} tests without a result", x=0.06, ha="left", fontsize=11.5, color=INK)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("log", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--unit", default="Mbit/s", help="What the log's download_rate column holds (check with whoever exported it)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t, cadence = tests(load(a.log))
    s = summarise(t, cadence)
    (a.out / "summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    t[["t", "lat", "lon", "speed_kph", "rate", "block"]].to_csv(a.out / "tests.csv", index=False)
    chart(t, s, a.unit, a.out / "field_throughput.png")
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
