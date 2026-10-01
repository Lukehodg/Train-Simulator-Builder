#!/usr/bin/env python3
"""Fit SINR against RSRP from Network Rail Global View 4G scanner logs (Rail Data Marketplace).

The first step of a signal -> SINR -> throughput model: how good a connection (SINR) a given signal strength
(RSRP) brings, per band and per network, as a typical value and a range.

Method
  1. Read the Global View 4G CSV in chunks. Every data row has one field more than the header, so it is read with
     index_col=False (otherwise every column shifts by one). Rows without a valid RSRP or SINR are dropped.
  2. The scanner logs every cell it hears. On each carrier a modem would use the strongest, so keep the strongest
     RSRP per (second, train, network, carrier). The network comes from the MNC, the band from the downlink frequency.
  3. Split by date: fit on readings before --split, check on readings from --split on.
  4. Per group (each band, pooling networks; each network, on its strongest carrier), bin RSRP from -135 to -65 dBm
     (about 97 % of readings; --rsrp-range) in 2 dB steps and
     take the 10th, 50th and 90th percentile of SINR in every bin with enough readings. Fit a smooth hinge to each
     percentile by weighted least squares (weight = readings in the bin):

         SINR(RSRP) = floor + slope * W * ln(1 + exp((RSRP - knee) / W)),   W = 10 dB

         floor  SINR at the edge of coverage, where neighbouring cells are about as strong as the serving one (dB)
         knee   RSRP above which SINR starts to climb (dBm)
         slope  dB of SINR gained per dB of RSRP above the knee

     Well below the knee SINR = floor; well above it, SINR = floor + slope * (RSRP - knee). The bend is gradual in the
     data; fixing its width W (rather than fitting it) keeps knee and slope from trading off against each other, at no
     visible cost to the fit (fit_rms_db in fit.csv: how far the binned percentiles sit from the curve).

     Percentiles rather than a mean and standard deviation: SINR is bounded and skewed, and its spread changes with
     RSRP, so mean +- SD misdescribes it. The 50th percentile is the typical value, the 10th what you can count on
     nine times out of ten, the 90th the headroom.
  5. Check every curve on the held-out readings: the share of readings below the 10th, 50th and 90th percentile
     curves (10, 50 and 90 % if the curves hold up on later data), and the mean absolute error of the median curve
     next to that of a single overall median (what knowing the RSRP is worth).

For each network two SINRs are fitted against the RSRP of its strongest carrier: the SINR on that carrier, and the
best SINR on any of the network's carriers in that second (closer to what a modem that picks its band gets).

Reading the parameters
  - A knee at the lower limit (-140 dBm) means SINR climbs steadily over the whole range with no floor in the data
    (typical of the 90th percentile): there the curve is in effect a straight line, floor and knee only fix its level.
  - Below about -125 dBm (few readings) the median dips 2-4 dB below the floor.

Caveats
  - Global View RSRP is not corrected for antenna and cable gain: it reads about 7-10 dB below a roof-antenna
    modem (e.g. Yellow Train data) at the same place. SINR is a ratio and barely affected, but feed these curves
    RSRP at Global View level.
  - Above about -65 dBm SINR depends on the place: it keeps rising on the low bands, but collapses on 1800-2600 MHz
    in dense inner-city areas (inner London), where many cells overlap. So the curves are fitted up to -65 dBm only
    and should not be extended beyond it (the hinge keeps rising);
    check.csv gives the share of readings above that and their median, and the charts show them as hollow points.

Requires: python 3.9+, numpy, pandas, pyarrow, scipy, matplotlib.

Usage:
  python fit_rsrp_sinr.py Global_View_4G.csv --out rsrp_sinr --split 2026-04-07
Outputs in --out: fit.csv (curve parameters), check.csv (held-out check), bands.png, networks.png, carriers.parquet
(the per-carrier table, reused on the next run with --cache).
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

QUANTILES = (0.1, 0.5, 0.9)
NETWORKS = {10: "O2", 15: "Vodafone", 20: "Three", 30: "EE", 33: "EE", 34: "EE"}          # UK MNCs (MCC 234)
NETWORK_COLOURS = {"EE": "#11917a", "O2": "#1f6fd1", "Three": "#a23aa8", "Vodafone": "#e25822"}
BANDS = [  # (name, downlink MHz from, to)
    ("700", 758, 790.5), ("800", 790.5, 822), ("900", 920, 961), ("1400", 1450, 1500),
    ("1800", 1800, 1881), ("2100", 2100, 2171), ("2600", 2600, 2691),
]
BIN_DB = 2.0
BEND_DB = 10.0                                                  # W: width of the hinge's bend (dB of RSRP)
PARAMS = ("floor_db", "knee_dbm", "slope_db_per_db")
BOUNDS = ([-30.0, -140.0, 0.0], [5.0, -60.0, 1.5])              # floor, knee, slope: physically sensible values
COLUMNS = ["train", "date", "time", "mcc", "mnc", "earfcn", "dlfreq", "rsrp", "sinr"]


# ---------------------------------------------------------------- reading
def read_carriers(csv: Path, chunksize: int = 2_000_000) -> pd.DataFrame:
    """One row per (second, train, network, carrier): the strongest cell on that carrier."""
    parts, n = [], 0
    for c in pd.read_csv(csv, encoding="utf-8-sig", index_col=False, usecols=COLUMNS, chunksize=chunksize,
                         dtype={"date": str, "time": str, "train": str}):
        n += len(c)
        for k in ("rsrp", "sinr", "mcc", "mnc", "dlfreq"):
            c[k] = pd.to_numeric(c[k], errors="coerce")
        c = c[c["rsrp"].between(-160, -20) & c["sinr"].between(-40, 60) & (c["mcc"] == 234)]   # 0 / -200: no reading
        c = c.assign(network=c["mnc"].map(NETWORKS)).dropna(subset=["network"])
        parts.append(_strongest_per_carrier(c))
    df = _strongest_per_carrier(pd.concat(parts))                                 # a second can straddle two chunks
    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df["band"] = band_of(df["dlfreq"].to_numpy())
    df = df.dropna(subset=["timestamp"]).drop(columns=["date", "time", "mcc", "mnc"]).reset_index(drop=True)
    print(f"{n:,} scanner rows -> {len(df):,} carrier-seconds")
    return df


def _strongest_per_carrier(c: pd.DataFrame) -> pd.DataFrame:
    return c.sort_values("rsrp", ascending=False).drop_duplicates(["date", "time", "train", "network", "earfcn"])


def band_of(dl_mhz: np.ndarray) -> np.ndarray:
    out = np.full(len(dl_mhz), "other", dtype=object)
    for name, lo, hi in BANDS:
        out[(dl_mhz >= lo) & (dl_mhz < hi)] = name
    return out


def per_network(carriers: pd.DataFrame) -> pd.DataFrame:
    """Per (second, train, network): RSRP and SINR of the strongest carrier, and the best SINR on any carrier."""
    key = ["timestamp", "train", "network"]
    strongest = carriers.sort_values("rsrp", ascending=False).drop_duplicates(key)[key + ["rsrp", "sinr"]]
    best = carriers.groupby(key, as_index=False)["sinr"].max().rename(columns={"sinr": "sinr_best"})
    return strongest.merge(best, on=key)


# ---------------------------------------------------------------- fitting
def hinge(rsrp, floor, knee, slope):
    """floor below the knee, rising at `slope` dB per dB above it, bending over about BEND_DB."""
    return floor + slope * BEND_DB * np.logaddexp(0.0, (np.asarray(rsrp, dtype=float) - knee) / BEND_DB)


def binned_quantiles(x: np.ndarray, y: np.ndarray, min_n: int) -> pd.DataFrame:
    b = np.floor(x / BIN_DB) * BIN_DB + BIN_DB / 2
    d = pd.DataFrame({"bin": b, "y": y})
    g = d.groupby("bin")["y"]
    q = g.quantile(list(QUANTILES)).unstack()
    q["n"] = g.size()
    return q[q["n"] >= min_n]


def fit_group(x: np.ndarray, y: np.ndarray, min_n: int) -> tuple[dict, pd.DataFrame]:
    """Hinge parameters per percentile, from the binned percentiles of SINR against RSRP."""
    q = binned_quantiles(x, y, min_n)
    params = {}
    if len(q) < 6:
        return params, q
    for p in QUANTILES:
        yq = q[p].to_numpy()
        lo, hi = np.array(BOUNDS[0]), np.array(BOUNDS[1])
        p0 = np.clip([yq[:3].mean(), -100.0, 0.5], lo + 1e-3, hi - 1e-3)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(hinge, q.index.to_numpy(), yq, p0=p0, sigma=1 / np.sqrt(q["n"].to_numpy()),
                                    bounds=BOUNDS, maxfev=20000)
        except RuntimeError:
            continue
        resid = yq - hinge(q.index.to_numpy(), *popt)
        params[p] = dict(zip(PARAMS, map(float, popt))) | {"fit_rms_db": float(np.sqrt(np.average(resid ** 2, weights=q["n"])))}
    return params, q


def check_group(params: dict, x: np.ndarray, y: np.ndarray, fit_median: float) -> dict:
    """How the curves hold up on readings they were not fitted on, against simply guessing the fit period's median."""
    out = {"readings": int(len(x)), "mae_without_rsrp_db": round(float(np.mean(np.abs(y - fit_median))), 2)}
    for p, v in params.items():
        out[f"below_p{int(p * 100)}"] = round(float(np.mean(y < hinge(x, **_args(v)))), 3)
    if 0.5 in params:
        out["mae_median_db"] = round(float(np.mean(np.abs(y - hinge(x, **_args(params[0.5]))))), 2)
    return out


def _args(v: dict) -> dict:
    return dict(zip(("floor", "knee", "slope"), (v[k] for k in PARAMS)))


# ---------------------------------------------------------------- charts
def plot(groups: list[tuple[str, str, dict, pd.DataFrame, pd.DataFrame]], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 4 if len(groups) > 4 else len(groups)
    rows = int(np.ceil(len(groups) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), sharex=True, sharey=True, squeeze=False)
    for ax, (name, colour, params, q, above) in zip(axes.flat, groups):
        xs = np.linspace(q.index.min(), q.index.max(), 200) if len(q) else np.array([])
        if {0.1, 0.9} <= params.keys():
            ax.fill_between(xs, hinge(xs, **_args(params[0.1])), hinge(xs, **_args(params[0.9])), color=colour, alpha=0.15,
                            linewidth=0, label="10th-90th percentile")
        if 0.5 in params:
            ax.plot(xs, hinge(xs, **_args(params[0.5])), color=colour, linewidth=2, label="median (fit)")
        ax.scatter(q.index, q[0.5], s=10, color=colour, zorder=3, label="median per 2 dB (data)")
        ax.scatter(q.index, q[0.1], s=6, color=colour, alpha=0.45, zorder=3)
        ax.scatter(q.index, q[0.9], s=6, color=colour, alpha=0.45, zorder=3)
        if len(above):
            ax.scatter(above.index, above[0.5], s=12, facecolors="none", edgecolors=colour, zorder=3, label="median above the fitted range")
        ax.set_title(name, fontsize=10)
        ax.grid(True, color="#dddddd", linewidth=0.6)
        ax.axhline(0, color="#999999", linewidth=0.8)
    for ax in axes.flat[len(groups):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("RSRP (dBm, Global View level)")
    for ax in axes[:, 0]:
        ax.set_ylabel("SINR (dB)")
    axes.flat[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", type=Path, help="Global View 4G CSV")
    ap.add_argument("--out", type=Path, default=Path("rsrp_sinr"), help="output folder")
    ap.add_argument("--split", default="2026-04-07", help="fit on readings before this date, check on the rest")
    ap.add_argument("--min-bin", type=int, default=200, help="fewest readings in a 2 dB bin for it to count")
    ap.add_argument("--rsrp-range", default="-135,-65", help="RSRP range fitted and checked (dBm); above it SINR depends on the place")
    ap.add_argument("--cache", action="store_true", help="reuse OUT/carriers.parquet from an earlier run")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cache = a.out / "carriers.parquet"
    carriers = pd.read_parquet(cache) if a.cache and cache.exists() else read_carriers(a.csv)
    if not (a.cache and cache.exists()):
        carriers.to_parquet(cache, index=False)
    split = pd.Timestamp(a.split)
    lo, hi = (float(v) for v in a.rsrp_range.split(","))
    print(f"readings {carriers['timestamp'].min():%d %b %Y} - {carriers['timestamp'].max():%d %b %Y}; fit before {split:%d %b %Y}, check after")

    fits, checks, band_plots, net_plots = [], [], [], []

    def run(d: pd.DataFrame, col: str, group: str, sinr_of: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
        fit, test = d[d["timestamp"] < split], d[d["timestamp"] >= split]
        fin, tin = fit[fit["rsrp"].between(lo, hi)], test[test["rsrp"].between(lo, hi)]
        params, q = fit_group(fin["rsrp"].to_numpy(), fin[col].to_numpy(), a.min_bin)
        check = check_group(params, tin["rsrp"].to_numpy(), tin[col].to_numpy(), float(fin[col].median()))
        up = test[test["rsrp"] > hi]                                # not fitted: said how many, and what they show
        check |= {"share_above_range": round(len(up) / max(len(test), 1), 3),
                  "median_sinr_above_range_db": round(float(up[col].median()), 1) if len(up) else None}
        _collect(fits, checks, group, sinr_of, params, len(fin), check)
        above = binned_quantiles(fit.loc[fit["rsrp"] > hi, "rsrp"].to_numpy(), fit.loc[fit["rsrp"] > hi, col].to_numpy(), a.min_bin)
        return params, q, above

    for band, _, _ in BANDS:                                        # per band, networks pooled
        params, q, above = run(carriers[carriers["band"] == band], "sinr", f"band {band} MHz", "carrier")
        band_plots.append((f"{band} MHz", "#3b5b7a", params, q, above))

    nets = per_network(carriers)
    for net in sorted(NETWORK_COLOURS):
        for col, what in (("sinr", "strongest carrier"), ("sinr_best", "best carrier")):
            params, q, above = run(nets[nets["network"] == net], col, net, what)
            if col == "sinr_best":
                net_plots.append((f"{net}: best carrier's SINR", NETWORK_COLOURS[net], params, q, above))

    pd.DataFrame(fits).to_csv(a.out / "fit.csv", index=False)
    chk = pd.DataFrame(checks)
    chk.to_csv(a.out / "check.csv", index=False)
    plot(band_plots, a.out / "bands.png", "SINR against RSRP per band, Global View 4G (strongest cell on each carrier)")
    plot(net_plots, a.out / "networks.png", "Best SINR on any of the network's carriers against its strongest carrier's RSRP")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(pd.DataFrame(fits).query("percentile == 50").round(1).to_string(index=False))
        print(chk.to_string(index=False))
    print(f"done in {time.time() - t0:.0f}s -> {a.out}/")


def _collect(fits: list, checks: list, group: str, sinr_of: str, params: dict, n_fit: int, check: dict) -> None:
    for p, v in params.items():
        fits.append({"group": group, "sinr_of": sinr_of, "percentile": int(p * 100), **{k: round(x, 2) for k, x in v.items()}, "fit_readings": n_fit})
    checks.append({"group": group, "sinr_of": sinr_of, **check})


if __name__ == "__main__":
    main()
