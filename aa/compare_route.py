"""The EDGE Rail active antenna against passive rack-router installs, along one route (aa/METHOD.md sections 0-2).

First cut, before the Trainlab throughput logs and the real passive installs' details arrive: the link-efficiency
curve, carrier bandwidths, cell load and cable runs are the stated assumptions in ASSUME below. They are the same for
every profile, so the comparison isolates what differs between the installs: receive branches, layers, the cable
ahead of the modem and how many carriers the modem can aggregate.

Signal and interference per carrier come from Network Rail Global View 4G readings where the scanner passed (the
strongest cell on each carrier, median over passes), and elsewhere from the simulator's calibrated signal for each
network plus the fitted SINR-against-RSRP curves (scripts/fit_rsrp_sinr.py). 5G (n78) is not in this run: the
scanner does not measure 3.4-3.8 GHz.

    python aa/compare_route.py tpe_man_ncl --gv <Global_View_4G.csv or carriers parquet> --sinr-fit <fit.csv>

Writes data/aa/compare/<route>/ (git-ignored): samples.parquet, summary.csv, breakdown.csv and a chart.
"""
from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ALL_NETWORKS = ("ee", "o2", "three", "vodafone")
NETWORKS = ("ee", "vodafone", "three")   # the EDGE Rail preset's three units; every profile's networks by default
PROFILES = {      # aa/METHOD.md section 0; `bonding` names the combining product, whose efficiency is in ASSUME
    "A": {"label": "EDGE Rail active antenna", "branches": 4, "layers": 4, "cable": False, "carriers": 5,
          "bonding": "Fleet Connect", "networks": NETWORKS},
    "P4": {"label": "Rack router, 4x4 per modem", "branches": 4, "layers": 4, "cable": True, "carriers": 5,
           "bonding": "router", "networks": NETWORKS},
    "P2": {"label": "Rack router, 2x2 per modem", "branches": 2, "layers": 2, "cable": True, "carriers": 3,
           "bonding": "router", "networks": NETWORKS},
}
ASSUME = {
    "noise_figure_db": 7.0,            # modem and scanner alike
    "branch_correlation": 0.3,         # rho between receive branches (ports facing different sides keep it low)
    "alpha": 0.6, "sinr_min_db": -10.0, "eta_max": 5.5,   # attenuated Shannon, 256-QAM capable modems (3GPP TR 36.942)
    "network_layers": 2,               # UK LTE cells: mostly 2 transmit antennas
    "cell_share": 0.5,                 # share of each cell's capacity the train gets
    # share of the summed link capacity the train gets after combining the networks. PLACEHOLDERS: neither Motion
    # Applied (Fleet Connect) nor Icomera (SureWAN) / Nomad (Nomad Connect) publish a figure. Equal, so the central
    # result isolates the antennas; SCENARIOS sweeps them.
    "bonding": {"Fleet Connect": 0.85, "router": 0.85},
    "cable_m": 10.0, "fittings_db": 1.0,                  # passive installs: LMR-400-class coax + connectors/protection
    "bandwidth_mhz": {"700": 10, "800": 10, "900": 10, "1400": 20, "1800": 20, "2100": 15, "2600": 20},
}
BAND_MHZ = {"700": 773, "800": 806, "900": 942, "1400": 1472, "1800": 1842, "2100": 2140, "2600": 2655}
NO_SERVICE_MBPS = 2.0

# Sensitivity of the comparison to how the networks are combined (placeholder values until there is data): each
# scenario sets the bonding efficiency per product and the networks the rack router carries.
SCENARIOS = {
    "central (both 0.85)": {"bonding": {"Fleet Connect": 0.85, "router": 0.85}, "router_networks": NETWORKS},
    "Fleet Connect 0.95, router 0.75": {"bonding": {"Fleet Connect": 0.95, "router": 0.75}, "router_networks": NETWORKS},
    "Fleet Connect 0.75, router 0.95": {"bonding": {"Fleet Connect": 0.75, "router": 0.95}, "router_networks": NETWORKS},
    "router adds O2 (4 networks vs 3)": {"bonding": {"Fleet Connect": 0.85, "router": 0.85}, "router_networks": ALL_NETWORKS},
}


# ---------------------------------------------------------------- link maths (no data involved)
def noise_dbm(nf_db: float) -> float:
    """Thermal noise in one 15 kHz resource element, plus the receiver's noise figure."""
    return -174.0 + 10 * math.log10(15e3) + nf_db


def cable_loss_db(mhz, metres: float, fittings_db: float) -> np.ndarray:
    """LMR-400-class coax: 0.128 dB/m at 900 MHz, 0.223 at 2500 (skin and dielectric loss: a*sqrt(f) + b*f)."""
    f = np.array([900.0, 2500.0])
    a, b = np.linalg.solve(np.c_[np.sqrt(f), f], [0.128, 0.223])
    mhz = np.asarray(mhz, dtype=float)
    return metres * (a * np.sqrt(mhz) + b * mhz) + fittings_db


def eta(sinr_lin, alpha: float, sinr_min_db: float, eta_max: float) -> np.ndarray:
    """Spectral efficiency (bit/s/Hz per layer): attenuated Shannon, zero below sinr_min, capped at eta_max."""
    e = np.minimum(alpha * np.log2(1 + sinr_lin), eta_max)
    return np.where(sinr_lin < 10 ** (sinr_min_db / 10), 0.0, e)


def _gamma_nodes(shape: float, n: int = 24):
    from scipy.special import gamma, roots_genlaguerre

    x, w = roots_genlaguerre(n, shape - 1)
    return x / shape, w / gamma(shape)      # E[f(X)] = sum w f(x) for X ~ Gamma(shape, mean 1)


INTERFERENCE = {
    # how combining several receive branches treats interference; the truth lies between the two
    "like noise": "interference arrives independently on each branch, so combining lifts the signal over it as over noise",
    "correlated": "interference arrives as correlated across the branches as the signal, so combining lifts the signal over noise only",
}


def carrier_mbps(s, i, n, bandwidth_mhz, branches: int, layers: int, a: dict, interference: str = "like noise") -> np.ndarray:
    """Expected throughput of one carrier (Mbit/s) from per-branch signal, interference and noise (linear, same units):
    Rayleigh fading on each branch, combined over `branches`, with the best number of layers r for the conditions.
    Each layer gets 1/r of the signal and the combining gain of m = M - r + 1 branches (diversity order m, shrunk by
    branch correlation)."""
    s, i, n = (np.asarray(v, dtype=float) for v in (s, i, n))
    best = np.zeros_like(s)
    rho = a["branch_correlation"]
    for r in range(1, layers + 1):
        m = branches - r + 1
        x, w = _gamma_nodes(m / (1 + (m - 1) * rho))
        mean = (s / r) * m / (i + n) if interference == "like noise" else (s / r) / (i + n / m)
        e = (eta(mean[..., None] * x, a["alpha"], a["sinr_min_db"], a["eta_max"]) * w).sum(axis=-1)
        best = np.maximum(best, r * e)
    return np.asarray(bandwidth_mhz, dtype=float) * a["cell_share"] * best


def link_terms(rsrp_dbm, sinr_db, gv_offset_db: float, cable_db, a: dict):
    """Per-branch signal, interference and noise (linear) at the modem of an install with `cable_db` ahead of it, and
    the interference-to-noise ratio (dB) at the antenna, from a scanner reading (RSRP and SINR as logged, before the
    scanner's antenna and cable corrections; gv_offset_db brings its level to the antenna's)."""
    n = 10 ** (noise_dbm(a["noise_figure_db"]) / 10)
    s = 10 ** (np.asarray(rsrp_dbm, dtype=float) / 10)
    i = np.maximum(s / 10 ** (np.asarray(sinr_db, dtype=float) / 10) - n, 1e-3 * n)   # interference at the scanner
    k = 10 ** (gv_offset_db / 10)
    s, i = s * k, i * k                                                                  # at the roof antenna
    return s, i, n * 10 ** (np.asarray(cable_db, dtype=float) / 10), 10 * np.log10(i / n)


# ---------------------------------------------------------------- inputs
def read_gv_table(path: Path) -> pd.DataFrame:
    """Global View 4G readings, the strongest cell per (second, train, network, carrier): from the scanner's CSV, or
    from a parquet of such rows (e.g. a previous read)."""
    if path.suffix == ".parquet":
        c = pd.read_parquet(path).rename(columns={"Latitude": "latitude", "Longitude": "longitude", "net": "network", "timestamp": "ts"})
        return c[[k for k in ("ts", "train", "network", "band", "rsrp", "sinr", "latitude", "longitude") if k in c]]
    nets = {10: "o2", 15: "vodafone", 20: "three", 30: "ee", 33: "ee", 34: "ee"}
    parts = []
    cols = ["train", "date", "time", "mcc", "mnc", "earfcn", "dlfreq", "rsrp", "sinr", "Latitude", "Longitude"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ch in pd.read_csv(path, encoding="utf-8-sig", index_col=False, usecols=cols, chunksize=2_000_000,
                              dtype={"date": str, "time": str, "train": str}):
            for k in ("rsrp", "sinr", "mcc", "mnc", "dlfreq", "Latitude", "Longitude"):
                ch[k] = pd.to_numeric(ch[k], errors="coerce")
            ch = ch[ch.rsrp.between(-160, -20) & ch.sinr.between(-40, 60) & (ch.mcc == 234)]
            ch = ch.assign(network=ch.mnc.map(nets)).dropna(subset=["network"])
            parts.append(ch.sort_values("rsrp", ascending=False).drop_duplicates(["date", "time", "train", "network", "earfcn"]))
    c = pd.concat(parts).rename(columns={"Latitude": "latitude", "Longitude": "longitude"})
    c["ts"] = pd.to_datetime(c.date + " " + c.time, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    f = c.dlfreq
    c["band"] = np.select([f < 790.5, f < 822, (f > 920) & (f < 961), (f > 1450) & (f < 1500), (f > 1800) & (f < 1881),
                           (f > 2100) & (f < 2171), (f > 2600) & (f < 2691)], ["700", "800", "900", "1400", "1800", "2100", "2600"], "other")
    return c[["ts", "train", "network", "band", "rsrp", "sinr", "latitude", "longitude"]]


def national_bands(table: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Per (network, band), over every Global View reading: how far below the network's strongest carrier the band
    typically sits (dB), and the share of seconds it is heard. Stands in on routes the scanner barely covered."""
    t = table[table.band != "other"]
    key = ["ts", "train", "network"] if {"ts", "train"} <= set(t.columns) else ["latitude", "longitude", "network"]
    top = t.groupby(key).rsrp.transform("max")
    delta = (t.rsrp - top).groupby([t.network, t.band]).median()
    seconds = t.drop_duplicates(key).groupby("network").size()
    share = t.drop_duplicates(key + ["band"]).groupby(["network", "band"]).size() / seconds
    return delta, share


def load_gv(table: pd.DataFrame, samples: pd.DataFrame, proj, cache: Path) -> pd.DataFrame:
    """The Global View readings within 50 m of the route, each on its nearest sample."""
    from tcs.sources.measurements import attach_to_route

    if cache.exists():
        return pd.read_parquet(cache)
    m = attach_to_route(table[["network", "band", "rsrp", "sinr", "latitude", "longitude"]], samples, proj, max_distance_m=50)
    cache.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(cache, index=False)
    return m


def hinge(rsrp, floor, knee, slope, bend=10.0):
    return floor + slope * bend * np.logaddexp(0.0, (np.asarray(rsrp, dtype=float) - knee) / bend)


def sinr_curves(sinr_fit: Path) -> pd.DataFrame:
    """Median SINR-against-RSRP hinge per band, from scripts/fit_rsrp_sinr.py's fit.csv."""
    fit = pd.read_csv(sinr_fit)
    fit = fit[(fit.percentile == 50) & fit.group.str.startswith("band ")].assign(band=lambda f: f.group.str.extract(r"band (\d+)")[0])
    return fit.set_index("band")[["floor_db", "knee_dbm", "slope_db_per_db"]]


MIN_MEASURED_SAMPLES = 200        # a network's own band mix on a route needs this many measured samples; else national


def carriers_along(route: str, table: pd.DataFrame, curve: pd.DataFrame, national: tuple[pd.Series, pd.Series],
                   out: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """One row per (sample, network, band) a train would use: RSRP and SINR at Global View level, measured where the
    scanner passed and modelled elsewhere."""
    import yaml

    from tcs.config import ROOT, load_settings
    from tcs.model import calibration
    from tcs.model.simulate import simulate
    from tcs.pipeline.sample_route import load_bundle

    s = load_settings(route_id=route)
    it = s.paths()["interim"]
    b = load_bundle(it, s.route["country"])
    samples = b.samples
    offset = -float((yaml.safe_load(open(ROOT / "config" / "calibration.yaml")).get("current_check") or {}).get("level_offset_db", -7.1))
    obs, _ = simulate(s, samples, pd.read_parquet(it / "coverage_prior.parquet"), pd.read_parquet(it / "serving.parquet"),
                      calibration=calibration.for_route(s, it))
    meas = load_gv(table, samples, b.proj, out / "gv_readings.parquet")
    meas = meas[meas.band != "other"]
    measured = meas.groupby(["sample_id", "network", "band"], as_index=False).agg(rsrp=("rsrp", "median"), sinr=("sinr", "median"), n=("rsrp", "size"))
    measured["source"] = "measured"

    # where the scanner did not pass: the simulator's signal for the network, spread over the bands that network
    # uses on this route (each band's typical offset from the strongest; nationally where the route has too few
    # readings), SINR from the fitted median curve
    top = measured.groupby(["sample_id", "network"]).rsrp.transform("max")
    delta = (measured.rsrp - top).groupby([measured.network, measured.band]).median()
    share = measured.groupby(["network", "band"]).sample_id.nunique() / measured.groupby("network").sample_id.nunique()
    enough = measured.groupby("network").sample_id.nunique()
    own = {n for n in ALL_NETWORKS if enough.get(n, 0) >= MIN_MEASURED_SAMPLES}
    n_delta, n_share = national
    delta = pd.concat([delta[delta.index.get_level_values(0).isin(own)], n_delta[~n_delta.index.get_level_values(0).isin(own)]])
    share = pd.concat([share[share.index.get_level_values(0).isin(own)], n_share[~n_share.index.get_level_values(0).isin(own)]])
    model = obs[obs.provider_id.isin(ALL_NETWORKS)][["sample_id", "provider_id", "signal_primary", "available"]].rename(columns={"provider_id": "network"})
    have = set(zip(measured.sample_id, measured.network))
    model = model[[(a, n) not in have for a, n in zip(model.sample_id, model.network)]]
    rows = []
    for (net, band), sh in share.items():
        if net not in ALL_NETWORKS or sh < 0.2 or band not in curve.index:
            continue
        mm = model[model.network == net]
        rsrp = mm.signal_primary.to_numpy() - offset + delta[(net, band)]
        rows.append(pd.DataFrame({"sample_id": mm.sample_id.to_numpy(), "network": net, "band": band, "rsrp": rsrp,
                                  "sinr": hinge(rsrp, *curve.loc[band]), "n": 0, "source": "modelled",
                                  "available": mm.available.to_numpy()}))
    modelled = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["sample_id", "network", "band", "rsrp", "sinr", "n", "source", "available"])
    modelled = modelled[modelled.available.astype(bool)].drop(columns="available")   # the simulator's own no-service (tunnels)
    car = pd.concat([measured[measured.network.isin(ALL_NETWORKS)], modelled], ignore_index=True)

    # how far the modelled SINR is from the measured where both exist (the fallback's own error)
    chk = measured.merge(obs[["sample_id", "provider_id", "signal_primary"]].rename(columns={"provider_id": "network"}), on=["sample_id", "network"])
    chk = chk[chk.band.isin(curve.index)]
    meta = {"gv_offset_db": offset, "fallback_sinr_mae_db": float("nan"), "fallback_sinr_bias_db": float("nan"),
            "own_band_mix": sorted(own)}
    if len(chk):
        chk_r = chk.signal_primary.to_numpy() - offset + np.array([delta.get((n, bd), 0.0) for n, bd in zip(chk.network, chk.band)])
        chk_s = hinge(chk_r, *[curve.loc[chk.band, c].to_numpy() for c in ("floor_db", "knee_dbm", "slope_db_per_db")])
        meta |= {"fallback_sinr_mae_db": float(np.mean(np.abs(chk_s - chk.sinr))), "fallback_sinr_bias_db": float(np.median(chk_s - chk.sinr))}
    return car, samples, meta


# ---------------------------------------------------------------- the comparison
def compare(car: pd.DataFrame, samples: pd.DataFrame, gv_offset_db: float, a: dict = ASSUME, interference: str = "like noise") -> pd.DataFrame:
    """Per sample: throughput per network and for the train, for every profile."""
    car = car.copy()
    car["bw"] = car.band.map(a["bandwidth_mhz"]).astype(float)
    cable = cable_loss_db(car.band.map(BAND_MHZ), a["cable_m"], a["fittings_db"])
    for k, p in PROFILES.items():
        s_, i_, n_, inr = link_terms(car.rsrp, car.sinr, gv_offset_db, cable if p["cable"] else 0.0, a)
        car[f"T_{k}"] = carrier_mbps(s_, i_, n_, car.bw, p["branches"], min(p["layers"], a["network_layers"]), a, interference)
        car["inr_db"] = inr
    out = samples[["sample_id", "distance_m", "sim_seconds", "in_tunnel", "cutting_depth_m"]].copy()
    for k, p in PROFILES.items():
        top = car.sort_values(f"T_{k}", ascending=False).groupby(["sample_id", "network"]).head(p["carriers"])
        per = top.groupby(["sample_id", "network"])[f"T_{k}"].sum().unstack(fill_value=0.0)
        per = per.reindex(index=out.sample_id, columns=list(ALL_NETWORKS), fill_value=0.0)
        for n in ALL_NETWORKS:
            out[f"{k}_{n}"] = per[n].to_numpy()
        out[f"T_{k}"] = train_mbps(out, k, a["bonding"][p["bonding"]], p["networks"])
    src = car.groupby("sample_id").source.agg(lambda s: "measured" if (s == "measured").any() else "modelled")
    out["source"] = out.sample_id.map(src).fillna("none")
    out["inr_db"] = out.sample_id.map(car[car.source == "measured"].groupby("sample_id").inr_db.median())
    t = out.sim_seconds.to_numpy()
    out["dt_s"] = np.gradient(t) if len(t) > 1 else 0.0
    return out


def train_mbps(df: pd.DataFrame, profile: str, bonding: float, networks) -> np.ndarray:
    """The train's throughput: the networks the install carries, combined at the given efficiency."""
    return bonding * df[[f"{profile}_{n}" for n in networks]].sum(axis=1).to_numpy()


def with_scenario(df: pd.DataFrame, scenario: dict) -> pd.DataFrame:
    """The same per-network results, combined as a SCENARIOS entry says."""
    out = df.copy()
    for k, p in PROFILES.items():
        nets = p["networks"] if p["bonding"] == "Fleet Connect" else scenario["router_networks"]
        out[f"T_{k}"] = train_mbps(out, k, scenario["bonding"][p["bonding"]], nets)
    return out


def _wq(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(np.interp(q * c[-1], c, x[o]))


def kpis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    w = df.dt_s.to_numpy()
    for k, p in PROFILES.items():
        t = df[f"T_{k}"].to_numpy()
        dead = t < NO_SERVICE_MBPS
        runs, cur = [], 0.0
        for d, dt in zip(dead, w):
            if d:
                cur += dt
            elif cur:
                runs.append(cur)
                cur = 0.0
        if cur:
            runs.append(cur)
        rows.append({"profile": k, "install": p["label"],
                     "mean_mbps": float(np.average(t, weights=w)), "p10_mbps": _wq(t, w, 0.1), "p50_mbps": _wq(t, w, 0.5),
                     "share_time_10mbps": float(w[t >= 10].sum() / w.sum()), "share_time_50mbps": float(w[t >= 50].sum() / w.sum()),
                     "no_service_minutes": float(sum(runs) / 60), "no_service_spells": len(runs),
                     "longest_no_service_min": float(max(runs, default=0) / 60)})
    k = pd.DataFrame(rows).set_index("profile")
    for p in ("P2", "P4"):
        r = df.T_A / df[f"T_{p}"].where(df[f"T_{p}"] > 0)
        ok = r.notna()
        k.loc[p, "uplift_median_x"] = _wq(r[ok].to_numpy(), w[ok.to_numpy()], 0.5)
        k.loc[p, "uplift_p10_x"] = k.loc["A", "p10_mbps"] / max(k.loc[p, "p10_mbps"], 1e-9)
    return k


def breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Where the uplift comes from: by setting and by interference regime (time-weighted mean throughput)."""
    setting = np.where(df.in_tunnel, "tunnel", np.where(df.cutting_depth_m >= 4, "cutting >= 4 m", "open"))
    regime = pd.cut(df.inr_db, [-np.inf, 0, 15, np.inf], labels=["noise-limited (INR < 0 dB)", "mixed (0-15 dB)", "interference-limited (> 15 dB)"])
    out = []
    for name, key in (("setting", setting), ("regime", regime.astype(str).where(df.inr_db.notna(), "no reading")), ("source", df.source)):
        for v, g in df.groupby(key):
            w = g.dt_s
            row = {"by": name, "group": v, "share_of_time": float(w.sum() / df.dt_s.sum())}
            for k in PROFILES:
                row[f"mean_{k}_mbps"] = float(np.average(g[f"T_{k}"], weights=w))
            row["A_vs_P2_x"] = row["mean_A_mbps"] / max(row["mean_P2_mbps"], 1e-9)
            row["A_vs_P4_x"] = row["mean_A_mbps"] / max(row["mean_P4_mbps"], 1e-9)
            out.append(row)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- chart
COLOURS = {"A": "#2a78d6", "P4": "#eb6834", "P2": "#1baf7a"}     # dataviz categorical slots 1-3 (validated)


def chart(cases: dict[str, pd.DataFrame], stations: pd.DataFrame, title: str, path: Path) -> None:
    """Throughput per profile along the route (interference treated like noise), the active antenna's advantage over
    each passive install (shaded down to the case of correlated interference), and the measured interference above
    noise."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df, lo = cases["like noise"], cases["correlated"]
    km = df.distance_m.to_numpy() / 1000
    roll = lambda s: s.rolling(21, center=True, min_periods=5).median()       # ~1 km
    fig, ax = plt.subplots(3, 1, figsize=(13, 9.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.3, 1.2]})
    ink, muted, grid = "#222222", "#6b6b6b", "#e4e4e0"
    for a_ in ax:
        a_.grid(True, axis="y", color=grid, linewidth=0.6)
        a_.spines[["top", "right"]].set_visible(False)
        a_.tick_params(colors=muted, labelsize=8)
        unmeasured = (df.source != "measured").to_numpy()
        a_.fill_between(km, 0, 1, where=unmeasured, transform=a_.get_xaxis_transform(), color="#f1efe8", linewidth=0, zorder=0)
        a_.fill_between(km, 0, 1, where=df.in_tunnel.to_numpy(), transform=a_.get_xaxis_transform(), color="#d9d6cc", linewidth=0, zorder=0)
    # A and P4 nearly coincide: A drawn wide underneath, P4 thin on top, so both stay visible
    for k, ls, lw in (("A", "-", 4.0), ("P4", "-", 1.3), ("P2", "--", 1.6)):
        y = roll(df[f"T_{k}"])
        ax[0].plot(km, y, color=COLOURS[k], linewidth=lw, linestyle=ls, alpha=0.55 if k == "A" else 1.0)
        last = y.dropna().iloc[-1]
        ax[0].annotate(f"{k}  {PROFILES[k]['label']}", (km[-1], last), xytext=(6, {"A": 9, "P4": -3, "P2": -12}[k]), textcoords="offset points",
                       fontsize=8, color=ink, va="center")
    ax[0].set_ylabel("Train throughput, Mbit/s\n(1 km median)", color=ink, fontsize=9)
    ax[0].set_ylim(bottom=0)
    for k in ("P2", "P4"):
        r = roll(df.T_A) / roll(df[f"T_{k}"])
        ax[1].fill_between(km, roll(lo.T_A) / roll(lo[f"T_{k}"]), r, color=COLOURS[k], alpha=0.14, linewidth=0)
        ax[1].plot(km, r, color=COLOURS[k], linewidth=1.6, linestyle="--" if k == "P2" else "-")
        ax[1].annotate(f"A vs {k}", (km[-1], r.dropna().iloc[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8, color=ink, va="center")
    ax[1].axhline(1, color=muted, linewidth=0.8)
    ax[1].set_ylabel("Active antenna's\nadvantage (x)", color=ink, fontsize=9)
    ax[2].plot(km, df.inr_db.rolling(11, center=True, min_periods=1).median(), color="#4a3aa7", linewidth=1.2)
    ax[2].axhline(0, color=muted, linewidth=0.8)
    ax[2].annotate("0 dB: noise-limited below", (km[0], 0), xytext=(2, 4), textcoords="offset points", fontsize=7.5, color=muted)
    ax[2].set_ylabel("Interference above\nnoise (dB, measured)", color=ink, fontsize=9)
    ax[2].set_xlabel("km from origin", color=ink, fontsize=9)
    top = ax[0].secondary_xaxis("top")
    top.set_xticks(stations.distance_m.to_numpy() / 1000, stations.name.str.replace(" Piccadilly", "", regex=False).tolist(), fontsize=7.5, rotation=30, ha="left")
    top.tick_params(colors=muted)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=ink)
    fig.text(0.01, 0.005, "Throughput: interference treated like noise by antenna combining. Middle panel, shaded down to: interference as correlated across "
             "the antennas as the signal (the pessimistic end). "
             "Grey: no Global View readings (modelled); darker grey: tunnels.\n"
             "Assumptions as aa/compare_route.py ASSUME: 50 % of each cell, 2-layer LTE cells, 10 m coax + 1 dB on passive installs; 4G only (no n78).",
             fontsize=7, color=muted)
    fig.tight_layout(rect=(0, 0.035, 0.9, 1))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_route(route: str, table: pd.DataFrame, curve: pd.DataFrame, national, out: Path) -> tuple[list[dict], list[dict]]:
    """The comparison on one route: per-sample results, KPIs, breakdown, sensitivity and chart in `out`; one summary row
    per interference case, and one sensitivity row per case and SCENARIOS entry."""
    from tcs.config import load_settings

    s = load_settings(route_id=route)
    out.mkdir(parents=True, exist_ok=True)
    car, samples, meta = carriers_along(route, table, curve, national, out)
    cases = {c: compare(car, samples, meta["gv_offset_db"], interference=c) for c in INTERFERENCE}
    pd.concat([d.assign(case=c) for c, d in cases.items()]).to_parquet(out / "samples.parquet", index=False)
    k = pd.concat({c: kpis(d) for c, d in cases.items()}, names=["interference"])
    k.round(2).to_csv(out / "summary.csv")
    pd.concat([breakdown(d).assign(interference=c) for c, d in cases.items()]).round(3).to_csv(out / "breakdown.csv", index=False)
    stations = pd.read_parquet(s.paths()["processed"] / "stations.parquet")
    chart(cases, stations, f"{s.route['name']}: EDGE Rail active antenna vs passive rack-router installs", out / f"compare_{route}.png")
    df = cases["like noise"]
    rows = []
    for c, d in cases.items():
        kk = k.loc[c]
        rows.append({"route": route, "name": s.route["name"], "km": round(float(d.distance_m.max()) / 1000, 1), "interference": c,
                     "share_time_measured": float(d.dt_s[d.source == "measured"].sum() / d.dt_s.sum()),
                     "share_time_tunnel": float(d.dt_s[d.in_tunnel].sum() / d.dt_s.sum()),
                     "fallback_sinr_mae_db": meta["fallback_sinr_mae_db"],
                     **{f"{m}_{pr}": float(kk.loc[pr, m]) for pr in PROFILES for m in ("p50_mbps", "p10_mbps", "share_time_10mbps", "no_service_minutes")},
                     "A_vs_P2_median_x": float(kk.loc["P2", "uplift_median_x"]), "A_vs_P4_median_x": float(kk.loc["P4", "uplift_median_x"])})
    sens = []
    for c, d in cases.items():
        for name, scen in SCENARIOS.items():
            ks = kpis(with_scenario(d, scen))
            sens.append({"route": route, "name": s.route["name"], "interference": c, "scenario": name,
                         **{f"p50_mbps_{pr}": float(ks.loc[pr, "p50_mbps"]) for pr in PROFILES},
                         **{f"no_service_minutes_{pr}": float(ks.loc[pr, "no_service_minutes"]) for pr in PROFILES},
                         **{f"A_vs_{pr}_median_x": float(ks.loc[pr, "uplift_median_x"]) for pr in ("P2", "P4")},
                         **{f"A_vs_{pr}_p10_x": float(ks.loc[pr, "uplift_p10_x"]) for pr in ("P2", "P4")}})
    pd.DataFrame(sens).round(3).to_csv(out / "sensitivity.csv", index=False)
    print(f"{route}: {rows[0]['share_time_measured']:.0%} measured; A vs P2 {rows[1]['A_vs_P2_median_x']:.2f}-{rows[0]['A_vs_P2_median_x']:.2f}x, "
          f"A vs P4 {rows[1]['A_vs_P4_median_x']:.2f}-{rows[0]['A_vs_P4_median_x']:.2f}x; median A {rows[0]['p50_mbps_A']:.0f} Mbit/s"
          + ("" if len(df) else " (no samples)"))
    return rows, sens


def chart_routes(summary: pd.DataFrame, path: Path) -> None:
    """Every route: typical train throughput per install, and the active antenna's advantage as a range."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hi = summary[summary.interference == "like noise"].set_index("route")
    lo = summary[summary.interference == "correlated"].set_index("route")
    order = hi.sort_values("A_vs_P2_median_x").index
    hi, lo = hi.loc[order], lo.loc[order]
    y = np.arange(len(order))
    ink, muted, grid = "#222222", "#6b6b6b", "#e4e4e0"
    fig, ax = plt.subplots(1, 2, figsize=(13, 0.42 * len(order) + 2.2), sharey=True, gridspec_kw={"width_ratios": [1.2, 1]})
    for a_ in ax:
        a_.grid(True, axis="x", color=grid, linewidth=0.6)
        a_.spines[["top", "right"]].set_visible(False)
        a_.tick_params(colors=muted, labelsize=8)
    for k, size, marker in (("A", 90, "o"), ("P4", 22, "o"), ("P2", 40, "D")):
        ax[0].scatter(hi[f"p50_mbps_{k}"], y, s=size, color=COLOURS[k], marker=marker, zorder=3,
                      facecolors="none" if k == "A" else COLOURS[k], linewidths=1.6 if k == "A" else 0, label=f"{k}  {PROFILES[k]['label']}")
    ax[0].set_title("Typical train throughput (median over the journey)", loc="left", fontsize=9.5, color=ink)
    ax[0].set_xlabel("Mbit/s", color=muted, fontsize=8)
    ax[0].set_xlim(left=0)
    ax[0].legend(fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.04), ncol=3, handletextpad=0.3, columnspacing=1.2)
    labels = [f"{n}  ·  {m:.0%} measured" for n, m in zip(hi.name, hi.share_time_measured)]
    ax[0].set_yticks(y, labels, fontsize=8, color=ink)
    for k, off in (("P2", 0.13), ("P4", -0.13)):
        a, b = lo[f"A_vs_{k}_median_x"].to_numpy(), hi[f"A_vs_{k}_median_x"].to_numpy()
        ax[1].hlines(y + off, np.minimum(a, b), np.maximum(a, b), color=COLOURS[k], linewidth=5, alpha=0.85)
        ax[1].scatter(b, y + off, s=14, color=COLOURS[k], zorder=3)
        ax[1].annotate(f"vs {k}", (b[-1], y[-1] + off), xytext=(6, 0), textcoords="offset points", fontsize=8, color=ink, va="center")
    ax[1].axvline(1, color=muted, linewidth=0.8)
    ax[1].set_title("Active antenna's advantage (median)", loc="left", fontsize=9.5, color=ink)
    ax[1].set_xlabel("x  (bar: interference correlated across antennas → treated like noise)", color=muted, fontsize=8)
    fig.suptitle("EDGE Rail active antenna vs passive rack-router installs, every route (4G, first cut)", x=0.01, ha="left", fontsize=12, color=ink)
    fig.text(0.01, 0.005, "Throughput: interference treated like noise. Assumptions as aa/compare_route.py ASSUME (the same for every install). "
             "Unmeasured stretches modelled from the simulator.", fontsize=7, color=muted)
    fig.tight_layout(rect=(0, 0.02, 0.97, 0.97))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def chart_sensitivity(sens: pd.DataFrame, path: Path) -> None:
    """How far the active antenna's advantage moves with the combining assumptions: per scenario, the spread over
    routes for each comparison, with the median route in each interference case."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(SCENARIOS)
    ink, muted, grid = "#222222", "#6b6b6b", "#e4e4e0"
    fig, ax = plt.subplots(figsize=(11, 1.15 * len(names) + 1.8))
    ax.grid(True, axis="x", color=grid, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=muted, labelsize=8)
    for i, name in enumerate(names[::-1]):
        g = sens[sens.scenario == name]
        for k, off in (("P2", 0.18), ("P4", -0.18)):
            col = f"A_vs_{k}_median_x"
            y = i + off
            lo, hi = g[col].min(), g[col].max()
            ax.hlines(y, lo, hi, color=COLOURS[k], linewidth=6, alpha=0.35)
            for case, face in (("correlated", "none"), ("like noise", COLOURS[k])):
                v = g.loc[g.interference == case, col].median()
                ax.scatter(v, y, s=46, facecolors=face, edgecolors=COLOURS[k], linewidths=1.6, zorder=3)
            ax.annotate(f"vs {k}  {lo:.2f}-{hi:.2f}x", (hi, y), xytext=(6, 0), textcoords="offset points", fontsize=8, color=ink, va="center")
    ax.axvline(1, color=muted, linewidth=0.8)
    ax.set_yticks(range(len(names)), names[::-1], fontsize=8.5, color=ink)
    ax.set_xlabel("Active antenna's advantage, median over the journey (x)", color=ink, fontsize=9)
    fig.suptitle("Sensitivity to how the networks are combined (placeholder bonding values)", x=0.01, ha="left", fontsize=12, color=ink)
    fig.text(0.01, 0.01, "Bars: spread over all routes and both interference cases. Dots: median route; hollow = interference correlated across "
             "antennas, filled = treated like noise. vs P2: 2x2 rack router; vs P4: 4x4 rack router.", fontsize=7, color=muted)
    fig.tight_layout(rect=(0, 0.04, 0.93, 0.95))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("route", help="route id, or 'all'")
    ap.add_argument("--gv", type=Path, required=True, help="Global View 4G CSV, or a parquet of its per-carrier readings")
    ap.add_argument("--sinr-fit", type=Path, required=True, help="fit.csv from scripts/fit_rsrp_sinr.py")
    ap.add_argument("--out", type=Path, default=None, help="default data/aa/compare")
    a = ap.parse_args()
    from tcs.config import ROOT, list_routes

    root = a.out or ROOT / "data" / "aa" / "compare"
    table = read_gv_table(a.gv)
    national = national_bands(table)
    curve = sinr_curves(a.sinr_fit)
    ids = [r["id"] for r in list_routes()] if a.route == "all" else [a.route]
    rows, sens = [], []
    for rid in ids:
        r, se = run_route(rid, table, curve, national, root / rid)
        rows += r
        sens += se
    summary, sens = pd.DataFrame(rows), pd.DataFrame(sens)
    if a.route == "all":
        summary.round(3).to_csv(root / "summary_all_routes.csv", index=False)
        sens.round(3).to_csv(root / "sensitivity_all_routes.csv", index=False)
        chart_routes(summary, root / "compare_all_routes.png")
        chart_sensitivity(sens, root / "compare_sensitivity.png")
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        print(sens.groupby(["scenario", "interference"], sort=False)[["A_vs_P2_median_x", "A_vs_P4_median_x", "p50_mbps_A", "p50_mbps_P2",
                                                                       "p50_mbps_P4"]].agg(["min", "median", "max"]).round(2).to_string())
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        cols = ["name", "interference", "share_time_measured", "p50_mbps_A", "p50_mbps_P4", "p50_mbps_P2", "p10_mbps_A", "p10_mbps_P2",
                "A_vs_P2_median_x", "A_vs_P4_median_x", "no_service_minutes_A", "no_service_minutes_P2"]
        print(summary[cols].round(2).to_string(index=False))
    print(f"-> {root}/")


if __name__ == "__main__":
    main()
