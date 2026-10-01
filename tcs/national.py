"""National calibration: the cellular model fitted to field measurements pooled over every built route.

Measurements before `split` fit the model and those from `split` on test it, so the accuracy it reports is out of
sample. The fit first chooses the rail environment terms (how much signal a cutting costs, how far it carries into a
tunnel), then each network's score bias (calibration.fit, keeping the model's dB scale) on open-air points. A second test fits
on every route but one and tests on the one left out, which is how the calibration fares on a route it has never seen.

`tcs calibrate-national` writes the result to config/calibration.yaml: parameters, provenance and accuracy per route,
network and setting. Only these derived figures are written; the measurements themselves stay where they are.
"""
from __future__ import annotations

import copy
import datetime
import itertools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import Settings, load_settings
from .model import calibration
from .model.cellular import cellular_observations
from .pipeline.sample_route import load_bundle
from .sources.measurements import attach_to_route, load_measurements, near_route, route_bbox

USABLE_DBM = -110.0                 # LTE signal a modem can hold a data session on; the agreement figure uses it
CUTTING_M = 4.0                     # deeper than this counts as a cutting in the accuracy breakdown

# Environment terms tried (every combination); the defaults in simulation.yaml sit inside each range.
CUTTING_PENALTY_MAX = [0.14, 0.2, 0.28, 0.36, 0.44, 0.52, 0.6]
CUTTING_FULL_DEPTH_M = [3, 4, 6, 8, 10, 15, 20, 30]
PORTAL_DECAY_M = [None, 50, 100, 150, 200, 300, 450, 700]
# Below the score at which a data session holds (cellular.throughput.score_floor): deep inside a tunnel without in-tunnel
# coverage there is no usable service. The measurements barely tell floors around that threshold apart (Yellow Train
# 2018-19: within 0.02 dB of absolute error from 0.10 to 0.14), so a tie never lights a long tunnel.
TUNNEL_FLOOR = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]


@dataclass
class RouteData:
    settings: Settings
    samples: pd.DataFrame
    prior: pd.DataFrame
    serving: pd.DataFrame
    points: pd.DataFrame                   # sample_id, provider_id, period (fit | test), rsrp_dbm (median of the passes), n
    measurements: dict = field(default_factory=dict)   # per period: rows matched to the route, first and last day


def collect(route_ids: list[str], path, preset: str, split: pd.Timestamp, max_distance_m: float = 50.0,
            log=print) -> tuple[list[RouteData], dict[str, str]]:
    """Each built route with its measurements, reduced to one median per route point, network and period. Routes drawn
    from approximate (non-OSM) track geometry, and legs of an OSM route drawn straight, are left out: a measurement
    would land on the wrong route point."""
    routes, skipped = [], {}
    for rid in route_ids:
        s = load_settings(route_id=rid)
        interim = s.paths()["interim"]
        if not (interim / "coverage_prior.parquet").exists():
            skipped[rid] = "not built"
            continue
        b = load_bundle(interim, s.route["country"])
        if b.geometry_source != "osm":
            skipped[rid] = f"approximate track geometry ({b.geometry_source})"
            continue
        m = load_measurements(path, preset, s.operators, bbox=route_bbox(b.samples), near=near_route(b.samples, b.proj, max_distance_m))
        m = attach_to_route(m, b.samples, b.proj, max_distance_m=max_distance_m)
        m = m[~m["sample_id"].isin(straight_samples(s, b.samples, b.provenance.get("straight_legs") or []))]
        m = m[m["provider_id"].notna() & np.isfinite(m["rsrp_dbm"]) & m["timestamp"].notna()]
        if m.empty:
            skipped[rid] = "no measurements along the route"
            continue
        m = m.assign(period=np.where(m["timestamp"] < split, "fit", "test"))
        pts = m.groupby(["sample_id", "provider_id", "period"], as_index=False).agg(rsrp_dbm=("rsrp_dbm", "median"), n=("rsrp_dbm", "size"))
        per = {p: {"rows": int(len(g)), "first": str(g["timestamp"].min().date()), "last": str(g["timestamp"].max().date())}
               for p, g in m.groupby("period")}
        routes.append(RouteData(s, b.samples, pd.read_parquet(interim / "coverage_prior.parquet"), pd.read_parquet(interim / "serving.parquet"), pts, per))
        log(f"{rid}: {len(m):,} measurements within {max_distance_m:.0f} m -> {len(pts):,} route points x network x period")
    return routes, skipped


def straight_samples(s: Settings, samples: pd.DataFrame, legs: list[str]) -> set:
    """Route points on legs drawn as a straight line between two stations (OSM had no rail path there)."""
    if not legs:
        return set()
    st = pd.read_parquet(s.paths()["processed"] / "stations.parquet").set_index("crs")["distance_m"]
    out = set()
    for leg in legs:
        a, b = (st.get(x) for x in str(leg).split("-", 1))
        if a is not None and b is not None:
            lo, hi = sorted((float(a), float(b)))
            out |= set(samples.loc[samples["distance_m"].between(lo, hi), "sample_id"])
    return out


def with_environment(s: Settings, env: dict) -> Settings:
    out = copy.copy(s)
    out.sim = copy.deepcopy(s.sim)
    c = out.sim["cellular"]
    c["terrain"]["cutting_penalty_max"] = env["cutting_penalty_max"]
    c["terrain"]["cutting_full_depth_m"] = env["cutting_full_depth_m"]
    c["tunnels"]["portal_decay_m"] = env["portal_decay_m"]
    c["tunnels"]["default_score"] = env["tunnel_floor"]
    return out


def environment_of(s: Settings) -> dict:
    c = s.sim["cellular"]
    return {"cutting_penalty_max": c["terrain"]["cutting_penalty_max"], "cutting_full_depth_m": c["terrain"]["cutting_full_depth_m"],
            "portal_decay_m": c["tunnels"].get("portal_decay_m"), "tunnel_floor": c["tunnels"]["default_score"]}


def predict(r: RouteData, env: dict, cal: dict | None) -> pd.DataFrame:
    """The model's RSRP at each measured route point, with what the accuracy breakdowns need."""
    obs = cellular_observations(with_environment(r.settings, env), r.samples, r.prior, r.serving, calibration=cal)
    obs = obs[["sample_id", "provider_id", "quality_score", "signal_primary", "_in_tunnel", "reason_code"]]
    j = r.points.merge(obs, on=["sample_id", "provider_id"])
    j = j.merge(r.samples[["sample_id", "distance_m", "cutting_depth_m"]], on="sample_id")
    j["route_id"] = r.settings.route_id
    return j


def _pooled(routes: list[RouteData], env: dict, period: str, cal: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Model rows and measurement rows over several routes, sample ids made unique across them (for calibration.fit)."""
    obs, meas = [], []
    for k, r in enumerate(routes):
        o = cellular_observations(with_environment(r.settings, env), r.samples, r.prior, r.serving, calibration=cal)
        o = o.merge(r.samples[["sample_id", "distance_m"]], on="sample_id")
        o["sample_id"] = o["sample_id"] + k * 10_000_000
        p = r.points[r.points["period"] == period].copy()
        p["sample_id"] = p["sample_id"] + k * 10_000_000
        obs.append(o)
        meas.append(p.assign(kind="point"))
    return pd.concat(obs, ignore_index=True), pd.concat(meas, ignore_index=True)


def fit_networks(routes: list[RouteData], env: dict) -> dict:
    s = routes[0].settings.sim["cellular"]["rsrp_dbm"]
    obs, meas = _pooled(routes, env, "fit")
    # The model's dB scale is kept (fit_scale=False): only each network's score bias moves, so the -110 dBm usable line
    # and the signal bands in reports keep their meaning (see calibration.fit).
    return calibration.fit(obs, meas, nominal_rsrp=(float(s["at_zero"]), float(s["at_one"])), sections=False, fit_scale=False)


def accuracy(j: pd.DataFrame) -> dict:
    """Predicted minus measured RSRP over route points (each point once, however many passes measured it)."""
    if j.empty:
        return {"points": 0}
    err = (j["signal_primary"] - j["rsrp_dbm"]).to_numpy(dtype=float)
    pred, meas = j["signal_primary"].to_numpy(dtype=float), j["rsrp_dbm"].to_numpy(dtype=float)
    corr = float(np.corrcoef(pred, meas)[0, 1]) if len(j) > 2 and np.std(pred) > 0 and np.std(meas) > 0 else None
    return {"points": int(len(j)), "bias_db": round(float(err.mean()), 2), "mae_db": round(float(np.abs(err).mean()), 2),
            "rmse_db": round(float(np.sqrt((err ** 2).mean())), 2), "within_6db": round(float((np.abs(err) <= 6).mean()), 3),
            "correlation": round(corr, 3) if corr is not None else None,
            "usable_agreement": round(float(((pred >= USABLE_DBM) == (meas >= USABLE_DBM)).mean()), 3),
            "measured_median_dbm": round(float(np.median(meas)), 1), "predicted_median_dbm": round(float(np.median(pred)), 1)}


def setting(j: pd.DataFrame) -> pd.Series:
    return pd.Series(np.where(j["_in_tunnel"], "tunnel", np.where(j["cutting_depth_m"] > CUTTING_M, "cutting", "open")), index=j.index)


def breakdown(j: pd.DataFrame, names: dict[str, str]) -> dict:
    env = setting(j)
    return {"overall": accuracy(j),
            "networks": {names.get(p, p): accuracy(g) for p, g in j.groupby("provider_id")},
            "settings": {e: accuracy(j[env == e]) for e in ("open", "cutting", "tunnel") if (env == e).any()}}


def _abs_error(j: pd.DataFrame) -> float:
    """Total absolute error: the median-unbiased choice. Signal inside tunnels and cuttings is skewed (a few points with
    a line of sight, or placed a little off), and squared error would chase that tail."""
    return float((j["signal_primary"] - j["rsrp_dbm"]).abs().sum())


def tunnel_coverage(routes: list[RouteData], deep_m: float = 250.0, min_points: int = 30, min_networks: int = 3,
                    threshold_dbm: float = -90.0) -> list[dict]:
    """Tunnels whose interior (more than deep_m from either portal) was measured at threshold_dbm or better (median):
    signal that cannot come from outside, i.e. in-tunnel coverage. "listed": the route's das_tunnels already has it."""
    from .model.cellular import das_mask, portals

    out = []
    for r in routes:
        smp = r.samples.sort_values("distance_m").reset_index(drop=True)
        tun = smp["in_tunnel"].fillna(False).to_numpy(dtype=bool)
        _, _, d_before, d_after = portals(tun, smp["distance_m"].to_numpy(dtype=float))
        das = das_mask(set(r.settings.sim["cellular"]["tunnels"].get("das_tunnels") or []), smp["tunnel_name"].to_numpy(), smp["distance_m"].to_numpy() / 1000)
        smp = smp.assign(deep=tun & (np.fmin(d_before, d_after) > deep_m), das=das, run=np.cumsum(np.r_[True, tun[1:] != tun[:-1]]))
        p = r.points.merge(smp.loc[smp["deep"], ["sample_id", "run", "tunnel_name", "distance_m"]], on="sample_id")
        for _, g in p.groupby("run"):
            if len(g) >= min_points and g["provider_id"].nunique() >= min_networks and g["rsrp_dbm"].median() >= threshold_dbm:
                run = smp[smp["run"] == g["run"].iloc[0]]
                name = g["tunnel_name"].dropna()
                out.append({"route": r.settings.route_id, "tunnel": str(name.mode().iloc[0]) if len(name) else None,
                            "km": [round(float(run["distance_m"].min()) / 1000, 1), round(float(run["distance_m"].max()) / 1000, 1)],
                            "points": int(len(g)), "median_dbm": round(float(g["rsrp_dbm"].median()), 1), "listed": bool(run["das"].all())})
    return out


def search_environment(routes: list[RouteData], base: dict, log=print) -> tuple[dict, dict]:
    """The environment terms that best fit the fit-period points, each network's calibration refitted for every try.
    Cuttings are chosen on open-air points, then the tunnel terms on tunnel points (they do not affect each other)."""
    best = None
    for pmax, depth in itertools.product(CUTTING_PENALTY_MAX, CUTTING_FULL_DEPTH_M):
        env = dict(base, cutting_penalty_max=pmax, cutting_full_depth_m=depth)
        cal = fit_networks(routes, env)
        j = pd.concat([predict(r, env, cal) for r in routes])
        j = j[(j["period"] == "fit") & ~j["_in_tunnel"]]
        err = _abs_error(j)
        if best is None or err < best[0]:
            best = (err, env, cal)
    _, env, cal = best
    log(f"cuttings: penalty {env['cutting_penalty_max']} of score at {env['cutting_full_depth_m']} m deep and more")
    best = None
    usable = float(routes[0].settings.sim["cellular"]["throughput"]["score_floor"])
    for decay, floor in itertools.product(PORTAL_DECAY_M, [f for f in TUNNEL_FLOOR if f < usable]):
        e = dict(env, portal_decay_m=decay, tunnel_floor=floor)
        j = pd.concat([predict(r, e, cal) for r in routes])
        j = j[(j["period"] == "fit") & (j["reason_code"] == "TUNNEL")]   # tunnels without in-tunnel coverage
        if j.empty:
            break
        err = _abs_error(j)
        if best is None or err < best[0]:
            best = (err, e)
    if best is not None:
        env = best[1]
    log(f"tunnels: signal from the portals falls off over {env['portal_decay_m']} m to a floor of {env['tunnel_floor']}")
    return env, cal


def run(route_ids: list[str], path, preset: str, split: str, max_distance_m: float = 50.0, log=print) -> dict:
    """The whole national fit and test; returns the document for config/calibration.yaml."""
    cut = pd.Timestamp(split, tz="UTC")
    routes, skipped = collect(route_ids, path, preset, cut, max_distance_m=max_distance_m, log=log)
    if not routes:
        raise ValueError("no route has measurements to calibrate against")
    names = {op["id"]: op["name"] for op in routes[0].settings.operators}
    base = environment_of(load_settings())
    env, cal = search_environment(routes, base, log=log)

    test = pd.concat([predict(r, env, cal) for r in routes])
    test = test[test["period"] == "test"]
    before = pd.concat([predict(r, base, None) for r in routes])       # the model as it was, for comparison
    before = before[before["period"] == "test"]
    per_route = {}
    for k, r in enumerate(routes):
        rid = r.settings.route_id
        others = [x for i, x in enumerate(routes) if i != k]
        held = accuracy(predict(r, env, fit_networks(others, env)).query("period == 'test'")) if others else None
        per_route[rid] = {"calibrated": breakdown(test[test["route_id"] == rid], names)["overall"],
                          "uncalibrated": accuracy(before[before["route_id"] == rid]),
                          "held_out": held,
                          "settings": breakdown(test[test["route_id"] == rid], names)["settings"],
                          "measurements": {p: v["rows"] for p, v in r.measurements.items()}}
    all_ts = pd.concat([r.points for r in routes])
    held_rows = [v["held_out"] for v in per_route.values() if v["held_out"] and v["held_out"].get("points")]
    doc = {
        "calibration": {k: cal[k] for k in ("version", "section_km", "rsrp_input", "bias", "rsrp_map", "residual_std", "sections", "n_points")},
        "environment": env,
        "fit_period": _period(routes, "fit"),
        "test_period": _period(routes, "test"),
        "split": str(cut.date()),
        "max_distance_m": max_distance_m,
        "routes": [r.settings.route_id for r in routes],
        "skipped": skipped,
        # measured signal deep inside these tunnels can only come from in-tunnel coverage: list them in the route's das_tunnels
        "in_tunnel_coverage": tunnel_coverage(routes),
        "points": {"fit": int((all_ts["period"] == "fit").sum()), "test": int((all_ts["period"] == "test").sum())},
        "validation": {
            "calibrated": breakdown(test, names),
            "uncalibrated": breakdown(before, names),
            "held_out_routes": {"routes": len(held_rows), "mae_db_median": round(float(np.median([h["mae_db"] for h in held_rows])), 2) if held_rows else None,
                                "bias_db_median": round(float(np.median([h["bias_db"] for h in held_rows])), 2) if held_rows else None},
            "routes": per_route,
        },
    }
    return doc


def _period(routes: list[RouteData], period: str) -> dict:
    days = [r.measurements[period] for r in routes if period in r.measurements]
    return {"from": min(d["first"] for d in days), "to": max(d["last"] for d in days)} if days else {}


def write(doc: dict, out: Path, source: str, file: str, preset: str) -> dict:
    """config/calibration.yaml: provenance first, then the parameters and accuracy national.run() produced."""
    doc = {"source": source, "file": file, "preset": preset, "fitted_on": datetime.date.today().isoformat(), **doc}
    header = ("# National calibration of the cellular model, written by `tcs calibrate-national`: re-run it rather than editing.\n"
              "# Parameters and accuracy figures derived from the measurements; the measurements themselves are not kept here.\n"
              "# `environment` holds the rail-environment terms fitted with it: config/simulation.yaml must carry the same values.\n")
    Path(out).write_text(header + yaml.safe_dump(_plain(doc), sort_keys=False, allow_unicode=True, width=140), encoding="utf-8")
    return doc


def _plain(v):
    """numpy scalars and tuples -> plain YAML types; floats to 4 significant decimals so the file stays readable."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return round(float(v), 4)
    return v
