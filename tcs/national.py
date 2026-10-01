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
CUTTING_PENALTY_MAX = [0.0, 0.05, 0.1, 0.14, 0.2, 0.28, 0.36, 0.44, 0.52, 0.6]
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


CHECKS = ("current_check", "five_g")      # sections `tcs check-national` adds; re-fitting the calibration keeps them
HEADER = ("# National calibration of the cellular model, written by `tcs calibrate-national` (and its checks against later\n"
          "# measurements by `tcs check-national`): re-run them rather than editing. Parameters and accuracy figures derived\n"
          "# from the measurements; the measurements themselves are not kept here.\n"
          "# `environment` holds the rail-environment terms fitted with it: config/simulation.yaml must carry the same values.\n")


def write(doc: dict, out: Path, source: str, file: str, preset: str) -> dict:
    """config/calibration.yaml: provenance first, then the parameters and accuracy national.run() produced."""
    doc = {"source": source, "file": file, "preset": preset, "fitted_on": datetime.date.today().isoformat(), **doc}
    prev = yaml.safe_load(Path(out).read_text(encoding="utf-8")) if Path(out).exists() else {}
    for k in CHECKS:
        if k in (prev or {}) and k not in doc:
            doc[k] = prev[k]
    _dump(doc, out)
    return doc


def save_checks(out: Path, **sections) -> dict:
    """Add or replace check sections in an existing calibration file; the calibration itself is left as it is."""
    doc = yaml.safe_load(Path(out).read_text(encoding="utf-8"))
    doc.update(sections)
    _dump(doc, out)
    return doc


def _dump(doc: dict, out: Path) -> None:
    Path(out).write_text(HEADER + yaml.safe_dump(_plain(doc), sort_keys=False, allow_unicode=True, width=140), encoding="utf-8")


def check(route_ids: list[str], path, preset: str, split: str, max_distance_m: float = 50.0, log=print) -> dict:
    """The model as calibrated, against a later measurement set whose overall level may differ from the calibration
    data's (a scanner that logs signal before correcting for its antenna and cable). One level offset is fitted on the
    measurements before `split`; accuracy is then measured on those from `split`, per route, network and setting."""
    cut = pd.Timestamp(split, tz="UTC")
    routes, skipped = collect(route_ids, path, preset, cut, max_distance_m=max_distance_m, log=log)
    if not routes:
        raise ValueError("no route has measurements to check against")
    names = {op["id"]: op["name"] for op in routes[0].settings.operators}
    j = pd.concat([predict(r, environment_of(r.settings), calibration.for_route(r.settings, r.settings.paths()["interim"])) for r in routes])
    fit, test = j[j["period"] == "fit"], j[j["period"] == "test"]
    if fit.empty or test.empty:
        raise ValueError(f"measurements on only one side of {cut.date()}: nothing to fit the level on, or nothing to test")
    offset = float(np.median(fit["rsrp_dbm"] - fit["signal_primary"]))
    test = test.assign(signal_primary=test["signal_primary"] + offset)
    b = breakdown(test, names)
    log(f"level offset {offset:+.1f} dB; then {b['overall']['points']:,} test points, bias {b['overall']['bias_db']:+.1f} dB, MAE {b['overall']['mae_db']:.1f} dB")
    return {"preset": preset, "fit_period": _period(routes, "fit"), "test_period": _period(routes, "test"), "split": str(cut.date()),
            "level_offset_db": round(offset, 1), "max_distance_m": max_distance_m, "skipped": skipped, **b,
            "routes": {rid: accuracy(g) for rid, g in test.groupby("route_id")}}


# NR-ARFCN -> MHz (3GPP 38.104) and the band each falls in, for saying which 5G a scanner measured.
NR_BANDS = [(758, 803, "700 MHz"), (925, 960, "900 MHz"), (1805, 1880, "1800 MHz"), (2110, 2170, "2.1 GHz"),
            (2620, 2690, "2.6 GHz"), (3300, 3800, "3.4-3.8 GHz")]


def nr_mhz(arfcn: float) -> float:
    return arfcn * 0.005 if arfcn < 600000 else 3000 + (arfcn - 600000) * 0.015


def five_g(route_ids: list[str], path_4g, path_5g, preset_4g: str = "global_view_4g", preset_5g: str = "global_view_5g",
           usable_dbm: float = -110.0, max_distance_m: float = 50.0, log=print) -> dict:
    """Where 5G was measured along each route: per network, the share of the route points passed by a train carrying
    the 5G scanner (known from its 4G log, which records every point it passes) at which that network's 5G reached
    usable_dbm (SS-RSRP, median of the passes)."""
    out, pooled, chans, days = {}, {}, set(), []
    for rid in route_ids:
        s = load_settings(route_id=rid)
        interim = s.paths()["interim"]
        if not (interim / "coverage_prior.parquet").exists():
            continue
        b = load_bundle(interim, s.route["country"])
        if b.geometry_source != "osm":
            continue
        near, box = near_route(b.samples, b.proj, max_distance_m), route_bbox(b.samples)
        g5 = attach_to_route(load_measurements(path_5g, preset_5g, s.operators, bbox=box, near=near), b.samples, b.proj, max_distance_m=max_distance_m)
        g4 = attach_to_route(load_measurements(path_4g, preset_4g, s.operators, bbox=box, near=near), b.samples, b.proj, max_distance_m=max_distance_m)
        straight = straight_samples(s, b.samples, b.provenance.get("straight_legs") or [])
        g4 = g4[g4["device"].isin(set(g5["device"])) & ~g4["sample_id"].isin(straight)]
        passed = set(g4["sample_id"])
        if not passed:
            continue
        g5 = g5[g5["sample_id"].isin(passed)]
        chans |= set(g5["channel"].dropna().astype(int))
        days += [g4["timestamp"].min(), g4["timestamp"].max()]
        nets = {}
        for op in s.operators:
            best = g5[g5["provider_id"] == op["id"]].groupby("sample_id")["rsrp_dbm"].median()
            nets[op["name"]] = round(float((best >= usable_dbm).sum()) / len(passed), 3)
            pooled.setdefault(op["name"], [0, 0])
            pooled[op["name"]][0] += int((best >= usable_dbm).sum())
            pooled[op["name"]][1] += len(passed)
        out[rid] = {"points": len(passed), "networks": nets}
        log(f"{rid}: 5G usable at " + ", ".join(f"{k} {v:.0%}" for k, v in nets.items()) + f" of {len(passed):,} route points")
    mhz = sorted({round(nr_mhz(c)) for c in chans})
    bands = [name for lo, hi, name in NR_BANDS if any(lo <= f <= hi for f in mhz)]
    return {"preset": preset_5g, "usable_dbm": usable_dbm, "bands": bands,
            "unmeasured_bands": [name for _, _, name in NR_BANDS if name not in bands],
            "period": {"from": str(min(days).date()), "to": str(max(days).date())} if days else {},
            "networks": {k: round(a / n, 3) for k, (a, n) in pooled.items() if n}, "routes": out}


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
