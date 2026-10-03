"""Command line: `tcs run --offline` builds everything; individual steps are exposed for iteration."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from .config import ROOT, list_routes, load_settings

app = typer.Typer(add_completion=False, help="Train Route 3D Connectivity Simulation pipeline")
console = Console()


def _interim(settings) -> Path:
    return settings.paths()["interim"]


def _require_built(settings) -> None:
    if not (_interim(settings) / "samples.parquet").exists():
        raise typer.BadParameter(f"route '{settings.route_id}' has not been built yet: run `tcs run --route {settings.route_id}` first")


@app.command()
def run(offline: bool = typer.Option(False, help="Use cached/synthetic sources only (no network, no keys)"),
        route: str | None = typer.Option(None, help="Route id from config/routes (default: config/route.yaml)"),
        weather: str = typer.Option("nominal"), policy: str | None = typer.Option(None), limit_postcodes: int | None = typer.Option(None, help="Dev: cap Ofcom API calls"),
        copy_to_web: bool = typer.Option(True, help="Copy the web bundle into web/public/data/"),
        train: Path | None = typer.Option(None, help="Train Studio project (*.train.json) describing the onboard architecture"),
        preset: str | None = typer.Option(None, help="Scenario preset from simulation.yaml `presets` (e.g. edge_rail_fleet_connect)"),
        no_satcom: bool = typer.Option(False, "--no-satcom", help="Model the train without its satellite link")):
    """Full pipeline: route -> terrain -> coverage -> cells -> movement -> simulate -> export."""
    run_pipeline(offline=offline, route=route, weather=weather, policy=policy, limit_postcodes=limit_postcodes, copy_to_web=copy_to_web, train=train, preset=preset,
                 satcom=not no_satcom)


def run_pipeline(offline: bool = False, route: str | None = None, weather: str = "nominal", policy: str | None = None, limit_postcodes: int | None = None,
                 copy_to_web: bool = True, train: Path | None = None, preset: str | None = None, satcom: bool = True):
    from .model import calibration
    from .model.simulate import simulate
    from .pipeline.export import export_all
    from .pipeline.join_cells import candidate_cells, corridor_cells, serving_cells
    from .pipeline.join_coverage import coverage_prior
    from .pipeline.movement import movement
    from .pipeline.obstruction import enrich_terrain
    from .pipeline.sample_route import build_route, save_bundle

    s = load_settings(offline=offline, route_id=route)
    interim = _interim(s)
    console.rule(f"[bold]{s.route['name']} · {s.route['origin_crs']} → {s.route['destination_crs']}")
    if preset:
        pr = _apply_preset(s, preset)
        console.log(f"preset: {pr.get('label', preset)}")
    design_path = train or (Path(s.sim.get("train", {}).get("design_file")) if s.sim.get("train", {}).get("design_file") else None)
    if design_path:
        from .sources.train_studio import apply_to_settings, load_design

        design = load_design(design_path, s)
        apply_to_settings(s, design)
        console.log(f"train design '{design.title}': {len(design.carriages)} carriages · {design.cellular_units} cellular units · "
                    f"{design.satcom_units} satcom ({design.satcom_terminal or '-'}) · {design.aps_connected}/{design.aps_total} APs connected · "
                    f"{design.passengers} seats · policy {design.policy}")
        for w in design.warnings:
            console.log(f"[yellow]{w}")
    _check_overrides(s, policy, weather)
    if policy:
        s.sim["wan"]["policy"] = policy                        # recorded in meta.json: the viewer opens on the scenario simulated here
    if weather != "nominal":
        s.sim["weather"] = weather                             # absent for nominal, so a plain run's meta.sim is simulation.yaml itself
    if not satcom:
        s.sim["satcom_enabled"] = False                        # recorded in meta.json like the policy: the viewer opens without it
    b = build_route(s)
    console.log(f"route: {len(b.samples)} samples at {s.spacing_m:.0f} m, {b.samples['distance_m'].max() / 1000:.1f} km, geometry={b.geometry_source}")
    for w in b.warnings:
        console.log(f"[yellow]{w}")
    b.samples = enrich_terrain(s, b)
    prior, b.samples = coverage_prior(s, b, limit=limit_postcodes)
    cells = corridor_cells(s, b, b.samples)
    cand = candidate_cells(s, b.samples, cells)
    serving = serving_cells(s, b.samples, cand)
    b.samples, stations = movement(s, b.samples, b.stations)
    b.stations = stations
    save_bundle(b, interim)
    prior.to_parquet(interim / "coverage_prior.parquet", index=False)
    cells.to_parquet(interim / "cells.parquet", index=False)
    serving.to_parquet(interim / "serving.parquet", index=False)
    cal = calibration.for_route(s, interim)
    obs, rc = simulate(s, b.samples, prior, serving, calibration=cal, weather=weather, policy=policy)
    written = export_all(s, b, b.samples, stations, cells, obs, rc, prior)
    if copy_to_web:
        dest = ROOT / "web" / "public" / "data" / s.route_id
        dest.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(dest / "reports", ignore_errors=True)   # evidence packs describe the previous build; `tcs report-all` rebuilds them
        for f in ("route.arrow", "cells.arrow", "meta.json"):
            shutil.copy(s.paths()["web"] / f, dest / f)
        console.log(f"web bundle -> {dest}")
        write_index()
    _summary(rc, obs)
    for k, v in written.items():
        console.log(f"[dim]{k}[/dim] {v}")


def _apply_preset(s, preset: str) -> dict:
    """Apply a named scenario preset from simulation.yaml to the settings; returns the preset."""
    pr = s.sim.get("presets", {}).get(preset)
    if pr is None:
        raise typer.BadParameter(f"unknown preset '{preset}'; available: {', '.join(s.sim.get('presets', {}))}")
    if pr.get("vehicle_profile"):
        s.sim["vehicle"]["profile"] = pr["vehicle_profile"]
    if pr.get("policy"):
        s.sim["wan"]["policy"] = pr["policy"]
    if pr.get("bonding_efficiency"):
        s.sim["wan"]["bonding_efficiency"] = float(pr["bonding_efficiency"])
    if "satcom_enabled" in pr:
        s.sim["satcom_enabled"] = bool(pr["satcom_enabled"])
    if "fitted_networks" in pr:                                # e.g. three EDGE Rail antennas, one network each
        s.sim["cellular"]["fitted_networks"] = None if pr["fitted_networks"] is None else list(pr["fitted_networks"])
    s.sim["active_preset"] = preset
    return pr


def _check_overrides(s, policy: str | None, weather: str) -> None:
    """An unknown policy or weather used to run silently with the default one while being labelled as asked."""
    from .model.bonding import POLICIES

    if policy and policy not in POLICIES:
        raise typer.BadParameter(f"unknown policy '{policy}'; choose from {', '.join(POLICIES)}")
    known = {w for p in s.starlink["satcom"]["providers"] for w in p["availability"]["weather"]} or {"nominal"}
    if weather not in known:
        raise typer.BadParameter(f"unknown weather '{weather}'; choose from {', '.join(sorted(known))}")


def _summary(rc: pd.DataFrame, obs: pd.DataFrame) -> None:
    t = Table(title="Predicted onboard Wi-Fi by service class (share of route)")
    t.add_column("class"), t.add_column("share", justify="right")
    for cls, share in rc["service_class"].value_counts(normalize=True).sort_index().items():
        t.add_row(cls, f"{share:.1%}")
    console.print(t)
    t2 = Table(title="Per-link availability / median capacity")
    t2.add_column("link"), t2.add_column("available", justify="right"), t2.add_column("p50 Mbps", justify="right"), t2.add_column("mean confidence", justify="right")
    for pid, g in obs.groupby("provider_id"):
        t2.add_row(pid, f"{g['available'].mean():.1%}", f"{g.loc[g['available'], 'capacity_mbps'].median():.0f}", f"{g['confidence'].mean():.2f}")
    console.print(t2)


def write_index() -> Path:
    """Catalogue of built bundles for the viewer's route picker: web/public/data/index.json."""
    root = ROOT / "web" / "public" / "data"
    items = []
    for meta in sorted(root.glob("*/meta.json")):
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        st = m.get("stations", [])
        items.append({"id": m["route"]["id"], "name": m["route"].get("name"), "operator": m["route"].get("operator"), "origin": st[0]["name"] if st else None,
                      "destination": st[-1]["name"] if st else None, "length_km": round(m.get("length_m", 0) / 1000, 1), "duration_min": round(m.get("duration_s", 0) / 60),
                      "geometry_source": m.get("geometry_source"), "terrain_source": m.get("terrain_source"), "n_samples": m.get("n_samples")})
    out = root / "index.json"
    out.write_text(json.dumps({"routes": items}, indent=1), encoding="utf-8")
    return out


@app.command()
def serve(port: int = typer.Option(8000, help="Preferred port (the next free one is used if taken)"),
          built: bool = typer.Option(False, help="Serve web/dist (a production build) instead of the source bundles"),
          open_browser: bool = typer.Option(True, "--open/--no-open")):
    """Serve the viewer locally over HTTP (the page fetches Arrow bundles, so file:// will not work)."""
    from .serve import serve as _serve

    root = ROOT / "web" / ("dist" if built else "public")
    if built and not (root / "index.html").exists():
        raise typer.BadParameter("web/dist is empty - run `tcs package --no-zip` or `npm run build` in web/ first")
    if not built:
        # Source mode needs the built JS too; without it, tell the user rather than serving a blank page.
        if not (root / "index.html").exists():
            raise typer.BadParameter("serve --built, or run `npm run dev` in web/ for live editing")
    console.log(f"serving {root}")
    url, _ = _serve(root, port=port, open_browser=open_browser)
    console.log(f"[bold]{url}[/bold]  (Ctrl+C to stop)")


@app.command()
def package(out: Path | None = typer.Option(None, help="Output folder (default: dist/)"),
            zip_it: bool = typer.Option(True, "--zip/--no-zip"), reports: bool = typer.Option(True, "--reports/--no-reports", help="Keep the evidence packs the viewer's Report button offers"),
            skip_build: bool = typer.Option(False, help="Reuse an existing web/dist instead of running npm build")):
    """Package the viewer, the Train Builder and every built route into a zip that runs with nothing installed."""
    from .package import package as _package

    _package(out_dir=out, include_reports=reports, zip_it=zip_it, skip_build=skip_build)


@app.command()
def routes():
    """List the route catalogue (config/routes) and which ones have a built bundle."""
    built = {p.parent.name for p in (ROOT / "web" / "public" / "data").glob("*/meta.json")}
    t = Table(title="Routes")
    t.add_column("id"), t.add_column("name"), t.add_column("from → to"), t.add_column("built")
    for r in list_routes():
        t.add_row(r["id"], r["name"] or "", f"{r['origin']} → {r['destination']}", "yes" if r["id"] in built else "-")
    console.print(t)


@app.command("build-all")
def build_all(offline: bool = typer.Option(False), only: str | None = typer.Option(None, help="Comma-separated route ids"), skip_built: bool = typer.Option(False)):
    """Run the pipeline for every route in the catalogue (continues past failures; writes index.json)."""
    ids = [r["id"] for r in list_routes()]
    if only:
        wanted = {x.strip() for x in only.split(",") if x.strip()}
        unknown = sorted(wanted - set(ids))
        if unknown or not wanted:                              # a typo would otherwise build nothing and report success
            raise typer.BadParameter(f"unknown route id(s): {', '.join(unknown) or '(none given)'}; `tcs routes` lists them")
        ids = [i for i in ids if i in wanted]
    built = {p.parent.name for p in (ROOT / "web" / "public" / "data").glob("*/meta.json")}
    failures = []
    for rid in ids:
        if skip_built and rid in built:
            console.log(f"[dim]{rid}: already built, skipping")
            continue
        try:
            run_pipeline(offline=offline, route=rid)
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            console.log(f"[red]{rid} failed: {exc}")
            failures.append((rid, str(exc)))
    write_index()
    console.rule("build-all")
    console.log(f"{len(ids) - len(failures)}/{len(ids)} routes built" + (f"; failed: {', '.join(f for f, _ in failures)}" if failures else ""))
    if failures:
        raise typer.Exit(code=1)


@app.command()
def report(route: str | None = typer.Option(None, help="Route id (default: config/route.yaml)"), preset: str | None = typer.Option(None, help="Scenario preset, e.g. edge_rail_fleet_connect"),
           train: Path | None = typer.Option(None, help="Train Studio project (*.train.json)"), policy: str | None = typer.Option(None), weather: str = typer.Option("nominal"),
           out: Path | None = typer.Option(None, help="Output folder (default data/processed/<route>/reports)"),
           prepared_for: str | None = typer.Option(None, help="Client named on the cover (overrides config/report.yaml)"),
           prepared_by: str | None = typer.Option(None, help="Your organisation on the cover"),
           tender_ref: str | None = typer.Option(None, help="The client's tender reference"),
           classification: str | None = typer.Option(None, help="Classification in every page header"),
           doc_version: str | None = typer.Option(None, help="Document version, e.g. 1.0"),
           no_satcom: bool = typer.Option(False, "--no-satcom", help="Model the train without its satellite link")):
    """Tender evidence pack: Word report + Excel appendix for one route and scenario (re-simulates from the cached route data)."""
    info = {"prepared_for": prepared_for, "prepared_by": prepared_by, "tender_reference": tender_ref, "classification": classification, "version": doc_version}
    written = _build_report(route, preset, train, policy, weather, out, {k: v for k, v in info.items() if v is not None}, satcom=not no_satcom)
    for k, v in written.items():
        console.log(f"[bold]{k}[/bold] {v}")


def _build_report(route: str | None, preset: str | None, train: Path | None, policy: str | None, weather: str, out: Path | None,
                  info: dict[str, str] | None = None, satcom: bool = True) -> dict[str, Path]:
    from .model import calibration
    from .model.simulate import simulate
    from .pipeline.export import _lidar_meta
    from .pipeline.sample_route import load_bundle
    from .report import build_report

    s = load_settings(route_id=route)
    s.report.update(info or {})                                # cover details given on the command line win over config/report.yaml
    _require_built(s)
    interim, processed = _interim(s), s.paths()["processed"]
    label_parts = []
    variant = []                                               # what else sets this scenario apart, for the file name
    if preset:
        pr = _apply_preset(s, preset)
        label_parts.append(pr.get("label", preset))
    if train:
        from .sources.train_studio import apply_to_settings, load_design

        d = load_design(train, s)
        apply_to_settings(s, d)
        label_parts.append(f"train design '{d.title}'")
        variant.append(d.title)
    _check_overrides(s, policy, weather)
    if policy:
        s.sim["wan"]["policy"] = policy
        variant.append(policy)
    if weather != "nominal":
        label_parts.append(f"weather {weather}")
        variant.append(weather)
    if not satcom:
        s.sim["satcom_enabled"] = False
        label_parts.append("no satellite link")
        variant.append("no-satcom")
    if not label_parts:
        label_parts.append("baseline configuration")
    label = " · ".join(label_parts) + f" · policy {s.sim['wan']['policy'].lower().replace('_', ' ')} · {s.sim['vehicle']['profile'].lower().replace('_', ' ')}"
    b = load_bundle(interim, s.route["country"])
    prior = pd.read_parquet(interim / "coverage_prior.parquet")
    serving = pd.read_parquet(interim / "serving.parquet")
    stations = pd.read_parquet(processed / "stations.parquet")
    cal = calibration.for_route(s, interim)
    obs, rc = simulate(s, b.samples, prior, serving, calibration=cal, weather=weather)
    meta = json.loads((processed / "web" / "meta.json").read_text(encoding="utf-8"))
    meta["model_version"] = s.sim["model_version"]
    meta["calibration"] = calibration.describe(s, interim)     # as simulated now, not as when the bundle was built
    meta.update(_lidar_meta(b.samples))                        # likewise the LiDAR the samples carry
    vpath = processed / "validation_by_section.csv"
    validation = None
    if vpath.exists():                                         # only a validation of this build's predictions counts
        if vpath.stat().st_mtime >= (processed / "provider_observation.parquet").stat().st_mtime:
            validation = pd.read_csv(vpath)
        else:
            console.log(f"[yellow]{vpath.name} predates the current build of {s.route_id}; run tcs validate again to include it")
    baseline = None
    if preset or train or policy or weather != "nominal" or not satcom:   # the report compares this scenario with the baseline configuration
        from .report_docx import POLICIES, VEHICLES

        base = load_settings(route_id=route)
        obs_b, rc_b = simulate(base, b.samples, prior, serving, calibration=cal)
        v, p = base.sim["vehicle"]["profile"], base.sim["wan"]["policy"]
        baseline = {"title": f"{VEHICLES.get(v, v).lower()}, {POLICIES.get(p, p).split(' (')[0].lower()}", "obs": obs_b, "rc": rc_b}
    from .sensitivity import run as sensitivity

    sens = sensitivity(s, b.samples, prior, serving, calibration=cal, weather=weather)
    return build_report(s, meta, b.samples, obs, rc, stations, out or (processed / "reports"), label, validation, weather=weather, baseline=baseline,
                        sensitivity=sens, variant=variant)


REPORT_SCENARIOS = ("baseline", "edge_rail_fleet_connect")


@app.command("report-all")
def report_all(only: str | None = typer.Option(None, help="Comma-separated route ids (default: every built route)"),
               scenarios: str = typer.Option(",".join(REPORT_SCENARIOS), help="'baseline' and/or preset ids"),
               web_data: Path = typer.Option(ROOT / "web" / "public" / "data", help="Viewer data folder holding the route bundles")):
    """Evidence packs for every built route, placed beside its viewer bundle (<web-data>/<route>/reports/) with the
    index.json the viewer's Report button reads. Re-simulates from the cached route data: no API calls."""
    from .sources.base import now_iso

    ids = sorted(p.parent.name for p in web_data.glob("*/meta.json"))
    if only:
        ids = [i for i in ids if i in {x.strip() for x in only.split(",")}]
    wanted = [x.strip() for x in scenarios.split(",") if x.strip()]
    presets = load_settings().sim.get("presets", {})
    failures = []
    for rid in ids:
        out = web_data / rid / "reports"
        shutil.rmtree(out, ignore_errors=True)
        built = load_settings(route_id=rid).paths()["processed"] / "web" / "meta.json"
        if not built.is_file() or built.read_bytes() != (web_data / rid / "meta.json").read_bytes():
            # the pack re-simulates from data/processed: it must be the same build the viewer shows
            console.log(f"[red]{rid}: the viewer bundle and data/processed/{rid} come from different builds; rebuild the route with tcs run")
            failures += [f"{rid}/{sc}" for sc in wanted]
            continue
        items = []
        for sc in wanted:
            try:
                written = _build_report(rid, None if sc == "baseline" else sc, None, None, "nominal", out)
            except Exception as exc:  # noqa: BLE001 - keep going, report at the end
                console.log(f"[red]{rid} / {sc} failed: {exc}")
                failures.append(f"{rid}/{sc}")
                continue
            label = presets.get(sc, {}).get("label", "Baseline (config defaults)" if sc == "baseline" else sc)
            items.append({"scenario": sc, "label": label, **{kind: {"file": p.name, "bytes": p.stat().st_size} for kind, p in written.items()}})
        for png in out.glob("*.png"):
            png.unlink()   # the charts are already inside the Word report
        if items:
            (out / "index.json").write_text(json.dumps({"route": rid, "generated_at": now_iso(), "reports": items}, indent=1), encoding="utf-8")
            console.log(f"{rid}: {len(items)} report(s) -> {out}")
    console.rule("report-all")
    console.log(f"{len(ids) * len(wanted) - len(failures)}/{len(ids) * len(wanted)} reports built" + (f"; failed: {', '.join(failures)}" if failures else ""))
    if failures:
        raise typer.Exit(code=1)


@app.command("probe-ofcom")
def probe_ofcom(postcode: str = typer.Argument("N1C4TB")):
    """Print the raw Ofcom mobile coverage payload for a postcode (to confirm field names)."""
    from .sources.ofcom_coverage import probe

    s = load_settings()
    console.print_json(json.dumps(probe(postcode, s, s.paths()["raw"])))


@app.command()
def calibrate(measurements: Path = typer.Argument(..., help="CSV file"), route: str | None = typer.Option(None, help="Route id (default: config/route.yaml)"),
              preset: str = typer.Option("network_survey", help="network_survey|ofcom_drive|modem_log|yellow_train"), max_distance_m: float = typer.Option(250.0)):
    """Fit score->RSRP and per-operator bias from measurements attached to the route; saves calibration.json."""
    from .model import calibration
    from .model.cellular import cellular_observations
    from .pipeline.sample_route import load_bundle
    from .sources.measurements import attach_to_route, load_measurements, route_bbox

    s = load_settings(route_id=route)
    _require_built(s)
    interim = _interim(s)
    b = load_bundle(interim, s.route["country"])
    meas = load_measurements(measurements, preset, s.operators, bbox=route_bbox(b.samples))
    meas = attach_to_route(meas, b.samples, b.proj, max_distance_m=max_distance_m)
    # Fit against the uncalibrated model, not the last run's provider_observation: that already carries any earlier
    # calibration, so re-fitting it would measure only the residual and saving it would undo the first calibration.
    obs = cellular_observations(s, b.samples, pd.read_parquet(interim / "coverage_prior.parquet"), pd.read_parquet(interim / "serving.parquet"))
    obs = obs.merge(b.samples[["sample_id", "distance_m"]], on="sample_id")
    r = s.sim["cellular"]["rsrp_dbm"]
    cal = calibration.fit(obs, meas, nominal_rsrp=(float(r["at_zero"]), float(r["at_one"])))
    calibration.save(cal, interim / "calibration.json")
    console.print_json(json.dumps(cal))
    console.log(f"re-run `tcs run --route {s.route_id}` to apply the calibration")


@app.command()
def validate(measurements: Path = typer.Argument(...), route: str | None = typer.Option(None, help="Route id (default: config/route.yaml)"),
             preset: str = typer.Option("network_survey", help="network_survey|ofcom_drive|modem_log|yellow_train"), section_km: float = typer.Option(10.0)):
    """Predicted-vs-observed metrics by route section (MAE/RMSE, outage precision/recall, classification accuracy)."""
    from .pipeline.sample_route import load_bundle
    from .sources.measurements import attach_to_route, load_measurements, route_bbox
    from .validate.metrics import report

    s = load_settings(route_id=route)
    _require_built(s)
    b = load_bundle(_interim(s), s.route["country"])
    meas = attach_to_route(load_measurements(measurements, preset, s.operators, bbox=route_bbox(b.samples)), b.samples, b.proj)
    obs = pd.read_parquet(s.paths()["processed"] / "provider_observation.parquet").merge(b.samples[["sample_id", "distance_m"]], on="sample_id")
    rep = report(obs, meas, section_km=section_km)
    console.print(rep.to_string())
    rep.to_csv(s.paths()["processed"] / "validation_by_section.csv", index=False)


@app.command("calibrate-national")
def calibrate_national(measurements: Path = typer.Argument(..., help="Measurement file covering many routes (CSV or parquet)"),
                       preset: str = typer.Option("yellow_train", help="network_survey|ofcom_drive|modem_log|yellow_train"),
                       split: str = typer.Option("2019-01-01", help="Fit on measurements before this date, test on those from it"),
                       only: str | None = typer.Option(None, help="Comma-separated route ids (default: every built route)"),
                       max_distance_m: float = typer.Option(50.0, help="Farthest a measurement may be from the track"),
                       source: str = typer.Option("Network Rail Yellow Train LTE measurements (Rail Data Marketplace)", help="Named in reports"),
                       out: Path = typer.Option(ROOT / "config" / "calibration.yaml")):
    """Fit the cellular model to measurements pooled over every built route and test it on later ones; writes
    config/calibration.yaml (parameters, provenance, accuracy). Every route without its own calibration then uses it."""
    from . import national

    ids = [r["id"] for r in list_routes()]
    if only:
        ids = [i for i in ids if i in {x.strip() for x in only.split(",")}]
    doc = national.run(ids, measurements, preset, split, max_distance_m=max_distance_m, log=console.log)
    doc = national.write(doc, out, source=source, file=measurements.name, preset=preset)
    v = doc["validation"]
    for label, key in (("before calibration", "uncalibrated"), ("calibrated", "calibrated")):
        a = v[key]["overall"]
        console.log(f"{label}: {a['points']:,} test points, bias {a['bias_db']:+.1f} dB, MAE {a['mae_db']:.1f} dB, within 6 dB {a['within_6db']:.0%}")
    for tc in [x for x in doc.get("in_tunnel_coverage") or [] if not x["listed"]]:
        console.log(f"[yellow]{tc['route']}: {tc['tunnel'] or 'tunnel'} at km {tc['km'][0]}-{tc['km'][1]} measured {tc['median_dbm']} dBm deep inside "
                    f"({tc['points']} points): in-tunnel coverage; add it to the route's das_tunnels and re-run")
    env, cur = doc["environment"], national.environment_of(load_settings())
    if env != cur:
        console.log(f"[yellow]set these in config/simulation.yaml (cellular.terrain / cellular.tunnels / cellular.cell_distance): {env} (now {cur})")
    console.log(f"wrote {out}")


@app.command("locate-masts")
def locate_masts(measurements: Path = typer.Argument(..., help="Scanner logs naming the cell behind each reading, e.g. Global View 4G (CSV or parquet)"),
                 preset: str = typer.Option("global_view_4g"),
                 before: str | None = typer.Option(None, help="Use only readings before this date (to test the masts on later trips)"),
                 source: str = typer.Option("Network Rail Global View 4G measurements (Rail Data Marketplace)", help="Named in the file"),
                 workers: int | None = typer.Option(None, help="Processes (default: all cores but one)"),
                 out: Path = typer.Option(ROOT / "config" / "masts.csv")):
    """Place each network's 4G masts from how their signal rises and falls along the track (tcs/masts.py); writes
    config/masts.csv (positions only). Built routes then use these positions for serving-cell distance and handovers
    in place of OpenCellID's for the same masts: re-run the routes, then `tcs calibrate-national`."""
    from . import masts

    s = load_settings()
    r = masts.readings(measurements, s.operators, preset=preset, before=before)
    if r.empty:
        raise typer.BadParameter(f"no readings in {measurements} name their cell (E-UTRAN cell identity)")
    m = masts.locate(r, workers=workers, log=console.log)
    period = (str(pd.Timestamp(r["timestamp"].min()).date()), str(pd.Timestamp(r["timestamp"].max()).date()))
    masts.save(m, out, source=source, period=period)
    console.log(f"wrote {len(m):,} masts to {out} ({', '.join(f'{k} {v:,}' for k, v in m['network'].value_counts().items())})")


@app.command("check-national")
def check_national(four_g: Path = typer.Option(..., "--four-g", help="Later 4G measurements covering many routes (CSV or parquet)"),
                   five_g: Path | None = typer.Option(None, "--five-g", help="5G measurements from the same trains (CSV or parquet)"),
                   preset_4g: str = typer.Option("global_view_4g"), preset_5g: str = typer.Option("global_view_5g"),
                   split: str = typer.Option(..., help="Fit the level offset before this date, test from it"),
                   source: str = typer.Option("Network Rail Global View measurements (Rail Data Marketplace)", help="Named in reports"),
                   only: str | None = typer.Option(None, help="Comma-separated route ids (default: every built route)"),
                   out: Path = typer.Option(ROOT / "config" / "calibration.yaml")):
    """Check the calibrated model against later measurements (e.g. the 2026 Global View 4G logs) and summarise where 5G
    was measured; adds `current_check` and `five_g` to config/calibration.yaml for the reports and the viewer. The
    calibration itself is unchanged: a later set whose level differs (a scanner logging before its antenna and cable
    corrections) checks the pattern of strong and weak signal, after one level offset."""
    from . import national

    if not out.exists():
        raise typer.BadParameter(f"{out} does not exist: run `tcs calibrate-national` first")
    ids = [r["id"] for r in list_routes()]
    if only:
        ids = [i for i in ids if i in {x.strip() for x in only.split(",")}]
    sections = {"current_check": {"source": source, "file": four_g.name, **national.check(ids, four_g, preset_4g, split, log=console.log)}}
    if five_g:
        sections["five_g"] = {"source": source, "file": five_g.name, **national.five_g(ids, four_g, five_g, preset_4g, preset_5g, log=console.log)}
    national.save_checks(out, **sections)
    console.log(f"wrote {', '.join(sections)} to {out}")


@app.command()
def sources():
    """Show which live sources are configured (keys present) and which will fall back."""
    s = load_settings()
    t = Table(title="Data sources")
    t.add_column("source"), t.add_column("status"), t.add_column("fallback")
    t.add_row("OSM Overpass (route, stations, tunnels/cuttings)", "keyless", "synthetic spline (offline)")
    t.add_row("Copernicus DEM GLO-30 (terrain, horizon)", "keyless", "procedural terrain")
    t.add_row("EA / Welsh Government / Scottish LiDAR (cuttings, trees, bridges; GB)", "keyless" if (s.terrain.get("lidar") or {}).get("enabled", True)
              else "[yellow]off (terrain.lidar.enabled)", "the 30 m terrain model")
    t.add_row("OS Code-Point Open (corridor postcodes)", "keyless", "-")
    t.add_row("Ofcom API mobile coverage", "key set" if s.key("OFCOM_API_KEY") else "[yellow]OFCOM_API_KEY missing", "Connected Nations open data → synthetic prior")
    t.add_row("OpenCellID bulk", "token set" if s.key("OPENCELLID_TOKEN") else "[yellow]OPENCELLID_TOKEN missing", "synthetic cell sites")
    t.add_row("Starlink telemetry", "file" if s.satcom_providers and s.satcom_providers[0]["telemetry"].get("file") else "predictive only", "-")
    console.print(t)


if __name__ == "__main__":
    app()
